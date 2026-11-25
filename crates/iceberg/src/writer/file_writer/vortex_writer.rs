// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The writer for Vortex-RDF data files.
//!
//! The table schema must carry the quad terms as required string fields
//! `s`, `p`, `o` and `g`, each holding an N-Triples term spelling; the default
//! graph is `""`, never null.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::SchemaRef as ArrowSchemaRef;
use bytes::Bytes;
use vortex_rdf_core::{ArrowBuilder, Indexes, LayoutStrategy, VortexRdfError, VortexRdfStore};

use super::{FileWriter, FileWriterBuilder};
use crate::io::OutputFile;
use crate::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, Datum, NestedField, PrimitiveType, Schema,
    SchemaRef, Struct, Type,
};
use crate::writer::CurrentFileStatus;
use crate::{Error, ErrorKind, Result};

/// The only layout vortex-rdf's Arrow builder supports so far.
const LAYOUT: LayoutStrategy = LayoutStrategy::Dictionary;

/// The quad term columns, in the order vortex-rdf's code columns come in.
const QUAD_COLUMNS: [&str; 4] = ["s", "p", "o", "g"];

/// Builds a [`VortexWriter`] per output file.
#[derive(Debug, Clone)]
pub struct VortexWriterBuilder {
    schema: SchemaRef,
    indexes: Indexes,
}

impl VortexWriterBuilder {
    /// A builder for files of `schema`'s quads, with the vortex-rdf `indexes` built into each.
    pub fn new(schema: SchemaRef, indexes: Indexes) -> Self {
        Self { schema, indexes }
    }
}

impl FileWriterBuilder for VortexWriterBuilder {
    type R = VortexWriter;

    async fn build(&self, output_file: OutputFile) -> Result<Self::R> {
        Ok(VortexWriter {
            indexes: self.indexes.clone(),
            output_file,
            batches: Vec::new(),
            current_row_num: 0,
            buffered_bytes: 0,
        })
    }
}

/// Writes arrow record batches of quads into one Vortex-RDF file.
///
/// A vortex-rdf store is built from all its quads at once (sorted, dictionary
/// encoded), so batches are buffered and the file is written on [`close`](FileWriter::close).
#[derive(Debug)]
pub struct VortexWriter {
    indexes: Indexes,
    output_file: OutputFile,
    batches: Vec<RecordBatch>,
    current_row_num: usize,
    buffered_bytes: usize,
}

impl VortexWriter {
    /// Vortex files' logical schema consists of 4 columns s, p, o, g
    /// with each column typed to be a [Primitive String](PrimitiveType::String)
    /// without null values
    pub fn schema() -> SchemaRef {
        let fields = QUAD_COLUMNS
            .iter()
            .enumerate()
            .map(|(i, name)| {
                NestedField::required(i as i32 + 1, name, Type::Primitive(PrimitiveType::String))
                    .into()
            })
            .collect::<Vec<_>>();
        Arc::new(Schema::builder().with_fields(fields).build().unwrap())
    }
}

impl FileWriter for VortexWriter {
    async fn write(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.current_row_num += batch.num_rows();
        self.buffered_bytes += batch.get_array_memory_size();
        self.batches.push(batch.clone());
        Ok(())
    }

    async fn close(self) -> Result<Vec<DataFileBuilder>> {
        if self.batches.is_empty() {
            return Ok(vec![]);
        }

        let schema = VortexWriter::schema();
        let arrow_schema: ArrowSchemaRef = Arc::new(schema.as_ref().try_into()?);
        let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
            self.batches.into_iter().map(Ok),
            arrow_schema,
        ));
        let built = reader
            .into_vortex_array(LAYOUT, self.indexes)
            .await
            .map_err(vortex_err)?;
        let store = VortexRdfStore::from_built(built).map_err(vortex_err)?;
        let record_count = store.size().await.map_err(vortex_err)? as u64;
        let (lower_bounds, upper_bounds) = term_bounds(schema, &store).await?;
        let bytes = store.to_bytes().await.map_err(vortex_err)?;
        let file_size = bytes.len();

        self.output_file.write(Bytes::from(bytes)).await?;

        let field_ids = lower_bounds.keys().copied();
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(self.output_file.location().to_string())
            .file_format(DataFileFormat::Vortex)
            .partition(Struct::empty())
            .record_count(record_count)
            .file_size_in_bytes(file_size as u64)
            .value_counts(field_ids.clone().map(|id| (id, record_count)).collect())
            .null_value_counts(field_ids.map(|id| (id, 0)).collect())
            .lower_bounds(lower_bounds)
            .upper_bounds(upper_bounds);
        Ok(vec![builder])
    }
}

/// The smallest and largest term of each of `store`'s `s`, `p`, `o`, `g` columns, keyed by field id.
///
/// Term codes are ranks in the store's dictionary, which is sorted by the terms' bytes (the order Iceberg
/// compares strings in), so each column's smallest and largest code decode to its bounds.
async fn term_bounds(
    schema: SchemaRef,
    store: &VortexRdfStore,
) -> Result<(HashMap<i32, Datum>, HashMap<i32, Datum>)> {
    let unexpected = |msg: &str| Error::new(ErrorKind::Unexpected, msg.to_string());
    let columns = store
        .code_columns_gathered()
        .await
        .map_err(vortex_err)?
        .ok_or_else(|| unexpected("Vortex-RDF store has no dictionary code columns."))?;
    let dict = store
        .dict_reader()
        .ok_or_else(|| unexpected("Vortex-RDF store has no term dictionary."))?;

    let mut lower = HashMap::new();
    let mut upper = HashMap::new();
    for (name, codes) in QUAD_COLUMNS.into_iter().zip(columns) {
        let (Some(&min), Some(&max)) = (codes.iter().min(), codes.iter().max()) else {
            continue;
        };
        let field_id = schema
            .field_by_name(name)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Vortex-RDF schema has no {name:?} field."),
                )
            })?
            .id;

        let terms_res: Result<Vec<String>> = dict
            .decode_many(&[min, max])
            .await
            .map_err(vortex_err)?
            .into_iter()
            .map(|term| term.ok_or_else(|| unexpected("Vortex-RDF term code is out of range.")))
            .collect();
        let mut terms = terms_res?.into_iter();

        lower.insert(field_id, Datum::string(terms.next().unwrap()));
        upper.insert(field_id, Datum::string(terms.next().unwrap()));
    }
    Ok((lower, upper))
}

impl CurrentFileStatus for VortexWriter {
    fn current_file_path(&self) -> String {
        self.output_file.location().to_string()
    }

    fn current_row_num(&self) -> usize {
        self.current_row_num
    }

    /// The in-memory size of the buffered batches: the file is only encoded on close.
    fn current_written_size(&self) -> usize {
        self.buffered_bytes
    }
}

fn vortex_err(err: VortexRdfError) -> Error {
    Error::new(ErrorKind::DataInvalid, "Failed to build vortex-rdf file.").with_source(err)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use vortex_rdf_core::VortexRdfStore;

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    fn quad_schema() -> SchemaRef {
        let fields = ["s", "p", "o", "g"]
            .into_iter()
            .enumerate()
            .map(|(i, name)| {
                NestedField::required(i as i32 + 1, name, Type::Primitive(PrimitiveType::String))
                    .into()
            })
            .collect::<Vec<_>>();
        Arc::new(Schema::builder().with_fields(fields).build().unwrap())
    }

    fn batch(schema: &SchemaRef, rows: &[[Option<&str>; 4]]) -> RecordBatch {
        let columns = (0..4)
            .map(|c| Arc::new(rows.iter().map(|r| r[c]).collect::<StringArray>()) as ArrayRef)
            .collect();
        let arrow_schema = schema_to_arrow_schema(schema).unwrap();
        RecordBatch::try_new(Arc::new(arrow_schema), columns).unwrap()
    }

    async fn writer(file_io: &FileIO, schema: &SchemaRef) -> VortexWriter {
        let output = file_io.new_output("memory:///data/quads.vortex").unwrap();
        VortexWriterBuilder::new(schema.clone(), vec![])
            .build(output)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_vortex_writer_round_trip() {
        let file_io = FileIO::new_with_memory();
        let schema = quad_schema();
        let alice = [
            Some("<http://ex.org/alice>"),
            Some("<http://ex.org/knows>"),
            Some("<http://ex.org/bob>"),
            Some(""),
        ];
        let bob = [
            Some("<http://ex.org/bob>"),
            Some("<http://ex.org/name>"),
            Some("\"Bob\""),
            Some("<http://ex.org/g>"),
        ];

        let mut writer = writer(&file_io, &schema).await;
        writer.write(&batch(&schema, &[alice, bob])).await.unwrap();
        writer.write(&batch(&schema, &[alice])).await.unwrap();
        assert_eq!(writer.current_row_num(), 3);

        let mut builders = writer.close().await.unwrap();
        assert_eq!(builders.len(), 1);
        let data_file = builders
            .pop()
            .unwrap()
            .partition_spec_id(0)
            .build()
            .unwrap();
        assert_eq!(data_file.file_format(), DataFileFormat::Vortex);

        let bytes = file_io
            .new_input(data_file.file_path())
            .unwrap()
            .read()
            .await
            .unwrap();
        assert_eq!(data_file.file_size_in_bytes(), bytes.len() as u64);
        let store = VortexRdfStore::from_bytes_owned(bytes.to_vec())
            .await
            .unwrap();
        let quads = store.quads_vec().await.unwrap();
        assert_eq!(data_file.record_count(), quads.len() as u64);
        // vortex-rdf stores are quad sets: the duplicate `alice` quad is stored once.
        assert_eq!(quads.len(), 2);

        let ids = 1..=4;
        let bounds = |terms: [&str; 4]| {
            ids.clone()
                .zip(terms)
                .map(|(id, term)| (id, Datum::string(term)))
                .collect::<HashMap<_, _>>()
        };
        assert_eq!(
            (
                data_file.value_counts(),
                data_file.null_value_counts(),
                data_file.lower_bounds(),
                data_file.upper_bounds(),
            ),
            (
                &ids.clone().map(|id| (id, 2)).collect(),
                &ids.clone().map(|id| (id, 0)).collect(),
                &bounds([
                    "<http://ex.org/alice>",
                    "<http://ex.org/knows>",
                    "\"Bob\"",
                    "",
                ]),
                &bounds([
                    "<http://ex.org/bob>",
                    "<http://ex.org/name>",
                    "<http://ex.org/bob>",
                    "<http://ex.org/g>",
                ]),
            )
        );
    }

    #[tokio::test]
    async fn test_vortex_writer_empty_writes_no_file() {
        let file_io = FileIO::new_with_memory();
        let writer = writer(&file_io, &quad_schema()).await;
        assert!(writer.close().await.unwrap().is_empty());
        assert!(!file_io.exists("memory:///data/quads.vortex").await.unwrap());
    }
}
