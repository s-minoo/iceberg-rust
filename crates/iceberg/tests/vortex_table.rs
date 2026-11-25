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

//! Vortex-RDF data files written, committed and planned through an Iceberg table.

#![cfg(feature = "vortex")]

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::arrow::{ArrowReaderOptions, schema_to_arrow_schema};
use iceberg::expr::{Predicate, Reference};
use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
use iceberg::scan::FileScanTask;
use iceberg::spec::{DataFileFormat, Datum, NestedField, PrimitiveType, Schema, Type};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::VortexWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use vortex_rdf_core::VortexRdfStore;

const ALICE: [&str; 4] = [
    "<http://ex.org/alice>",
    "<http://ex.org/knows>",
    "<http://ex.org/bob>",
    "",
];
const BOB: [&str; 4] = [
    "<http://ex.org/bob>",
    "<http://ex.org/name>",
    "\"Bob\"",
    "<http://ex.org/g>",
];

fn quad_schema() -> Schema {
    let fields = ["s", "p", "o", "g"]
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            NestedField::required(i as i32 + 1, name, Type::Primitive(PrimitiveType::String)).into()
        })
        .collect::<Vec<_>>();
    Schema::builder().with_fields(fields).build().unwrap()
}

fn batch(schema: &Schema, rows: &[[&str; 4]]) -> RecordBatch {
    let columns = (0..4)
        .map(|c| Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r[c]))) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(schema_to_arrow_schema(schema).unwrap()), columns).unwrap()
}

/// A table holding one committed Vortex-RDF file of `ALICE`, `BOB` and a duplicate `ALICE`.
async fn table_with_vortex_file(warehouse: &str) -> Table {
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "memory",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse.to_string())]),
        )
        .await
        .unwrap();
    let namespace = NamespaceIdent::new("rdf".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("quads".to_string())
                .schema(quad_schema())
                .build(),
        )
        .await
        .unwrap();

    let schema = table.metadata().current_schema().clone();
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        VortexWriterBuilder::new(schema.clone(), vec![]),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        DefaultFileNameGenerator::new("data".to_string(), None, DataFileFormat::Vortex),
    );
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
    writer.write(batch(&schema, &[ALICE, BOB])).await.unwrap();
    writer.write(batch(&schema, &[ALICE])).await.unwrap();
    let data_files = writer.close().await.unwrap();

    let tx = Transaction::new(&table);
    let tx = tx.fast_append().add_data_files(data_files).apply(tx).unwrap();
    tx.commit(&catalog).await.unwrap()
}

async fn plan(table: &Table, filter: Option<Predicate>) -> Vec<FileScanTask> {
    let mut scan = table.scan();
    if let Some(filter) = filter {
        scan = scan.with_filter(filter);
    }
    scan.build()
        .unwrap()
        .plan_files()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap()
}

fn subject_is(term: &str) -> Option<Predicate> {
    Some(Reference::new("s").equal_to(Datum::string(term)))
}

#[tokio::test]
async fn test_committed_vortex_file_is_planned_and_readable() {
    let warehouse = tempfile::tempdir().unwrap();
    let table = table_with_vortex_file(warehouse.path().to_str().unwrap()).await;

    let tasks = plan(&table, None).await;
    assert_eq!(tasks.len(), 1);
    let task = &tasks[0];
    assert_eq!(
        (
            task.data_file_format(),
            task.data_file_path().ends_with(".vortex"),
            task.record_count(),
        ),
        (DataFileFormat::Vortex, true, Some(2))
    );

    let bytes = table
        .file_io()
        .new_input(task.data_file_path())
        .unwrap()
        .read()
        .await
        .unwrap();
    let store = VortexRdfStore::from_bytes_owned(bytes.to_vec())
        .await
        .unwrap();
    let mut subjects: Vec<String> = store
        .quads_vec()
        .await
        .unwrap()
        .iter()
        .map(|q| q.subject.to_string())
        .collect();
    subjects.sort();
    assert_eq!(subjects, [ALICE[0], BOB[0]]);
}

#[tokio::test]
async fn test_scan_prunes_vortex_file_by_term_bounds() {
    let warehouse = tempfile::tempdir().unwrap();
    let table = table_with_vortex_file(warehouse.path().to_str().unwrap()).await;

    assert_eq!(plan(&table, subject_is(BOB[0])).await.len(), 1);
    assert_eq!(plan(&table, subject_is("<http://ex.org/zzz>")).await.len(), 0);
    assert_eq!(plan(&table, subject_is("<http://aaa>")).await.len(), 0);
}

#[tokio::test]
async fn test_arrow_scan_of_vortex_file_errors() {
    let warehouse = tempfile::tempdir().unwrap();
    let table = table_with_vortex_file(warehouse.path().to_str().unwrap()).await;

    let scan = table.scan().build().unwrap();
    let result = match scan.to_arrow(ArrowReaderOptions::default()).await {
        Ok(stream) => stream.try_collect::<Vec<_>>().await.map(|_| ()),
        Err(err) => Err(err),
    };
    assert!(result.is_err());
}
