// Copyright 2022 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    collections::HashMap,
    sync::{atomic::AtomicU64, mpsc, Arc},
    thread,
    time::Duration,
};

use api_version::{ApiV2, KvFormat, RawValue};
use dashmap::DashMap;
use engine_rocks::{raw::FlushOptions, util::get_cf_handle, RocksEngine};
use engine_traits::{CF_DEFAULT, CF_WRITE};
use keys::DATA_PREFIX_KEY;
use keyspace_meta::KeyspaceLevelGCService;
use kvproto::{
    keyspacepb,
    kvrpcpb::*,
    metapb::{Peer, Region},
};
use pd_client::FeatureGate;
use raft::StateRole;
use raftstore::{
    coprocessor::{
        region_info_accessor::MockRegionInfoProvider, CoprocessorHost, RegionChangeEvent,
    },
    RegionInfoAccessor,
};
use tikv::{
    config::DbConfig,
    server::gc_worker::{
        compaction_filter::{
            GC_COMPACTION_FILTERED, GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED,
            GC_COMPACTION_FILTER_MVCC_DELETION_MET, GC_COMPACTION_FILTER_PERFORM,
            GC_COMPACTION_FILTER_SKIP,
        },
        rawkv_compaction_filter::make_key,
        AutoGcConfig, GcConfig, GcWorker, MockSafePointProvider, PrefixedEngine, TestGcRunner,
        STAT_RAW_KEYMODE, STAT_TXN_KEYMODE,
    },
    storage::{
        kv::{Modify, TestEngineBuilder, WriteData},
        mvcc::{tests::must_get, MVCC_VERSIONS_HISTOGRAM},
        txn::tests::{must_commit, must_prewrite_delete, must_prewrite_put},
        Engine,
    },
};
use txn_types::{Key, TimeStamp};

#[test]
fn test_txn_create_compaction_filter() {
    GC_COMPACTION_FILTER_PERFORM.reset();
    GC_COMPACTION_FILTER_SKIP.reset();

    let mut cfg = DbConfig::default();
    cfg.writecf.disable_auto_compactions = true;
    cfg.writecf.dynamic_level_bytes = false;
    let dir = tempfile::TempDir::new().unwrap();
    let builder = TestEngineBuilder::new().path(dir.path());
    let mut engine = builder.build_with_cfg(&cfg).unwrap();
    let raw_engine = engine.get_rocksdb();

    let mut gc_runner = TestGcRunner::new(0);
    let value = vec![b'v'; 512];

    must_prewrite_put(&mut engine, b"zkey", &value, b"zkey", 100);
    must_commit(&mut engine, b"zkey", 100, 110);

    gc_runner
        .safe_point(TimeStamp::new(1).into_inner())
        .gc(&raw_engine);
    assert_eq!(
        GC_COMPACTION_FILTER_PERFORM
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get(),
        1
    );
    assert_eq!(
        GC_COMPACTION_FILTER_SKIP
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get(),
        1
    );

    GC_COMPACTION_FILTER_PERFORM.reset();
    GC_COMPACTION_FILTER_SKIP.reset();
}

fn make_keypsace_txnkv_key(keyspace_id: u32, user_key: Vec<u8>) -> Vec<u8> {
    let mut combined_vec = Vec::from(DATA_PREFIX_KEY);
    let keyspace_txnkv_prefix: Vec<u8> = ApiV2::get_keyspace_id_to_txnkv_prefix(keyspace_id);
    combined_vec.extend_from_slice(&keyspace_txnkv_prefix);
    combined_vec.extend_from_slice(&user_key);
    combined_vec
}

fn make_combined_key(mut a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    a.extend_from_slice(&b);
    a
}

#[test]
fn test_txn_mvcc_filtered_v2() {
    let combined_vec = Vec::from(DATA_PREFIX_KEY);
    let user_key = b"key";
    let api_v1_mvcc_key = make_combined_key(combined_vec, user_key.to_vec());
    test_txn_mvcc_filtered(None, api_v1_mvcc_key);
    let keyspace_txnkv_mvcc_key = make_keypsace_txnkv_key(1, user_key.to_vec());
    test_txn_mvcc_filtered(Some(1), keyspace_txnkv_mvcc_key);
}

fn test_txn_mvcc_filtered(keyspace_id: Option<u32>, key: Vec<u8>) {
    MVCC_VERSIONS_HISTOGRAM.reset();
    GC_COMPACTION_FILTERED.reset();

    let mut engine = TestEngineBuilder::new().build().unwrap();
    let raw_engine = engine.get_rocksdb();
    let value = vec![b'v'; 512];
    let mut gc_runner = TestGcRunner::new(0);
    gc_runner.keyspace_level_gc_service = make_keyspace_level_gc_service().clone();

    // GC can't delete keys after the given safe point.
    must_prewrite_put(&mut engine, key.as_slice(), &value, key.as_slice(), 100);
    must_commit(&mut engine, key.as_slice(), 100, 110);

    gc_runner
        .update_gc_safe_point(keyspace_id, 50)
        .gc(&raw_engine);
    must_get(&mut engine, key.as_slice(), 110, &value);

    // GC can't delete keys before the safe ponit if they are latest versions.
    gc_runner
        .update_gc_safe_point(keyspace_id, 200)
        .gc(&raw_engine);
    must_get(&mut engine, key.as_slice(), 110, &value);

    must_prewrite_put(&mut engine, key.as_slice(), &value, key.as_slice(), 120);
    must_commit(&mut engine, key.as_slice(), 120, 130);

    // GC can't delete the latest version before the safe ponit.
    gc_runner
        .update_gc_safe_point(keyspace_id, 115)
        .gc(&raw_engine);
    must_get(&mut engine, key.as_slice(), 110, &value);

    // GC a version will also delete the key on default CF.
    gc_runner
        .update_gc_safe_point(keyspace_id, 200)
        .gc(&raw_engine);
    assert_eq!(
        MVCC_VERSIONS_HISTOGRAM
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get_sample_sum(),
        4_f64
    );
    assert_eq!(
        GC_COMPACTION_FILTERED
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get(),
        1
    );

    MVCC_VERSIONS_HISTOGRAM.reset();
    GC_COMPACTION_FILTERED.reset();
}

// make_keyspace_level_gc_service is used to construct the required keyspace
// metas, mappings, and keyspace level GC service.
fn make_keyspace_level_gc_service() -> Arc<KeyspaceLevelGCService> {
    let mut keyspace_config = HashMap::new();
    keyspace_config.insert(
        keyspace_meta::KEYSPACE_CONFIG_KEY_GC_MGMT_TYPE.to_string(),
        keyspace_meta::GC_MGMT_TYPE_KEYSPACE_LEVEL_GC.to_string(),
    );

    // Init keyspace_1 and keyspace_2.
    let keyspace_1_meta = keyspacepb::KeyspaceMeta {
        id: 1,
        name: "test_keyspace_1".to_string(),
        state: Default::default(),
        created_at: 0,
        state_changed_at: 0,
        config: keyspace_config.clone(),
        unknown_fields: Default::default(),
        cached_size: Default::default(),
    };

    let keyspace_2_meta = keyspacepb::KeyspaceMeta {
        id: 2,
        name: "test_keyspace_2".to_string(),
        state: Default::default(),
        created_at: 0,
        state_changed_at: 0,
        config: keyspace_config,
        unknown_fields: Default::default(),
        cached_size: Default::default(),
    };

    // Init keyspace level GC cache.
    let keyspace_level_gc_map = DashMap::new();
    // make data ts < props.min_ts
    keyspace_level_gc_map.insert(1_u32, 60_u64);
    keyspace_level_gc_map.insert(2_u32, 69_u64);

    let keyspace_level_gc_map = Arc::new(keyspace_level_gc_map);

    // Init the mapping from keyspace id to keyspace meta.
    let keyspace_id_meta_map = DashMap::new();

    // Make data ts < props.min_ts( props.min_ts = 70).
    keyspace_id_meta_map.insert(keyspace_1_meta.id, keyspace_1_meta);
    keyspace_id_meta_map.insert(keyspace_2_meta.id, keyspace_2_meta);

    let keyspace_id_meta_cache = Arc::new(keyspace_id_meta_map);

    Arc::new(KeyspaceLevelGCService::new(
        Arc::clone(&keyspace_level_gc_map),
        Arc::clone(&keyspace_id_meta_cache),
    ))
}

#[test]
fn test_txn_gc_keys_handled() {
    let store_id = 1;
    GC_COMPACTION_FILTER_MVCC_DELETION_MET.reset();
    GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED.reset();

    let engine = TestEngineBuilder::new().build().unwrap();
    let mut prefixed_engine = PrefixedEngine(engine.clone());

    let (tx, _rx) = mpsc::channel();
    let feature_gate = FeatureGate::default();
    feature_gate.set_version("5.0.0").unwrap();
    let mut gc_worker = GcWorker::new(
        prefixed_engine.clone(),
        tx,
        GcConfig::default(),
        feature_gate,
        Arc::new(MockRegionInfoProvider::new(vec![])),
        Arc::new(KeyspaceLevelGCService::default()),
    );
    gc_worker.start(store_id).unwrap();

    let mut r1 = Region::default();
    r1.set_id(1);
    r1.mut_region_epoch().set_version(1);
    r1.set_start_key(b"".to_vec());
    r1.set_end_key(b"".to_vec());
    r1.mut_peers().push(Peer::default());
    r1.mut_peers()[0].set_store_id(store_id);

    let sp_provider = MockSafePointProvider(200);
    let mut host = CoprocessorHost::<RocksEngine>::default();
    let ri_provider = RegionInfoAccessor::new(&mut host, Arc::new(|| false));
    let auto_gc_cfg = AutoGcConfig::new(sp_provider, ri_provider, 1);
    let safe_point = Arc::new(AtomicU64::new(500));

    gc_worker.start_auto_gc(auto_gc_cfg, safe_point).unwrap();
    host.on_region_changed(&r1, RegionChangeEvent::Create, StateRole::Leader);

    let db = engine.kv_engine().unwrap().as_inner().clone();
    let cf = get_cf_handle(&db, CF_WRITE).unwrap();

    for i in 0..3 {
        let k = format!("k{:02}", i).into_bytes();
        must_prewrite_put(&mut prefixed_engine, &k, b"value", &k, 101);
        must_commit(&mut prefixed_engine, &k, 101, 102);
        must_prewrite_delete(&mut prefixed_engine, &k, &k, 151);
        must_commit(&mut prefixed_engine, &k, 151, 152);
    }

    let mut fopts = FlushOptions::default();
    fopts.set_wait(true);
    db.flush_cf(cf, &fopts).unwrap();

    db.compact_range_cf(cf, None, None);

    // This compaction can schedule gc task
    db.compact_range_cf(cf, None, None);
    thread::sleep(Duration::from_millis(100));

    assert_eq!(
        GC_COMPACTION_FILTER_MVCC_DELETION_MET
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get(),
        6
    );

    assert_eq!(
        GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED
            .with_label_values(&[STAT_TXN_KEYMODE])
            .get(),
        3
    );

    GC_COMPACTION_FILTER_MVCC_DELETION_MET.reset();
    GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED.reset();
}

#[test]
fn test_raw_mvcc_filtered() {
    MVCC_VERSIONS_HISTOGRAM.reset();
    GC_COMPACTION_FILTERED.reset();

    let mut cfg = DbConfig::default();
    cfg.defaultcf.disable_auto_compactions = true;
    cfg.defaultcf.dynamic_level_bytes = false;

    let engine = TestEngineBuilder::new()
        .api_version(ApiVersion::V2)
        .build_with_cfg(&cfg)
        .unwrap();
    let raw_engine = engine.get_rocksdb();
    let mut gc_runner = TestGcRunner::new(0);

    let user_key = b"r\0aaaaaaaaaaa";

    let test_raws = vec![
        (user_key, 100, false),
        (user_key, 90, false),
        (user_key, 70, false),
    ];

    let modifies = test_raws
        .into_iter()
        .map(|(key, ts, is_delete)| {
            (
                make_key(key, ts),
                ApiV2::encode_raw_value(RawValue {
                    user_value: &[0; 10][..],
                    expire_ts: Some(TimeStamp::max().into_inner()),
                    is_delete,
                }),
            )
        })
        .map(|(k, v)| Modify::Put(CF_DEFAULT, Key::from_encoded_slice(k.as_slice()), v))
        .collect();

    let ctx = Context {
        api_version: ApiVersion::V2,
        ..Default::default()
    };
    let batch = WriteData::from_modifies(modifies);

    engine.write(&ctx, batch).unwrap();

    gc_runner.safe_point(80).gc_raw(&raw_engine);

    assert_eq!(
        MVCC_VERSIONS_HISTOGRAM
            .with_label_values(&[STAT_RAW_KEYMODE])
            .get_sample_sum(),
        1_f64
    );
    assert_eq!(
        GC_COMPACTION_FILTERED
            .with_label_values(&[STAT_RAW_KEYMODE])
            .get(),
        1
    );

    MVCC_VERSIONS_HISTOGRAM.reset();
    GC_COMPACTION_FILTERED.reset();
}

#[test]
fn test_raw_gc_keys_handled() {
    let store_id = 1;
    GC_COMPACTION_FILTER_MVCC_DELETION_MET.reset();
    GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED.reset();

    let engine = TestEngineBuilder::new()
        .api_version(ApiVersion::V2)
        .build()
        .unwrap();
    let prefixed_engine = PrefixedEngine(engine.clone());

    let (tx, _rx) = mpsc::channel();
    let feature_gate = FeatureGate::default();
    let mut gc_worker = GcWorker::new(
        prefixed_engine,
        tx,
        GcConfig::default(),
        feature_gate,
        Arc::new(MockRegionInfoProvider::new(vec![])),
        Arc::new(KeyspaceLevelGCService::default()),
    );
    gc_worker.start(store_id).unwrap();

    let mut r1 = Region::default();
    r1.set_id(1);
    r1.mut_region_epoch().set_version(1);
    r1.set_start_key(b"".to_vec());
    r1.set_end_key(b"".to_vec());
    r1.mut_peers().push(Peer::default());
    r1.mut_peers()[0].set_store_id(store_id);

    let sp_provider = MockSafePointProvider(200);
    let mut host = CoprocessorHost::<RocksEngine>::default();
    let ri_provider = RegionInfoAccessor::new(&mut host, Arc::new(|| false));
    let auto_gc_cfg = AutoGcConfig::new(sp_provider, ri_provider, store_id);
    let safe_point = Arc::new(AtomicU64::new(500));

    gc_worker.start_auto_gc(auto_gc_cfg, safe_point).unwrap();
    host.on_region_changed(&r1, RegionChangeEvent::Create, StateRole::Leader);

    let db = engine.kv_engine().unwrap().as_inner().clone();

    let user_key_del = b"r\0aaaaaaaaaaa";

    // If it's deleted, it will call async scheduler GcTask.
    let test_raws = vec![
        (user_key_del, 9, true),
        (user_key_del, 5, false),
        (user_key_del, 1, false),
    ];

    let modifies = test_raws
        .into_iter()
        .map(|(key, ts, is_delete)| {
            (
                make_key(key, ts),
                ApiV2::encode_raw_value(RawValue {
                    user_value: &[0; 10][..],
                    expire_ts: Some(TimeStamp::max().into_inner()),
                    is_delete,
                }),
            )
        })
        .map(|(k, v)| Modify::Put(CF_DEFAULT, Key::from_encoded_slice(k.as_slice()), v))
        .collect();

    let ctx = Context {
        api_version: ApiVersion::V2,
        ..Default::default()
    };

    let batch = WriteData::from_modifies(modifies);

    engine.write(&ctx, batch).unwrap();

    let cf = get_cf_handle(&db, CF_DEFAULT).unwrap();
    let mut fopts = FlushOptions::default();
    fopts.set_wait(true);
    db.flush_cf(cf, &fopts).unwrap();

    db.compact_range_cf(cf, None, None);

    thread::sleep(Duration::from_millis(100));

    assert_eq!(
        GC_COMPACTION_FILTER_MVCC_DELETION_MET
            .with_label_values(&[STAT_RAW_KEYMODE])
            .get(),
        1
    );
    assert_eq!(
        GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED
            .with_label_values(&[STAT_RAW_KEYMODE])
            .get(),
        1
    );

    GC_COMPACTION_FILTER_MVCC_DELETION_MET.reset();
    GC_COMPACTION_FILTER_MVCC_DELETION_HANDLED.reset();
}
