
use super::*;
use std::env;
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn unique_tempdir(label: &str) -> PathBuf {
    let mut dir = env::temp_dir();
    dir.push(format!(
        "ax-engine-disk-cache-test-{}-{}-{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    dir
}

/// Build a no-prefill-token entry from a raw payload. Tests that
/// only care about the kv_cache payload don't need to think about
/// the prefill-token slot or producer cost metadata.
fn payload_only(bytes: &[u8]) -> DiskPrefixCacheEntry {
    DiskPrefixCacheEntry {
        payload: bytes.to_vec(),
        prefill_output_token: None,
        producer_cold_prefill_us: 0,
        producer_serialize_us: 0,
    }
}

/// Schema-v3 key helper preserving the older tests' shape:
/// `fingerprint_seed` uniquifies the artifact fingerprint the way the
/// v2 token_hash argument used to uniquify the key. This is a test
/// cache-key differentiator, not cryptographic salt material.
fn test_key(
    model_id: &str,
    route_policy: &str,
    layer_layout: &str,
    block_size_tokens: u32,
    token_count: u32,
    fingerprint_seed: u64,
    tokens: &[u32],
) -> Vec<u8> {
    canonical_key_bytes(&DiskPrefixKeyFields {
        model_id,
        artifact_fingerprint_sha256: &format!("{fingerprint_seed:064x}"),
        route_policy,
        layer_layout,
        kv_payload_version: 3,
        block_size_tokens,
        token_count,
        tokens,
    })
}

#[test]
fn insert_then_get_roundtrip() {
    let dir = unique_tempdir("roundtrip");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-a",
        "policy-a",
        "layout-a",
        16,
        4,
        0xdead_beef,
        &[11, 22, 33, 44],
    );
    let payload = b"PAYLOAD-FOR-CACHE".to_vec();
    cache
        .insert(&key_bytes, &payload_only(&payload))
        .expect("insert");
    let got = cache.get(&key_bytes).expect("get").expect("hit");
    assert_eq!(got.payload, payload);
    assert_eq!(
        got.prefill_output_token, None,
        "no prefill token written → reads back as None",
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn open_sweeps_stale_temp_files_but_keeps_entries() {
    let dir = unique_tempdir("stale-tmp-sweep");
    // First open seeds the directory with one live entry.
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-a",
        "policy-a",
        "layout-a",
        16,
        4,
        0xdead_beef,
        &[11, 22, 33, 44],
    );
    cache
        .insert(&key_bytes, &payload_only(b"PAYLOAD"))
        .expect("insert");
    // Simulate a crash mid-insert from a previous process.
    let stale_tmp = dir.join(format!("{}.tmp.99999", key_sha256_hex(b"other-key")));
    fs::write(&stale_tmp, b"partial").expect("write stale tmp");
    drop(cache);

    let cache = DiskPrefixCache::open(&dir).expect("reopen");
    assert!(!stale_tmp.exists(), "stale temp file must be swept at open");
    let got = cache.get(&key_bytes).expect("get").expect("hit");
    assert_eq!(got.payload, b"PAYLOAD".to_vec());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn roundtrip_preserves_prefill_output_token() {
    // Regression for the M4-discovered cross-restart correctness
    // bug: when the producing prefill captured a greedy prefill
    // output token, the on-disk format must carry it so the L2
    // restore path can avoid recomputing at decode step 0.
    let dir = unique_tempdir("prefill-tok-roundtrip");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key("m", "p", "l", 16, 4, 0xfeed_d00d, &[11, 22, 33, 44]);
    let entry = DiskPrefixCacheEntry {
        payload: b"payload".to_vec(),
        prefill_output_token: Some(987_654),
        producer_cold_prefill_us: 111,
        producer_serialize_us: 22,
    };
    cache.insert(&key_bytes, &entry).expect("insert");
    let got = cache.get(&key_bytes).expect("get").expect("hit");
    assert_eq!(got.payload, entry.payload);
    assert_eq!(got.prefill_output_token, Some(987_654));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn get_miss_returns_none() {
    let dir = unique_tempdir("miss");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-b",
        "policy-b",
        "layout-b",
        16,
        4,
        0xfeed_face,
        &[11, 22, 33, 44],
    );
    assert!(cache.get(&key_bytes).expect("get").is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn key_mismatch_returns_miss() {
    let dir = unique_tempdir("collision");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_a = test_key(
        "model-c",
        "policy-c",
        "layout-c",
        16,
        4,
        1,
        &[11, 22, 33, 44],
    );
    let key_b = test_key(
        "model-c",
        "policy-c",
        "layout-c",
        16,
        4,
        2,
        &[11, 22, 33, 44],
    );
    cache
        .insert(&key_a, &payload_only(b"payload-a"))
        .expect("insert");
    // Different key, but same filename slot would only happen on a
    // SHA256 collision (vanishingly improbable). Simulate by writing
    // key_a's content under key_b's filename: we manually swap so the
    // parser must reject the on-disk content as a hash collision.
    let path_a = cache.path_for(&key_a);
    let path_b = cache.path_for(&key_b);
    fs::rename(&path_a, &path_b).expect("rename");
    let result = cache.get(&key_b).expect("get");
    assert!(
        result.is_none(),
        "key mismatch should be reported as a miss"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_payload_returns_miss() {
    let dir = unique_tempdir("corrupt");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-d",
        "policy-d",
        "layout-d",
        16,
        4,
        9,
        &[11, 22, 33, 44],
    );
    cache
        .insert(&key_bytes, &payload_only(b"payload-correct"))
        .expect("insert");
    // Flip a byte in the payload region (last byte of file).
    let path = cache.path_for(&key_bytes);
    let mut raw = fs::read(&path).expect("read");
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    fs::write(&path, raw).expect("write corrupted");
    let result = cache.get(&key_bytes).expect("get");
    assert!(result.is_none(), "corrupted payload should miss");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn contains_returns_true_after_insert() {
    let dir = unique_tempdir("contains-hit");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-c1",
        "policy-c1",
        "layout-c1",
        16,
        4,
        0xc04e_7415,
        &[11, 22, 33, 44],
    );
    assert!(!cache.contains(&key_bytes), "before insert: must not exist");
    cache
        .insert(&key_bytes, &payload_only(b"payload-x"))
        .expect("insert");
    assert!(cache.contains(&key_bytes), "after insert: must exist");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn contains_does_not_validate_payload_integrity() {
    // The probe path relies on `contains` being O(1)-cheap — it does
    // NOT read the file or check the SHA256. A subsequent `get` is
    // what surfaces a corrupt file as a cache miss. This test locks
    // that contract: after deliberately corrupting a file, contains
    // still returns true, while get returns None.
    let dir = unique_tempdir("contains-corrupt");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-c2",
        "policy-c2",
        "layout-c2",
        16,
        4,
        0xc0c0_dead,
        &[11, 22, 33, 44],
    );
    cache
        .insert(&key_bytes, &payload_only(b"payload-valid"))
        .expect("insert");
    // Flip the last byte (payload region) to corrupt the SHA256.
    let path = cache.path_for(&key_bytes);
    let mut raw = fs::read(&path).expect("read");
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    fs::write(&path, raw).expect("write corrupted");

    assert!(
        cache.contains(&key_bytes),
        "contains is existence-only; must remain true post-corruption"
    );
    assert!(
        cache.get(&key_bytes).expect("get").is_none(),
        "get must surface corruption as a miss"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn eviction_drops_oldest_when_entry_budget_exceeded() {
    let dir = unique_tempdir("evict-entries");
    // Allow only 2 entries — third insert must evict the oldest.
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 2,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

    let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
    let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
    let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);

    // Insert in order a, b, c with sleeps so mtimes are strictly
    // increasing (1-second filesystem resolution is the worst-case
    // we need to defeat).
    cache
        .insert(&key_a, &payload_only(b"payload-a"))
        .expect("insert a");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    cache
        .insert(&key_b, &payload_only(b"payload-b"))
        .expect("insert b");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let outcome = cache
        .insert(&key_c, &payload_only(b"payload-c"))
        .expect("insert c");

    assert_eq!(
        outcome.evictions, 1,
        "third insert must evict exactly one entry"
    );
    assert!(!cache.contains(&key_a), "oldest entry (a) must be evicted");
    assert!(cache.contains(&key_b), "b must survive");
    assert!(cache.contains(&key_c), "c must survive");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn eviction_drops_oldest_when_byte_budget_exceeded() {
    let dir = unique_tempdir("evict-bytes");
    // Make the budget tighter than two payloads but bigger than
    // one. The actual per-file size is the payload + header
    // (~56 bytes), so we pick a budget that fits exactly one
    // entry comfortably and not two.
    let payload_size: usize = 4096;
    let per_file = (FIXED_HEADER_LEN + 32 + payload_size) as u64;
    let policy = DiskPrefixCachePolicy {
        max_bytes: per_file + (per_file / 4), // ~1.25 × single file
        max_entries: usize::MAX,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

    let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
    let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
    let payload = vec![0u8; payload_size];

    cache
        .insert(&key_a, &payload_only(&payload))
        .expect("insert a");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let outcome = cache
        .insert(&key_b, &payload_only(&payload))
        .expect("insert b");

    assert_eq!(
        outcome.evictions, 1,
        "byte-budget overflow must evict exactly one entry"
    );
    assert!(!cache.contains(&key_a), "oldest entry must be evicted");
    assert!(cache.contains(&key_b), "newest entry must survive");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn eviction_skips_non_axkv_files() {
    // Operator-placed junk in the directory must not crash the
    // walk and must not be counted toward the eviction budget.
    let dir = unique_tempdir("evict-junk");
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 1,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
    fs::write(dir.join("NOTES.md"), b"hello").expect("write junk");

    let key_a = test_key("m", "p", "l", 16, 4, 0xaa, &[11, 22, 33, 44]);
    cache
        .insert(&key_a, &payload_only(b"payload-a"))
        .expect("insert a");

    // Junk file must remain untouched.
    assert!(dir.join("NOTES.md").is_file());
    assert!(cache.contains(&key_a));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn lock_file_is_ignored_by_eviction_budget() {
    // Opening the cache creates the sentinel lock file. It must
    // never count as a stored entry, otherwise max_entries=1 would
    // evict the only real cache entry immediately.
    let dir = unique_tempdir("lock-file-budget");
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 1,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x10cc, &[11, 22, 33, 44]);

    let outcome = cache
        .insert(&key, &payload_only(b"payload-a"))
        .expect("insert");

    assert_eq!(outcome.evictions, 0, "lock file must not force eviction");
    assert!(cache.contains(&key), "real entry must survive");
    assert!(dir.join(LOCK_FILE_NAME).is_file(), "lock sentinel exists");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_policy_evicts_existing_entries_to_budget_on_open() {
    // Pre-populate three entries under the permissive default
    // policy, then reopen with a tight `max_entries=1` budget.
    // The new instance must trim back to one entry without
    // requiring a fresh insert. Mtimes must be staggered so the
    // initial-sweep ordering is deterministic.
    let dir = unique_tempdir("reopen-evict");
    {
        let cache = DiskPrefixCache::open(&dir).expect("open default");
        let key_a = test_key("m", "p", "l", 16, 4, 0xa1, &[11, 22, 33, 44]);
        let key_b = test_key("m", "p", "l", 16, 4, 0xb2, &[11, 22, 33, 44]);
        let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);
        cache
            .insert(&key_a, &payload_only(b"payload-a"))
            .expect("insert a");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache
            .insert(&key_b, &payload_only(b"payload-b"))
            .expect("insert b");
        std::thread::sleep(std::time::Duration::from_millis(1100));
        cache
            .insert(&key_c, &payload_only(b"payload-c"))
            .expect("insert c");
    }

    let tight = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 1,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, tight).expect("reopen");

    let remaining: Vec<_> = fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == ENTRY_EXTENSION)
        })
        .collect();
    assert_eq!(
        remaining.len(),
        1,
        "with_policy must trim existing entries down to the policy on open",
    );

    // The newest of the three entries (c) must survive.
    let key_c = test_key("m", "p", "l", 16, 4, 0xc3, &[11, 22, 33, 44]);
    assert!(cache.contains(&key_c), "newest entry must survive");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn temp_file_guard_removes_tmp_on_drop() {
    // The guard's job is to clean up a partially written
    // `.tmp.<pid>` file if insert fails between create and
    // rename. We simulate that by manually creating a tmp-shaped
    // file, dropping the guard without disarming, and checking
    // it was reaped. This proves the guard's drop path runs
    // independently of whether insert itself errored.
    let dir = unique_tempdir("tmp-guard");
    fs::create_dir_all(&dir).expect("mkdir");
    let tmp = dir.join("orphan.tmp.999");
    fs::write(&tmp, b"fake-partial-write").expect("seed tmp");
    assert!(tmp.is_file(), "seed");

    {
        let _guard = TempFileGuard::new(&tmp);
        // Drop without disarming.
    }
    assert!(!tmp.exists(), "temp guard must remove the tmp file on drop");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn temp_file_guard_disarmed_does_not_remove() {
    let dir = unique_tempdir("tmp-guard-disarm");
    fs::create_dir_all(&dir).expect("mkdir");
    let tmp = dir.join("survivor.tmp.999");
    fs::write(&tmp, b"keep-me").expect("seed tmp");

    {
        let mut guard = TempFileGuard::new(&tmp);
        guard.disarm();
    }
    assert!(tmp.is_file(), "disarmed guard must not remove the tmp file",);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn concurrent_inserts_and_eviction_preserve_cache_consistency() {
    // M3B advisory-lock stress proxy. Eight threads share an
    // `Arc<DiskPrefixCache>` with a tight `max_entries=8` budget
    // and hammer it with 25 inserts each (200 total) plus parallel
    // explicit eviction sweeps. The lock serializes
    // `insert -> rename -> eviction` per thread; without it,
    // concurrent evictors would race against in-flight renames and
    // produce orphan `.tmp.<pid>` files, dropped entries, or
    // double-removals. After all threads join we assert:
    //   - no `.tmp.*` orphan remains (rename either succeeded
    //     atomically or the TempFileGuard reaped the temp);
    //   - the on-disk entry count stays within the policy budget;
    //   - every surviving `.axkv` file parses cleanly when looked
    //     up by its canonical key — i.e. no torn writes.
    use std::sync::Arc;
    use std::thread;

    let dir = unique_tempdir("multi-thread-stress");
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 8,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = Arc::new(DiskPrefixCache::with_policy(&dir, policy).expect("open"));

    let threads_per_role = 4;
    let inserts_per_thread = 25;
    let mut handles = Vec::with_capacity(threads_per_role * 2);

    // Inserter threads: each writes `inserts_per_thread` entries
    // with unique keys, returning the key bytes it produced.
    for worker_id in 0..threads_per_role {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || -> Vec<Vec<u8>> {
            let mut written = Vec::with_capacity(inserts_per_thread);
            for op in 0..inserts_per_thread {
                let token_hash = ((worker_id as u64) << 32) | op as u64;
                let key = test_key("m", "p", "l", 16, 4, token_hash, &[11, 22, 33, 44]);
                let payload = format!("payload-w{worker_id}-op{op}").into_bytes();
                cache
                    .insert(
                        &key,
                        &DiskPrefixCacheEntry {
                            payload,
                            prefill_output_token: Some(op as u32),
                            producer_cold_prefill_us: 0,
                            producer_serialize_us: 0,
                        },
                    )
                    .expect("insert");
                written.push(key);
            }
            written
        }));
    }

    // Evictor threads: hammer explicit eviction sweeps in parallel
    // with the inserts. These compete for the same lock, so they
    // must not panic or leave the directory in an inconsistent
    // state regardless of the interleaving with inserts.
    for _ in 0..threads_per_role {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || -> Vec<Vec<u8>> {
            for _ in 0..inserts_per_thread {
                let _ = cache.evict_until_within_policy();
                thread::sleep(std::time::Duration::from_micros(50));
            }
            Vec::new()
        }));
    }

    // Collect every key we know was at-least-once written.
    let all_written: Vec<Vec<u8>> = handles
        .into_iter()
        .flat_map(|h| h.join().expect("worker panicked"))
        .collect();
    assert_eq!(
        all_written.len(),
        threads_per_role * inserts_per_thread,
        "every insert must have completed without error",
    );

    // No `.tmp.*` orphans left behind.
    let tmp_orphans: Vec<_> = fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        tmp_orphans.is_empty(),
        "no .tmp.* files should remain after concurrent stress; found {tmp_orphans:?}",
    );

    // Entry count within budget. The advisory lock serializes
    // insert+evict so the budget must hold strictly, not "give or
    // take one in flight".
    let surviving_axkv: Vec<_> = fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .extension()
                .is_some_and(|ext| ext == ENTRY_EXTENSION)
        })
        .collect();
    assert!(
        surviving_axkv.len() <= 8,
        "policy budget breached: {} entries > max_entries=8",
        surviving_axkv.len(),
    );

    // Every surviving file must parse cleanly when looked up
    // through the public `get` API. We can't predict which keys
    // survived eviction, but for each `axkv` file on disk we can
    // recover the canonical key it claims to hold by scanning all
    // produced keys.
    let mut clean_hits = 0;
    for key in &all_written {
        if cache.contains(key)
            && cache
                .get(key)
                .expect("get")
                .filter(|entry| !entry.payload.is_empty())
                .is_some()
        {
            clean_hits += 1;
        }
    }
    assert_eq!(
        clean_hits,
        surviving_axkv.len(),
        "every surviving .axkv must parse cleanly via get() — torn writes detected",
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn atomic_rename_temp_files_cleaned() {
    // The .tmp.* file should not survive a successful insert.
    let dir = unique_tempdir("atomic");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_bytes = test_key(
        "model-e",
        "policy-e",
        "layout-e",
        16,
        4,
        42,
        &[11, 22, 33, 44],
    );
    cache
        .insert(&key_bytes, &payload_only(b"payload-e"))
        .expect("insert");
    let entries: Vec<_> = fs::read_dir(&dir)
        .expect("read_dir")
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(entries.is_empty(), "no .tmp.* file should remain");
    let _ = fs::remove_dir_all(&dir);
}

// ── SSD memory optimisation: env-var configuration & budget tests ──

struct EnvVarGuard {
    saved: Vec<(&'static str, Option<String>)>,
}

impl EnvVarGuard {
    fn capture(keys: &[&'static str]) -> Self {
        Self {
            saved: keys.iter().map(|&key| (key, env::var(key).ok())).collect(),
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            // SAFETY: env-mutating tests hold ENV_LOCK for the whole scope,
            // and this guard restores the original value before releasing it.
            unsafe {
                match value {
                    Some(v) => env::set_var(key, v),
                    None => env::remove_var(key),
                }
            }
        }
    }
}

#[test]
fn policy_from_env_respects_max_bytes_override() {
    let key = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::set_var(key, "4096");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(policy.max_bytes, 4096, "env override must be respected");
}

#[test]
fn policy_from_env_respects_max_entries_override() {
    let key = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::set_var(key, "42");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(policy.max_entries, 42, "env override must be respected");
}

#[test]
fn policy_from_env_keeps_page_store_default_off_and_parses_page_controls() {
    const ENABLED: &str = "AX_MLX_PREFIX_CACHE_DISK_PAGE_STORE";
    const BYTES: &str = "AX_MLX_PREFIX_CACHE_DISK_PAGE_BYTES";
    const GRACE: &str = "AX_MLX_PREFIX_CACHE_DISK_PAGE_GC_GRACE_MS";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[ENABLED, BYTES, GRACE]);
    // SAFETY: ENV_LOCK serializes all environment-mutating tests and the
    // guard restores the original values before releasing the lock.
    unsafe {
        env::remove_var(ENABLED);
        env::remove_var(BYTES);
        env::remove_var(GRACE);
    }
    assert!(!DiskPrefixCachePolicy::from_env().page_store);
    // SAFETY: same lock/restore discipline as above.
    unsafe {
        env::set_var(ENABLED, "true");
        env::set_var(BYTES, "131072");
        env::set_var(GRACE, "1234");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert!(policy.page_store);
    assert_eq!(policy.page_bytes, 131_072);
    assert_eq!(policy.page_gc_grace_ms, 1234);
}

#[test]
fn policy_from_env_respects_combined_overrides() {
    let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
    let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::set_var(key_bytes, "8192");
        env::set_var(key_entries, "7");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(policy.max_bytes, 8192);
    assert_eq!(policy.max_entries, 7);
}

#[test]
fn policy_from_env_falls_back_to_defaults() {
    let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
    let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::remove_var(key_bytes);
        env::remove_var(key_entries);
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(
        policy.max_bytes, DEFAULT_DISK_CACHE_MAX_BYTES,
        "default max_bytes must be 8 GiB"
    );
    assert_eq!(
        policy.max_entries, DEFAULT_DISK_CACHE_MAX_ENTRIES,
        "default max_entries must be 1024"
    );
}

#[test]
fn policy_from_env_ignores_malformed_values() {
    let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
    let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::set_var(key_bytes, "not-a-number");
        env::set_var(key_entries, "xyz");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(
        policy.max_bytes, DEFAULT_DISK_CACHE_MAX_BYTES,
        "malformed max_bytes must fall back to default"
    );
    assert_eq!(
        policy.max_entries, DEFAULT_DISK_CACHE_MAX_ENTRIES,
        "malformed max_entries must fall back to default"
    );
}

#[test]
fn policy_from_env_zero_means_disabled() {
    // `0` must mean "disable the disk tier" — matching the in-memory
    // tier's env semantics — not silently fall back to the 8 GiB
    // default, which is the opposite of what an operator setting 0
    // intends.
    let key_bytes = "AX_MLX_PREFIX_CACHE_DISK_MAX_BYTES";
    let key_entries = "AX_MLX_PREFIX_CACHE_DISK_MAX_ENTRIES";
    let _env_lock = ENV_LOCK.lock().expect("env lock");
    let _env_guard = EnvVarGuard::capture(&[key_bytes, key_entries]);
    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::set_var(key_bytes, "0");
        env::set_var(key_entries, "0");
    }
    let policy = DiskPrefixCachePolicy::from_env();
    assert_eq!(policy.max_bytes, 0, "zero max_bytes must be preserved");
    assert_eq!(policy.max_entries, 0, "zero max_entries must be preserved");
    assert!(!policy.enabled(), "zero budgets disable the disk tier");

    // SAFETY: ENV_LOCK is held for this test's whole scope, and the
    // EnvVarGuard restores the captured vars on drop.
    unsafe {
        env::remove_var(key_bytes);
        env::remove_var(key_entries);
    }
    assert!(
        DiskPrefixCachePolicy::from_env().enabled(),
        "default policy is enabled"
    );
}

#[test]
fn cross_session_persistence_writes_and_reopens() {
    // Simulate two independent sessions sharing the same cache directory.
    // Session 1 writes; session 2 opens the same dir and reads.
    let dir = unique_tempdir("cross-session");
    let key_bytes = test_key(
        "model-x",
        "policy-x",
        "layout-x",
        16,
        4,
        0xcafe,
        &[11, 22, 33, 44],
    );
    let payload = b"session-1-payload".to_vec();
    let entry = DiskPrefixCacheEntry {
        payload: payload.clone(),
        prefill_output_token: Some(42),
        producer_cold_prefill_us: 0,
        producer_serialize_us: 0,
    };

    // Session 1: open, write, drop.
    {
        let cache1 = DiskPrefixCache::open(&dir).expect("session 1 open");
        cache1.insert(&key_bytes, &entry).expect("session 1 insert");
    }

    // Session 2: reopen, read.
    {
        let cache2 = DiskPrefixCache::open(&dir).expect("session 2 open");
        let got = cache2.get(&key_bytes).expect("session 2 get").expect("hit");
        assert_eq!(got.payload, payload);
        assert_eq!(got.prefill_output_token, Some(42));
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_stores_and_measures_large_payload() {
    // Verify that a realistically-sized KV cache payload (simulated)
    // is stored and measured correctly for SSD capacity accounting.
    let dir = unique_tempdir("large-payload");
    let policy = DiskPrefixCachePolicy {
        max_bytes: 1024 * 1024, // 1 MiB budget
        max_entries: usize::MAX,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

    // 256 KiB payload — should fit within 1 MiB budget
    let payload_256k = vec![0xAB_u8; 256 * 1024];
    let key1 = test_key("m", "p", "l", 16, 4, 1, &[11, 22, 33, 44]);
    cache
        .insert(&key1, &payload_only(&payload_256k))
        .expect("insert 256k");

    // Second 256 KiB — still within budget
    let key2 = test_key("m", "p", "l", 16, 4, 2, &[11, 22, 33, 44]);
    let outcome = cache
        .insert(&key2, &payload_only(&payload_256k))
        .expect("insert second 256k");
    assert_eq!(outcome.evictions, 0, "two 256k payloads must fit in 1 MiB");

    // Verify both are readable
    assert!(cache.get(&key1).expect("get").is_some());
    assert!(cache.get(&key2).expect("get").is_some());

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn disk_cache_evicts_when_payload_exceeds_byte_budget() {
    // With a tight budget, inserting a large payload forces eviction.
    let dir = unique_tempdir("evict-large");
    let budget = 4096u64; // 4 KiB
    let policy = DiskPrefixCachePolicy {
        max_bytes: budget,
        max_entries: usize::MAX,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");

    let key1 = test_key("m", "p", "l", 16, 4, 0xaa, &[11, 22, 33, 44]);
    cache
        .insert(&key1, &payload_only(&vec![0u8; 2048]))
        .expect("insert first");
    std::thread::sleep(std::time::Duration::from_millis(1100));

    let key2 = test_key("m", "p", "l", 16, 4, 0xbb, &[11, 22, 33, 44]);
    let outcome = cache
        .insert(&key2, &payload_only(&vec![0u8; 2048]))
        .expect("insert second");
    // Two 2048-byte payloads + header overhead > 4096 budget
    assert!(outcome.evictions >= 1, "must evict at least one entry");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_policy_trims_existing_entries_on_reopen() {
    // Pre-populate 5 entries, then reopen with max_entries=2.
    // Only 2 newest should survive.
    let dir = unique_tempdir("reopen-trim");
    {
        let cache = DiskPrefixCache::open(&dir).expect("open default");
        for i in 0..5u64 {
            let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
            cache
                .insert(&key, &payload_only(format!("payload-{i}").as_bytes()))
                .expect("insert");
            std::thread::sleep(std::time::Duration::from_millis(1050));
        }
    }

    let tight = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 2,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, tight).expect("reopen tight");

    // Entries 0-2 must be evicted; 3-4 must survive.
    for i in 0..3 {
        let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
        assert!(!cache.contains(&key), "entry {i} must be evicted");
    }
    for i in 3..5 {
        let key = test_key("m", "p", "l", 16, 4, i, &[11, 22, 33, 44]);
        assert!(cache.contains(&key), "entry {i} must survive");
    }

    let _ = fs::remove_dir_all(&dir);
}
#[test]
fn token_content_mismatch_returns_miss() {
    // Simulated 64-bit FNV token_hash collision: two different prompts
    // with the SAME token_hash and token_count. Schema v2 commits to
    // the token content via SHA-256, so the embedded-key comparison
    // must reject the swapped file instead of restoring the wrong KV.
    let dir = unique_tempdir("fnv-collision");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_x = test_key("m", "p", "l", 16, 4, 7, &[1, 2, 3, 4]);
    let key_y = test_key("m", "p", "l", 16, 4, 7, &[9, 9, 9, 9]);
    assert_ne!(
        key_x, key_y,
        "same token_hash but different tokens must produce different keys"
    );
    cache
        .insert(&key_x, &payload_only(b"payload-x"))
        .expect("insert");
    fs::rename(cache.path_for(&key_x), cache.path_for(&key_y)).expect("rename");
    assert!(
        cache.get(&key_y).expect("get").is_none(),
        "token-content mismatch must miss, never restore the wrong KV"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_entry_is_unlinked_on_get() {
    // A file that fails validation must be reclaimed, not left
    // consuming budget until mtime eviction reaches it.
    let dir = unique_tempdir("corrupt-unlink");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0xbad, &[11, 22, 33, 44]);
    cache
        .insert(&key, &payload_only(b"payload"))
        .expect("insert");
    let path = cache.path_for(&key);
    let mut raw = fs::read(&path).expect("read");
    let last = raw.len() - 1;
    raw[last] ^= 0xFF;
    fs::write(&path, raw).expect("corrupt");

    assert!(cache.get(&key).expect("get").is_none(), "corruption misses");
    assert!(
        !path.exists(),
        "corrupt entry must be unlinked by the failed get"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn get_refreshes_mtime_so_eviction_approximates_lru() {
    // Hot entries must survive eviction over cold-but-newer ones.
    let dir = unique_tempdir("lru-touch");
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 2,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
    let key_a = test_key("m", "p", "l", 16, 4, 0xa, &[11, 22, 33, 44]);
    let key_b = test_key("m", "p", "l", 16, 4, 0xb, &[11, 22, 33, 44]);
    let key_c = test_key("m", "p", "l", 16, 4, 0xc, &[11, 22, 33, 44]);

    cache.insert(&key_a, &payload_only(b"a")).expect("insert a");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    cache.insert(&key_b, &payload_only(b"b")).expect("insert b");
    std::thread::sleep(std::time::Duration::from_millis(1100));
    // Touch a: it is now more recently used than b.
    assert!(cache.get(&key_a).expect("get").is_some());
    std::thread::sleep(std::time::Duration::from_millis(1100));
    cache.insert(&key_c, &payload_only(b"c")).expect("insert c");

    assert!(cache.contains(&key_a), "hot entry must survive");
    assert!(!cache.contains(&key_b), "cold entry must be evicted");
    assert!(cache.contains(&key_c), "new entry must survive");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn oversized_payload_is_skipped_not_self_evicting() {
    let dir = unique_tempdir("oversized");
    let policy = DiskPrefixCachePolicy {
        max_bytes: 1024,
        max_entries: usize::MAX,
        ..DiskPrefixCachePolicy::default()
    };
    let cache = DiskPrefixCache::with_policy(&dir, policy).expect("open");
    let key_small = test_key("m", "p", "l", 16, 4, 1, &[11, 22, 33, 44]);
    cache
        .insert(&key_small, &payload_only(b"small"))
        .expect("insert small");

    let key_big = test_key("m", "p", "l", 16, 4, 2, &[11, 22, 33, 44]);
    let outcome = cache
        .insert(&key_big, &payload_only(&vec![0u8; 4096]))
        .expect("oversized insert returns Ok");
    assert_eq!(outcome.evictions, 0, "skipped store must not evict");
    assert!(
        !cache.contains(&key_big),
        "oversized entry must not be written"
    );
    assert!(
        cache.contains(&key_small),
        "existing entries must not be sacrificed for an undurable store"
    );
    let _ = fs::remove_dir_all(&dir);
}
#[test]
fn canonical_key_v3_changes_for_every_identity_field() {
    let base = DiskPrefixKeyFields {
        model_id: "m",
        artifact_fingerprint_sha256: "aa",
        route_policy: "p",
        layer_layout: "l",
        kv_payload_version: 3,
        block_size_tokens: 16,
        token_count: 4,
        tokens: &[1, 2, 3, 4],
    };
    let reference = canonical_key_bytes(&base);
    let variants = [
        canonical_key_bytes(&DiskPrefixKeyFields {
            model_id: "m2",
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            artifact_fingerprint_sha256: "bb",
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            route_policy: "p2",
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            layer_layout: "l2",
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            kv_payload_version: 4,
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            block_size_tokens: 32,
            ..base
        }),
        canonical_key_bytes(&DiskPrefixKeyFields {
            tokens: &[1, 2, 3, 5],
            ..base
        }),
    ];
    for (idx, variant) in variants.iter().enumerate() {
        assert_ne!(
            &reference, variant,
            "identity field {idx} must change the canonical key"
        );
    }
    // Length-prefixed strings: shifting a boundary must not alias.
    let shifted = canonical_key_bytes(&DiskPrefixKeyFields {
        model_id: "mp",
        route_policy: "",
        ..base
    });
    assert_ne!(reference, shifted, "string boundaries must be unambiguous");
}

#[test]
fn v3_entry_roundtrips_producer_metadata() {
    let dir = unique_tempdir("producer-meta");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x11, &[11, 22, 33, 44]);
    let entry = DiskPrefixCacheEntry {
        payload: b"payload-meta".to_vec(),
        prefill_output_token: Some(7),
        producer_cold_prefill_us: 123_456,
        producer_serialize_us: 7_890,
    };
    cache.insert(&key, &entry).expect("insert");
    let got = cache.get(&key).expect("get").expect("hit");
    assert_eq!(got.payload, entry.payload);
    assert_eq!(got.prefill_output_token, Some(7));
    assert_eq!(got.producer_cold_prefill_us, 123_456);
    assert_eq!(got.producer_serialize_us, 7_890);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn v3_read_rejects_unknown_flags_and_trailing_bytes() {
    let dir = unique_tempdir("flags-trailing");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x22, &[11, 22, 33, 44]);
    cache
        .insert(&key, &payload_only(b"payload"))
        .expect("insert");
    let path = cache.path_for(&key);
    let pristine = fs::read(&path).expect("read");

    // Unknown flag bit set (offset 12..16). Bit 0 is the defined
    // page-manifest flag; bit 1 is reserved and must fail closed.
    let mut flagged = pristine.clone();
    flagged[12] |= 0x02;
    fs::write(&path, &flagged).expect("write");
    assert!(
        cache.get(&key).expect("get").is_none(),
        "unknown flags are from the future and must fail closed"
    );

    // Trailing bytes beyond the declared sections.
    let mut trailing = pristine.clone();
    trailing.push(0);
    fs::write(&path, &trailing).expect("write");
    assert!(
        cache.get(&key).expect("get").is_none(),
        "trailing bytes outside declared sections are invalid"
    );

    // Header corruption on the checksum must miss.
    let mut bad_hash = pristine.clone();
    bad_hash[28] ^= 0xFF;
    fs::write(&path, &bad_hash).expect("write");
    assert!(cache.get(&key).expect("get").is_none());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn v3_read_rejects_symlinked_entry() {
    let dir = unique_tempdir("symlink");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key_real = test_key("m", "p", "l", 16, 4, 0x33, &[11, 22, 33, 44]);
    let key_link = test_key("m", "p", "l", 16, 4, 0x44, &[11, 22, 33, 44]);
    cache
        .insert(&key_real, &payload_only(b"payload"))
        .expect("insert");
    std::os::unix::fs::symlink(cache.path_for(&key_real), cache.path_for(&key_link))
        .expect("symlink");
    assert!(
        cache.get(&key_link).expect("get").is_none(),
        "symlinked entries must be rejected"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn stale_v2_format_entry_is_a_miss_and_unlinked() {
    let dir = unique_tempdir("stale-v2");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x55, &[11, 22, 33, 44]);
    // Hand-craft a v2-shaped file at the key's path.
    let mut raw = Vec::new();
    raw.extend_from_slice(FILE_MAGIC);
    raw.extend_from_slice(&2u32.to_le_bytes());
    raw.extend_from_slice(&[0u8; 48]); // v2 header remainder
    fs::write(cache.path_for(&key), &raw).expect("write v2");
    assert!(
        cache.get(&key).expect("get").is_none(),
        "older format versions are clean misses"
    );
    assert!(
        !cache.path_for(&key).exists(),
        "stale-version entries are reclaimed lazily"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn fresh_root_and_entries_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let dir = unique_tempdir("perms");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    assert_eq!(
        fs::metadata(&dir).expect("dir meta").permissions().mode() & 0o777,
        0o700,
        "fresh cache root must be owner-only"
    );
    let key = test_key("m", "p", "l", 16, 4, 0x66, &[11, 22, 33, 44]);
    cache
        .insert(&key, &payload_only(b"payload"))
        .expect("insert");
    assert_eq!(
        fs::metadata(cache.path_for(&key))
            .expect("entry meta")
            .permissions()
            .mode()
            & 0o777,
        0o600,
        "entries must be owner-only"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn admission_emits_every_closed_reason() {
    let policy = DiskPrefixCachePolicy {
        min_prefix_tokens: 2048,
        max_entry_bytes: 1024 * 1024,
        min_savings_us: 10_000,
        ..DiskPrefixCachePolicy::default()
    };
    let throughput = DiskThroughputSnapshot {
        write_bytes_per_us: 100.0,   // 100 B/µs
        restore_bytes_per_us: 200.0, // 200 B/µs
    };

    let disabled = DiskPrefixCachePolicy {
        admission: DiskAdmissionMode::Disabled,
        ..policy
    };
    assert_eq!(
        disabled
            .evaluate_admission(4096, 1024, Some(1), Some(throughput))
            .0,
        DiskAdmissionReason::Disabled
    );

    // Oversize rejects even in `always` (diagnostic mode obeys caps).
    let always = DiskPrefixCachePolicy {
        admission: DiskAdmissionMode::Always,
        ..policy
    };
    assert_eq!(
        always
            .evaluate_admission(4096, 2 * 1024 * 1024, None, None)
            .0,
        DiskAdmissionReason::EntryTooLarge
    );
    assert_eq!(
        always.evaluate_admission(1, 1024, None, None).0,
        DiskAdmissionReason::AdmittedAlways,
        "always bypasses the value model and the prefix floor"
    );

    assert_eq!(
        policy
            .evaluate_admission(100, 1024, Some(1), Some(throughput))
            .0,
        DiskAdmissionReason::PrefixTooShort
    );
    assert_eq!(
        policy
            .evaluate_admission(4096, 1024, None, Some(throughput))
            .0,
        DiskAdmissionReason::NoCostModel
    );
    assert_eq!(
        policy
            .evaluate_admission(4096, 1024, Some(1_000_000), None)
            .0,
        DiskAdmissionReason::NoCostModel
    );

    // 1 MiB entry: restore ≈ 5.2 ms, write ≈ 10.5 ms. A 10 s cold
    // prefill clears write + min_savings comfortably.
    let (reason, estimate) =
        policy.evaluate_admission(4096, 1024 * 1024, Some(10_000_000), Some(throughput));
    assert_eq!(reason, DiskAdmissionReason::AdmittedPositiveValue);
    assert!(estimate.restore_us > 0 && estimate.write_us > 0);

    // A 6 ms cold prefill loses to lifecycle cost.
    assert_eq!(
        policy
            .evaluate_admission(4096, 1024 * 1024, Some(6_000), Some(throughput))
            .0,
        DiskAdmissionReason::PredictedNoSavings
    );

    // Saturation: absurd byte counts must not panic or admit.
    assert_eq!(
        policy
            .evaluate_admission(
                u32::MAX,
                u64::MAX,
                Some(u64::MAX),
                Some(DiskThroughputSnapshot {
                    write_bytes_per_us: f64::MIN_POSITIVE,
                    restore_bytes_per_us: f64::MIN_POSITIVE,
                }),
            )
            .0,
        DiskAdmissionReason::EntryTooLarge
    );

    // Every reason has a distinct stable code and label, and `ALL`
    // covers the whole enum in code order.
    let mut codes: Vec<u32> = DiskAdmissionReason::ALL.iter().map(|r| r.code()).collect();
    assert!(
        codes.windows(2).all(|pair| pair[0] < pair[1]),
        "ALL in code order"
    );
    codes.dedup();
    assert_eq!(
        codes.len(),
        DiskAdmissionReason::ALL.len(),
        "codes must be distinct"
    );
    let mut labels: Vec<&str> = DiskAdmissionReason::ALL.iter().map(|r| r.label()).collect();
    labels.sort_unstable();
    labels.dedup();
    assert_eq!(
        labels.len(),
        DiskAdmissionReason::ALL.len(),
        "labels must be distinct"
    );
}

#[test]
fn get_timed_reports_stage_timings_and_bytes() {
    let dir = unique_tempdir("stage-timings");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x77, &[11, 22, 33, 44]);
    let payload = vec![0x5Au8; 128 * 1024];
    cache.insert(&key, &payload_only(&payload)).expect("insert");
    let (entry, timings) = cache.get_timed(&key).expect("get").expect("hit");
    assert_eq!(entry.payload, payload);
    assert!(
        timings.bytes_read >= payload.len() as u64,
        "bytes_read covers header + key + payload"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn get_restored_timed_streams_native_kv_without_payload_vec() {
    let dir = unique_tempdir("restored-native");
    let cache = DiskPrefixCache::open(&dir).expect("open");
    let key = test_key("m", "p", "l", 16, 4, 0x88, &[1, 2, 3, 4]);
    let native = crate::kv_cache::MlxKVCache::new(2);
    let payload = native.serialize_to_bytes();
    cache
        .insert_parts(&key, &payload, Some(7), 50_000, 1_000)
        .expect("insert");
    let (restored, timings) = cache
        .get_restored_timed(&key)
        .expect("get_restored")
        .expect("hit");
    assert_eq!(restored.prefill_output_token, Some(7));
    assert_eq!(restored.producer_cold_prefill_us, 50_000);
    assert_eq!(restored.cache.seq_len(), 0);
    assert!(
        timings.bytes_read >= payload.len() as u64,
        "streamed read covers header + key + payload"
    );
    // Corrupt trailing payload must miss (fail closed).
    let path = cache.path_for(&key);
    let mut raw = fs::read(&path).expect("read");
    if let Some(last) = raw.last_mut() {
        *last ^= 0xff;
    }
    fs::write(&path, raw).expect("corrupt");
    assert!(
        cache.get_restored_timed(&key).expect("io").is_none(),
        "corrupt entry is a miss"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn native_restore_unlinks_checksummed_invalid_kv_payload() {
    for page_store in [false, true] {
        let dir = unique_tempdir("invalid-native-payload");
        let cache = DiskPrefixCache::with_policy(
            &dir,
            DiskPrefixCachePolicy {
                page_store,
                page_gc_grace_ms: 0,
                ..DiskPrefixCachePolicy::default()
            },
        )
        .expect("open");
        let key = test_key("m", "p", "l", 4, 4, 0x89, &[1, 2, 3, 4]);
        cache
            .insert(&key, &payload_only(b"invalid native KV"))
            .expect("insert");
        assert!(cache.get(&key).expect("opaque get").is_some());

        assert!(cache.get_restored_timed(&key).expect("restore").is_none());
        assert!(
            !cache.contains(&key),
            "invalid native payload must be reclaimed"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn native_cleanup_preserves_a_healthy_replacement() {
    for page_store in [false, true] {
        let dir = unique_tempdir("native-cleanup-replacement");
        let cache = DiskPrefixCache::with_policy(
            &dir,
            DiskPrefixCachePolicy {
                page_store,
                ..DiskPrefixCachePolicy::default()
            },
        )
        .expect("open");
        let key = test_key("m", "p", "l", 4, 4, 0x90, &[1, 2, 3, 4]);
        cache
            .insert(&key, &payload_only(b"invalid native KV"))
            .expect("insert invalid");
        let path = cache.path_for(&key);
        assert!(matches!(
            cache.read_entry_restored(&path, &key),
            Err(ReadEntryError::Invalid)
        ));

        // Simulate an atomic replacement between the failed read and
        // the cleanup lock acquisition.
        let payload = crate::kv_cache::MlxKVCache::new(2).serialize_to_bytes();
        cache
            .insert_parts(&key, &payload, Some(7), 0, 0)
            .expect("replace");
        cache.remove_unparseable_entry(&path, &key, true);

        let (restored, _) = cache
            .get_restored_timed(&key)
            .expect("restore")
            .expect("hit");
        assert_eq!(restored.prefill_output_token, Some(7));
        assert!(cache.contains(&key));
        let _ = fs::remove_dir_all(&dir);
    }
}

#[test]
fn page_manifest_round_trip_survives_restart_with_flag_disabled() {
    let dir = unique_tempdir("page-restart");
    let key = test_key("m", "p", "standard-fa", 4, 4, 0x99, &[1, 2, 3, 4]);
    let payload = b"abcdefghabcdefgh-tail".to_vec();
    {
        let cache = DiskPrefixCache::with_policy(
            &dir,
            DiskPrefixCachePolicy {
                page_store: true,
                page_bytes: 8,
                page_gc_grace_ms: 0,
                ..DiskPrefixCachePolicy::default()
            },
        )
        .expect("open page store");
        cache
            .insert_parts(&key, &payload, Some(17), 42_000, 900)
            .expect("insert page manifest");
        let raw = fs::read(cache.path_for(&key)).expect("manifest file");
        assert_eq!(
            u32::from_le_bytes(raw[12..16].try_into().expect("flags")),
            PAGE_MANIFEST_FLAG
        );
        let blobs = cache.page_store.as_ref().expect("page store").blob_paths();
        assert_eq!(blobs.len(), 2, "duplicate 8-byte page is stored once");
        let hit = cache.get(&key).expect("get").expect("hit");
        assert_eq!(hit.payload, payload);
        assert_eq!(hit.prefill_output_token, Some(17));
    }

    // Disabling new page writes must not strand already durable entries.
    let reopened = DiskPrefixCache::open(&dir).expect("reopen legacy-write mode");
    assert!(!reopened.policy().page_store);
    let hit = reopened.get(&key).expect("reopen get").expect("reopen hit");
    assert_eq!(hit.payload, payload);
    assert_eq!(hit.producer_cold_prefill_us, 42_000);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn page_manifest_streams_native_restore_without_payload_vec() {
    let dir = unique_tempdir("page-native");
    let cache = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: true,
            page_bytes: 16,
            page_gc_grace_ms: 0,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open");
    let key = test_key("m", "p", "standard-fa", 4, 4, 0xaa, &[1, 2, 3, 4]);
    let native = crate::kv_cache::MlxKVCache::new(2);
    let payload = native.serialize_to_bytes();
    cache
        .insert_parts(&key, &payload, Some(23), 50_000, 1_000)
        .expect("insert");
    let (restored, timings) = cache
        .get_restored_timed(&key)
        .expect("restore")
        .expect("hit");
    assert_eq!(restored.cache.seq_len(), 0);
    assert_eq!(restored.prefill_output_token, Some(23));
    assert!(timings.bytes_read >= payload.len() as u64);

    let trailing_key = test_key("m", "p", "standard-fa", 4, 4, 0xab, &[1, 2, 3, 4]);
    let mut trailing = payload.clone();
    trailing.push(0);
    cache
        .insert(&trailing_key, &payload_only(&trailing))
        .expect("insert trailing payload");
    assert!(
        cache
            .get_restored_timed(&trailing_key)
            .expect("trailing payload is a miss")
            .is_none(),
        "page restore must reject bytes the native payload did not consume"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn page_gc_keeps_shared_blob_until_last_manifest_is_removed() {
    let dir = unique_tempdir("page-refcount");
    let cache = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: true,
            page_bytes: 8,
            page_gc_grace_ms: 0,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open");
    let key_a = test_key("m", "p", "standard-fa", 4, 2, 1, &[1, 2]);
    let key_b = test_key("m", "p", "standard-fa", 4, 2, 2, &[1, 3]);
    cache
        .insert(&key_a, &payload_only(b"abcdefgh-A"))
        .expect("insert a");
    cache
        .insert(&key_b, &payload_only(b"abcdefgh-B"))
        .expect("insert b");
    let page_store = cache.page_store.as_ref().expect("page store");
    assert_eq!(page_store.blob_paths().len(), 3);

    fs::remove_file(cache.path_for(&key_a)).expect("remove first manifest");
    cache.evict_until_within_policy();
    assert_eq!(
        page_store.blob_paths().len(),
        2,
        "shared page and remaining unique page stay live"
    );
    assert_eq!(
        cache.get(&key_b).expect("get b").expect("b hit").payload,
        b"abcdefgh-B"
    );

    fs::remove_file(cache.path_for(&key_b)).expect("remove last manifest");
    cache.evict_until_within_policy();
    assert!(page_store.blob_paths().is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn page_store_enforces_manifest_count_and_unique_page_byte_budgets() {
    let dir = unique_tempdir("page-budgets");
    let base = DiskPrefixCachePolicy {
        page_store: true,
        page_bytes: 8,
        page_gc_grace_ms: 0,
        max_bytes: u64::MAX,
        max_entry_bytes: u64::MAX,
        max_entries: 2,
        ..DiskPrefixCachePolicy::default()
    };
    let key_a = test_key("m", "p", "standard-fa", 4, 2, 11, &[1, 2]);
    let key_b = test_key("m", "p", "standard-fa", 4, 2, 12, &[1, 3]);
    {
        let cache = DiskPrefixCache::with_policy(&dir, base).expect("open");
        cache
            .insert(&key_a, &payload_only(b"abcdefgh-A"))
            .expect("insert a");
        cache
            .insert(&key_b, &payload_only(b"abcdefgh-B"))
            .expect("insert b");
    }

    let count_limited = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            max_entries: 1,
            ..base
        },
    )
    .expect("count-limited reopen");
    let survivors =
        usize::from(count_limited.contains(&key_a)) + usize::from(count_limited.contains(&key_b));
    assert_eq!(survivors, 1);
    assert_eq!(
        count_limited
            .page_store
            .as_ref()
            .expect("page store")
            .blob_paths()
            .len(),
        2,
        "one shared and one surviving unique page remain"
    );

    let manifest_bytes = fs::read_dir(&dir)
        .expect("root")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|ext| ext == ENTRY_EXTENSION)
        })
        .filter_map(|entry| entry.metadata().ok().map(|meta| meta.len()))
        .sum::<u64>();
    let page_bytes = count_limited
        .page_store
        .as_ref()
        .expect("page store")
        .blob_paths()
        .into_iter()
        .filter_map(|path| fs::metadata(path).ok().map(|meta| meta.len()))
        .sum::<u64>();
    let physical_bytes = manifest_bytes.saturating_add(page_bytes);
    drop(count_limited);

    let byte_limited = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            max_bytes: physical_bytes.saturating_sub(1),
            max_entry_bytes: u64::MAX,
            max_entries: usize::MAX,
            ..base
        },
    )
    .expect("byte-limited reopen");
    assert!(!byte_limited.contains(&key_a));
    assert!(!byte_limited.contains(&key_b));
    assert!(
        byte_limited
            .page_store
            .as_ref()
            .expect("page store")
            .blob_paths()
            .is_empty(),
        "unique page bytes participate in the byte budget"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn corrupt_page_fails_closed_and_reclaims_manifest() {
    let dir = unique_tempdir("page-corrupt");
    let cache = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: true,
            page_bytes: 8,
            page_gc_grace_ms: 0,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open");
    let key = test_key("m", "p", "standard-fa", 4, 2, 3, &[1, 2]);
    cache
        .insert(&key, &payload_only(b"abcdefgh-tail"))
        .expect("insert");
    let page_store = cache.page_store.as_ref().expect("page store");
    let blob = page_store.blob_paths().into_iter().next().expect("blob");
    fs::write(blob, b"corrupt!").expect("corrupt page");
    assert!(cache.get(&key).expect("corruption is a miss").is_none());
    assert!(!cache.path_for(&key).exists());
    assert!(page_store.blob_paths().is_empty());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn page_manifest_is_a_miss_not_a_delete_for_a_reader_without_page_store() {
    let dir = unique_tempdir("page-unsupported-reader");
    let key = test_key("m", "p", "standard-fa", 4, 2, 21, &[1, 2]);
    // Opened before any `.pages` directory exists and with page writes
    // off: this cache holds no page store at all, mirroring a process
    // that started before another process enabled page mode.
    let reader = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: false,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open reader");
    assert!(reader.page_store.is_none());
    let writer = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: true,
            page_bytes: 8,
            page_gc_grace_ms: 0,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open page writer");
    writer
        .insert(&key, &payload_only(b"abcdefgh-tail"))
        .expect("insert page manifest");

    // The reader cannot serve the entry, but the miss must not delete
    // the writer's valid manifest (previously `Invalid` fed the
    // corrupt-entry cleanup under the exclusive lock).
    assert!(reader.get(&key).expect("reader get").is_none());
    assert!(
        reader
            .get_restored_timed(&key)
            .expect("reader restore")
            .is_none()
    );
    assert!(reader.path_for(&key).exists());
    let hit = writer.get(&key).expect("writer get").expect("writer hit");
    assert_eq!(hit.payload, b"abcdefgh-tail");
    let _ = fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn page_gc_aborts_the_sweep_when_a_manifest_is_unreadable() {
    use std::os::unix::fs::PermissionsExt;

    let dir = unique_tempdir("page-gc-abort");
    let cache = DiskPrefixCache::with_policy(
        &dir,
        DiskPrefixCachePolicy {
            page_store: true,
            page_bytes: 8,
            page_gc_grace_ms: 0,
            ..DiskPrefixCachePolicy::default()
        },
    )
    .expect("open");
    let key = test_key("m", "p", "standard-fa", 4, 2, 31, &[1, 2]);
    cache
        .insert(&key, &payload_only(b"abcdefgh-tail"))
        .expect("insert");
    let page_store = cache.page_store.as_ref().expect("page store");
    let blob_count = page_store.blob_paths().len();
    assert!(blob_count > 0);

    // A manifest that cannot be opened must abort the sweep: its pages
    // are still referenced, and an incomplete live set must not delete
    // them.
    let manifest = cache.path_for(&key);
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o000)).expect("chmod 000");
    cache.gc_page_store();
    assert_eq!(
        page_store.blob_paths().len(),
        blob_count,
        "a sweep with an incomplete live set must keep referenced pages",
    );

    // Readable again: the manifest marks its pages and nothing is
    // collected; once the manifest is gone, the sweep reclaims.
    fs::set_permissions(&manifest, fs::Permissions::from_mode(0o600)).expect("chmod 600");
    cache.gc_page_store();
    assert_eq!(page_store.blob_paths().len(), blob_count);
    fs::remove_file(&manifest).expect("remove manifest");
    cache.gc_page_store();
    assert!(page_store.blob_paths().is_empty());
    let _ = fs::remove_dir_all(&dir);
}
