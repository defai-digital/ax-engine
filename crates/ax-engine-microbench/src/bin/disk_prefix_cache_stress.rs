use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_mlx::disk_prefix_cache::{
    DiskPrefixCache, DiskPrefixCacheEntry, DiskPrefixCachePolicy, canonical_key_bytes,
};
use serde_json::json;

const SCHEMA_VERSION: &str = "ax.disk_prefix_cache_stress.v1";
const DEFAULT_WORKERS: usize = 4;
const DEFAULT_ITERATIONS: usize = 24;
const DEFAULT_PAYLOAD_BYTES: usize = 4096;
const DEFAULT_OVERLAP_KEYS: usize = 8;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = if args.iter().any(|arg| arg == "--worker") {
        run_worker(&args)
    } else {
        run_orchestrator(&args)
    };
    match result {
        Ok(()) => {}
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

fn run_orchestrator(args: &[String]) -> Result<(), String> {
    let output = value_after(args, "--output").map(PathBuf::from);
    let workers = usize_after(args, "--workers").unwrap_or(DEFAULT_WORKERS);
    let iterations = usize_after(args, "--iterations").unwrap_or(DEFAULT_ITERATIONS);
    let payload_bytes = usize_after(args, "--payload-bytes").unwrap_or(DEFAULT_PAYLOAD_BYTES);
    let overlap_keys = usize_after(args, "--overlap-keys").unwrap_or(DEFAULT_OVERLAP_KEYS);
    if workers == 0 || iterations == 0 || payload_bytes == 0 || overlap_keys == 0 {
        return Err("workers, iterations, payload-bytes, and overlap-keys must be positive".into());
    }

    let root = unique_dir("ax-disk-prefix-stress")?;
    let concurrent_dir = root.join("concurrent");
    let eviction_dir = root.join("eviction");
    fs::create_dir_all(&concurrent_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&eviction_dir).map_err(|e| e.to_string())?;

    let concurrent = run_concurrent_stress(
        &concurrent_dir,
        workers,
        iterations,
        payload_bytes,
        overlap_keys,
    )?;
    let eviction = run_eviction_pressure(&eviction_dir, payload_bytes)?;

    let corruption_load_failures = concurrent["workers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["corruption_load_failures"].as_u64())
        .sum::<u64>();
    let read_misses = concurrent["workers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["read_misses"].as_u64())
        .sum::<u64>();
    let worker_failures = concurrent["workers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["exit_status"].as_i64().unwrap_or(1) != 0)
        .count();
    let eviction_policy_passed = eviction["policy_passed"].as_bool().unwrap_or(false);
    let verdict = if corruption_load_failures == 0
        && read_misses == 0
        && worker_failures == 0
        && eviction_policy_passed
    {
        "PASS"
    } else {
        "FAIL"
    };

    let artifact = json!({
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": unix_now(),
        "host": {
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        },
        "aggregate": {
            "verdict": verdict,
            "concurrent_workers": workers,
            "iterations_per_worker": iterations,
            "overlap_keys": overlap_keys,
            "corruption_load_failures": corruption_load_failures,
            "read_misses": read_misses,
            "worker_failures": worker_failures,
            "eviction_policy_passed": eviction_policy_passed,
        },
        "concurrent_stress": concurrent,
        "eviction_pressure": eviction,
    });

    let text = serde_json::to_string_pretty(&artifact).map_err(|e| e.to_string())? + "\n";
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        fs::write(&path, text).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
    } else {
        print!("{text}");
    }
    if verdict == "PASS" {
        Ok(())
    } else {
        Err("disk prefix-cache stress failed; see artifact details".into())
    }
}

fn run_concurrent_stress(
    cache_dir: &Path,
    workers: usize,
    iterations: usize,
    payload_bytes: usize,
    overlap_keys: usize,
) -> Result<serde_json::Value, String> {
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: workers * iterations + overlap_keys + 16,
    };
    DiskPrefixCache::with_policy(cache_dir, policy).map_err(|e| e.to_string())?;

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut children = Vec::new();
    for worker_id in 0..workers {
        let child = Command::new(&exe)
            .arg("--worker")
            .arg("--cache-dir")
            .arg(cache_dir)
            .arg("--worker-id")
            .arg(worker_id.to_string())
            .arg("--iterations")
            .arg(iterations.to_string())
            .arg("--payload-bytes")
            .arg(payload_bytes.to_string())
            .arg("--overlap-keys")
            .arg(overlap_keys.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn worker {worker_id}: {e}"))?;
        children.push((worker_id, child));
    }

    let mut rows = Vec::new();
    for (worker_id, child) in children {
        let output = child
            .wait_with_output()
            .map_err(|e| format!("wait worker {worker_id}: {e}"))?;
        let mut row: serde_json::Value = if output.stdout.is_empty() {
            json!({"worker_id": worker_id})
        } else {
            serde_json::from_slice(&output.stdout)
                .map_err(|e| format!("parse worker {worker_id} JSON: {e}"))?
        };
        row["exit_status"] = json!(output.status.code().unwrap_or(-1));
        if !output.stderr.is_empty() {
            row["stderr"] = json!(String::from_utf8_lossy(&output.stderr).to_string());
        }
        rows.push(row);
    }

    Ok(json!({
        "cache_dir": cache_dir,
        "policy": {
            "max_bytes": policy.max_bytes,
            "max_entries": policy.max_entries,
        },
        "workers": rows,
        "files_after": list_axkv_files(cache_dir),
    }))
}

fn run_worker(args: &[String]) -> Result<(), String> {
    let cache_dir = required_path(args, "--cache-dir")?;
    let worker_id = usize_after(args, "--worker-id").ok_or("--worker-id is required")?;
    let iterations = usize_after(args, "--iterations").unwrap_or(DEFAULT_ITERATIONS);
    let payload_bytes = usize_after(args, "--payload-bytes").unwrap_or(DEFAULT_PAYLOAD_BYTES);
    let overlap_keys = usize_after(args, "--overlap-keys").unwrap_or(DEFAULT_OVERLAP_KEYS);
    let cache = DiskPrefixCache::with_policy(
        cache_dir,
        DiskPrefixCachePolicy {
            max_bytes: u64::MAX,
            max_entries: usize::MAX,
        },
    )
    .map_err(|e| e.to_string())?;

    let mut insert_errors = 0u64;
    let mut read_errors = 0u64;
    let mut read_misses = 0u64;
    let mut corruption_load_failures = 0u64;
    let mut disk_evictions = 0u64;
    for i in 0..iterations {
        let shared_key = i % overlap_keys;
        let key = stress_key(shared_key);
        let expected = stress_entry(shared_key, payload_bytes);
        match cache.insert(&key, &expected) {
            Ok(outcome) => disk_evictions += u64::from(outcome.evictions),
            Err(_) => {
                insert_errors += 1;
                continue;
            }
        }
        match cache.get(&key) {
            Ok(Some(actual)) => {
                if actual.payload != expected.payload
                    || actual.prefill_output_token != expected.prefill_output_token
                {
                    corruption_load_failures += 1;
                }
            }
            Ok(None) => read_misses += 1,
            Err(_) => read_errors += 1,
        }
    }

    let row = json!({
        "worker_id": worker_id,
        "iterations": iterations,
        "insert_errors": insert_errors,
        "read_errors": read_errors,
        "read_misses": read_misses,
        "corruption_load_failures": corruption_load_failures,
        "disk_evictions": disk_evictions,
    });
    println!(
        "{}",
        serde_json::to_string(&row).map_err(|e| e.to_string())?
    );

    if insert_errors == 0 && read_errors == 0 && read_misses == 0 && corruption_load_failures == 0 {
        Ok(())
    } else {
        Err("worker observed cache failure".into())
    }
}

fn run_eviction_pressure(
    cache_dir: &Path,
    payload_bytes: usize,
) -> Result<serde_json::Value, String> {
    let policy = DiskPrefixCachePolicy {
        max_bytes: u64::MAX,
        max_entries: 2,
    };
    let cache = DiskPrefixCache::with_policy(cache_dir, policy).map_err(|e| e.to_string())?;
    let mut total_evictions = 0u64;
    let inserted = 5usize;
    for key_id in 0..inserted {
        let outcome = cache
            .insert(
                &stress_key(10_000 + key_id),
                &stress_entry(10_000 + key_id, payload_bytes),
            )
            .map_err(|e| e.to_string())?;
        total_evictions += u64::from(outcome.evictions);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let files = list_axkv_files(cache_dir);
    let newest_a = cache.contains(&stress_key(10_000 + inserted - 1));
    let newest_b = cache.contains(&stress_key(10_000 + inserted - 2));
    let oldest = cache.contains(&stress_key(10_000));
    let policy_passed = files.len() <= policy.max_entries
        && total_evictions >= 3
        && newest_a
        && newest_b
        && !oldest;
    Ok(json!({
        "cache_dir": cache_dir,
        "policy": {
            "max_bytes": policy.max_bytes,
            "max_entries": policy.max_entries,
        },
        "inserted_entries": inserted,
        "total_evictions": total_evictions,
        "files_after": files,
        "newest_entries_survived": newest_a && newest_b,
        "oldest_entry_evicted": !oldest,
        "policy_passed": policy_passed,
    }))
}

fn stress_key(key_id: usize) -> Vec<u8> {
    canonical_key_bytes(
        "stress-model",
        "stress-route",
        "stress-layout",
        16,
        16 * (key_id as u32 + 1),
        0x5eed_0000_u64 + key_id as u64,
    )
}

fn stress_entry(key_id: usize, payload_bytes: usize) -> DiskPrefixCacheEntry {
    let payload = (0..payload_bytes)
        .map(|offset| ((key_id + offset) & 0xff) as u8)
        .collect();
    DiskPrefixCacheEntry {
        payload,
        prefill_output_token: Some((key_id as u32).wrapping_add(17)),
    }
}

fn list_axkv_files(cache_dir: &Path) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return rows;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "axkv") {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        rows.push(json!({
            "name": entry.file_name().to_string_lossy(),
            "size_bytes": metadata.len(),
        }));
    }
    rows.sort_by_key(|row| {
        row["name"]
            .as_str()
            .map(ToOwned::to_owned)
            .unwrap_or_default()
    });
    rows
}

fn required_path(args: &[String], flag: &str) -> Result<PathBuf, String> {
    value_after(args, flag)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} is required"))
}

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

fn usize_after(args: &[String], flag: &str) -> Option<usize> {
    value_after(args, flag).and_then(|value| value.parse().ok())
}

fn unique_dir(label: &str) -> Result<PathBuf, String> {
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0)
    ));
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
