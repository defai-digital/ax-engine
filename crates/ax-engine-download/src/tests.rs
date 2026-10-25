#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;
use std::net::{TcpListener, TcpStream};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use std::thread::JoinHandle;

static NEXT: AtomicUsize = AtomicUsize::new(0);
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "ax-native-download-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&root).unwrap();
        Self(fs::canonicalize(root).unwrap())
    }
    fn options(&self) -> Options {
        Options {
            repository: "AutomatosX/test-pack".into(),
            revision: Some(COMMIT.into()),
            destination: Some(self.0.join("model")),
            cache: self.0.join("cache"),
            force: false,
            local_only: false,
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Server {
    hub: Hub,
    requests: Arc<Mutex<Vec<String>>>,
    thread: Option<JoinHandle<()>>,
}
impl Server {
    fn new(replies: Vec<Vec<u8>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = requests.clone();
        let thread = std::thread::spawn(move || {
            for reply in replies {
                let start = Instant::now();
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e)
                            if e.kind() == std::io::ErrorKind::WouldBlock
                                && start.elapsed() < Duration::from_secs(5) =>
                        {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(_) => return,
                    }
                };
                // Accepted sockets inherit O_NONBLOCK on macOS. The fixture
                // polls accept, but request parsing deliberately uses blocking IO.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                seen.lock().unwrap().push(read_request(&mut stream));
                stream.write_all(&reply).unwrap();
                stream.shutdown(std::net::Shutdown::Write).unwrap();
                let _ = stream.take(4096).read_to_end(&mut Vec::new());
            }
        });
        let hub = Hub {
            agent: ureq::AgentBuilder::new()
                .redirects(0)
                .timeout(Duration::from_secs(5))
                .build(),
            base: Url::parse(&format!("http://{address}")).unwrap(),
        };
        Self {
            hub,
            requests,
            thread: Some(thread),
        }
    }
    fn finish(mut self) -> Vec<String> {
        self.thread.take().unwrap().join().unwrap();
        let requests = self.requests.lock().unwrap().clone();
        for request in &requests {
            assert!(!request.to_lowercase().contains("authorization:"));
            assert!(!request.to_lowercase().contains("cookie:"));
        }
        requests
    }
}
fn read_request(stream: &mut TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut byte = [0];
    while bytes.len() < 32_768 && !bytes.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(0) => break,
            Ok(_) => bytes.push(byte[0]),
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => panic!("Mock request read failed: {error}"),
        }
    }
    String::from_utf8(bytes).unwrap()
}
fn response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
    let mut bytes = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n",
        body.len()
    )
    .into_bytes();
    bytes.extend(body);
    bytes
}
fn member(path: &str, bytes: &[u8], lfs: bool) -> Value {
    if lfs {
        json!({"rfilename":path,"size":bytes.len(),"lfs":{"sha256":format!("{:x}",Sha256::digest(bytes))}})
    } else {
        let mut digest = sha1::Sha1::new();
        digest.update(format!("blob {}\0", bytes.len()));
        digest.update(bytes);
        json!({"rfilename":path,"size":bytes.len(),"blobId":format!("{:x}",digest.finalize())})
    }
}
fn inventory() -> Value {
    json!({"sha":COMMIT,"private":false,"gated":false,"siblings":[member("config.json", b"{}",false),member("model.safetensors",b"weights",true)]})
}
fn replies(value: Value) -> Vec<Vec<u8>> {
    vec![
        response("200 OK", "", value.to_string().as_bytes()),
        response("200 OK", "", b"{}"),
        response("200 OK", "", b"weights"),
    ]
}
fn prepare(path: &Path) -> std::result::Result<(), String> {
    assert_eq!(fs::read(path.join("config.json")).unwrap(), b"{}");
    assert_eq!(
        fs::read(path.join("model.safetensors")).unwrap(),
        b"weights"
    );
    fs::write(path.join(MANIFEST), "{}").unwrap();
    Ok(())
}
#[test]
fn anonymous_download_checks_git_and_lfs_digests_and_pins_all_files() {
    let temp = Temp::new();
    let server = Server::new(replies(inventory()));
    let options = temp.options();
    let mut events = Vec::new();
    let result = download_with(&server.hub, &options, prepare, |d, t, f| {
        events.push((d, t, f.to_string()))
    })
    .unwrap();
    assert_eq!(result.commit.as_deref(), Some(COMMIT));
    assert_eq!(
        read_json(&result.destination.join(RECEIPT)).unwrap()["resolved_revision"],
        COMMIT
    );
    let requests = server.finish();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].contains(&format!("/revision/{COMMIT}?blobs=true")));
    assert!(requests[1].contains(&format!("/resolve/{COMMIT}/config.json")));
    assert!(
        requests[2].contains(&format!("/resolve/{COMMIT}/model.safetensors")),
        "{requests:?}"
    );
    assert_eq!(events.last().unwrap(), &(100, 100, "Ready".into()));
}
#[test]
fn third_party_and_force_offline_fail_before_network() {
    let temp = Temp::new();
    let mut options = temp.options();
    options.repository = "other/model".into();
    assert_eq!(
        download(&options, prepare, |_, _, _| {}).unwrap_err().code,
        "UNSUPPORTED_REPOSITORY"
    );
    options.repository = "AutomatosX/model".into();
    options.force = true;
    options.local_only = true;
    assert_eq!(
        download(&options, prepare, |_, _, _| {}).unwrap_err().code,
        "INVALID_ARGUMENTS"
    );
}
#[test]
fn private_gated_and_wrong_revision_fail_before_any_file_download() {
    for (field, value) in [
        ("private", json!(true)),
        ("gated", json!("auto")),
        ("sha", json!("1111111111111111111111111111111111111111")),
    ] {
        let temp = Temp::new();
        let mut metadata = inventory();
        metadata[field] = value;
        let server = Server::new(vec![response(
            "200 OK",
            "",
            metadata.to_string().as_bytes(),
        )]);
        assert!(download_with(&server.hub, &temp.options(), prepare, |_, _, _| {}).is_err());
        assert_eq!(server.finish().len(), 1);
        assert!(!temp.0.join("model").exists());
    }
}
#[test]
fn unsafe_and_duplicate_repository_paths_fail_before_file_download() {
    for path in [
        "../escape",
        "/absolute",
        "folder/../escape",
        "a\\b",
        "a//b",
        ".ax-native-state.json",
        "CONFIG.JSON",
    ] {
        let temp = Temp::new();
        let mut metadata = inventory();
        metadata["siblings"][1]["rfilename"] = json!(path);
        let server = Server::new(vec![response(
            "200 OK",
            "",
            metadata.to_string().as_bytes(),
        )]);
        assert_eq!(
            download_with(&server.hub, &temp.options(), prepare, |_, _, _| {})
                .unwrap_err()
                .code,
            "INVALID_METADATA"
        );
        assert_eq!(server.finish().len(), 1);
    }
}
#[test]
fn access_errors_are_actionable_and_do_not_suggest_login() {
    for (status, expected) in [
        (401, "ACCESS_DENIED"),
        (403, "ACCESS_DENIED"),
        (404, "REPOSITORY_NOT_FOUND"),
    ] {
        let temp = Temp::new();
        let server = Server::new(vec![response(
            &format!("{status} Error"),
            "",
            b"secret upstream diagnostic",
        )]);
        let error = download_with(&server.hub, &temp.options(), prepare, |_, _, _| {}).unwrap_err();
        assert_eq!(error.code, expected);
        assert!(!error.message.contains("secret"));
        assert!(!error.message.contains("login"));
        server.finish();
    }
}
#[test]
fn partial_transfer_resumes_and_full_response_restarts() {
    for ranged in [true, false] {
        let temp = Temp::new();
        let part = temp.0.join("partial");
        fs::write(&part, b"wei").unwrap();
        let response = if ranged {
            response(
                "206 Partial Content",
                "Content-Range: bytes 3-6/7\r\n",
                b"ghts",
            )
        } else {
            response("200 OK", "", b"weights")
        };
        let server = Server::new(vec![response]);
        let member = Member {
            path: "weights".into(),
            size: 7,
            digest: format!("{:x}", Sha256::digest(b"weights")),
        };
        fetch_file(
            &server.hub,
            server.hub.url(&["file"]).unwrap(),
            &member,
            &part,
            0,
            7,
            &mut |_, _, _| {},
        )
        .unwrap();
        assert_eq!(fs::read(&part).unwrap(), b"weights");
        assert!(
            server.finish()[0]
                .to_lowercase()
                .contains("range: bytes=3-")
        );
    }
}
#[test]
fn bad_range_preserves_partial_and_checksum_mismatch_discards_it() {
    let temp = Temp::new();
    let part = temp.0.join("partial");
    fs::write(&part, b"wei").unwrap();
    let server = Server::new(vec![response(
        "206 Partial Content",
        "Content-Range: bytes 2-6/7\r\n",
        b"ghts",
    )]);
    let member = Member {
        path: "weights".into(),
        size: 7,
        digest: format!("{:x}", Sha256::digest(b"weights")),
    };
    assert_eq!(
        fetch_file(
            &server.hub,
            server.hub.url(&["file"]).unwrap(),
            &member,
            &part,
            0,
            7,
            &mut |_, _, _| {}
        )
        .unwrap_err()
        .code,
        "INVALID_RESPONSE"
    );
    assert_eq!(fs::read(&part).unwrap(), b"wei");
    server.finish();
    let server = Server::new(vec![response("200 OK", "", b"invalid")]);
    assert_eq!(
        fetch_file(
            &server.hub,
            server.hub.url(&["file"]).unwrap(),
            &member,
            &part,
            0,
            7,
            &mut |_, _, _| {}
        )
        .unwrap_err()
        .code,
        "INTEGRITY_ERROR"
    );
    assert!(!part.exists());
    server.finish();
}
#[test]
fn interrupted_transfer_keeps_resumable_bytes() {
    let temp = Temp::new();
    let part = temp.0.join("partial");
    let server = Server::new(vec![
        b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nwei".to_vec(),
    ]);
    let member = Member {
        path: "weights".into(),
        size: 7,
        digest: format!("{:x}", Sha256::digest(b"weights")),
    };
    let error = fetch_file(
        &server.hub,
        server.hub.url(&["file"]).unwrap(),
        &member,
        &part,
        0,
        7,
        &mut |_, _, _| {},
    )
    .unwrap_err();
    assert!(error.retryable);
    assert_eq!(fs::read(part).unwrap(), b"wei");
    server.finish();
}
#[test]
fn manifest_failure_preserves_installed_model_and_retry_uses_staged_bytes() {
    let temp = Temp::new();
    let mut options = temp.options();
    let destination = options.destination.clone().unwrap();
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("old"), b"keep").unwrap();
    write_json(
        &destination.join(RECEIPT),
        &json!({"repo_id":options.repository,"revision":COMMIT}),
    )
    .unwrap();
    options.force = true;
    let server = Server::new(replies(inventory()));
    let error = download_with(
        &server.hub,
        &options,
        |_| Err("unsupported model".into()),
        |_, _, _| {},
    )
    .unwrap_err();
    assert_eq!(error.code, "INVALID_MODEL");
    assert_eq!(fs::read(destination.join("old")).unwrap(), b"keep");
    server.finish();
    let server = Server::new(vec![response(
        "200 OK",
        "",
        inventory().to_string().as_bytes(),
    )]);
    download_with(&server.hub, &options, prepare, |_, _, _| {}).unwrap();
    assert!(!destination.join("old").exists());
    assert_eq!(server.finish().len(), 1);
}
#[test]
fn unrelated_destination_and_busy_lock_are_preserved() {
    let temp = Temp::new();
    let mut options = temp.options();
    options.force = true;
    let destination = options.destination.as_ref().unwrap();
    fs::create_dir(destination).unwrap();
    fs::write(destination.join("user-data"), b"keep").unwrap();
    let server = Server::new(vec![response(
        "200 OK",
        "",
        inventory().to_string().as_bytes(),
    )]);
    assert_eq!(
        download_with(&server.hub, &options, prepare, |_, _, _| {})
            .unwrap_err()
            .code,
        "DESTINATION_CONFLICT"
    );
    server.finish();
    fs::remove_dir_all(destination).unwrap();
    let lock = File::create(temp.0.join(".model.ax-download.lock")).unwrap();
    lock.lock_exclusive().unwrap();
    let server = Server::new(vec![response(
        "200 OK",
        "",
        inventory().to_string().as_bytes(),
    )]);
    assert_eq!(
        download_with(&server.hub, &options, prepare, |_, _, _| {})
            .unwrap_err()
            .code,
        "DOWNLOAD_BUSY"
    );
    server.finish();
}
#[test]
fn legacy_hf_cache_offline_import_accepts_internal_blobs_only() {
    use std::os::unix::fs::symlink;
    for external in [false, true] {
        let temp = Temp::new();
        let mut options = temp.options();
        options.local_only = true;
        let repository = options.cache.join("models--AutomatosX--test-pack");
        let snapshot = repository.join("snapshots").join(COMMIT);
        fs::create_dir_all(&snapshot).unwrap();
        let blobs = repository.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        fs::write(blobs.join("config"), b"{}").unwrap();
        fs::write(blobs.join("weights"), b"weights").unwrap();
        symlink("../../blobs/config", snapshot.join("config.json")).unwrap();
        let weights = if external {
            let p = temp.0.join("outside");
            fs::write(&p, b"weights").unwrap();
            p
        } else {
            blobs.join("weights")
        };
        symlink(weights, snapshot.join("model.safetensors")).unwrap();
        let result = download(&options, prepare, |_, _, _| {});
        if external {
            assert_eq!(result.unwrap_err().code, "INVALID_MODEL");
        } else {
            assert!(result.is_ok());
        }
    }
}
#[test]
fn retry_headers_and_terminal_failure_contract() {
    let response = ureq::Response::new(429, "Busy", "{}").unwrap();
    assert_eq!(retry_delay(&response, 1), 4);
    let response: ureq::Response = "HTTP/1.1 429 Busy\r\nRetry-After: 14\r\n\r\n"
        .parse()
        .unwrap();
    assert_eq!(retry_delay(&response, 0), 14);
    let response: ureq::Response = "HTTP/1.1 429 Busy\r\nRateLimit: \"resolvers\";r=0;t=27\r\n\r\n"
        .parse()
        .unwrap();
    assert_eq!(retry_delay(&response, 0), 27);
    let failure = Err(http_failure(429, None));
    let value = summary("AutomatosX/model", Some(COMMIT), &failure, None);
    assert_eq!(value["status"], "download_failed");
    assert_eq!(value["failure"]["code"], "RATE_LIMITED");
    assert_eq!(value["failure"]["retryable"], true);
}

#[test]
fn failed_default_download_is_never_visible_as_a_snapshot() {
    let temp = Temp::new();
    let mut options = temp.options();
    options.destination = None;
    let server = Server::new(replies(inventory()));
    assert!(
        download_with(
            &server.hub,
            &options,
            |_| Err("invalid manifest".into()),
            |_, _, _| {}
        )
        .is_err()
    );
    let repository = options.cache.join("models--AutomatosX--test-pack");
    assert_eq!(
        fs::read_dir(repository.join("snapshots")).unwrap().count(),
        0
    );
    assert!(repository.join(".ax-downloads").is_dir());
    server.finish();
}

#[test]
fn explicit_completed_download_can_be_used_offline_without_hub_cache() {
    let temp = Temp::new();
    let mut options = temp.options();
    let server = Server::new(replies(inventory()));
    download_with(&server.hub, &options, prepare, |_, _, _| {}).unwrap();
    server.finish();
    assert!(!options.cache.exists());
    options.local_only = true;
    let result = download(&options, prepare, |_, _, _| {}).unwrap();
    assert_eq!(result.destination, options.destination.unwrap());
    assert_eq!(result.commit.as_deref(), Some(COMMIT));
}

#[test]
fn foreign_backup_and_linked_stage_cannot_be_removed_or_written() {
    use std::os::unix::fs::symlink;
    for linked in [false, true] {
        let temp = Temp::new();
        let options = temp.options();
        let outside = temp.0.join("keep");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("data"), b"preserve").unwrap();
        if linked {
            symlink(&outside, temp.0.join(".model.ax-download")).unwrap();
        } else {
            fs::rename(&outside, temp.0.join(".model.ax-download.previous")).unwrap();
        }
        let server = Server::new(vec![response(
            "200 OK",
            "",
            inventory().to_string().as_bytes(),
        )]);
        assert!(download_with(&server.hub, &options, prepare, |_, _, _| {}).is_err());
        let preserved = if linked {
            outside
        } else {
            temp.0.join(".model.ax-download.previous")
        };
        assert_eq!(fs::read(preserved.join("data")).unwrap(), b"preserve");
        server.finish();
    }
}

#[test]
#[ignore = "Read-only public Hub integration: downloads config.json only, never weights"]
fn public_hub_pinned_metadata_and_config_download() {
    let temp = Temp::new();
    let options = Options {
        repository: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP".into(),
        revision: Some("b2c5354f779e430d0c1733143db848a72b71c16e".into()),
        destination: None,
        cache: temp.0.clone(),
        force: false,
        local_only: false,
    };
    let hub = Hub::public().unwrap();
    let inventory = hub.inventory(&options, &mut |_, _, _| {}).unwrap();
    let member = inventory
        .files
        .iter()
        .find(|file| file.path == "config.json")
        .unwrap();
    assert!(member.size < 1024 * 1024);
    let url = hub
        .url(&[
            "AutomatosX",
            "AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
            "resolve",
            &inventory.commit,
            "config.json",
        ])
        .unwrap();
    fetch_file(
        &hub,
        url,
        member,
        &temp.0.join("config.json"),
        0,
        member.size,
        &mut |_, _, _| {},
    )
    .unwrap();
    assert!(read_json(&temp.0.join("config.json")).unwrap().is_object());
}

#[test]
fn journal_recovers_an_interrupted_swap_of_an_empty_destination() {
    let temp = Temp::new();
    let options = temp.options();
    let destination = options.destination.as_ref().unwrap();
    let backup = temp.0.join(".model.ax-download.previous");
    fs::create_dir(&backup).unwrap();
    write_json(
        &temp.0.join(".model.ax-download.transaction.json"),
        &json!({"repository":options.repository,"destination":destination}),
    )
    .unwrap();
    let server = Server::new(replies(inventory()));
    download_with(&server.hub, &options, prepare, |_, _, _| {}).unwrap();
    assert!(destination.join(MANIFEST).is_file());
    assert!(!backup.exists());
    assert!(!temp.0.join(".model.ax-download.transaction.json").exists());
    server.finish();
}
