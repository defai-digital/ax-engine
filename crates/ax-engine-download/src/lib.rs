//! Anonymous downloads of public AutomatosX model packs. No Python, login,
//! token discovery, custom Hub endpoint, or inference is involved here.
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};
use url::Url;

const RECEIPT: &str = ".ax-engine-download.json";
const STATE: &str = ".ax-native-state.json";
const MANIFEST: &str = "model-manifest.json";

#[derive(Debug)]
pub struct Failure {
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
    pub http_status: Option<u16>,
}
impl Failure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: false,
            http_status: None,
        }
    }
    pub fn json(&self) -> Value {
        json!({"code":self.code,"retryable":self.retryable,"http_status":self.http_status})
    }
}
impl From<std::io::Error> for Failure {
    fn from(error: std::io::Error) -> Self {
        let code = if matches!(error.raw_os_error(), Some(28 | 69 | 122)) {
            "DISK_INSUFFICIENT"
        } else {
            "IO_ERROR"
        };
        Self::new(code, error.to_string())
    }
}
type Result<T> = std::result::Result<T, Failure>;

pub struct Options {
    pub repository: String,
    pub revision: Option<String>,
    pub destination: Option<PathBuf>,
    pub cache: PathBuf,
    pub force: bool,
    pub local_only: bool,
}
#[derive(Debug)]
pub struct Completed {
    pub destination: PathBuf,
    pub commit: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Member {
    path: String,
    size: u64,
    digest: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
struct Inventory {
    repository: String,
    commit: String,
    files: Vec<Member>,
}

pub fn cache_root() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HF_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::var_os("XDG_CACHE_HOME")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| {
                            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                                .join(".cache")
                        })
                        .join("huggingface")
                })
                .join("hub")
        })
}

/// Restrict product downloads to the publisher's namespace. Catalog admission
/// and pinned product revisions remain the responsibility of the caller.
pub fn supported_repository(repository: &str) -> bool {
    repository.strip_prefix("AutomatosX/").is_some_and(|name| {
        !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
    })
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn relative(value: &str) -> Result<&Path> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || value.chars().any(char::is_control)
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
        || value
            .split('/')
            .any(|part| part.starts_with(".ax-") || part == ".git")
    {
        return Err(Failure::new(
            "INVALID_METADATA",
            "The repository contains an unsafe file path.",
        ));
    }
    Ok(path)
}
fn absolute(path: &Path) -> Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    if path.components().any(|c| matches!(c, Component::ParentDir)) || path.file_name().is_none() {
        return Err(Failure::new(
            "INVALID_DESTINATION",
            "Choose a dedicated model folder without parent traversal.",
        ));
    }
    Ok(path)
}
/// Refuse linked writable paths. Read-only legacy Hub snapshot links are handled
/// separately, and may resolve only inside their repository cache directory.
fn no_links(path: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    for part in path.components() {
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err(Failure::new(
                    "INVALID_DESTINATION",
                    "A writable download path contains a symbolic link.",
                ));
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}
fn read_json(path: &Path) -> Result<Value> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(Failure::new(
            "INVALID_METADATA",
            "Model metadata is too large.",
        ));
    }
    serde_json::from_slice(&bytes)
        .map_err(|_| Failure::new("INVALID_METADATA", "Model metadata is invalid JSON."))
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    no_links(path)?;
    let temp = path.with_extension("json.new");
    no_links(&temp)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .or_else(|error| {
            if error.kind() != std::io::ErrorKind::AlreadyExists {
                return Err(error);
            }
            fs::remove_file(&temp)?;
            OpenOptions::new().write(true).create_new(true).open(&temp)
        })?;
    serde_json::to_writer(&mut file, value)
        .map_err(|_| Failure::new("IO_ERROR", "Cannot write model metadata."))?;
    file.sync_all()?;
    fs::rename(temp, path)?;
    Ok(())
}
fn digest_file(path: &Path, member: &Member) -> Result<bool> {
    if !path.is_file() || fs::metadata(path)?.len() != member.size {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut sha256 = Sha256::new();
    let mut sha1 = sha1::Sha1::new();
    sha1.update(format!("blob {}\0", member.size));
    let mut buffer = [0u8; 256 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if member.digest.len() == 64 {
            sha256.update(&buffer[..count]);
        } else {
            sha1.update(&buffer[..count]);
        }
    }
    let actual = if member.digest.len() == 64 {
        format!("{:x}", sha256.finalize())
    } else {
        format!("{:x}", sha1.finalize())
    };
    Ok(actual.eq_ignore_ascii_case(&member.digest))
}

struct Hub {
    agent: ureq::Agent,
    base: Url,
}
impl Hub {
    fn public() -> Result<Self> {
        Ok(Self {
            agent: ureq::AgentBuilder::new()
                .redirects(0)
                .timeout_connect(Duration::from_secs(20))
                .timeout_read(Duration::from_secs(60))
                .user_agent("AX-Engine-Public-Downloader/1")
                .build(),
            base: Url::parse("https://huggingface.co")
                .map_err(|_| Failure::new("INVALID_ENDPOINT", "Invalid public Hub endpoint."))?,
        })
    }
    fn url(&self, segments: &[&str]) -> Result<Url> {
        let mut url = self.base.clone();
        url.path_segments_mut()
            .map_err(|_| Failure::new("INVALID_METADATA", "Invalid Hub endpoint."))?
            .extend(segments);
        Ok(url)
    }
    fn request(
        &self,
        url: Url,
        offset: u64,
        progress: &mut impl FnMut(u64, u64, &str),
    ) -> Result<ureq::Response> {
        let mut url = url;
        for _ in 0..8 {
            // Never load or forward credentials. CDN redirects need only their
            // server-provided signed URL; transport errors never expose that URL.
            if !url.username().is_empty() || url.password().is_some() || !self.allowed(&url) {
                return Err(Failure::new(
                    "INVALID_REDIRECT",
                    "The Hub returned an unsafe download redirect.",
                ));
            }
            let mut response = None;
            for attempt in 0..4 {
                let mut request = self
                    .agent
                    .get(url.as_str())
                    .set("Accept-Encoding", "identity");
                if offset > 0 {
                    request = request.set("Range", &format!("bytes={offset}-"));
                }
                match request.call() {
                    Ok(value) => {
                        response = Some(value);
                        break;
                    }
                    Err(ureq::Error::Status(status, value))
                        if (status == 429 || status == 408 || status >= 500) && attempt < 3 =>
                    {
                        let wait = retry_delay(&value, attempt);
                        for remaining in (1..=wait).rev() {
                            progress(
                                0,
                                0,
                                &format!("Download service is busy. Retrying in {remaining}s"),
                            );
                            std::thread::sleep(Duration::from_secs(1));
                        }
                    }
                    Err(ureq::Error::Status(status, value)) => {
                        return Err(http_failure(status, value.header("X-Error-Code")));
                    }
                    Err(ureq::Error::Transport(_)) => {
                        return Err(Failure {
                        code: "NETWORK_ERROR",
                        message:
                            "Cannot reach Hugging Face. Check your connection and retry to resume."
                                .into(),
                        retryable: true,
                        http_status: None,
                    });
                    }
                }
            }
            let response = response
                .ok_or_else(|| Failure::new("NETWORK_ERROR", "Download retries were exhausted."))?;
            if matches!(response.status(), 301 | 302 | 303 | 307 | 308) {
                url = response
                    .header("Location")
                    .and_then(|value| url.join(value).ok())
                    .ok_or_else(|| {
                        Failure::new("INVALID_REDIRECT", "The Hub returned an invalid redirect.")
                    })?;
                continue;
            }
            return Ok(response);
        }
        Err(Failure::new(
            "INVALID_REDIRECT",
            "Too many download redirects.",
        ))
    }
    fn allowed(&self, url: &Url) -> bool {
        if url.scheme() == "https" {
            return true;
        }
        #[cfg(test)]
        if url.scheme() == "http" && url.host_str() == Some("127.0.0.1") {
            return true;
        }
        false
    }
    fn inventory(
        &self,
        options: &Options,
        progress: &mut impl FnMut(u64, u64, &str),
    ) -> Result<Inventory> {
        let (owner, name) = options.repository.split_once('/').ok_or_else(|| {
            Failure::new(
                "UNSUPPORTED_REPOSITORY",
                "Choose an AX Engine public model.",
            )
        })?;
        let revision = options.revision.as_deref().unwrap_or("main");
        let mut url = self.url(&["api", "models", owner, name, "revision", revision])?;
        url.set_query(Some("blobs=true"));
        let response = self.request(url, 0, progress)?;
        let mut bytes = Vec::new();
        response
            .into_reader()
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(Failure::new(
                "INVALID_METADATA",
                "The repository listing is too large.",
            ));
        }
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
            Failure::new(
                "INVALID_METADATA",
                "The Hub returned invalid model metadata.",
            )
        })?;
        if value.get("private") != Some(&json!(false))
            || value.get("gated").is_some_and(|v| v != &json!(false))
        {
            return Err(Failure::new(
                "ACCESS_DENIED",
                "AX Engine downloads only public, unrestricted AutomatosX models. Choose another public model.",
            ));
        }
        let commit = value["sha"]
            .as_str()
            .filter(|s| hex(s, 40))
            .ok_or_else(|| Failure::new("INVALID_METADATA", "Missing immutable model revision."))?;
        if hex(revision, 40) && !revision.eq_ignore_ascii_case(commit) {
            return Err(Failure::new(
                "INVALID_METADATA",
                "The Hub returned a different model revision.",
            ));
        }
        let rows = value["siblings"]
            .as_array()
            .ok_or_else(|| Failure::new("INVALID_METADATA", "Missing model file listing."))?;
        let mut files = Vec::new();
        let mut names = HashSet::new();
        for row in rows {
            let path = row["rfilename"]
                .as_str()
                .ok_or_else(|| Failure::new("INVALID_METADATA", "Missing model file name."))?;
            relative(path)?;
            if !names.insert(path.to_lowercase()) {
                return Err(Failure::new(
                    "INVALID_METADATA",
                    "Duplicate or case-colliding model file names.",
                ));
            }
            let size = row["size"]
                .as_u64()
                .ok_or_else(|| Failure::new("INVALID_METADATA", "Missing model file size."))?;
            let digest = if row.get("lfs").is_some_and(|v| !v.is_null()) {
                row["lfs"]["sha256"].as_str().filter(|s| hex(s, 64))
            } else {
                row["blobId"].as_str().filter(|s| hex(s, 40))
            }
            .ok_or_else(|| Failure::new("INVALID_METADATA", "Missing model content checksum."))?;
            files.push(Member {
                path: path.into(),
                size,
                digest: digest.into(),
            });
        }
        if !files.iter().any(|f| f.path == "config.json")
            || !files.iter().any(|f| f.path.ends_with(".safetensors"))
        {
            return Err(Failure::new(
                "INVALID_MODEL",
                "The repository does not contain an MLX safetensors model.",
            ));
        }
        Ok(Inventory {
            repository: options.repository.clone(),
            commit: commit.into(),
            files,
        })
    }
}
fn retry_delay(response: &ureq::Response, attempt: u32) -> u64 {
    response
        .header("Retry-After")
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            response.header("RateLimit").and_then(|s| {
                s.split(';')
                    .find_map(|part| part.trim().strip_prefix("t=")?.parse::<u64>().ok())
            })
        })
        .unwrap_or(2u64.pow(attempt + 1))
        .clamp(1, 600)
}
fn http_failure(status: u16, code: Option<&str>) -> Failure {
    let (kind, message) = match (status, code) {
        (_, Some("RevisionNotFound")) => (
            "REVISION_NOT_FOUND",
            "This model revision is unavailable. Refresh the model catalog.",
        ),
        (401 | 403, _) => (
            "ACCESS_DENIED",
            "This model is not publicly downloadable. AX Engine supports public, unrestricted AutomatosX models only.",
        ),
        (404, _) => (
            "REPOSITORY_NOT_FOUND",
            "This public model or revision could not be found. Refresh the model catalog.",
        ),
        (429, _) => (
            "RATE_LIMITED",
            "Hugging Face is limiting downloads. Retry later to resume; no token is required.",
        ),
        _ => (
            "NETWORK_ERROR",
            "The model download service is unavailable. Retry to resume.",
        ),
    };
    Failure {
        code: kind,
        message: message.into(),
        retryable: status == 429 || status == 408 || status >= 500,
        http_status: Some(status),
    }
}

/// Prepare must generate and validate the native manifest without evaluating
/// weights. It runs against staging; failures leave the installed model intact.
pub fn download(
    options: &Options,
    prepare: impl FnMut(&Path) -> std::result::Result<(), String>,
    progress: impl FnMut(u64, u64, &str),
) -> Result<Completed> {
    download_with(&Hub::public()?, options, prepare, progress)
}
fn download_with(
    hub: &Hub,
    options: &Options,
    mut prepare: impl FnMut(&Path) -> std::result::Result<(), String>,
    mut progress: impl FnMut(u64, u64, &str),
) -> Result<Completed> {
    if options.force && options.local_only {
        return Err(Failure::new(
            "INVALID_ARGUMENTS",
            "--force cannot be combined with --local-only.",
        ));
    }
    let local = Path::new(&options.repository).is_dir();
    if !local && !supported_repository(&options.repository) {
        return Err(Failure::new(
            "UNSUPPORTED_REPOSITORY",
            "Built-in downloads support public AutomatosX models only. Download other models yourself and use ax-engine serve with a local model path.",
        ));
    }
    if let Some(revision) = &options.revision {
        relative(revision)?;
    }
    let mut offline_destination = None;
    if options.local_only
        && let Some(destination) = &options.destination
    {
        let receipt = read_json(&destination.join(RECEIPT)).unwrap_or(Value::Null);
        if receipt["repo_id"].as_str() == Some(&options.repository)
            && receipt["revision"] == json!(options.revision)
        {
            offline_destination = Some((
                fs::canonicalize(destination)?,
                receipt["resolved_revision"].as_str().map(String::from),
            ));
        }
    }
    if local && options.destination.is_none() {
        let source = fs::canonicalize(&options.repository)?;
        prepare(&source).map_err(|e| Failure::new("INVALID_MODEL", e))?;
        return Ok(Completed {
            destination: source,
            commit: None,
        });
    }
    let cache = absolute(&options.cache)?;
    let repo_cache = cache.join(format!("models--{}", options.repository.replace('/', "--")));
    let mut source = None;
    let inventory = if let Some((path, commit)) = offline_destination {
        source = Some(path);
        commit.map(|commit| Inventory {
            repository: options.repository.clone(),
            commit,
            files: Vec::new(),
        })
    } else if local {
        source = Some(fs::canonicalize(&options.repository)?);
        None
    } else if options.local_only {
        let revision = options.revision.as_deref().unwrap_or("main");
        let commit = if hex(revision, 40) {
            revision.to_owned()
        } else {
            relative(revision)?;
            fs::read_to_string(repo_cache.join("refs").join(revision))
                .unwrap_or_default()
                .trim()
                .to_owned()
        };
        if !hex(&commit, 40) {
            return Err(Failure::new(
                "LOCAL_CACHE_MISSING",
                "No matching local snapshot is available; offline mode never connects to the Hub.",
            ));
        }
        let snapshot = repo_cache.join("snapshots").join(&commit);
        if !snapshot.is_dir() {
            return Err(Failure::new(
                "LOCAL_CACHE_MISSING",
                "The requested model revision is not cached.",
            ));
        }
        source = Some(snapshot);
        Some(Inventory {
            repository: options.repository.clone(),
            commit,
            files: Vec::new(),
        })
    } else {
        progress(0, 100, "Checking public model files");
        Some(hub.inventory(options, &mut progress)?)
    };
    let destination = absolute(&options.destination.clone().unwrap_or_else(|| {
        repo_cache.join("snapshots").join(
            inventory
                .as_ref()
                .map(|i| i.commit.as_str())
                .unwrap_or("local"),
        )
    }))?;
    // Canonicalize the existing parent (macOS /tmp and /var are system links),
    // then forbid links within the writable transaction paths.
    let parent = destination
        .parent()
        .ok_or_else(|| Failure::new("INVALID_DESTINATION", "Invalid model destination."))?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    let name = destination
        .file_name()
        .ok_or_else(|| Failure::new("INVALID_DESTINATION", "Invalid model folder."))?;
    let destination = parent.join(name);
    no_links(&destination)?;
    // Keep incomplete default-cache downloads outside snapshots so other Hub
    // clients cannot discover or load an uncommitted model directory.
    let transaction_parent = if options.destination.is_none() {
        parent
            .parent()
            .ok_or_else(|| Failure::new("INVALID_DESTINATION", "Invalid cache directory."))?
            .join(".ax-downloads")
    } else {
        parent.clone()
    };
    no_links(&transaction_parent)?;
    fs::create_dir_all(&transaction_parent)?;
    let prefix = format!(".{}.ax-download", name.to_string_lossy());
    let stage = transaction_parent.join(&prefix);
    let backup = transaction_parent.join(format!("{prefix}.previous"));
    let journal = transaction_parent.join(format!("{prefix}.transaction.json"));
    let transaction = json!({"repository": options.repository, "destination": destination});
    let lock_path = transaction_parent.join(format!("{prefix}.lock"));
    no_links(&lock_path)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock_path)?;
    lock.try_lock_exclusive().map_err(|_| {
        Failure::new(
            "DOWNLOAD_BUSY",
            "Another download is using this destination.",
        )
    })?;
    no_links(&stage)?;
    no_links(&backup)?;
    no_links(&journal)?;
    if backup.exists() {
        if read_json(&journal).ok().as_ref() != Some(&transaction) {
            return Err(Failure::new(
                "DESTINATION_CONFLICT",
                "An unrelated backup folder occupies the download transaction path.",
            ));
        }
        if !destination.exists() {
            fs::rename(&backup, &destination)?;
        } else {
            if read_json(&destination.join(RECEIPT))?["repo_id"].as_str()
                != Some(&options.repository)
            {
                return Err(Failure::new(
                    "DESTINATION_CONFLICT",
                    "Recovery found an unrelated destination. The previous model was preserved.",
                ));
            }
            fs::remove_dir_all(&backup)?;
        }
    }
    if source
        .as_ref()
        .is_some_and(|s| fs::canonicalize(s).ok() == fs::canonicalize(&destination).ok())
    {
        prepare(&destination).map_err(|e| Failure::new("INVALID_MODEL", e))?;
        return Ok(Completed {
            destination,
            commit: inventory.map(|i| i.commit),
        });
    }
    if destination.exists() && fs::read_dir(&destination)?.next().is_some() {
        let receipt = read_json(&destination.join(RECEIPT)).unwrap_or(Value::Null);
        let is_snapshot = destination
            .starts_with(fs::canonicalize(&repo_cache).unwrap_or(repo_cache.clone()))
            && destination
                .parent()
                .is_some_and(|p| p.ends_with("snapshots"));
        if receipt["repo_id"].as_str() != Some(&options.repository) && !is_snapshot {
            return Err(Failure::new(
                "DESTINATION_CONFLICT",
                "This folder contains another model or unrelated files. Choose an empty model folder.",
            ));
        }
        if !options.force && !is_snapshot && receipt["revision"] != json!(options.revision) {
            return Err(Failure::new(
                "DESTINATION_CONFLICT",
                "This folder contains another revision. Use --force to replace this model.",
            ));
        }
    }
    let identity = json!({"repository":options.repository,"commit":inventory.as_ref().map(|i| &i.commit),"revision":options.revision});
    if stage.exists() {
        if read_json(&stage.join(STATE))? != identity {
            return Err(Failure::new(
                "DESTINATION_CONFLICT",
                "A different interrupted download uses this folder. Choose another destination.",
            ));
        }
    } else {
        fs::create_dir(&stage)?;
        write_json(&stage.join(STATE), &identity)?;
    }
    let operation: Result<()> = (|| {
        if let Some(source) = &source {
            let boundary = if local {
                source.clone()
            } else {
                fs::canonicalize(&repo_cache)?
            };
            copy_tree(source, &stage, &boundary)?;
        } else if let Some(inventory) = &inventory {
            fetch_files(hub, options, inventory, &stage, &destination, &mut progress)?;
        }
        progress(95, 100, "Validating model manifest");
        prepare(&stage).map_err(|e| Failure::new("INVALID_MODEL", e))?;
        write_json(
            &stage.join(RECEIPT),
            &json!({"schema_version":"ax.download_provenance.v1","repo_id":options.repository,"revision":options.revision,"resolved_revision":inventory.as_ref().map(|i| &i.commit)}),
        )?;
        if destination.exists() {
            write_json(&journal, &transaction)?;
            fs::rename(&destination, &backup)?;
        }
        if let Err(error) = fs::rename(&stage, &destination) {
            if backup.exists() {
                fs::rename(&backup, &destination)?;
            }
            return Err(error.into());
        }
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        if journal.exists() {
            fs::remove_file(&journal)?;
        }
        Ok(())
    })();
    operation?;
    if options.destination.is_none()
        && let Some(inventory) = &inventory
    {
        let revision = options.revision.as_deref().unwrap_or("main");
        if !hex(revision, 40) {
            relative(revision)?;
            let path = repo_cache.join("refs").join(revision);
            no_links(&path)?;
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let temp = path.with_extension(format!("ax-new-{}", std::process::id()));
            no_links(&temp)?;
            // Replacement avoids modifying any existing linked inode.
            if temp.exists() {
                fs::remove_file(&temp)?;
            }
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp)?;
            file.write_all(inventory.commit.as_bytes())?;
            file.sync_all()?;
            fs::rename(temp, path)?;
        }
    }
    progress(100, 100, "Ready");
    Ok(Completed {
        destination,
        commit: inventory.map(|i| i.commit),
    })
}

fn copy_tree(source: &Path, destination: &Path, boundary: &Path) -> Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_string = name.to_string_lossy();
        if name_string.starts_with(".ax-") || name_string == ".git" {
            continue;
        }
        relative(&name_string)?;
        let resolved = fs::canonicalize(entry.path())?;
        if !resolved.starts_with(boundary) {
            return Err(Failure::new(
                "INVALID_MODEL",
                "A model link points outside its model or cache folder.",
            ));
        }
        let target = destination.join(name);
        no_links(&target)?;
        if resolved.is_dir() {
            if entry.file_type()?.is_symlink() {
                return Err(Failure::new(
                    "INVALID_MODEL",
                    "Linked model directories are unsupported.",
                ));
            }
            fs::create_dir_all(&target)?;
            copy_tree(&resolved, &target, boundary)?;
        } else {
            if target.exists() {
                fs::remove_file(&target)?;
            }
            fs::copy(resolved, target)?;
        }
    }
    Ok(())
}
fn fetch_files(
    hub: &Hub,
    options: &Options,
    inventory: &Inventory,
    stage: &Path,
    existing: &Path,
    progress: &mut impl FnMut(u64, u64, &str),
) -> Result<()> {
    let total = inventory
        .files
        .iter()
        .try_fold(0u64, |sum, f| sum.checked_add(f.size))
        .ok_or_else(|| Failure::new("INVALID_METADATA", "Model size overflow."))?;
    let mut missing = 0u64;
    for member in &inventory.files {
        let target = stage.join(relative(&member.path)?);
        no_links(&target)?;
        if !digest_file(&target, member)? {
            let part = stage
                .join(".ax-parts")
                .join(format!("{:x}", Sha256::digest(member.path.as_bytes())));
            no_links(&part)?;
            let partial_size = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            let reusable = if partial_size < member.size {
                partial_size
            } else if digest_file(&part, member)? {
                member.size
            } else {
                0
            };
            missing = missing.saturating_add(member.size - reusable);
        }
    }
    if fs2::available_space(stage)? < missing.saturating_add(64 * 1024 * 1024) {
        return Err(Failure::new(
            "DISK_INSUFFICIENT",
            "Not enough free disk space to stage and verify this model. Existing models were preserved.",
        ));
    }
    let parts = stage.join(".ax-parts");
    no_links(&parts)?;
    fs::create_dir_all(&parts)?;
    let (owner, name) = inventory
        .repository
        .split_once('/')
        .ok_or_else(|| Failure::new("INVALID_METADATA", "Invalid repository."))?;
    let mut done = 0u64;
    for member in &inventory.files {
        let target = stage.join(&member.path);
        if let Some(parent) = target.parent() {
            no_links(parent)?;
            fs::create_dir_all(parent)?;
        }
        if !digest_file(&target, member)? {
            let previous = existing.join(&member.path);
            if !options.force && previous.is_file() && digest_file(&previous, member)? {
                // Never hard-link staged mutable artifacts to a previous install.
                if target.exists() {
                    fs::remove_file(&target)?;
                }
                fs::copy(&previous, &target)?;
            } else {
                let part = parts.join(format!("{:x}", Sha256::digest(member.path.as_bytes())));
                no_links(&part)?;
                let mut url = hub.url(&[owner, name, "resolve", &inventory.commit])?;
                url.path_segments_mut()
                    .map_err(|_| Failure::new("INVALID_METADATA", "Invalid download URL."))?
                    .extend(member.path.split('/'));
                let mut last = None;
                for attempt in 0..3 {
                    match fetch_file(hub, url.clone(), member, &part, done, total, progress) {
                        Ok(()) => {
                            last = None;
                            break;
                        }
                        Err(error) => {
                            let retry = error.retryable;
                            last = Some(error);
                            if !retry || attempt == 2 {
                                break;
                            }
                        }
                    }
                }
                if let Some(error) = last {
                    return Err(error);
                }
                fs::rename(&part, &target)?;
            }
        }
        done += member.size;
        progress(
            done.saturating_mul(90),
            total.saturating_mul(100),
            &member.path,
        );
    }
    fs::remove_dir_all(parts)?;
    Ok(())
}
fn fetch_file(
    hub: &Hub,
    url: Url,
    member: &Member,
    part: &Path,
    done: u64,
    total: u64,
    progress: &mut impl FnMut(u64, u64, &str),
) -> Result<()> {
    let mut offset = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if offset == member.size && digest_file(part, member)? {
        return Ok(());
    }
    if offset >= member.size && part.exists() {
        fs::remove_file(part)?;
        offset = 0;
    }
    let response = hub.request(url, offset, progress)?;
    let append = offset > 0 && response.status() == 206;
    if append {
        let expected = format!(
            "bytes {}-{}/{}",
            offset,
            member.size.saturating_sub(1),
            member.size
        );
        if response.header("Content-Range") != Some(expected.as_str()) {
            return Err(Failure::new(
                "INVALID_RESPONSE",
                "The server returned an inconsistent resume range.",
            ));
        }
    } else if response.status() == 200 {
        offset = 0;
    } else {
        return Err(Failure::new(
            "INVALID_RESPONSE",
            "The server returned an unexpected download response.",
        ));
    }
    if response
        .header("Content-Encoding")
        .is_some_and(|v| v != "identity")
    {
        return Err(Failure::new(
            "INVALID_RESPONSE",
            "Encoded model transfer cannot be verified.",
        ));
    }
    // Remove rather than truncate stale partial files, avoiding linked inodes.
    if !append && part.exists() {
        fs::remove_file(part)?;
    }
    #[cfg(unix)]
    if append {
        use std::os::unix::fs::MetadataExt;
        if fs::metadata(part)?.nlink() != 1 {
            return Err(Failure::new(
                "INVALID_DESTINATION",
                "A partial download has multiple links.",
            ));
        }
    }
    let mut file = if append {
        OpenOptions::new().append(true).open(part)?
    } else {
        OpenOptions::new().write(true).create_new(true).open(part)?
    };
    let mut reader = response.into_reader();
    let mut buffer = [0u8; 256 * 1024];
    let mut last_progress = Instant::now();
    loop {
        let count = reader.read(&mut buffer).map_err(|_| Failure {
            code: "NETWORK_ERROR",
            message: "The transfer was interrupted. Retry to resume.".into(),
            retryable: true,
            http_status: None,
        })?;
        if count == 0 {
            break;
        }
        offset = offset.saturating_add(count as u64);
        if offset > member.size {
            fs::remove_file(part)?;
            return Err(Failure::new(
                "INTEGRITY_ERROR",
                "The download exceeds its published size.",
            ));
        }
        file.write_all(&buffer[..count])?;
        if last_progress.elapsed() >= Duration::from_millis(150) {
            progress(
                done.saturating_add(offset).saturating_mul(90),
                total.saturating_mul(100),
                &member.path,
            );
            last_progress = Instant::now();
        }
    }
    file.sync_all()?;
    if offset < member.size {
        return Err(Failure {
            code: "NETWORK_ERROR",
            message: "The download is incomplete. Retry to resume.".into(),
            retryable: true,
            http_status: None,
        });
    }
    if !digest_file(part, member)? {
        fs::remove_file(part)?;
        return Err(Failure::new(
            "INTEGRITY_ERROR",
            "The model checksum does not match the pinned revision. Retry the download.",
        ));
    }
    Ok(())
}

/// Preserve the CLI/GUI terminal NDJSON contract.
pub fn summary(
    repository: &str,
    revision: Option<&str>,
    result: &std::result::Result<Completed, Failure>,
    destination: Option<&Path>,
) -> Value {
    let dest = result
        .as_ref()
        .ok()
        .map(|c| c.destination.as_path())
        .or(destination);
    let failure = result.as_ref().err();
    let mut value = json!({
        "schema_version":"ax.download_model.v1", "repo_id":repository,"revision":revision,
        "resolved_revision":result.as_ref().ok().and_then(|c| c.commit.as_deref()),
        "dest":dest,"manifest_path":dest.map(|d|d.join(MANIFEST)),
        "manifest_present":dest.is_some_and(|d|d.join(MANIFEST).is_file()),
        "config_present":dest.is_some_and(|d|d.join("config.json").is_file()),
        "safetensors_count":dest.and_then(|d|fs::read_dir(d).ok()).map(|rows|rows.flatten().filter(|e|e.path().extension().is_some_and(|x|x=="safetensors")).count()).unwrap_or(0),
        "status":if result.is_ok(){"ready"}else{"download_failed"},
        "errors":failure.map(|f|vec![f.message.clone()]).unwrap_or_default(),
        "server_command":dest.map(|d|json!(["ax-engine-server","--mlx","--mlx-model-artifacts-dir",d,"--port","31418"]))
    });
    if let Some(failure) = failure {
        value["failure"] = failure.json();
    }
    value
}

#[cfg(test)]
mod tests;
