//! Immutable content-addressed payload pages for the durable prefix cache.
//!
//! The outer `.axkv` file remains the atomic, key-bound commit record. With
//! the page-manifest flag set, its payload is a small manifest that names
//! immutable `.axpg` blobs. Blobs are published before the manifest, so a
//! crash can leave only an unreachable page; it can never expose a partial
//! cache hit.

use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

pub(crate) const PAGE_MANIFEST_FLAG: u32 = 1;
const PAGE_MANIFEST_MAGIC: &[u8; 4] = b"AXPM";
const PAGE_MANIFEST_VERSION: u32 = 1;
const PAGE_MANIFEST_HEADER_LEN: usize = 96;
const PAGE_DESCRIPTOR_LEN: usize = 36;
const PAGE_EXTENSION: &str = "axpg";
const PAGE_DIR_NAME: &str = ".pages";
const PAGE_HASH_DOMAIN: &[u8] = b"ax.mlx.disk_prefix_page.v1\0";
const PAYLOAD_HASH_DOMAIN: &[u8] = b"ax.mlx.disk_prefix_payload.v1\0";
const KEY_HASH_DOMAIN: &[u8] = b"ax.mlx.disk_prefix_page_key.v1\0";

#[derive(Debug)]
pub(crate) enum PageStoreError {
    Io(std::io::Error),
    Invalid,
}

impl From<std::io::Error> for PageStoreError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PageDescriptor {
    pub(crate) len: u32,
    pub(crate) hash: [u8; 32],
}

#[derive(Clone, Debug)]
pub(crate) struct PageManifest {
    pub(crate) page_bytes: u32,
    pub(crate) payload_len: u64,
    pub(crate) payload_hash: [u8; 32],
    pub(crate) pages: Vec<PageDescriptor>,
}

impl PageManifest {
    pub(crate) fn decode(
        bytes: &[u8],
        expected_key: Option<&[u8]>,
        max_payload_bytes: u64,
    ) -> Result<Self, PageStoreError> {
        if bytes.len() < PAGE_MANIFEST_HEADER_LEN || &bytes[0..4] != PAGE_MANIFEST_MAGIC {
            return Err(PageStoreError::Invalid);
        }
        let version = read_u32(bytes, 4)?;
        let header_len = read_u32(bytes, 8)? as usize;
        let page_bytes = read_u32(bytes, 12)?;
        let page_count = read_u32(bytes, 16)? as usize;
        let reserved = read_u32(bytes, 20)?;
        let payload_len = read_u64(bytes, 24)?;
        let payload_hash: [u8; 32] = bytes
            .get(32..64)
            .ok_or(PageStoreError::Invalid)?
            .try_into()
            .map_err(|_| PageStoreError::Invalid)?;
        let key_hash: [u8; 32] = bytes
            .get(64..96)
            .ok_or(PageStoreError::Invalid)?
            .try_into()
            .map_err(|_| PageStoreError::Invalid)?;
        let descriptor_bytes = page_count
            .checked_mul(PAGE_DESCRIPTOR_LEN)
            .and_then(|n| n.checked_add(PAGE_MANIFEST_HEADER_LEN))
            .ok_or(PageStoreError::Invalid)?;
        if version != PAGE_MANIFEST_VERSION
            || header_len != PAGE_MANIFEST_HEADER_LEN
            || reserved != 0
            || page_bytes == 0
            || page_count == 0
            || payload_len == 0
            || payload_len > max_payload_bytes
            || descriptor_bytes != bytes.len()
        {
            return Err(PageStoreError::Invalid);
        }
        if let Some(key) = expected_key
            && key_hash != key_hash_bytes(key)
        {
            return Err(PageStoreError::Invalid);
        }

        let mut pages = Vec::with_capacity(page_count);
        let mut sum = 0u64;
        for index in 0..page_count {
            let offset = PAGE_MANIFEST_HEADER_LEN + index * PAGE_DESCRIPTOR_LEN;
            let len = read_u32(bytes, offset)?;
            let hash: [u8; 32] = bytes
                .get(offset + 4..offset + PAGE_DESCRIPTOR_LEN)
                .ok_or(PageStoreError::Invalid)?
                .try_into()
                .map_err(|_| PageStoreError::Invalid)?;
            let is_last = index + 1 == page_count;
            if len == 0 || len > page_bytes || (!is_last && len != page_bytes) {
                return Err(PageStoreError::Invalid);
            }
            sum = sum
                .checked_add(u64::from(len))
                .ok_or(PageStoreError::Invalid)?;
            pages.push(PageDescriptor { len, hash });
        }
        if sum != payload_len {
            return Err(PageStoreError::Invalid);
        }
        Ok(Self {
            page_bytes,
            payload_len,
            payload_hash,
            pages,
        })
    }

    pub(crate) fn page_hashes_hex(&self) -> Vec<String> {
        self.pages.iter().map(|page| hash_hex(&page.hash)).collect()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PageStore {
    dir: PathBuf,
    page_bytes: usize,
    gc_grace: std::time::Duration,
}

impl PageStore {
    pub(crate) fn exists(root: &Path) -> bool {
        root.join(PAGE_DIR_NAME).is_dir()
    }

    pub(crate) fn estimated_manifest_len(page_bytes: usize, payload_len: usize) -> Option<usize> {
        if page_bytes == 0 || payload_len == 0 {
            return None;
        }
        payload_len
            .div_ceil(page_bytes)
            .checked_mul(PAGE_DESCRIPTOR_LEN)
            .and_then(|bytes| bytes.checked_add(PAGE_MANIFEST_HEADER_LEN))
    }

    pub(crate) fn open(
        root: &Path,
        page_bytes: usize,
        gc_grace_ms: u64,
    ) -> Result<Self, PageStoreError> {
        if page_bytes == 0 || page_bytes > u32::MAX as usize {
            return Err(PageStoreError::Invalid);
        }
        let dir = root.join(PAGE_DIR_NAME);
        fs::create_dir_all(&dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self {
            dir,
            page_bytes,
            gc_grace: std::time::Duration::from_millis(gc_grace_ms),
        })
    }

    pub(crate) fn publish_payload(
        &self,
        payload: &[u8],
        key_bytes: &[u8],
    ) -> Result<Vec<u8>, PageStoreError> {
        if payload.is_empty() {
            return Err(PageStoreError::Invalid);
        }
        let mut pages = Vec::with_capacity(payload.len().div_ceil(self.page_bytes));
        for chunk in payload.chunks(self.page_bytes) {
            let hash = page_hash_bytes(chunk);
            self.publish_page(&hash, chunk)?;
            pages.push(PageDescriptor {
                len: u32::try_from(chunk.len()).map_err(|_| PageStoreError::Invalid)?,
                hash,
            });
        }
        let payload_hash = payload_hash_bytes(payload);
        let key_hash = key_hash_bytes(key_bytes);
        let mut manifest = Vec::with_capacity(
            PAGE_MANIFEST_HEADER_LEN + pages.len().saturating_mul(PAGE_DESCRIPTOR_LEN),
        );
        manifest.extend_from_slice(PAGE_MANIFEST_MAGIC);
        manifest.extend_from_slice(&PAGE_MANIFEST_VERSION.to_le_bytes());
        manifest.extend_from_slice(&(PAGE_MANIFEST_HEADER_LEN as u32).to_le_bytes());
        manifest.extend_from_slice(&(self.page_bytes as u32).to_le_bytes());
        manifest.extend_from_slice(
            &u32::try_from(pages.len())
                .map_err(|_| PageStoreError::Invalid)?
                .to_le_bytes(),
        );
        manifest.extend_from_slice(&0u32.to_le_bytes());
        manifest.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        manifest.extend_from_slice(&payload_hash);
        manifest.extend_from_slice(&key_hash);
        for page in pages {
            manifest.extend_from_slice(&page.len.to_le_bytes());
            manifest.extend_from_slice(&page.hash);
        }
        Ok(manifest)
    }

    fn publish_page(&self, hash: &[u8; 32], bytes: &[u8]) -> Result<(), PageStoreError> {
        let final_path = self.page_path(hash);
        if validate_page_file(&final_path, bytes.len() as u64, hash).is_ok() {
            return Ok(());
        }
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let temp_path = self.dir.join(format!(
            "{}.tmp.{}.{}",
            hash_hex(hash),
            std::process::id(),
            unique
        ));
        let mut guard = PageTempGuard::new(&temp_path);
        {
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(&temp_path)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        fs::rename(&temp_path, &final_path)?;
        guard.disarm();
        if let Ok(dir) = fs::File::open(&self.dir) {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    pub(crate) fn read_payload(
        &self,
        manifest: PageManifest,
    ) -> Result<(Vec<u8>, PageReadStats), PageStoreError> {
        let capacity =
            usize::try_from(manifest.payload_len).map_err(|_| PageStoreError::Invalid)?;
        let mut out = Vec::with_capacity(capacity);
        let mut reader = self.reader(manifest);
        reader.read_to_end(&mut out).map_err(PageStoreError::Io)?;
        let stats = reader.finish()?;
        Ok((out, stats))
    }

    pub(crate) fn reader(&self, manifest: PageManifest) -> PagePayloadReader {
        debug_assert!(manifest.page_bytes > 0);
        PagePayloadReader {
            page_dir: self.dir.clone(),
            manifest,
            page_index: 0,
            page_read: 0,
            current: None,
            page_hasher: None,
            payload_hasher: payload_hasher(),
            stats: PageReadStats::default(),
        }
    }

    pub(crate) fn page_size(&self, hash_hex: &str) -> Option<u64> {
        fs::symlink_metadata(self.dir.join(format!("{hash_hex}.{PAGE_EXTENSION}")))
            .ok()
            .filter(|meta| meta.file_type().is_file())
            .map(|meta| meta.len())
    }

    #[cfg(test)]
    pub(crate) fn blob_paths(&self) -> Vec<PathBuf> {
        fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == PAGE_EXTENSION))
            .collect()
    }

    pub(crate) fn gc_unreferenced(
        &self,
        live_hashes: &HashSet<String>,
    ) -> Result<u32, PageStoreError> {
        let now = std::time::SystemTime::now();
        let mut removed = 0u32;
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if file_name.contains(".tmp.") {
                let old_enough = entry
                    .metadata()
                    .ok()
                    .and_then(|meta| meta.modified().ok())
                    .and_then(|modified| now.duration_since(modified).ok())
                    .is_some_and(|age| age >= self.gc_grace);
                if old_enough && fs::remove_file(&path).is_ok() {
                    removed = removed.saturating_add(1);
                }
                continue;
            }
            let Some(hash) = file_name.strip_suffix(&format!(".{PAGE_EXTENSION}")) else {
                continue;
            };
            if hash.len() != 64 || live_hashes.contains(hash) {
                continue;
            }
            let meta = match fs::symlink_metadata(&path) {
                Ok(meta) if meta.file_type().is_file() => meta,
                _ => continue,
            };
            let old_enough = meta
                .modified()
                .ok()
                .and_then(|modified| now.duration_since(modified).ok())
                .is_some_and(|age| age >= self.gc_grace);
            if old_enough && fs::remove_file(&path).is_ok() {
                removed = removed.saturating_add(1);
            }
        }
        if removed > 0
            && let Ok(dir) = fs::File::open(&self.dir)
        {
            let _ = dir.sync_all();
        }
        Ok(removed)
    }

    fn page_path(&self, hash: &[u8; 32]) -> PathBuf {
        self.dir
            .join(format!("{}.{PAGE_EXTENSION}", hash_hex(hash)))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PageReadStats {
    pub(crate) read_wall_us: u64,
    pub(crate) checksum_wall_us: u64,
    pub(crate) bytes_read: u64,
}

pub(crate) struct PagePayloadReader {
    page_dir: PathBuf,
    manifest: PageManifest,
    page_index: usize,
    page_read: usize,
    current: Option<fs::File>,
    page_hasher: Option<Sha256>,
    payload_hasher: Sha256,
    stats: PageReadStats,
}

impl PagePayloadReader {
    pub(crate) fn finish(mut self) -> Result<PageReadStats, PageStoreError> {
        // Do not drain here. Native restore must reject a valid cache payload
        // followed by manifest-declared trailing bytes, matching the legacy
        // v3 `Take::limit() == 0` contract.
        if self.stats.bytes_read != self.manifest.payload_len
            || self.page_index != self.manifest.pages.len()
        {
            return Err(PageStoreError::Invalid);
        }
        let checksum_started = std::time::Instant::now();
        let digest = self.payload_hasher.finalize();
        self.stats.checksum_wall_us = self
            .stats
            .checksum_wall_us
            .saturating_add(elapsed_us(checksum_started));
        if digest.as_slice() != self.manifest.payload_hash {
            return Err(PageStoreError::Invalid);
        }
        Ok(self.stats)
    }

    fn open_current(&mut self) -> std::io::Result<()> {
        let descriptor = self
            .manifest
            .pages
            .get(self.page_index)
            .ok_or_else(invalid_data)?;
        let path = self
            .page_dir
            .join(format!("{}.{}", hash_hex(&descriptor.hash), PAGE_EXTENSION));
        let started = std::time::Instant::now();
        let meta = fs::symlink_metadata(&path)?;
        if !meta.file_type().is_file() || meta.len() != u64::from(descriptor.len) {
            return Err(invalid_data());
        }
        self.current = Some(fs::File::open(path)?);
        self.stats.read_wall_us = self.stats.read_wall_us.saturating_add(elapsed_us(started));
        self.page_read = 0;
        self.page_hasher = Some(page_hasher());
        Ok(())
    }

    fn finish_current(&mut self) -> std::io::Result<()> {
        let descriptor = self
            .manifest
            .pages
            .get(self.page_index)
            .ok_or_else(invalid_data)?;
        // Metadata was checked before open, but also prove the page did not
        // grow between stat and read. Cooperative writers never mutate a
        // published blob; an extra byte is corruption.
        let read_started = std::time::Instant::now();
        let mut trailing = [0u8; 1];
        if self
            .current
            .as_mut()
            .ok_or_else(invalid_data)?
            .read(&mut trailing)?
            != 0
        {
            return Err(invalid_data());
        }
        self.stats.read_wall_us = self
            .stats
            .read_wall_us
            .saturating_add(elapsed_us(read_started));
        let checksum_started = std::time::Instant::now();
        let digest = self.page_hasher.take().ok_or_else(invalid_data)?.finalize();
        self.stats.checksum_wall_us = self
            .stats
            .checksum_wall_us
            .saturating_add(elapsed_us(checksum_started));
        if digest.as_slice() != descriptor.hash {
            return Err(invalid_data());
        }
        self.current = None;
        self.page_index = self.page_index.saturating_add(1);
        self.page_read = 0;
        Ok(())
    }
}

impl Read for PagePayloadReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if buf.is_empty() || self.page_index >= self.manifest.pages.len() {
            return Ok(0);
        }
        if self.current.is_none() {
            self.open_current()?;
        }
        let descriptor = &self.manifest.pages[self.page_index];
        let remaining = descriptor.len as usize - self.page_read;
        let take = remaining.min(buf.len());
        let started = std::time::Instant::now();
        let read = self
            .current
            .as_mut()
            .ok_or_else(invalid_data)?
            .read(&mut buf[..take])?;
        self.stats.read_wall_us = self.stats.read_wall_us.saturating_add(elapsed_us(started));
        if read == 0 {
            return Err(invalid_data());
        }
        let checksum_started = std::time::Instant::now();
        self.page_hasher
            .as_mut()
            .ok_or_else(invalid_data)?
            .update(&buf[..read]);
        self.payload_hasher.update(&buf[..read]);
        self.stats.checksum_wall_us = self
            .stats
            .checksum_wall_us
            .saturating_add(elapsed_us(checksum_started));
        self.page_read += read;
        self.stats.bytes_read = self.stats.bytes_read.saturating_add(read as u64);
        if self.page_read == descriptor.len as usize {
            self.finish_current()?;
        }
        Ok(read)
    }
}

fn validate_page_file(path: &Path, expected_len: u64, expected_hash: &[u8; 32]) -> Result<(), ()> {
    let meta = fs::symlink_metadata(path).map_err(|_| ())?;
    if !meta.file_type().is_file() || meta.len() != expected_len {
        return Err(());
    }
    let mut file = fs::File::open(path).map_err(|_| ())?;
    let mut hasher = page_hasher();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf).map_err(|_| ())?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    if hasher.finalize().as_slice() == expected_hash {
        Ok(())
    } else {
        Err(())
    }
}

fn page_hasher() -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(PAGE_HASH_DOMAIN);
    hasher
}

fn payload_hasher() -> Sha256 {
    let mut hasher = Sha256::new();
    hasher.update(PAYLOAD_HASH_DOMAIN);
    hasher
}

fn page_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    digest_with_hasher(page_hasher(), bytes)
}

fn payload_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    digest_with_hasher(payload_hasher(), bytes)
}

fn key_hash_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(KEY_HASH_DOMAIN);
    digest_with_hasher(hasher, bytes)
}

fn digest_with_hasher(mut hasher: Sha256, bytes: &[u8]) -> [u8; 32] {
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, PageStoreError> {
    bytes
        .get(offset..offset + 4)
        .ok_or(PageStoreError::Invalid)?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| PageStoreError::Invalid)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, PageStoreError> {
    bytes
        .get(offset..offset + 8)
        .ok_or(PageStoreError::Invalid)?
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| PageStoreError::Invalid)
}

fn invalid_data() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid durable KV page")
}

fn elapsed_us(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

struct PageTempGuard {
    path: Option<PathBuf>,
}

impl PageTempGuard {
    fn new(path: &Path) -> Self {
        Self {
            path: Some(path.to_path_buf()),
        }
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for PageTempGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ax-page-store-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or_default()
        ))
    }

    #[test]
    fn manifest_round_trip_streams_and_deduplicates_pages() {
        let root = tempdir("round-trip");
        let store = PageStore::open(&root, 8, 0).expect("open");
        let payload = b"abcdefghabcdefghXYZ";
        let key = b"canonical-key";
        let encoded = store.publish_payload(payload, key).expect("publish");
        let manifest = PageManifest::decode(&encoded, Some(key), 1024).expect("decode");
        assert_eq!(manifest.pages.len(), 3);
        assert_eq!(manifest.pages[0].hash, manifest.pages[1].hash);
        assert_eq!(
            fs::read_dir(&store.dir)
                .expect("page dir")
                .filter_map(Result::ok)
                .filter(|entry| entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == PAGE_EXTENSION))
                .count(),
            2
        );
        let (restored, stats) = store.read_payload(manifest).expect("restore");
        assert_eq!(restored, payload);
        assert_eq!(stats.bytes_read, payload.len() as u64);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn manifest_rejects_wrong_key_and_corrupt_page() {
        let root = tempdir("corrupt");
        let store = PageStore::open(&root, 8, 0).expect("open");
        let encoded = store
            .publish_payload(b"abcdefghXYZ", b"key-a")
            .expect("publish");
        assert!(PageManifest::decode(&encoded, Some(b"key-b"), 1024).is_err());
        let manifest = PageManifest::decode(&encoded, Some(b"key-a"), 1024).expect("decode");
        let path = store.page_path(&manifest.pages[0].hash);
        fs::write(path, b"corrupt!").expect("corrupt");
        assert!(store.read_payload(manifest).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ordered_payload_hash_rejects_reordered_valid_pages() {
        let root = tempdir("reordered");
        let store = PageStore::open(&root, 8, 0).expect("open");
        let mut encoded = store
            .publish_payload(b"abcdefghABCDEFGHxyz", b"key")
            .expect("publish");
        let first = encoded
            [PAGE_MANIFEST_HEADER_LEN..PAGE_MANIFEST_HEADER_LEN + PAGE_DESCRIPTOR_LEN]
            .to_vec();
        let second = encoded[PAGE_MANIFEST_HEADER_LEN + PAGE_DESCRIPTOR_LEN
            ..PAGE_MANIFEST_HEADER_LEN + 2 * PAGE_DESCRIPTOR_LEN]
            .to_vec();
        encoded[PAGE_MANIFEST_HEADER_LEN..PAGE_MANIFEST_HEADER_LEN + PAGE_DESCRIPTOR_LEN]
            .copy_from_slice(&second);
        encoded[PAGE_MANIFEST_HEADER_LEN + PAGE_DESCRIPTOR_LEN
            ..PAGE_MANIFEST_HEADER_LEN + 2 * PAGE_DESCRIPTOR_LEN]
            .copy_from_slice(&first);
        let manifest = PageManifest::decode(&encoded, Some(b"key"), 1024).expect("shape valid");
        assert!(store.read_payload(manifest).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn gc_respects_reference_set() {
        let root = tempdir("gc");
        let store = PageStore::open(&root, 8, 0).expect("open");
        let encoded = store
            .publish_payload(b"abcdefghXYZ", b"key")
            .expect("publish");
        let manifest = PageManifest::decode(&encoded, Some(b"key"), 1024).expect("decode");
        let live: HashSet<String> = manifest.page_hashes_hex().into_iter().collect();
        assert_eq!(store.gc_unreferenced(&live).expect("live gc"), 0);
        assert_eq!(
            store.gc_unreferenced(&HashSet::new()).expect("orphan gc"),
            2
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn gc_grace_delays_orphan_reclamation() {
        let root = tempdir("gc-grace");
        let store = PageStore::open(&root, 8, 60_000).expect("open");
        store
            .publish_payload(b"abcdefghXYZ", b"key")
            .expect("publish");
        assert_eq!(store.gc_unreferenced(&HashSet::new()).expect("grace gc"), 0);
        assert_eq!(store.blob_paths().len(), 2);

        let immediate = PageStore::open(&root, 8, 0).expect("reopen immediate");
        assert_eq!(
            immediate
                .gc_unreferenced(&HashSet::new())
                .expect("immediate gc"),
            2
        );
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn page_directory_and_blobs_are_owner_only_and_symlinks_are_rejected() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let root = tempdir("permissions");
        let store = PageStore::open(&root, 8, 0).expect("open");
        let encoded = store
            .publish_payload(b"abcdefghXYZ", b"key")
            .expect("publish");
        let manifest = PageManifest::decode(&encoded, Some(b"key"), 1024).expect("manifest");
        assert_eq!(
            fs::metadata(&store.dir)
                .expect("dir metadata")
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        let blob = store.page_path(&manifest.pages[0].hash);
        assert_eq!(
            fs::metadata(&blob)
                .expect("blob metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        let target = root.join("attacker-page");
        fs::write(&target, b"abcdefgh").expect("target");
        fs::remove_file(&blob).expect("remove blob");
        symlink(&target, &blob).expect("symlink");
        assert!(store.read_payload(manifest).is_err());
        let _ = fs::remove_dir_all(root);
    }
}
