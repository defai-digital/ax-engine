//! Resolve artifact files within a pack or its own Hugging Face blob store.

use std::io;
use std::path::{Component, Path, PathBuf};

pub(crate) fn resolve_file(
    root: &Path,
    canonical_root: &Path,
    relative: &Path,
) -> io::Result<PathBuf> {
    let joined = root.join(relative);
    let canonical = joined.canonicalize()?;
    if canonical.is_file()
        && (canonical.starts_with(canonical_root)
            || is_snapshot_blob_link(root, canonical_root, relative, &joined, &canonical))
    {
        return Ok(canonical);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "artifact file {} escapes root {} or is not a regular file",
            canonical.display(),
            canonical_root.display()
        ),
    ))
}

fn hex_name(path: &Path, lengths: &[usize]) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            lengths.contains(&name.len())
                && name
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
}

fn is_snapshot_blob_link(
    root: &Path,
    canonical_root: &Path,
    relative: &Path,
    joined: &Path,
    canonical: &Path,
) -> bool {
    let Some(snapshots) = canonical_root.parent() else {
        return false;
    };
    if snapshots.file_name().is_none_or(|name| name != "snapshots")
        || !hex_name(canonical_root, &[40])
    {
        return false;
    }
    let Some(repo) = snapshots.parent() else {
        return false;
    };
    let blobs = repo.join("blobs");
    if !blobs.symlink_metadata().is_ok_and(|meta| meta.is_dir())
        || canonical.parent() != Some(blobs.as_path())
        || !hex_name(canonical, &[40, 64])
        || !joined
            .symlink_metadata()
            .is_ok_and(|meta| meta.is_symlink())
    {
        return false;
    }
    // Only a file link addressed inside this snapshot may reach its blob store.
    // A manifest must not grant itself access by naming an absolute blob or ../.
    let inside = if relative.is_absolute() {
        relative
            .strip_prefix(root)
            .or_else(|_| relative.strip_prefix(canonical_root))
            .ok()
    } else {
        Some(relative)
    };
    let Some(inside) = inside else {
        return false;
    };
    let mut cursor = root.to_path_buf();
    let mut components = inside.components().peekable();
    while let Some(component) = components.next() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => {
                cursor.push(name);
                if components.peek().is_some()
                    && !cursor.symlink_metadata().is_ok_and(|meta| meta.is_dir())
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

#[cfg(all(test, unix))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "ax-artifact-path-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
        fn snapshot(&self) -> PathBuf {
            let root = self
                .0
                .join("models--owner--model/snapshots")
                .join("a".repeat(40));
            std::fs::create_dir_all(&root).unwrap();
            std::fs::create_dir(root.parent().unwrap().parent().unwrap().join("blobs")).unwrap();
            root
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn resolves_local_files_and_in_pack_links() {
        let f = Fixture::new();
        let file = f.0.join("weights.safetensors");
        std::fs::write(&file, b"fixture").unwrap();
        symlink("weights.safetensors", f.0.join("alias.safetensors")).unwrap();
        for name in [
            Path::new("weights.safetensors"),
            Path::new("alias.safetensors"),
            &file,
        ] {
            assert_eq!(resolve_file(&f.0, &f.0, name).unwrap(), file);
        }
        assert!(resolve_file(&f.0, &f.0, Path::new(".")).is_err());
    }

    #[test]
    fn resolves_snapshot_blob_links_without_allowing_direct_blob_paths() {
        let f = Fixture::new();
        let root = f.snapshot();
        let blobs = root.parent().unwrap().parent().unwrap().join("blobs");
        for length in [40, 64] {
            let name = "b".repeat(length);
            let blob = blobs.join(&name);
            std::fs::write(&blob, b"fixture").unwrap();
            let link = root.join(format!("weights-{length}.safetensors"));
            symlink(Path::new("../../blobs").join(&name), &link).unwrap();
            assert_eq!(resolve_file(&root, &root, &link).unwrap(), blob);
            assert_eq!(
                resolve_file(&root, &root, Path::new(link.file_name().unwrap())).unwrap(),
                blob
            );
            assert!(resolve_file(&root, &root, &blob).is_err());
            assert!(resolve_file(&root, &root, &Path::new("../../blobs").join(&name)).is_err());
        }
    }

    #[test]
    fn rejects_foreign_blobs_and_symlinked_blob_directories() {
        let f = Fixture::new();
        let root = f.snapshot();
        let foreign = f.0.join("foreign");
        std::fs::create_dir(&foreign).unwrap();
        let blob = foreign.join("c".repeat(64));
        std::fs::write(&blob, b"fixture").unwrap();
        let link = root.join("weights.safetensors");
        symlink(&blob, &link).unwrap();
        assert!(resolve_file(&root, &root, &link).is_err());
        let blobs = root.parent().unwrap().parent().unwrap().join("blobs");
        std::fs::remove_dir(&blobs).unwrap();
        symlink(&foreign, &blobs).unwrap();
        assert!(resolve_file(&root, &root, &link).is_err());
    }

    #[test]
    fn rejects_blob_link_chains_and_directory_links() {
        let f = Fixture::new();
        let root = f.snapshot();
        let blobs = root.parent().unwrap().parent().unwrap().join("blobs");
        let outside = f.0.join("c".repeat(64));
        std::fs::write(&outside, b"fixture").unwrap();
        let blob = blobs.join("d".repeat(64));
        symlink(&outside, &blob).unwrap();
        let link = root.join("weights.safetensors");
        symlink(&blob, &link).unwrap();
        assert!(resolve_file(&root, &root, &link).is_err());
        std::fs::remove_file(&blob).unwrap();
        std::fs::write(&blob, b"fixture").unwrap();
        symlink(&root, root.join("nested")).unwrap();
        assert!(resolve_file(&root, &root, Path::new("nested/weights.safetensors")).is_err());
        assert_eq!(resolve_file(&root, &root, &link).unwrap(), blob);
    }
}
