//! Strict descriptor and byte decoder for one pinned research artifact.
use super::codec::verify_file;
use ax_engine_core::{
    artifact_path,
    ncm::{NcmError, NcmFinishReason, NcmOutput},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub const REVISION: &str = "2d5f137713fe047abb6fcffe780aad419486f3dd";
pub const SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Deserialize)]
struct FilePin {
    sha256: String,
    size_bytes: u64,
}
#[derive(Deserialize)]
struct Lock {
    files: BTreeMap<String, FilePin>,
}

#[derive(Clone, Debug)]
pub struct NcmArtifactDescriptorV1 {
    root: PathBuf,
    pub identity: String,
    pub weight_bytes: u64,
    files: BTreeMap<String, PathBuf>,
}
impl NcmArtifactDescriptorV1 {
    pub fn from_research_pin(root: &Path) -> Result<Self, NcmError> {
        let lock: Lock =
            serde_json::from_str(include_str!("tinystories-lock.json")).map_err(runtime)?;
        let canonical = root.canonicalize().map_err(runtime)?;
        let roles: BTreeSet<_> = lock.files.keys().map(String::as_str).collect();
        validate_root(root, &canonical, &roles)?;
        let mut hash = Sha256::new();
        hash.update(b"ax.ncm.tinystories.f32.greedy.hexbytes.v1\0");
        let mut weight_bytes = 0;
        let mut files = BTreeMap::new();
        for (name, pin) in &lock.files {
            let path =
                artifact_path::resolve_file(root, &canonical, Path::new(name)).map_err(runtime)?;
            verify_file(&path, pin.size_bytes, &pin.sha256).map_err(runtime)?;
            files.insert(name.clone(), path);
            hash.update(name.as_bytes());
            hash.update([0]);
            hash.update(pin.sha256.as_bytes());
            hash.update(pin.size_bytes.to_le_bytes());
            if name.ends_with(".safetensors") {
                weight_bytes += pin.size_bytes;
            }
        }
        Ok(Self {
            root: canonical,
            files,
            identity: format!("{:x}", hash.finalize()),
            weight_bytes,
        })
    }
    pub fn file(&self, name: &str) -> Result<&Path, NcmError> {
        self.files
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| runtime("unlisted component role"))
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
}

fn validate_root(root: &Path, canonical: &Path, roles: &BTreeSet<&str>) -> Result<(), NcmError> {
    for item in std::fs::read_dir(root).map_err(runtime)? {
        let item = item.map_err(runtime)?;
        let name = item.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| runtime("non-UTF8 artifact filename"))?;
        let documentation = name.ends_with(".md")
            || name.ends_with(".txt")
            || name.starts_with("LICENSE")
            || name == ".gitattributes";
        if !roles.contains(name) && !documentation {
            return Err(runtime(format!("unlisted artifact input: {name}")));
        }
        artifact_path::resolve_file(root, canonical, Path::new(name)).map_err(runtime)?;
    }
    Ok(())
}

pub fn validate_unigram(values: &[f32]) -> Result<u32, NcmError> {
    let mass: f64 = values.iter().map(|&x| f64::from(x)).sum();
    if values.len() != 4096
        || values.iter().any(|&x| !x.is_finite() || x < 0.0)
        || (mass - 1.0).abs() > 1e-6
    {
        return Err(runtime("expected normalized F32[4096] L0 probabilities"));
    }
    // A strict greater-than comparison preserves the lowest-index tie rule.
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BpeFile {
    vocab: Vec<String>,
    merges: Vec<[String; 2]>,
    special_tokens: Vec<String>,
}

pub struct ByteDecoder {
    vocab: Vec<Vec<u8>>,
}
impl ByteDecoder {
    pub fn load(path: &Path) -> Result<Self, NcmError> {
        let bytes = std::fs::read(path).map_err(runtime)?;
        if bytes.len() > 1024 * 1024 {
            return Err(runtime("BPE file exceeds size cap"));
        }
        Self::parse(&bytes)
    }
    fn parse(bytes: &[u8]) -> Result<Self, NcmError> {
        let bpe: BpeFile = serde_json::from_slice(bytes).map_err(runtime)?;
        if bpe.vocab.len() != 8192
            || bpe.merges.len() != 7934
            || bpe.special_tokens != ["<|endoftext|>", "<|pad|>"]
        {
            return Err(runtime("wrong TinyStories BPE schema"));
        }
        let vocab = bpe
            .vocab
            .iter()
            .map(|s| unhex(s))
            .collect::<Result<Vec<_>, _>>()?;
        if vocab[0] != b"<|endoftext|>" || vocab[1] != b"<|pad|>" {
            return Err(runtime("EOS/PAD byte identity mismatch"));
        }
        let set: BTreeSet<_> = vocab.iter().cloned().collect();
        if set.len() != 8192 {
            return Err(runtime("duplicate BPE bytes"));
        }
        for [left, right] in bpe.merges {
            let left = unhex(&left)?;
            let right = unhex(&right)?;
            let joined = [left.as_slice(), right.as_slice()].concat();
            if !set.contains(&left) || !set.contains(&right) || !set.contains(&joined) {
                return Err(runtime("invalid merge reference"));
            }
        }
        Ok(Self { vocab })
    }
    pub fn decode(&self, raw: Vec<u32>) -> Result<NcmOutput, NcmError> {
        if raw.len() != 128 || raw.iter().any(|&id| id >= 8192) {
            return Err(runtime("expected 128 BPE positions"));
        }
        let visible_bpe_ids = raw
            .iter()
            .copied()
            .take_while(|&id| id != 0)
            .filter(|&id| id != 1)
            .collect::<Vec<_>>();
        let mut bytes = Vec::new();
        for &id in &visible_bpe_ids {
            bytes.extend_from_slice(&self.vocab[id as usize]);
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let finish_reason = if raw.contains(&0) {
            NcmFinishReason::Stop
        } else {
            NcmFinishReason::Length
        };
        Ok(NcmOutput {
            raw_bpe_ids: raw,
            visible_bpe_ids,
            text,
            finish_reason,
        })
    }
}
fn unhex(s: &str) -> Result<Vec<u8>, NcmError> {
    if s.is_empty() || !s.len().is_multiple_of(2) || s.len() > 65536 {
        return Err(runtime("invalid BPE hex length"));
    }
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let digit = |x: u8| match x {
                b'0'..=b'9' => Some(x - b'0'),
                b'a'..=b'f' => Some(x - b'a' + 10),
                _ => None,
            };
            match (digit(pair[0]), digit(pair[1])) {
                (Some(a), Some(b)) => Ok(16 * a + b),
                _ => Err(runtime("invalid BPE hex byte")),
            }
        })
        .collect()
}
pub(super) fn runtime(error: impl std::fmt::Display) -> NcmError {
    NcmError::Runtime(error.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn unigram_rejects_invalid_mass_values_shape_and_preserves_two_way_tie() {
        let mut x = vec![0.0; 4096];
        x[3] = 0.5;
        x[7] = 0.5;
        assert_eq!(validate_unigram(&x).unwrap(), 3);
        x[0] = f32::NAN;
        assert!(validate_unigram(&x).is_err());
        x[0] = -0.1;
        assert!(validate_unigram(&x).is_err());
        assert!(validate_unigram(&[1.0]).is_err());
        assert!(validate_unigram(&vec![0.; 4096]).is_err());
        x[0] = 0.1;
        assert!(validate_unigram(&x).is_err());
    }
    #[test]
    fn byte_decode_handles_special_empty_split_utf8_and_whitespace() {
        let mut vocab = vec![b"x".to_vec(); 8192];
        vocab[2] = vec![0xc3];
        vocab[3] = vec![0xa9];
        vocab[4] = vec![0xff];
        vocab[5] = b"\n".to_vec();
        let d = ByteDecoder { vocab };
        let mut row = vec![1; 128];
        row[..6].copy_from_slice(&[5, 2, 1, 3, 0, 4]);
        let o = d.decode(row).unwrap();
        assert_eq!(o.text, "\n\u{e9}");
        assert_eq!(o.visible_bpe_ids, vec![5, 2, 3]);
        assert_eq!(d.decode(vec![0; 128]).unwrap().text, "");
        assert_eq!(d.decode(vec![1; 128]).unwrap().text, "");
        let mut row = vec![1; 128];
        row[0] = 4;
        assert_eq!(d.decode(row).unwrap().text, "\u{fffd}");
        assert!(d.decode(vec![8192; 128]).is_err());
    }
    #[test]
    fn malformed_hex_and_tokenizer_schema_fail() {
        for s in ["", "0", "zz", "AA"] {
            assert!(unhex(s).is_err());
        }
        assert_eq!(unhex("c3a9").unwrap(), vec![195, 169]);
        assert!(ByteDecoder::parse(b"{}").is_err());
    }
}
