pub mod header;
pub mod mmap;
pub mod tensor;

pub use header::{GgufError, GgufHeader, MetadataValue};
pub use mmap::MappedModel;
pub use tensor::{GgmlType, TensorInfo};

/// GGUF magic bytes: "GGUF" as little-endian u32.
/// 'G'=0x47, 'G'=0x47, 'U'=0x55, 'F'=0x46 → LE u32 = 0x46554747
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Supported GGUF format version.
pub const GGUF_VERSION: u32 = 3;
