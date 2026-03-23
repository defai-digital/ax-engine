use std::path::Path;

use ax_core::gguf::MappedModel;

pub mod baseline;
pub mod perf;
pub mod profile;
pub mod report;
pub mod soak;

pub(crate) fn init_kernel_profile(model_path: &str, mapped: &MappedModel) {
    let profile_model_name = Path::new(model_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .or_else(|| mapped.header.get_str("general.name").map(str::to_owned))
        .or_else(|| mapped.header.architecture().map(str::to_owned))
        .unwrap_or_else(|| "default".to_string());
    let profile_quant = mapped
        .predominant_quant()
        .map(|dtype| dtype.to_string())
        .unwrap_or_else(|| "default".to_string());
    ax_metal::init_global_profile(&profile_model_name, &profile_quant);
}
