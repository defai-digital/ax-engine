use super::*;
use crate::gguf::tensor::GgmlType;

#[test]
fn test_arch_name() {
    let fwd = Qwen3MoeForward;
    assert_eq!(fwd.arch_name(), "qwen3moe");
}

#[test]
fn test_moe_gpu_expert_dtype_supported() {
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q4K
    ));
    assert!(Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q5K
    ));
    // Q6K/Q8_0 disabled: f32 kernels compound numerical error across layers
    assert!(!Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q6K
    ));
    assert!(!Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::Q8_0
    ));
    assert!(!Qwen3MoeForward::moe_gpu_expert_dtype_supported(
        GgmlType::F32
    ));
}
