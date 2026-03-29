use super::*;

#[test]
fn test_qwen3_forward_arch_name() {
    let fwd = Qwen3Forward;
    assert_eq!(fwd.arch_name(), "qwen3");
}

#[test]
fn test_qwen35_layout_rejected_by_architecture() {
    assert!(unsupported_qwen3_layout_reason("qwen35", false, false).is_some());
}

#[test]
fn test_qwen35_layout_rejected_by_fused_qkv() {
    assert!(unsupported_qwen3_layout_reason("qwen3", true, false).is_some());
}

#[test]
fn test_qwen35_layout_rejected_by_ssm_tensors() {
    assert!(unsupported_qwen3_layout_reason("qwen3", false, true).is_some());
}

#[test]
fn test_plain_qwen3_layout_remains_supported() {
    assert!(unsupported_qwen3_layout_reason("qwen3", false, false).is_none());
}
