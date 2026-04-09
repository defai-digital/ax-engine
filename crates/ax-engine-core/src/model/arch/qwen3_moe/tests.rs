use super::*;

#[test]
fn test_arch_name() {
    let fwd = Qwen3MoeForward;
    assert_eq!(fwd.arch_name(), "qwen3moe");
}
