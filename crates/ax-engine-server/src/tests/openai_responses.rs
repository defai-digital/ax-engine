use crate::openai::responses::openai_finish_reason;
use ax_engine_sdk::GenerateFinishReason;

#[test]
fn finish_reason_preserves_only_real_openai_terminal_labels() {
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Stop)),
        Some("stop")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::MaxOutputTokens)),
        Some("length")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::ContentFilter)),
        Some("content_filter")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Cancelled)),
        None
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Error)),
        None
    );
    assert_eq!(openai_finish_reason(None), None);
}
