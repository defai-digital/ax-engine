use crate::openai::responses::openai_finish_reason;
use ax_engine_sdk::GenerateFinishReason;

#[test]
fn finish_reason_maps_terminal_labels_without_hiding_cancellations() {
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
        Some("cancel")
    );
    assert_eq!(
        openai_finish_reason(Some(GenerateFinishReason::Error)),
        None
    );
    assert_eq!(openai_finish_reason(None), None);
}
