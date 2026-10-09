//! Weight-free controls for the Flash Next fallback disposition, the decode
//! gate order, and the paused catch-up buffer.
//!
//! Runner paths that need the real draft head live in `flash_next_tests.rs`
//! behind synthetic-artifact gates; everything here is pure so it runs in the
//! ordinary `cargo test` pass.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;

fn row_array(value: f32) -> MlxArray {
    let data = [value];
    MlxArray::from_raw_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data.as_slice()),
        &[1, 1, 1],
        MlxDtype::Float32,
    )
}

#[test]
fn fallback_disposition_truth_table() {
    use FlashNextMtpFallbackDisposition::{Drop, RetainPaused};
    let expected = [
        (FlashNextMtpFallbackReason::NotStrictGreedy, Drop),
        (FlashNextMtpFallbackReason::ThinkControl, RetainPaused),
        (FlashNextMtpFallbackReason::PendingDirect, RetainPaused),
        (FlashNextMtpFallbackReason::NoBudget, Drop),
        (FlashNextMtpFallbackReason::CursorUnavailable, Drop),
        (FlashNextMtpFallbackReason::ComponentsUnavailable, Drop),
        (FlashNextMtpFallbackReason::StepError, Drop),
    ];
    assert_eq!(expected.len(), FLASH_NEXT_MTP_FALLBACK_REASON_COUNT);
    for (reason, disposition) in expected {
        assert_eq!(
            flash_next_mtp_fallback_disposition(reason, false),
            disposition,
            "{reason:?} without the kill switch"
        );
        // The kill switch restores the legacy drop-on-block policy for every
        // reason, including the two the paused state exists for.
        assert_eq!(
            flash_next_mtp_fallback_disposition(reason, true),
            Drop,
            "{reason:?} with AX_MLX_FLASH_NEXT_STICKY_FALLBACK=1"
        );
    }
    // Every reason keeps its own route bucket: a paused step is attributed to
    // the cursor gap it serves from, never to a new counter.
    for reason in FlashNextMtpFallbackReason::ALL {
        assert!(!reason.route_key().is_empty());
    }
}

#[test]
fn fallback_disposition_of_decode_blocks_matches_the_reason() {
    for (block, reason) in [
        (
            FlashNextMtpDecodeBlock::NotStrictGreedy,
            FlashNextMtpFallbackReason::NotStrictGreedy,
        ),
        (
            FlashNextMtpDecodeBlock::ThinkControl,
            FlashNextMtpFallbackReason::ThinkControl,
        ),
        (
            FlashNextMtpDecodeBlock::PendingDirect,
            FlashNextMtpFallbackReason::PendingDirect,
        ),
        (
            FlashNextMtpDecodeBlock::NoBudget,
            FlashNextMtpFallbackReason::NoBudget,
        ),
        (
            FlashNextMtpDecodeBlock::CursorUnavailable,
            FlashNextMtpFallbackReason::CursorUnavailable,
        ),
        (
            FlashNextMtpDecodeBlock::CursorPaused,
            FlashNextMtpFallbackReason::CursorUnavailable,
        ),
    ] {
        assert_eq!(block.fallback_reason(), reason);
    }
}

#[test]
fn decode_block_gate_order_is_fixed() {
    // Nothing blocks: strict greedy, no think window, no pending token, budget
    // left, aligned cursor.
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, false, 8, false, true),
        None
    );
    // NotStrictGreedy dominates every other reason.
    assert_eq!(
        flash_next_mtp_decode_block(false, true, true, true, 0, true, false),
        Some(FlashNextMtpDecodeBlock::NotStrictGreedy)
    );
    // Think control beats a pending token, exhausted budget and the cursor gap.
    assert_eq!(
        flash_next_mtp_decode_block(true, true, false, true, 0, true, false),
        Some(FlashNextMtpDecodeBlock::ThinkControl)
    );
    assert_eq!(
        flash_next_mtp_decode_block(true, false, true, true, 0, true, false),
        Some(FlashNextMtpDecodeBlock::ThinkControl)
    );
    // A pending direct token beats the budget and the cursor gap.
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, true, 0, true, false),
        Some(FlashNextMtpDecodeBlock::PendingDirect)
    );
    // An exhausted budget beats the cursor gap.
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, false, 0, false, false),
        Some(FlashNextMtpDecodeBlock::NoBudget)
    );
    // Paused beats aligned: a paused cursor keeps pausing even if the draft
    // history would verify, and it is never read as a lost cursor.
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, false, 8, true, false),
        Some(FlashNextMtpDecodeBlock::CursorPaused)
    );
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, false, 8, true, true),
        Some(FlashNextMtpDecodeBlock::CursorPaused)
    );
    // Last gate: the cursor is not at the trunk boundary.
    assert_eq!(
        flash_next_mtp_decode_block(true, false, false, false, 8, false, false),
        Some(FlashNextMtpDecodeBlock::CursorUnavailable)
    );
}

#[test]
fn pause_buffer_pairs_tokens_with_rows_in_trunk_order() {
    let mut buffer = FlashNextCursorPauseBuffer::default();
    assert!(buffer.is_empty());
    assert!(buffer.push(17, row_array(0.25)));
    assert!(buffer.push(31, row_array(0.5)));
    let (tokens, rows) = buffer.try_absorb_parts().unwrap();
    assert_eq!(tokens, vec![17, 31]);
    assert_eq!(rows.shape(), [1, 2, 1]);
    mlx_sys::eval(&[&rows]);
    assert_eq!(rows.data_f32(), [0.25, 0.5]);
}

#[test]
fn pause_buffer_caps_at_the_row_limit() {
    let mut buffer = FlashNextCursorPauseBuffer::default();
    for index in 0..FLASH_NEXT_CURSOR_PAUSE_ROW_CAP {
        assert!(buffer.push(index as u32, row_array(index as f32)));
    }
    assert!(!buffer.push(9_999, row_array(0.0)));
    assert_eq!(buffer.rows.len(), FLASH_NEXT_CURSOR_PAUSE_ROW_CAP);
    let (tokens, _rows) = buffer.try_absorb_parts().unwrap();
    assert_eq!(tokens.len(), FLASH_NEXT_CURSOR_PAUSE_ROW_CAP);
    assert_eq!(tokens.last().copied(), Some(255));
}

#[test]
fn pause_overflow_drops_the_cursor_and_counts_it() {
    let mut state = FlashNextMtpRequestState::default();
    for index in 0..FLASH_NEXT_CURSOR_PAUSE_ROW_CAP {
        assert!(state.extend_pause(index as u32, row_array(0.0)));
    }
    assert!(state.paused.is_some());
    assert!(!state.extend_pause(7, row_array(0.0)));
    assert!(
        state.paused.is_none(),
        "an overflowed buffer must not survive the drop"
    );
    assert_eq!(state.telemetry.cursor_pause_overflows, 1);
    // No cursor was live, so nothing inflates the real drop counter.
    assert_eq!(state.telemetry.cursor_dropped, 0);
}

#[test]
fn drop_cursor_clears_the_pause_buffer() {
    let mut state = FlashNextMtpRequestState::default();
    assert!(state.extend_pause(11, row_array(0.0)));
    assert!(state.paused.is_some());
    state.drop_cursor();
    assert!(state.paused.is_none());
    assert!(state.take_pause_absorb_parts().is_none());
    assert_eq!(state.telemetry.cursor_dropped, 0);
}

#[test]
fn paused_cursor_is_not_prefix_snapshot_eligible() {
    let mut state = FlashNextMtpRequestState::default();
    assert!(!state.prefix_snapshot_eligible(), "no cursor, no sidecar");
    state.paused = Some(FlashNextCursorPauseBuffer::default());
    assert!(!state.prefix_snapshot_eligible());
    // The paused state is what fences the sidecar; the cursor itself is not
    // constructible without head weights here, so the cursor-present arm of
    // `prefix_snapshot_eligible` is covered by the artifact-backed tests.
    assert!(state.paused.is_some());
}

#[test]
fn pause_absorb_rejects_foreign_row_shapes() {
    let mut buffer = FlashNextCursorPauseBuffer::default();
    assert!(buffer.push(17, row_array(0.25)));
    // A foreign row (a different width) must fail closed, not panic inside the
    // concatenate: the flush path drops the cursor on this error.
    let wide = [1.0f32, 2.0];
    buffer.rows.push((
        31,
        MlxArray::from_raw_data(
            wide.as_ptr().cast(),
            std::mem::size_of_val(wide.as_slice()),
            &[1, 1, 2],
            MlxDtype::Float32,
        ),
    ));
    assert!(buffer.try_absorb_parts().is_err());
}

#[test]
fn pause_absorb_parts_are_taken_once() {
    let mut state = FlashNextMtpRequestState::default();
    assert!(state.extend_pause(3, row_array(1.0)));
    let taken = state
        .take_pause_absorb_parts()
        .expect("buffered rows")
        .unwrap();
    assert_eq!(taken.0, vec![3]);
    assert!(
        state.take_pause_absorb_parts().is_none(),
        "a catch-up attempt consumes the buffer"
    );
    assert!(state.paused.is_none());
}

#[test]
fn sticky_kill_switch_restores_legacy_drop_on_every_block() {
    // The kill switch is a `OnceLock`-cached env read, so parity is asserted
    // through the scoped override instead of mutating process state.
    let env_default = crate::fastpath::flash_next_sticky_fallback_env_enabled();
    {
        let _scope = crate::fastpath::scoped_flash_next_sticky_fallback(true);
        assert!(crate::fastpath::flash_next_sticky_fallback_enabled());
        for reason in FlashNextMtpFallbackReason::ALL {
            assert_eq!(
                flash_next_mtp_fallback_disposition(
                    reason,
                    crate::fastpath::flash_next_sticky_fallback_enabled(),
                ),
                FlashNextMtpFallbackDisposition::Drop,
                "{reason:?}"
            );
        }
        {
            // Nested scopes restore the outer selection, not the environment.
            let _inner = crate::fastpath::scoped_flash_next_sticky_fallback(false);
            assert!(!crate::fastpath::flash_next_sticky_fallback_enabled());
        }
        assert!(crate::fastpath::flash_next_sticky_fallback_enabled());
    }
    assert_eq!(
        crate::fastpath::flash_next_sticky_fallback_enabled(),
        env_default
    );
}
