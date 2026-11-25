//! Prefill scheduling rules of the MLX runner as pure functions.
//!
//! The runner decides, per prefill item, how wide its forwards are and
//! whether it ends a forward early to store a prefix snapshot. Those rules
//! depend on the model family, the MTP policy, the host and a handful of
//! switches. Each rule lives here as a function of plain values, so it has a
//! table test, and the runner only gathers the inputs and carries the result
//! out.

/// What decides whether a prefill item ends a forward to store a prefix
/// snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CaptureInputs {
    /// The item belongs to a semantic-cut capture, which stores its own cut.
    pub semantic_capture: bool,
    /// The model has linear-attention (recurrent) layers, whose state cannot
    /// be cut out of a longer prefill afterwards.
    pub linear_attention: bool,
    /// The runner restores a stored prefix only as a whole prompt
    /// (Flash Next without MTP), so no snapshot inside a prompt is readable.
    pub exact_prefix_only: bool,
    /// The item runs the Flash Next MTP prefill, which absorbs draft history.
    pub mtp_prefill: bool,
    /// `AX_MLX_LINEAR_PREFIX_GRID_RUNG`.
    pub grid_rung_enabled: bool,
    /// The in-memory prefix cache accepts stores.
    pub prefix_cache_enabled: bool,
}

/// The snapshots a prefill item may stop for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CapturePolicy {
    /// At the last prefill-chunk grid rung the item crosses.
    pub grid_rung: bool,
    /// At the last block boundary of the prompt.
    pub block_boundary: bool,
}

/// Which interior snapshots a prefill item captures.
///
/// A snapshot is taken only where a later request can restore it: on a
/// linear-attention model, with the prefix cache on, outside a semantic-cut
/// capture, and not on a runner that restores whole prompts only. The grid
/// rung additionally needs its switch and the ordinary prefill, because the
/// MTP prefill keeps its own schedule up to the block boundary.
pub(crate) fn capture_policy(inputs: CaptureInputs) -> CapturePolicy {
    let interior = !inputs.semantic_capture
        && inputs.linear_attention
        && !inputs.exact_prefix_only
        && inputs.prefix_cache_enabled;
    CapturePolicy {
        grid_rung: interior && !inputs.mtp_prefill && inputs.grid_rung_enabled,
        block_boundary: interior,
    }
}

/// A loaded Flash Next trunk, as far as the long-prompt chunk depends on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlashNextTrunk {
    /// Routed experts are streamed from disk rather than resident.
    pub expert_stream: bool,
    /// Longest context at which a forward of the base chunk fits the
    /// sparse-attention index-score workspace.
    pub score_context_limit: usize,
}

/// What decides whether a long prompt keeps the runner's base prefill chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WideChunkInputs {
    /// `AX_MLX_FLASH_NEXT_LONG_PROMPT_WIDE_CHUNK`.
    pub enabled: bool,
    /// The host has a Neural Accelerator GPU (M5 and later), where the wide
    /// chunk was measured.
    pub neural_accelerator: bool,
    /// The Flash Next trunk, when this runner serves one.
    pub trunk: Option<FlashNextTrunk>,
    /// The two-sequence prefill split is active and owns the schedule.
    pub split_prefill: bool,
}

/// Prefill chunk for `remaining` prompt tokens after `prior` cached ones.
///
/// `scaled` is the family's long-prompt scaling of `base`. Flash Next keeps
/// `base` instead while the whole prompt fits the index-score workspace at
/// that chunk: each of its forwards streams every routed expert stack once,
/// so fewer, wider forwards are cheaper. Every other case returns `scaled`.
pub(crate) fn long_prompt_chunk(
    base: usize,
    scaled: usize,
    prior: usize,
    remaining: usize,
    wide: WideChunkInputs,
) -> usize {
    let keeps_base = scaled < base
        && wide.enabled
        && wide.neural_accelerator
        && !wide.split_prefill
        && wide.trunk.is_some_and(|trunk| {
            !trunk.expert_stream && prior.saturating_add(remaining) <= trunk.score_context_limit
        });
    if keeps_base { base } else { scaled }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINEAR_WITH_CACHE: CaptureInputs = CaptureInputs {
        semantic_capture: false,
        linear_attention: true,
        exact_prefix_only: false,
        mtp_prefill: false,
        grid_rung_enabled: true,
        prefix_cache_enabled: true,
    };

    #[test]
    fn interior_captures_follow_what_a_later_request_can_restore() {
        let both = CapturePolicy {
            grid_rung: true,
            block_boundary: true,
        };
        let boundary_only = CapturePolicy {
            grid_rung: false,
            block_boundary: true,
        };
        let none = CapturePolicy::default();
        let cases: [(&str, CaptureInputs, CapturePolicy); 7] = [
            ("linear model, cache on", LINEAR_WITH_CACHE, both),
            (
                "full-attention model",
                CaptureInputs {
                    linear_attention: false,
                    ..LINEAR_WITH_CACHE
                },
                none,
            ),
            (
                "prefix cache off",
                CaptureInputs {
                    prefix_cache_enabled: false,
                    ..LINEAR_WITH_CACHE
                },
                none,
            ),
            (
                "semantic-cut capture",
                CaptureInputs {
                    semantic_capture: true,
                    ..LINEAR_WITH_CACHE
                },
                none,
            ),
            (
                "Flash Next without MTP: whole prompts only",
                CaptureInputs {
                    exact_prefix_only: true,
                    ..LINEAR_WITH_CACHE
                },
                none,
            ),
            (
                "Flash Next MTP prefill keeps its schedule up to the boundary",
                CaptureInputs {
                    mtp_prefill: true,
                    ..LINEAR_WITH_CACHE
                },
                boundary_only,
            ),
            (
                "grid rung switched off",
                CaptureInputs {
                    grid_rung_enabled: false,
                    ..LINEAR_WITH_CACHE
                },
                boundary_only,
            ),
        ];
        for (name, inputs, expected) in cases {
            assert_eq!(capture_policy(inputs), expected, "{name}");
        }
    }

    const RESIDENT_TRUNK: FlashNextTrunk = FlashNextTrunk {
        expert_stream: false,
        score_context_limit: 32_768,
    };
    const M5_FLASH_NEXT: WideChunkInputs = WideChunkInputs {
        enabled: true,
        neural_accelerator: true,
        trunk: Some(RESIDENT_TRUNK),
        split_prefill: false,
    };

    #[test]
    fn long_prompts_keep_the_base_chunk_only_on_a_resident_flash_next_trunk() {
        // (name, base, scaled, prior, remaining, inputs, expected)
        let cases: [(&str, usize, usize, usize, usize, WideChunkInputs, usize); 11] = [
            (
                "short prompt is not scaled",
                2048,
                2048,
                0,
                1500,
                M5_FLASH_NEXT,
                2048,
            ),
            (
                "long prompt keeps the base chunk",
                2048,
                512,
                0,
                4443,
                M5_FLASH_NEXT,
                2048,
            ),
            (
                "exactly at the workspace limit",
                2048,
                512,
                0,
                32_768,
                M5_FLASH_NEXT,
                2048,
            ),
            (
                "one token past the limit",
                2048,
                512,
                0,
                32_769,
                M5_FLASH_NEXT,
                512,
            ),
            (
                "cached prefix counts toward the limit",
                2048,
                512,
                30_000,
                2769,
                M5_FLASH_NEXT,
                512,
            ),
            (
                "switch off",
                2048,
                512,
                0,
                4443,
                WideChunkInputs {
                    enabled: false,
                    ..M5_FLASH_NEXT
                },
                512,
            ),
            (
                "no Neural Accelerator",
                2048,
                512,
                0,
                4443,
                WideChunkInputs {
                    neural_accelerator: false,
                    ..M5_FLASH_NEXT
                },
                512,
            ),
            (
                "another model family",
                2048,
                512,
                0,
                4443,
                WideChunkInputs {
                    trunk: None,
                    ..M5_FLASH_NEXT
                },
                512,
            ),
            (
                "streamed experts",
                2048,
                512,
                0,
                4443,
                WideChunkInputs {
                    trunk: Some(FlashNextTrunk {
                        expert_stream: true,
                        ..RESIDENT_TRUNK
                    }),
                    ..M5_FLASH_NEXT
                },
                512,
            ),
            (
                "two-sequence prefill split",
                2048,
                512,
                0,
                4443,
                WideChunkInputs {
                    split_prefill: true,
                    ..M5_FLASH_NEXT
                },
                512,
            ),
            (
                "a scaled chunk above the base is left alone",
                1024,
                2048,
                0,
                4443,
                M5_FLASH_NEXT,
                2048,
            ),
        ];
        for (name, base, scaled, prior, remaining, inputs, expected) in cases {
            assert_eq!(
                long_prompt_chunk(base, scaled, prior, remaining, inputs),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn prior_plus_remaining_cannot_overflow_into_a_wide_chunk() {
        assert_eq!(
            long_prompt_chunk(2048, 512, usize::MAX, 4443, M5_FLASH_NEXT),
            512
        );
    }
}
