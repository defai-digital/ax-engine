#![allow(clippy::expect_used)]

use ax_engine_core::gemma4_unified::{
    Gemma4UnifiedAudioRuntimeInput, Gemma4UnifiedImageRuntimeInput, Gemma4UnifiedModality,
    Gemma4UnifiedRuntimeInputs, Gemma4UnifiedSoftTokenRange, Gemma4UnifiedTokenSpan,
    Gemma4UnifiedVideoRuntimeInput,
};
use ax_engine_core::{
    MiniCpmV46ImageRuntimeInput, MiniCpmV46RuntimeInputs, NemotronOmniAudioRuntimeInput,
    NemotronOmniImageRuntimeInput, NemotronOmniRuntimeInputs, Qwen3VlImageRuntimeInput,
    Qwen3VlRuntimeInputs,
};

const FINGERPRINT: &str = "fixed-model-fingerprint";

fn qwen() -> Qwen3VlRuntimeInputs {
    Qwen3VlRuntimeInputs {
        images: vec![Qwen3VlImageRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 2,
            patches: vec![0.25; 8 * 1536],
            num_patches: 8,
            patch_dim: 1536,
            grid_t: 1,
            height: 32,
            width: 64,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            is_video: false,
        }],
    }
}

fn span(modality: Gemma4UnifiedModality, soft_token_count: u32) -> Gemma4UnifiedTokenSpan {
    Gemma4UnifiedTokenSpan {
        modality,
        placeholder_index: 1,
        replacement_start: 1,
        soft_token_count,
        replacement_token_count: soft_token_count + 2,
    }
}

fn gemma_image() -> Gemma4UnifiedRuntimeInputs {
    Gemma4UnifiedRuntimeInputs {
        images: vec![Gemma4UnifiedImageRuntimeInput {
            span: span(Gemma4UnifiedModality::Image, 2),
            pixel_values: vec![0.25; 24],
            pixel_position_ids: vec![[0, 0], [0, 1]],
        }],
        ..Default::default()
    }
}

fn gemma_video() -> Gemma4UnifiedRuntimeInputs {
    Gemma4UnifiedRuntimeInputs {
        videos: vec![Gemma4UnifiedVideoRuntimeInput {
            span: Gemma4UnifiedTokenSpan {
                replacement_token_count: 12,
                ..span(Gemma4UnifiedModality::Video, 4)
            },
            soft_token_ranges: vec![
                Gemma4UnifiedSoftTokenRange {
                    start: 2,
                    soft_token_count: 2,
                },
                Gemma4UnifiedSoftTokenRange {
                    start: 8,
                    soft_token_count: 2,
                },
            ],
            pixel_values: vec![0.25; 48],
            pixel_position_ids: vec![[0, 0], [0, 1], [0, 0], [0, 1]],
            frame_count: 2,
        }],
        ..Default::default()
    }
}

fn minicpm() -> MiniCpmV46RuntimeInputs {
    MiniCpmV46RuntimeInputs {
        images: vec![MiniCpmV46ImageRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 2,
            pixel_values: vec![0.25; 56 * 112 * 3],
            height: 56,
            width: 112,
            patch_size: 14,
            spatial_downsample_factor: 4,
        }],
    }
}

fn nemotron() -> NemotronOmniRuntimeInputs {
    NemotronOmniRuntimeInputs {
        images: vec![NemotronOmniImageRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 2,
            pixel_values: vec![0.25; 3 * 32 * 64],
            height: 32,
            width: 64,
            patch_size: 16,
            spatial_downsample_factor: 2,
        }],
        ..Default::default()
    }
}

// Every collision pair must be accepted by the production request contract.
// Tensor values and soft-token counts stay identical within each pair.
macro_rules! assert_valid_miss {
    ($before:expr, $after:expr) => {{
        $before
            .validate_for_prompt_len(32)
            .expect("valid original media");
        $after
            .validate_for_prompt_len(32)
            .expect("valid changed media");
        assert_ne!(
            $before.media_prefix_key(FINGERPRINT),
            $after.media_prefix_key(FINGERPRINT)
        );
    }};
}

#[test]
fn qwen_media_identity_binds_grid_and_scatter() {
    let original = qwen();
    let mut changed = original.clone();
    changed.images[0].height = 64;
    changed.images[0].width = 32;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].placeholder_index = 8;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].grid_t = 2;
    changed.images[0].width = 32;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].is_video = true;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].temporal_patch_size = 1;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].patch_size = 8;
    changed.images[0].num_patches = 32;
    changed.images[0].patch_dim = 384;
    changed.images[0].spatial_merge_size = 4;
    assert_valid_miss!(original, changed);
}

#[test]
fn gemma_media_identity_binds_positions_and_scatter() {
    let original = gemma_image();
    let mut changed = original.clone();
    changed.images[0].pixel_position_ids = vec![[0, 0], [1, 0]];
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].span.replacement_start = 8;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].span.placeholder_index = 8;
    assert_valid_miss!(original, changed);
}

#[test]
fn gemma_video_media_identity_binds_ranges_positions_and_frames() {
    let original = gemma_video();
    let mut changed = original.clone();
    changed.videos[0].soft_token_ranges[1].start = 9;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.videos[0].span.replacement_token_count = 13;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.videos[0].pixel_position_ids = vec![[0, 0], [1, 0], [0, 0], [1, 0]];
    assert_valid_miss!(original, changed);
    // The contiguous legacy representation also carries independent frame geometry.
    let mut original = original.clone();
    original.videos[0].soft_token_ranges.clear();
    original.videos[0].span.replacement_token_count = 6;
    let mut changed = original.clone();
    changed.videos[0].frame_count = 1;
    assert_valid_miss!(original, changed);
}

#[test]
fn minicpm_media_identity_binds_geometry_and_scatter() {
    let original = minicpm();
    let mut changed = original.clone();
    changed.images[0].height = 112;
    changed.images[0].width = 56;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].placeholder_index = 8;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].patch_size = 28;
    changed.images[0].spatial_downsample_factor = 2;
    assert_valid_miss!(original, changed);
}

#[test]
fn nemotron_media_identity_binds_geometry_and_scatter() {
    let original = nemotron();
    let mut changed = original.clone();
    changed.images[0].height = 64;
    changed.images[0].width = 32;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].placeholder_index = 8;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.images[0].patch_size = 32;
    changed.images[0].spatial_downsample_factor = 1;
    assert_valid_miss!(original, changed);
}

#[test]
fn audio_media_identity_binds_rate_and_scatter() {
    let original = NemotronOmniRuntimeInputs {
        audios: vec![NemotronOmniAudioRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 2,
            samples: vec![0.25; 24],
            sample_rate: 16_000,
        }],
        ..Default::default()
    };
    let mut changed = original.clone();
    changed.audios[0].sample_rate = 48_000;
    assert_valid_miss!(original, changed);
    let mut changed = original.clone();
    changed.audios[0].placeholder_index = 8;
    assert_valid_miss!(original, changed);

    let original = Gemma4UnifiedRuntimeInputs {
        audios: vec![Gemma4UnifiedAudioRuntimeInput {
            span: span(Gemma4UnifiedModality::Audio, 2),
            input_features: vec![0.25; 24],
            frame_count: 2,
            feature_count: 12,
        }],
        ..Default::default()
    };
    let mut changed = original.clone();
    changed.audios[0].span.replacement_start = 8;
    assert_valid_miss!(original, changed);
}

#[test]
fn media_identity_separates_families_and_modalities() {
    let image = gemma_image();
    let audio = Gemma4UnifiedRuntimeInputs {
        audios: vec![Gemma4UnifiedAudioRuntimeInput {
            span: span(Gemma4UnifiedModality::Audio, 2),
            input_features: image.images[0].pixel_values.clone(),
            frame_count: 2,
            feature_count: 12,
        }],
        ..Default::default()
    };
    assert_valid_miss!(image, audio);

    let mini = minicpm();
    let omni = NemotronOmniRuntimeInputs {
        images: vec![NemotronOmniImageRuntimeInput {
            placeholder_index: 2,
            soft_token_count: 2,
            pixel_values: mini.images[0].pixel_values.clone(),
            height: 56,
            width: 112,
            patch_size: 14,
            spatial_downsample_factor: 4,
        }],
        ..Default::default()
    };
    assert_valid_miss!(mini, omni);
}

#[test]
fn media_identity_preserves_empty_sentinel_and_order() {
    assert_eq!(
        Qwen3VlRuntimeInputs::default().media_prefix_key(FINGERPRINT),
        ""
    );
    assert_eq!(
        Gemma4UnifiedRuntimeInputs::default().media_prefix_key(FINGERPRINT),
        ""
    );
    assert_eq!(
        MiniCpmV46RuntimeInputs::default().media_prefix_key(FINGERPRINT),
        ""
    );
    assert_eq!(
        NemotronOmniRuntimeInputs::default().media_prefix_key(FINGERPRINT),
        ""
    );
    let mut original = qwen();
    let mut second = original.images[0].clone();
    second.placeholder_index = 8;
    original.images.push(second);
    let mut changed = original.clone();
    changed.images.reverse();
    assert_valid_miss!(original, changed);
    assert_eq!(
        original.media_prefix_key(FINGERPRINT),
        original.clone().media_prefix_key(FINGERPRINT)
    );
    assert_ne!(
        original.media_prefix_key(FINGERPRINT),
        original.media_prefix_key("other-model")
    );
}

#[test]
fn runtime_media_identity_retires_v1_and_hashes_tensor_tail_bits() {
    let original = qwen();
    let legacy = ax_engine_core::ordered_media_digests_key(&[ax_engine_core::media_digest_f32(
        &original.images[0].patches,
        2,
        FINGERPRINT,
    )]);
    assert_eq!(
        legacy,
        "bcbc45b5783138d62c1221ac76b5a070cbcc991bcb62ee55d3bc00f6cc2592eb"
    );
    assert_ne!(original.media_prefix_key(FINGERPRINT), legacy);
    let mut changed = original.clone();
    *changed.images[0].patches.last_mut().expect("tensor tail") = -0.0;
    assert_valid_miss!(original, changed);
    let mut positive_zero = changed.clone();
    *positive_zero.images[0]
        .patches
        .last_mut()
        .expect("tensor tail") = 0.0;
    assert_valid_miss!(changed, positive_zero);
}
