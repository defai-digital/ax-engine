/// Number of trailing tokens that stay unique per request in a shared-prefix
/// scenario; everything before the tail is shared, so prefix-reuse
/// measurements scale with the manifest's `input_tokens_target` instead of a
/// fixed 64-token stub.
pub(crate) const SHARED_PREFIX_UNIQUE_TAIL_TOKENS: u32 = 64;

pub(crate) fn scenario_shared_prefix_target(target_len: u32) -> u32 {
    target_len.saturating_sub(SHARED_PREFIX_UNIQUE_TAIL_TOKENS)
}

pub(crate) fn synthetic_prompt_tokens(
    target_len: u32,
    prompt_ref: Option<&str>,
    prefix_group: Option<&str>,
    shared_prefix_target: u32,
    body_group: Option<&str>,
    ordinal: u32,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(target_len as usize);
    let shared_prefix_len = prefix_group.map_or(0, |_| target_len.min(shared_prefix_target));
    let shared_seed = prefix_group.map(stable_hash32).unwrap_or(17);
    let body_seed = body_group.map(stable_hash32).unwrap_or_else(|| {
        stable_hash32(prompt_ref.unwrap_or("prompt")).wrapping_add(ordinal * 131)
    });

    for index in 0..shared_prefix_len {
        tokens.push(shared_seed.wrapping_add(index + 1));
    }

    for index in shared_prefix_len..target_len {
        tokens.push(body_seed.wrapping_add(index + 1));
    }

    tokens
}

pub(crate) fn synthetic_prompt_text(
    target_len: u32,
    prompt_ref: Option<&str>,
    prefix_group: Option<&str>,
    shared_prefix_target: u32,
    body_group: Option<&str>,
    ordinal: u32,
) -> String {
    let shared_prefix_len = prefix_group.map_or(0, |_| target_len.min(shared_prefix_target));
    let shared_seed = prefix_group.map(stable_hash32).unwrap_or(17);
    let body_seed = body_group.map(stable_hash32).unwrap_or_else(|| {
        stable_hash32(prompt_ref.unwrap_or("prompt")).wrapping_add(ordinal * 131)
    });

    let mut parts = Vec::with_capacity(target_len as usize);
    for index in 0..shared_prefix_len {
        parts.push(format!("shared{}_{}", shared_seed, index + 1));
    }

    for index in shared_prefix_len..target_len {
        parts.push(format!("body{}_{}", body_seed, index + 1));
    }

    parts.join(" ")
}

pub(crate) fn synthetic_text_output_tokens(text: &str, target_len: u32) -> Vec<u32> {
    synthetic_prompt_tokens(target_len, Some(text), None, 0, None, 0)
}

pub(crate) fn replay_prompt_target(prompt_ref: &str) -> u32 {
    if prompt_ref.contains("long") {
        1024
    } else if prompt_ref.contains("variant") {
        320
    } else {
        256
    }
}

fn stable_hash32(input: &str) -> u32 {
    let mut hash = 2_166_136_261u32;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash
}
