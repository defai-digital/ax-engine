use super::*;

pub(crate) fn saturating_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

pub(crate) fn saturating_u32_from_u64(value: u64) -> u32 {
    value.min(u32::MAX as u64) as u32
}

pub(crate) fn saturating_u32_from_u128(value: u128) -> u32 {
    value.min(u32::MAX as u128) as u32
}

pub(crate) fn elapsed_us(started: Instant) -> u32 {
    saturating_u32_from_u128(started.elapsed().as_micros())
}

pub(crate) fn kib_ceil(bytes: u64) -> u32 {
    if bytes == 0 {
        0
    } else {
        saturating_u32_from_u64(bytes.saturating_add(1023) / 1024)
    }
}
