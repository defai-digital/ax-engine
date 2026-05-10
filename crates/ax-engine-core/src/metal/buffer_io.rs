use std::mem::{size_of, size_of_val};

use metal::{
    Buffer, CommandQueue, Device, MTLPixelFormat, MTLRegion, MTLResourceOptions, MTLTextureType,
    TextureDescriptor,
};
use objc::rc::autoreleasepool;

pub(super) const SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH: u32 = 16_384;
const SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS: u32 = 4;

pub(super) fn new_zeroed_shared_buffer<T>(device: &Device, element_count: u32) -> Buffer {
    let byte_count = element_count.max(1) as usize * size_of::<T>();
    let zeros = vec![0_u8; byte_count];
    device.new_buffer_with_data(
        zeros.as_ptr().cast(),
        zeros.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) fn new_shared_buffer_with_data<T>(device: &Device, values: &[T]) -> Buffer {
    if values.is_empty() {
        let zeros = vec![0_u8; size_of::<T>().max(1)];
        return device.new_buffer_with_data(
            zeros.as_ptr().cast(),
            zeros.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
    }

    device.new_buffer_with_data(
        values.as_ptr().cast(),
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub(super) fn copy_shared_buffer_prefix<T>(
    command_queue: &CommandQueue,
    src: &Buffer,
    dst: &Buffer,
    element_count: u32,
) {
    copy_shared_buffer_range::<T>(command_queue, src, 0, dst, 0, element_count);
}

pub(super) fn copy_shared_buffer_range<T>(
    command_queue: &CommandQueue,
    src: &Buffer,
    src_element_offset: u32,
    dst: &Buffer,
    dst_element_offset: u32,
    element_count: u32,
) {
    if element_count == 0 {
        return;
    }

    let byte_count = element_count as u64 * size_of::<T>() as u64;
    let src_offset = src_element_offset as u64 * size_of::<T>() as u64;
    let dst_offset = dst_element_offset as u64 * size_of::<T>() as u64;
    autoreleasepool(|| {
        let command_buffer = command_queue.new_command_buffer();
        command_buffer.set_label("ax.phase1.dispatch_arena_cache_copy");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(src, src_offset, dst, dst_offset, byte_count);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

pub(super) fn read_shared_buffer_prefix(buffer: &Buffer, element_count: u32) -> Vec<f32> {
    read_shared_typed_buffer_prefix(buffer, element_count, MTLPixelFormat::R32Float)
}

pub(super) fn read_shared_u32_buffer_prefix(buffer: &Buffer, element_count: u32) -> Vec<u32> {
    read_shared_typed_buffer_prefix(buffer, element_count, MTLPixelFormat::R32Uint)
}

fn read_shared_typed_buffer_prefix<T: Copy + Default>(
    buffer: &Buffer,
    element_count: u32,
    pixel_format: MTLPixelFormat,
) -> Vec<T> {
    if element_count == 0 {
        return Vec::new();
    }
    let mut values = Vec::with_capacity(element_count as usize);
    let mut offset = 0_u32;
    while offset < element_count {
        let max_chunk_len = element_count
            .saturating_sub(offset)
            .min(SHARED_BUFFER_READBACK_MAX_TEXTURE_WIDTH);
        let chunk_len = max_chunk_len - (max_chunk_len % SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS);
        if chunk_len == 0 {
            break;
        }
        let bytes_per_row = chunk_len as u64 * size_of::<T>() as u64;
        let texture = new_readback_texture(
            buffer,
            offset as u64 * size_of::<T>() as u64,
            bytes_per_row,
            chunk_len,
            pixel_format,
        );

        if texture.width() < chunk_len as u64 {
            return Vec::new();
        }

        let mut chunk_values = vec![T::default(); chunk_len as usize];
        texture.get_bytes(
            chunk_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, chunk_len as u64, 1),
            0,
        );
        values.extend_from_slice(&chunk_values);
        offset = offset.saturating_add(chunk_len);
    }
    if offset < element_count {
        let tail_count = element_count.saturating_sub(offset);
        let tail_window = SHARED_BUFFER_READBACK_ALIGNMENT_ELEMENTS.max(tail_count);
        let device = buffer.device().to_owned();
        let command_queue = device.new_command_queue();
        let staging_buffer = new_zeroed_shared_buffer::<T>(&device, tail_window);
        copy_shared_buffer_range::<T>(
            &command_queue,
            buffer,
            offset,
            &staging_buffer,
            0,
            tail_count,
        );
        let bytes_per_row = tail_window as u64 * size_of::<T>() as u64;
        let texture =
            new_readback_texture(&staging_buffer, 0, bytes_per_row, tail_window, pixel_format);
        if texture.width() < tail_window as u64 {
            return Vec::new();
        }
        let mut tail_values = vec![T::default(); tail_window as usize];
        texture.get_bytes(
            tail_values.as_mut_ptr().cast(),
            bytes_per_row,
            MTLRegion::new_2d(0, 0, tail_window as u64, 1),
            0,
        );
        values.extend_from_slice(&tail_values[..tail_count as usize]);
    }
    values
}

fn new_readback_texture(
    buffer: &Buffer,
    buffer_offset: u64,
    bytes_per_row: u64,
    width: u32,
    pixel_format: MTLPixelFormat,
) -> metal::Texture {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(pixel_format);
    descriptor.set_width(width as u64);
    descriptor.set_height(1);
    buffer.new_texture_with_descriptor(&descriptor, buffer_offset, bytes_per_row)
}
