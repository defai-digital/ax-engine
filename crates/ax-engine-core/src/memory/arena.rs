/// Pre-allocated arena for scratch buffers in the forward pass.
///
/// Allocations are carved out sequentially and can be reused after `reset()`.
/// This avoids churn from repeated temporary `Vec` allocations in hot paths.
#[derive(Debug, Clone)]
pub struct Arena {
    buf: Vec<f32>,
    cursor: usize,
}

impl Arena {
    /// Create a new arena with capacity for `capacity_f32` values.
    pub fn with_capacity(capacity_f32: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity_f32),
            cursor: 0,
        }
    }

    /// Allocate a scratch buffer of `n` f32 values.
    ///
    /// The returned slice is zero-initialized.
    pub fn alloc_f32(&mut self, n: usize) -> &mut [f32] {
        let start = self.cursor;
        let end = start.saturating_add(n);

        if end > self.buf.len() {
            self.buf.resize(end, 0.0);
        } else {
            self.buf[start..end].fill(0.0);
        }

        self.cursor = end;
        &mut self.buf[start..end]
    }

    /// Reset the arena for the next forward pass (no dealloc — just reset pointer).
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn alloc_returns_requested_len() {
        let mut arena = Arena::with_capacity(8);
        let s = arena.alloc_f32(5);
        assert_eq!(s.len(), 5);
        assert!(s.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn reset_reuses_memory() {
        let mut arena = Arena::with_capacity(4);
        let s1 = arena.alloc_f32(4);
        s1[0] = 42.0;
        arena.reset();
        let s2 = arena.alloc_f32(2);
        assert_eq!(s2.len(), 2);
        assert_eq!(s2[0], 0.0);
    }

    #[test]
    fn grows_when_needed() {
        let mut arena = Arena::with_capacity(1);
        let s = arena.alloc_f32(16);
        assert_eq!(s.len(), 16);
    }
}
