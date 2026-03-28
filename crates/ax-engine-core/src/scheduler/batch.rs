//! Micro-batching window for collecting concurrent requests.
//!
//! `BatchCollector` accumulates items within a configurable time window
//! (default 2–5ms) and returns them as a batch. This amortizes per-request
//! overhead and enables batched matrix operations.
//!
//! # Usage pattern
//!
//! ```text
//! Producer threads:    collector.submit(request)
//!                         ↓
//! Collector thread:    let batch = collector.collect()
//!                         ↓
//!                      process(batch)   // e.g. batched matmul
//! ```
//!
//! The collector blocks on `collect()` until at least one item arrives,
//! then waits for the remainder of the window to gather more items.
//! Returns early if `max_batch_size` is reached.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Batch collector configuration.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Collection window duration. Items arriving within this window
    /// after the first item are grouped into a single batch.
    /// Default: 3ms (midpoint of 2–5ms range).
    pub window: Duration,
    /// Maximum batch size. Triggers early collection when reached.
    /// Default: 8.
    pub max_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            window: Duration::from_millis(3),
            max_batch_size: 8,
        }
    }
}

/// Micro-batching collector that groups items within a time window.
///
/// Thread-safe: multiple producers can call `submit()` concurrently
/// while a single consumer calls `collect()`.
pub struct BatchCollector<T> {
    config: BatchConfig,
    queue: Mutex<VecDeque<T>>,
    notify: Condvar,
}

impl<T> BatchCollector<T> {
    /// Create a new batch collector with the given configuration.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            queue: Mutex::new(VecDeque::new()),
            notify: Condvar::new(),
        }
    }

    /// Submit an item to the batch queue.
    ///
    /// Returns immediately. The item will be included in the next
    /// batch returned by `collect()`.
    pub fn submit(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(item);
        self.notify.notify_one();
    }

    /// Collect a batch of items.
    ///
    /// 1. Blocks until at least one item arrives
    /// 2. Starts the collection window
    /// 3. Returns when the window expires or `max_batch_size` is reached
    ///
    /// Always returns at least one item (blocks until available).
    pub fn collect(&self) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();

        // Wait for the first item
        while queue.is_empty() {
            queue = self.notify.wait(queue).unwrap();
        }

        // First item received — start collection window
        let deadline = Instant::now() + self.config.window;
        Self::drain_until(&self.notify, queue, deadline, self.config.max_batch_size)
    }

    /// Collect a batch with an overall timeout.
    ///
    /// Like `collect()`, but returns an empty batch if no items arrive
    /// within `timeout`.
    pub fn collect_timeout(&self, timeout: Duration) -> Vec<T> {
        let deadline = Instant::now() + timeout;
        let mut queue = self.queue.lock().unwrap();

        // Wait for the first item (with timeout)
        while queue.is_empty() {
            let now = Instant::now();
            if now >= deadline {
                return Vec::new();
            }
            let (guard, timeout_result) = self.notify.wait_timeout(queue, deadline - now).unwrap();
            queue = guard;
            if timeout_result.timed_out() && queue.is_empty() {
                return Vec::new();
            }
        }

        // First item received — collect for the shorter of window or remaining timeout
        let window_deadline = Instant::now() + self.config.window;
        let effective_deadline = window_deadline.min(deadline);
        Self::drain_until(
            &self.notify,
            queue,
            effective_deadline,
            self.config.max_batch_size,
        )
    }

    /// Try to collect a batch without blocking.
    ///
    /// Returns whatever items are currently in the queue (may be empty).
    pub fn try_collect(&self) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();
        queue.drain(..).collect()
    }

    /// Number of items currently waiting in the queue.
    pub fn pending(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Drain items from the queue until deadline or max_batch_size.
    ///
    /// Takes ownership of the lock guard. Returns the collected batch.
    fn drain_until(
        notify: &Condvar,
        mut queue: std::sync::MutexGuard<'_, VecDeque<T>>,
        deadline: Instant,
        max_batch_size: usize,
    ) -> Vec<T> {
        loop {
            if queue.len() >= max_batch_size {
                return queue.drain(..max_batch_size).collect();
            }

            let now = Instant::now();
            if now >= deadline {
                return queue.drain(..).collect();
            }

            let (guard, _) = notify.wait_timeout(queue, deadline - now).unwrap();
            queue = guard;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.window, Duration::from_millis(3));
        assert_eq!(config.max_batch_size, 8);
    }

    #[test]
    fn test_try_collect_empty() {
        let collector = BatchCollector::<u32>::new(BatchConfig::default());
        let batch = collector.try_collect();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_submit_and_try_collect() {
        let collector = BatchCollector::new(BatchConfig::default());
        collector.submit(1u32);
        collector.submit(2);
        collector.submit(3);
        let batch = collector.try_collect();
        assert_eq!(batch, vec![1, 2, 3]);
        assert_eq!(collector.pending(), 0);
    }

    #[test]
    fn test_pending_count() {
        let collector = BatchCollector::new(BatchConfig::default());
        assert_eq!(collector.pending(), 0);
        collector.submit(1u32);
        assert_eq!(collector.pending(), 1);
        collector.submit(2);
        assert_eq!(collector.pending(), 2);
        collector.try_collect();
        assert_eq!(collector.pending(), 0);
    }

    #[test]
    fn test_collect_blocks_until_item() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_millis(1),
            max_batch_size: 8,
        }));

        let c = collector.clone();
        let handle = thread::spawn(move || c.collect());

        // Submit after a short delay
        thread::sleep(Duration::from_millis(20));
        collector.submit(42u32);

        let batch = handle.join().unwrap();
        assert_eq!(batch, vec![42]);
    }

    #[test]
    fn test_collect_gathers_within_window() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_millis(50),
            max_batch_size: 8,
        }));

        let c = collector.clone();
        let handle = thread::spawn(move || c.collect());

        // Submit items over 30ms (within the 50ms window)
        collector.submit(1u32);
        thread::sleep(Duration::from_millis(10));
        collector.submit(2);
        thread::sleep(Duration::from_millis(10));
        collector.submit(3);

        let batch = handle.join().unwrap();
        assert!(batch.len() >= 2, "expected >=2 items, got {}", batch.len());
        assert!(batch.contains(&1));
    }

    #[test]
    fn test_collect_respects_max_batch_size() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_secs(10), // long window
            max_batch_size: 3,
        }));

        // Pre-fill with more than max items
        collector.submit(1u32);
        collector.submit(2);
        collector.submit(3);
        collector.submit(4); // overflow, will be in next batch

        let c = collector.clone();
        let handle = thread::spawn(move || c.collect());

        let batch = handle.join().unwrap();
        assert_eq!(batch.len(), 3, "should cap at max_batch_size");
        assert_eq!(batch, vec![1, 2, 3]);

        // Remaining item is still in queue
        assert_eq!(collector.pending(), 1);
    }

    #[test]
    fn test_collect_timeout_returns_empty_on_timeout() {
        let collector = BatchCollector::<u32>::new(BatchConfig::default());
        let start = Instant::now();
        let batch = collector.collect_timeout(Duration::from_millis(20));
        let elapsed = start.elapsed();
        assert!(batch.is_empty());
        assert!(
            elapsed >= Duration::from_millis(15),
            "should have waited ~20ms, waited {elapsed:?}"
        );
    }

    #[test]
    fn test_collect_timeout_returns_items() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_millis(5),
            max_batch_size: 8,
        }));

        let c = collector.clone();
        let handle = thread::spawn(move || c.collect_timeout(Duration::from_secs(1)));

        thread::sleep(Duration::from_millis(10));
        collector.submit(99u32);

        let batch = handle.join().unwrap();
        assert_eq!(batch, vec![99]);
    }

    #[test]
    fn test_concurrent_submits() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_millis(50),
            max_batch_size: 100,
        }));

        let n_threads = 4;
        let items_per_thread = 10;

        // Concurrent submitters
        let mut submitters = Vec::new();
        for t in 0..n_threads {
            let c = collector.clone();
            submitters.push(thread::spawn(move || {
                for i in 0..items_per_thread {
                    c.submit(t * 100 + i);
                }
            }));
        }

        for s in submitters {
            s.join().unwrap();
        }

        // All items should be in the queue
        let batch = collector.try_collect();
        assert_eq!(
            batch.len(),
            n_threads * items_per_thread,
            "should collect all {} items, got {}",
            n_threads * items_per_thread,
            batch.len()
        );
    }

    #[test]
    fn test_multiple_batches() {
        let collector = BatchCollector::new(BatchConfig {
            window: Duration::from_millis(1),
            max_batch_size: 2,
        });

        collector.submit(1u32);
        collector.submit(2);
        collector.submit(3);
        collector.submit(4);

        let batch1 = collector.try_collect();
        assert_eq!(batch1.len(), 4);

        // Submit more
        collector.submit(5);
        let batch2 = collector.try_collect();
        assert_eq!(batch2, vec![5]);
    }

    #[test]
    fn test_window_timing() {
        let window = Duration::from_millis(20);
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window,
            max_batch_size: 100,
        }));

        let c = collector.clone();
        let start = Instant::now();
        let handle = thread::spawn(move || {
            let batch = c.collect();
            (batch, start.elapsed())
        });

        // Submit first item immediately
        collector.submit(1u32);

        let (batch, elapsed) = handle.join().unwrap();
        assert!(!batch.is_empty());
        // Should return after approximately the window duration
        assert!(
            elapsed >= Duration::from_millis(15),
            "returned too early: {elapsed:?}"
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "returned too late: {elapsed:?}"
        );
    }

    #[test]
    fn test_max_batch_triggers_early_return() {
        let collector = Arc::new(BatchCollector::new(BatchConfig {
            window: Duration::from_secs(10), // very long window
            max_batch_size: 4,
        }));

        let c = collector.clone();
        let start = Instant::now();
        let handle = thread::spawn(move || {
            let batch = c.collect();
            (batch, start.elapsed())
        });

        // Submit max items quickly
        for i in 0..4u32 {
            collector.submit(i);
        }

        let (batch, elapsed) = handle.join().unwrap();
        assert_eq!(batch.len(), 4);
        // Should return much faster than the 10s window
        assert!(
            elapsed < Duration::from_millis(100),
            "max_batch_size should trigger early return, took {elapsed:?}"
        );
    }
}
