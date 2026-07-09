//! Global (not per-client) request-rate limiting for the HTTP router.
//!
//! Hand-rolled to match the existing house style for load-shedding — see the
//! concurrency semaphore in `routes.rs` — rather than pulling in a new crate
//! for a single token bucket. This is a single global bucket, not a
//! per-client/per-IP limiter: the server binds to `127.0.0.1` by default,
//! and per-key limiting would need `ConnectInfo` extraction that isn't wired
//! into the router today.

use std::time::Instant;

use parking_lot::Mutex;

/// Requests-per-second and burst capacity for the global rate limiter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateLimitConfig {
    pub(crate) rps: f64,
    pub(crate) burst: f64,
}

/// A single global token bucket.
pub(crate) struct TokenBucket {
    state: Mutex<(f64, Instant)>,
}

impl TokenBucket {
    pub(crate) fn new(initial_tokens: f64) -> Self {
        Self {
            state: Mutex::new((initial_tokens, Instant::now())),
        }
    }

    /// Attempt to consume one token, lazily refilling based on elapsed time
    /// since the last call. Returns `false` (consuming nothing) when the
    /// bucket is empty.
    pub(crate) fn try_acquire(&self, cfg: &RateLimitConfig) -> bool {
        let mut guard = self.state.lock();
        let (tokens, last) = &mut *guard;
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(*last).as_secs_f64();
        *tokens = (*tokens + elapsed * cfg.rps).min(cfg.burst);
        *last = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_burst_then_exhausts() {
        let cfg = RateLimitConfig {
            rps: 1.0,
            burst: 2.0,
        };
        let bucket = TokenBucket::new(cfg.burst);
        assert!(bucket.try_acquire(&cfg));
        assert!(bucket.try_acquire(&cfg));
        assert!(!bucket.try_acquire(&cfg), "bucket should be exhausted");
    }

    #[test]
    fn starts_empty_when_constructed_with_zero_tokens() {
        let cfg = RateLimitConfig {
            rps: 1000.0,
            burst: 1.0,
        };
        let bucket = TokenBucket::new(0.0);
        assert!(!bucket.try_acquire(&cfg), "should start with no tokens");
    }

    #[test]
    fn refills_over_time() {
        let cfg = RateLimitConfig {
            rps: 1000.0,
            burst: 1.0,
        };
        let bucket = TokenBucket::new(0.0);
        assert!(!bucket.try_acquire(&cfg), "starts empty");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            bucket.try_acquire(&cfg),
            "should have refilled within 10ms at 1000rps"
        );
    }

    #[test]
    fn never_refills_past_burst_capacity() {
        let cfg = RateLimitConfig {
            rps: 10_000.0,
            burst: 1.0,
        };
        let bucket = TokenBucket::new(1.0);
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Even though elapsed time * rps would produce many tokens, the
        // bucket must clamp to `burst` — only one acquire should succeed.
        assert!(bucket.try_acquire(&cfg));
        assert!(!bucket.try_acquire(&cfg), "must clamp at burst capacity");
    }
}
