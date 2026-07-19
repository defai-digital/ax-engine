//! Per-client request-rate limiting for the HTTP router.
//!
//! Hand-rolled to match the existing house style for load-shedding — see the
//! concurrency semaphore in `routes.rs` — rather than pulling in a new crate
//! for a token bucket. Clients are keyed by bearer token when present, else
//! peer IP from `ConnectInfo`, else a shared `"default"` key (used by unit
//! tests that call the router via `oneshot` without connect info).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Instant;

use axum::extract::ConnectInfo;
use axum::http::{Request, header};
use parking_lot::Mutex;

/// Hard cap on tracked client buckets to bound memory under IP churn.
const MAX_CLIENT_BUCKETS: usize = 4_096;

/// Requests-per-second and burst capacity for the rate limiter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RateLimitConfig {
    pub(crate) rps: f64,
    pub(crate) burst: f64,
}

/// A single token bucket (one client).
struct TokenBucket {
    tokens: f64,
    last: Instant,
}

impl TokenBucket {
    fn new(initial_tokens: f64) -> Self {
        Self {
            tokens: initial_tokens,
            last: Instant::now(),
        }
    }

    /// Attempt to consume one token, lazily refilling based on elapsed time
    /// since the last call. Returns `false` (consuming nothing) when the
    /// bucket is empty.
    fn try_acquire(&mut self, cfg: &RateLimitConfig) -> bool {
        let now = Instant::now();
        let elapsed = now.saturating_duration_since(self.last).as_secs_f64();
        self.tokens = (self.tokens + elapsed * cfg.rps).min(cfg.burst);
        self.last = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Per-client token buckets keyed by API key or peer IP.
pub(crate) struct ClientRateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
    initial_tokens: f64,
}

impl ClientRateLimiter {
    pub(crate) fn new(initial_tokens: f64) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            initial_tokens,
        }
    }

    /// Attempt to acquire one token for the client identified by `request`.
    pub(crate) fn try_acquire<B>(&self, request: &Request<B>, cfg: &RateLimitConfig) -> bool {
        let key = client_key(request);
        let mut buckets = self.buckets.lock();
        if buckets.len() >= MAX_CLIENT_BUCKETS && !buckets.contains_key(&key) {
            // Evict ~25% of entries when the map is full so a single noisy
            // client population cannot grow memory unboundedly. Insertion
            // order is not tracked; HashMap iteration is enough for a
            // bounded, best-effort sweep.
            let evict = (MAX_CLIENT_BUCKETS / 4).max(1);
            let victims: Vec<String> = buckets.keys().take(evict).cloned().collect();
            for victim in victims {
                buckets.remove(&victim);
            }
        }
        let bucket = buckets
            .entry(key)
            .or_insert_with(|| TokenBucket::new(self.initial_tokens));
        bucket.try_acquire(cfg)
    }
}

/// Resolve the rate-limit key for a request.
///
/// Prefer the bearer token (per-API-key fairness when keys are shared across
/// many IPs), then the peer IP from `ConnectInfo`, then a shared default so
/// `oneshot` tests without connect info still share a single bucket.
pub(crate) fn client_key<B>(request: &Request<B>) -> String {
    if let Some(auth) = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
    {
        let token = auth
            .strip_prefix("Bearer ")
            .or_else(|| auth.strip_prefix("bearer "))
            .map(str::trim)
            .filter(|token| !token.is_empty());
        if let Some(token) = token {
            return format!("key:{token}");
        }
    }
    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return format!("ip:{}", addr.ip());
    }
    "default".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;

    fn empty_request() -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap()
    }

    fn request_with_bearer(token: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri("/health")
            .header(header::AUTHORIZATION, format!("Bearer {token}"))
            .body(Body::empty())
            .unwrap()
    }

    fn request_with_peer(ip: &str) -> Request<Body> {
        let addr: SocketAddr = format!("{ip}:12345").parse().unwrap();
        let mut request = empty_request();
        request.extensions_mut().insert(ConnectInfo(addr));
        request
    }

    #[test]
    fn allows_burst_then_exhausts_per_client() {
        let cfg = RateLimitConfig {
            rps: 1.0,
            burst: 2.0,
        };
        let limiter = ClientRateLimiter::new(cfg.burst);
        let request = empty_request();
        assert!(limiter.try_acquire(&request, &cfg));
        assert!(limiter.try_acquire(&request, &cfg));
        assert!(
            !limiter.try_acquire(&request, &cfg),
            "bucket should be exhausted"
        );
    }

    #[test]
    fn separate_clients_have_independent_buckets() {
        let cfg = RateLimitConfig {
            rps: 0.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(cfg.burst);
        let alice = request_with_bearer("alice");
        let bob = request_with_bearer("bob");
        assert!(limiter.try_acquire(&alice, &cfg));
        assert!(
            !limiter.try_acquire(&alice, &cfg),
            "alice should be exhausted"
        );
        assert!(
            limiter.try_acquire(&bob, &cfg),
            "bob must not be starved by alice"
        );
    }

    #[test]
    fn peer_ip_keys_are_independent() {
        let cfg = RateLimitConfig {
            rps: 0.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(cfg.burst);
        let a = request_with_peer("10.0.0.1");
        let b = request_with_peer("10.0.0.2");
        assert!(limiter.try_acquire(&a, &cfg));
        assert!(!limiter.try_acquire(&a, &cfg));
        assert!(limiter.try_acquire(&b, &cfg));
    }

    #[test]
    fn bearer_token_takes_precedence_over_peer_ip() {
        let cfg = RateLimitConfig {
            rps: 0.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(cfg.burst);
        let mut from_a = request_with_bearer("shared");
        from_a
            .extensions_mut()
            .insert(ConnectInfo("10.0.0.1:1".parse::<SocketAddr>().unwrap()));
        let mut from_b = request_with_bearer("shared");
        from_b
            .extensions_mut()
            .insert(ConnectInfo("10.0.0.2:1".parse::<SocketAddr>().unwrap()));
        assert!(limiter.try_acquire(&from_a, &cfg));
        assert!(
            !limiter.try_acquire(&from_b, &cfg),
            "same bearer key must share one bucket across IPs"
        );
    }

    #[test]
    fn starts_empty_when_constructed_with_zero_tokens() {
        let cfg = RateLimitConfig {
            rps: 1000.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(0.0);
        assert!(
            !limiter.try_acquire(&empty_request(), &cfg),
            "should start with no tokens"
        );
    }

    #[test]
    fn refills_over_time() {
        let cfg = RateLimitConfig {
            rps: 1000.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(0.0);
        let request = empty_request();
        assert!(!limiter.try_acquire(&request, &cfg), "starts empty");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            limiter.try_acquire(&request, &cfg),
            "should have refilled within 10ms at 1000rps"
        );
    }

    #[test]
    fn never_refills_past_burst_capacity() {
        let cfg = RateLimitConfig {
            rps: 10_000.0,
            burst: 1.0,
        };
        let limiter = ClientRateLimiter::new(1.0);
        let request = empty_request();
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Even though elapsed time * rps would produce many tokens, the
        // bucket must clamp to `burst` — only one acquire should succeed.
        assert!(limiter.try_acquire(&request, &cfg));
        assert!(
            !limiter.try_acquire(&request, &cfg),
            "must clamp at burst capacity"
        );
    }

    #[test]
    fn client_key_default_without_auth_or_peer() {
        assert_eq!(client_key(&empty_request()), "default");
    }
}
