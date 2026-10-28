//! Bounded speculative work accounting for n-gram prefetch.
//!
//! Aggregate ceilings bound speculative jobs and staged bytes across all
//! tables through one shared global instance (conservative defaults: 4 jobs
//! and 64 MiB, overridable with `AX_FLASH_NEXT_NGRAM_PREFETCH_JOBS` and
//! `AX_FLASH_NEXT_NGRAM_PREFETCH_BYTES`). Deterministic tests use isolated
//! instances so no test mutates the global ceilings. Each admitted prefetch
//! holds a job guard for its worker lifetime and a byte reservation for its
//! full staging planes; both release via Drop on every exit path including
//! spawn failure, panic, errors, and cancellation. Completed-but-unclaimed
//! staged planes keep their byte reservation until claimed (transferred to
//! demand ownership) or dropped. Demand reads never acquire speculative
//! budget. No reference code is reused here.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Cancellation flag shared between a table and its speculative workers.
/// Workers check between positional reads and exit promptly after finishing
/// any already-started OS read.
#[derive(Debug, Default)]
pub(crate) struct CancelFlag {
    cancelled: AtomicBool,
}

impl CancelFlag {
    pub(crate) fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Context for a single positional row read observed by test hooks.
#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ReadHookContext {
    /// True for speculative prefetch workers, false for synchronous demand.
    pub is_speculative: bool,
    /// Global row id requested (for counting deduplication).
    pub row: u64,
    /// Plane index in staging order (weight, then scales, biases).
    pub plane: usize,
    /// Caller position of the representative read (first occurrence).
    pub position: usize,
}

/// Test hook invoked before each positional row read. Production installs
/// none. Hooks may block (gates), count, or delay deterministically via
/// mutexes and condvars; they must not sleep.
pub(crate) type ReadHook = Arc<dyn Fn(&ReadHookContext) + Send + Sync>;

const DEFAULT_MAX_JOBS: usize = 4;
const DEFAULT_MAX_STAGED_BYTES: usize = 64 * 1024 * 1024;
const ENV_JOBS: &str = "AX_FLASH_NEXT_NGRAM_PREFETCH_JOBS";
const ENV_BYTES: &str = "AX_FLASH_NEXT_NGRAM_PREFETCH_BYTES";
const MAX_JOBS_HARD: usize = 1024;
const MAX_BYTES_HARD: usize = 8 * 1024 * 1024 * 1024;

fn parse_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse::<usize>().ok()
}

/// Aggregate accounting for speculative jobs and staged bytes.
///
/// Production tables share the global instance so ceilings span tables.
/// Tests build isolated instances with [`Self::new`] and never touch the
/// global ceilings, so ceiling tests run in parallel without races.
pub(crate) struct SpecAccounting {
    jobs: AtomicUsize,
    bytes: AtomicUsize,
    max_jobs: AtomicUsize,
    max_bytes: AtomicUsize,
}

impl SpecAccounting {
    fn from_env() -> Self {
        let jobs = parse_env_usize(ENV_JOBS).unwrap_or(DEFAULT_MAX_JOBS);
        let bytes = parse_env_usize(ENV_BYTES).unwrap_or(DEFAULT_MAX_STAGED_BYTES);
        Self::new(jobs, bytes)
    }

    pub(crate) fn new(max_jobs: usize, max_bytes: usize) -> Self {
        Self {
            jobs: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            max_jobs: AtomicUsize::new(max_jobs.min(MAX_JOBS_HARD)),
            max_bytes: AtomicUsize::new(max_bytes.min(MAX_BYTES_HARD)),
        }
    }

    #[cfg(test)]
    pub(crate) fn current_jobs(&self) -> usize {
        self.jobs.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn current_staged_bytes(&self) -> usize {
        self.bytes.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn max_jobs(&self) -> usize {
        self.max_jobs.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub(crate) fn max_staged_bytes(&self) -> usize {
        self.max_bytes.load(Ordering::SeqCst)
    }

    /// Try to admit one speculative job without blocking. Returns None when
    /// the aggregate ceiling is reached; the caller skips the prefetch and
    /// demand reads synchronously.
    pub(crate) fn try_acquire_job(self: &Arc<Self>) -> Option<JobGuard> {
        loop {
            let current = self.jobs.load(Ordering::SeqCst);
            let max = self.max_jobs.load(Ordering::SeqCst);
            if current >= max {
                return None;
            }
            if self
                .jobs
                .compare_exchange_weak(
                    current,
                    current.saturating_add(1),
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                return Some(JobGuard {
                    accounting: Arc::clone(self),
                });
            }
        }
    }

    /// Try to reserve staged bytes without blocking. Returns None when the
    /// aggregate ceiling would be exceeded.
    pub(crate) fn try_reserve_bytes(self: &Arc<Self>, bytes: usize) -> Option<ByteGuard> {
        if bytes == 0 {
            return Some(ByteGuard {
                accounting: Arc::clone(self),
                bytes: 0,
            });
        }
        loop {
            let current = self.bytes.load(Ordering::SeqCst);
            let max = self.max_bytes.load(Ordering::SeqCst);
            let next = current.checked_add(bytes)?;
            if next > max {
                return None;
            }
            if self
                .bytes
                .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Some(ByteGuard {
                    accounting: Arc::clone(self),
                    bytes,
                });
            }
        }
    }
}

static GLOBAL: OnceLock<Arc<SpecAccounting>> = OnceLock::new();

pub(crate) fn global_accounting() -> Arc<SpecAccounting> {
    GLOBAL
        .get_or_init(|| Arc::new(SpecAccounting::from_env()))
        .clone()
}

/// RAII holder for one speculative job slot. Dropping releases the slot on
/// every path including spawn failure (closure dropped), panic (unwinding),
/// errors, and cancellation. Detached workers hold this until they exit so
/// live work stays counted.
pub(crate) struct JobGuard {
    accounting: Arc<SpecAccounting>,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        self.accounting.jobs.fetch_sub(1, Ordering::SeqCst);
    }
}

/// RAII reservation for staged bytes. Dropping releases the bytes.
/// Successful workers move this into their ready result so unclaimed staged
/// planes stay counted until claimed or dropped.
pub(crate) struct ByteGuard {
    accounting: Arc<SpecAccounting>,
    bytes: usize,
}

#[cfg(test)]
impl ByteGuard {
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for ByteGuard {
    fn drop(&mut self) {
        self.accounting
            .bytes
            .fetch_sub(self.bytes, Ordering::SeqCst);
    }
}

/// Completed speculative staging with its byte reservation. Dropping releases
/// the bytes without use (mismatched, replaced, or dropped tables).
pub(crate) struct StagedReady {
    pub planes: Vec<Vec<u8>>,
    pub bytes: ByteGuard,
}

impl StagedReady {
    /// Transfer staged planes to synchronous demand ownership: release the
    /// speculative byte reservation and return the planes for dequantize.
    pub(crate) fn into_planes(self) -> Vec<Vec<u8>> {
        let Self { planes, bytes } = self;
        drop(bytes);
        planes
    }
}

/// Outcome stored by a speculative worker for non-blocking claim.
pub(crate) enum SpecOutcome {
    Ready(StagedReady),
    Failed(String),
    Cancelled,
}

/// Shared slot where a worker publishes its outcome. The table takes the
/// pending work (including this slot) without blocking; the worker holds the
/// other clone and stores exactly once before exiting.
pub(crate) type SpecSlot = Arc<Mutex<Option<SpecOutcome>>>;

pub(crate) fn new_slot() -> SpecSlot {
    Arc::new(Mutex::new(None))
}

/// Reap a speculative thread without ever blocking the caller. Finished
/// threads are joined (observing panic without propagating); still-running
/// threads are detached but remain globally accounted via their guards and
/// must check cancellation promptly after their current OS read.
pub(crate) fn reap_if_finished(handle: std::thread::JoinHandle<()>) {
    if handle.is_finished() {
        let _ = handle.join();
    } else {
        drop(handle);
    }
}
