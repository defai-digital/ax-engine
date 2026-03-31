//! Request scheduler with fixed thread pool for decode operations.
//!
//! Provides a fixed-size Rayon thread pool for CPU decode work with:
//! - Configurable thread count (defaults to P-core count)
//! - Thermal-aware concurrency limits
//! - Active job tracking
//!
//! Decode operations (N=1 matvec) are memory-bound and benefit from
//! consistent CPU scheduling on performance cores.

pub mod batch;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::thermal::{ThermalMonitor, ThermalState, ThrottleAction, throttle_action};

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of worker threads for decode.
    /// Default: detected performance core count.
    pub decode_threads: usize,
    /// Enable thermal pressure monitoring and throttle advisory.
    pub thermal_throttle: bool,
    /// Pin decode threads to performance cores via QoS class.
    ///
    /// When true (default), each worker thread is set to
    /// `QOS_CLASS_USER_INTERACTIVE` which hints the macOS scheduler
    /// to run it on a performance core rather than an efficiency core.
    pub pin_to_pcores: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            decode_threads: performance_core_count(),
            thermal_throttle: true,
            pin_to_pcores: true,
        }
    }
}

/// Request scheduler with fixed thread pool.
///
/// Wraps a Rayon `ThreadPool` sized for decode operations. The pool is
/// created once and threads persist for the lifetime of the scheduler
/// (no per-request spawn/join overhead).
///
/// When thermal throttling is enabled, `recommended_concurrency()` returns
/// a reduced thread count under thermal pressure.
pub struct Scheduler {
    pool: rayon::ThreadPool,
    config: SchedulerConfig,
    thermal: Option<Mutex<ThermalMonitor>>,
    active_jobs: Arc<AtomicUsize>,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: SchedulerConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(config.decode_threads > 0, "decode_threads must be > 0");

        let pin = config.pin_to_pcores;
        let mut builder = rayon::ThreadPoolBuilder::new()
            .num_threads(config.decode_threads)
            .thread_name(|i| format!("ax-decode-{i}"));

        if pin {
            builder = builder.start_handler(|_thread_index| {
                pin_current_thread_to_pcore();
            });
        }

        let pool = builder
            .build()
            .map_err(|e| anyhow::anyhow!("failed to create thread pool: {e}"))?;

        let thermal = if config.thermal_throttle {
            Some(Mutex::new(ThermalMonitor::new()))
        } else {
            None
        };

        tracing::info!(
            "Scheduler initialized: {} decode threads, thermal_throttle={}, pin_to_pcores={}",
            config.decode_threads,
            config.thermal_throttle,
            config.pin_to_pcores,
        );

        Ok(Self {
            pool,
            config,
            thermal,
            active_jobs: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Execute a closure synchronously on the thread pool.
    ///
    /// Blocks the caller until the closure completes. The closure runs
    /// on a pool worker thread.
    pub fn execute<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.active_jobs.fetch_add(1, Ordering::Relaxed);
        let _guard = ActiveJobGuard(&self.active_jobs);
        self.pool.install(f)
    }

    /// Spawn a closure on the thread pool (fire-and-forget).
    ///
    /// Returns immediately. Use `execute()` if you need the result.
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let active = self.active_jobs.clone();
        active.fetch_add(1, Ordering::Relaxed);
        self.pool.spawn(move || {
            let _guard = ActiveJobGuard(&active);
            f();
        });
    }

    /// Number of threads in the pool.
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Number of currently active jobs.
    pub fn active_jobs(&self) -> usize {
        self.active_jobs.load(Ordering::Relaxed)
    }

    /// Recommended max concurrency based on current thermal state.
    ///
    /// Returns the pool size under normal conditions. Reduces concurrency
    /// when thermal pressure is elevated:
    /// - Nominal/Fair: full pool size
    /// - Serious: half pool size (min 1)
    /// - Critical: 1 thread (minimum)
    pub fn recommended_concurrency(&self) -> usize {
        let state = match &self.thermal {
            Some(monitor) => monitor.lock().unwrap_or_else(|e| e.into_inner()).poll(),
            None => ThermalState::Nominal,
        };
        concurrency_for_state(self.config.decode_threads, state)
    }

    /// Poll thermal state and return the throttle action.
    ///
    /// Returns `None` if thermal throttling is disabled.
    pub fn poll_thermal(&self) -> Option<ThrottleAction> {
        self.thermal
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()).recommend())
    }

    /// Current thermal state, if monitoring is enabled.
    pub fn thermal_state(&self) -> Option<ThermalState> {
        self.thermal
            .as_ref()
            .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()).last_state())
    }
}

/// Compute recommended concurrency for a thermal state.
fn concurrency_for_state(max_threads: usize, state: ThermalState) -> usize {
    match throttle_action(state) {
        ThrottleAction::None => max_threads,
        ThrottleAction::ReduceBatch => (max_threads / 2).max(1),
        ThrottleAction::CpuOnly => 1,
    }
}

/// Configure rayon's global thread pool with P-core pinning.
///
/// Call once at process startup, before any parallel work. Sets the global
/// pool size to the P-core count and pins each worker thread to performance
/// cores via QoS class.
///
/// Returns the number of threads configured.
pub fn init_global_threadpool() -> usize {
    let n = performance_core_count();
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .start_handler(|_| {
            pin_current_thread_to_pcore();
        })
        .build_global();
    if let Err(e) = result {
        tracing::warn!("Failed to configure rayon global pool: {e}");
    }
    tracing::info!(
        threads = n,
        "Rayon global thread pool initialized (P-core pinned)"
    );
    n
}

/// Detect the number of performance cores on Apple Silicon.
///
/// Uses `sysctlbyname("hw.perflevel0.logicalcpu")` which returns the
/// P-core count on Apple Silicon Macs. Falls back to half of total
/// cores if the sysctl is unavailable.
pub fn performance_core_count() -> usize {
    let count = unsafe { sysctl_perf_cores() };
    if count > 0 {
        count
    } else {
        // Fallback: half of available cores (assume ~50% are P-cores)
        let total = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        (total / 2).max(1)
    }
}

/// Read P-core count via sysctlbyname.
///
/// Returns 0 on failure.
unsafe fn sysctl_perf_cores() -> usize {
    unsafe {
        let mut value: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        let ret = sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            &mut value as *mut i32 as *mut std::ffi::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if ret == 0 && value > 0 {
            value as usize
        } else {
            0
        }
    }
}

unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::ffi::c_int;
}

/// Drop guard that decrements active_jobs counter on drop (including panic unwind).
struct ActiveJobGuard<'a>(&'a AtomicUsize);

impl Drop for ActiveJobGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

// --- QoS class constants (from <sys/qos.h>) ---

/// `QOS_CLASS_USER_INTERACTIVE` — highest priority, scheduled on P-cores.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

/// `QOS_CLASS_DEFAULT` — default QoS, no core preference.
#[cfg(test)]
const QOS_CLASS_DEFAULT: u32 = 0x15;

/// Pin the calling thread to performance cores via QoS class.
///
/// Sets the current thread's QoS to `QOS_CLASS_USER_INTERACTIVE`, which
/// hints the macOS scheduler to run it on a performance core (P-core)
/// rather than an efficiency core (E-core) on Apple Silicon.
///
/// This is a best-effort hint — the kernel may still migrate the thread
/// under extreme load or thermal pressure. Logs a warning on failure.
pub fn pin_current_thread_to_pcore() {
    // SAFETY: pthread_set_qos_class_self_np is a standard POSIX-extension
    // on macOS. It modifies only the calling thread's scheduling attributes.
    let ret = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    if ret != 0 {
        tracing::warn!(
            "failed to set QOS_CLASS_USER_INTERACTIVE on thread {:?}: errno={}",
            std::thread::current().name().unwrap_or("?"),
            ret,
        );
    }
}

/// Read the QoS class of the calling thread.
///
/// Returns `(qos_class, relative_priority)`. Useful for verifying that
/// `pin_current_thread_to_pcore()` took effect.
pub fn current_thread_qos() -> (u32, i32) {
    let mut qos: u32 = 0;
    let mut rel: i32 = 0;
    // SAFETY: reading QoS attributes of the calling thread.
    unsafe {
        pthread_get_qos_class_np(pthread_self(), &mut qos, &mut rel);
    }
    (qos, rel)
}

unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> std::ffi::c_int;
    fn pthread_get_qos_class_np(
        thread: *mut std::ffi::c_void,
        qos_class: *mut u32,
        relative_priority: *mut i32,
    ) -> std::ffi::c_int;
    fn pthread_self() -> *mut std::ffi::c_void;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_config_default() {
        let config = SchedulerConfig::default();
        assert!(config.decode_threads > 0, "should detect at least 1 core");
        assert!(config.thermal_throttle);
        assert!(config.pin_to_pcores);
    }

    #[test]
    fn test_scheduler_creation() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        assert_eq!(sched.num_threads(), 2);
        assert_eq!(sched.active_jobs(), 0);
    }

    #[test]
    fn test_scheduler_creation_with_thermal() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: true,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        assert_eq!(sched.num_threads(), 2);
        assert!(sched.thermal_state().is_some());
    }

    #[test]
    fn test_scheduler_zero_threads_rejected() {
        let config = SchedulerConfig {
            decode_threads: 0,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        assert!(Scheduler::new(config).is_err());
    }

    #[test]
    fn test_scheduler_execute() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        let result = sched.execute(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_scheduler_execute_on_pool_thread() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        let name = sched.execute(|| {
            std::thread::current()
                .name()
                .unwrap_or("unknown")
                .to_string()
        });
        assert!(
            name.starts_with("ax-decode-"),
            "expected pool thread name, got: {name}"
        );
    }

    #[test]
    fn test_scheduler_spawn() {
        use std::sync::Arc;

        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();

        let flag = Arc::new(AtomicUsize::new(0));
        let flag_clone = flag.clone();
        sched.spawn(move || {
            flag_clone.store(1, Ordering::SeqCst);
        });

        // Wait for the spawned task to complete
        for _ in 0..100 {
            if flag.load(Ordering::SeqCst) == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(flag.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_scheduler_active_jobs_tracking() {
        use std::sync::Barrier;

        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = std::sync::Arc::new(Scheduler::new(config).unwrap());
        assert_eq!(sched.active_jobs(), 0);

        // Use a barrier to hold the job active while we check the counter
        let barrier = std::sync::Arc::new(Barrier::new(2));
        let barrier_clone = barrier.clone();
        let sched_clone = sched.clone();

        let handle = std::thread::spawn(move || {
            sched_clone.execute(move || {
                barrier_clone.wait(); // wait for main thread to check
            });
        });

        // Give the execute a moment to start
        std::thread::sleep(std::time::Duration::from_millis(50));
        let active = sched.active_jobs();
        barrier.wait(); // release the job
        handle.join().unwrap();

        assert!(active >= 1, "expected at least 1 active job, got {active}");
        assert_eq!(sched.active_jobs(), 0);
    }

    #[test]
    fn test_concurrency_for_state() {
        assert_eq!(concurrency_for_state(8, ThermalState::Nominal), 8);
        assert_eq!(concurrency_for_state(8, ThermalState::Fair), 8);
        assert_eq!(concurrency_for_state(8, ThermalState::Serious), 4);
        assert_eq!(concurrency_for_state(8, ThermalState::Critical), 1);
    }

    #[test]
    fn test_concurrency_for_state_small_pool() {
        // Even with 1 thread, Serious should return 1 (not 0)
        assert_eq!(concurrency_for_state(1, ThermalState::Serious), 1);
        assert_eq!(concurrency_for_state(1, ThermalState::Critical), 1);
        assert_eq!(concurrency_for_state(2, ThermalState::Serious), 1);
    }

    #[test]
    fn test_recommended_concurrency() {
        let config = SchedulerConfig {
            decode_threads: 4,
            thermal_throttle: true,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        let conc = sched.recommended_concurrency();
        // On a non-thermally-stressed machine, expect full pool size
        assert!((1..=4).contains(&conc));
    }

    #[test]
    fn test_poll_thermal_disabled() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        assert!(sched.poll_thermal().is_none());
        assert!(sched.thermal_state().is_none());
    }

    #[test]
    fn test_poll_thermal_enabled() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: true,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();
        let action = sched.poll_thermal();
        assert!(action.is_some());
    }

    #[test]
    fn test_performance_core_count() {
        let count = performance_core_count();
        assert!(count >= 1, "should detect at least 1 P-core, got {count}");
        // On Apple Silicon, typically 4-16 P-cores
        assert!(count <= 32, "P-core count unexpectedly high: {count}");
    }

    #[test]
    fn test_scheduler_thread_names() {
        let config = SchedulerConfig {
            decode_threads: 3,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();

        // Collect thread names from all 3 workers
        let names: Vec<String> = sched.execute(|| {
            use rayon::prelude::*;
            (0..3)
                .into_par_iter()
                .map(|_| {
                    std::thread::current()
                        .name()
                        .unwrap_or("unknown")
                        .to_string()
                })
                .collect()
        });

        for name in &names {
            assert!(
                name.starts_with("ax-decode-"),
                "unexpected thread name: {name}"
            );
        }
    }

    // --- P-core pinning (QoS) tests ---

    #[test]
    fn test_pin_current_thread_to_pcore() {
        // Pin the current thread and verify the QoS class changed
        pin_current_thread_to_pcore();
        let (qos, _rel) = current_thread_qos();
        assert_eq!(
            qos, QOS_CLASS_USER_INTERACTIVE,
            "expected QOS_CLASS_USER_INTERACTIVE (0x{:02x}), got 0x{:02x}",
            QOS_CLASS_USER_INTERACTIVE, qos,
        );
    }

    #[test]
    fn test_current_thread_qos_returns_valid() {
        let (qos, rel) = current_thread_qos();
        // QoS class should be one of the known values (0x09..0x21)
        assert!(
            (0x09..=0x21).contains(&qos),
            "unexpected QoS class: 0x{qos:02x}"
        );
        // relative_priority should be in [-15, 0]
        assert!(
            (-15..=0).contains(&rel),
            "unexpected relative_priority: {rel}"
        );
    }

    #[test]
    fn test_pin_is_idempotent() {
        pin_current_thread_to_pcore();
        let (qos1, _) = current_thread_qos();
        pin_current_thread_to_pcore();
        let (qos2, _) = current_thread_qos();
        assert_eq!(qos1, qos2, "pinning twice should produce same result");
    }

    #[test]
    fn test_scheduler_with_pinning_creates_pool() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: true,
        };
        let sched = Scheduler::new(config).unwrap();
        assert_eq!(sched.num_threads(), 2);
    }

    #[test]
    fn test_scheduler_pinned_threads_have_interactive_qos() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: true,
        };
        let sched = Scheduler::new(config).unwrap();

        // Execute on a pool thread and check its QoS
        let (qos, _rel) = sched.execute(current_thread_qos);
        assert_eq!(
            qos, QOS_CLASS_USER_INTERACTIVE,
            "pool thread should have QOS_CLASS_USER_INTERACTIVE, got 0x{qos:02x}",
        );
    }

    #[test]
    fn test_scheduler_unpinned_threads_have_default_qos() {
        let config = SchedulerConfig {
            decode_threads: 2,
            thermal_throttle: false,
            pin_to_pcores: false,
        };
        let sched = Scheduler::new(config).unwrap();

        // Without pinning, pool threads should have default QoS
        let (qos, _rel) = sched.execute(current_thread_qos);
        assert_eq!(
            qos, QOS_CLASS_DEFAULT,
            "unpinned pool thread should have QOS_CLASS_DEFAULT (0x{:02x}), got 0x{qos:02x}",
            QOS_CLASS_DEFAULT,
        );
    }

    #[test]
    fn test_scheduler_all_threads_pinned() {
        let config = SchedulerConfig {
            decode_threads: 4,
            thermal_throttle: false,
            pin_to_pcores: true,
        };
        let sched = Scheduler::new(config).unwrap();

        // Verify all 4 threads got pinned
        let qos_classes: Vec<u32> = sched.execute(|| {
            use rayon::prelude::*;
            (0..4)
                .into_par_iter()
                .map(|_| {
                    let (qos, _) = current_thread_qos();
                    qos
                })
                .collect()
        });

        for (i, &qos) in qos_classes.iter().enumerate() {
            assert_eq!(
                qos, QOS_CLASS_USER_INTERACTIVE,
                "thread {i} should be pinned, got QoS 0x{qos:02x}",
            );
        }
    }
}
