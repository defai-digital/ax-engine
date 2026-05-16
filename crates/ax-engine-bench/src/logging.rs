use std::env;

use tracing_subscriber::EnvFilter;

pub(crate) fn init_tracing() {
    let filter = env::var("AX_BENCH_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUST_LOG")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });

    let Some(filter) = filter else {
        return;
    };
    let Ok(env_filter) = EnvFilter::try_new(filter) else {
        return;
    };

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_ansi(false)
        .compact()
        .try_init();
}
