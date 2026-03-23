use anyhow::{Context, bail};

/// Memory budget gate.
///
/// Uses host total RAM as a conservative upper bound and reserves headroom
/// for KV cache, intermediate buffers, and OS memory pressure.
pub struct MemoryBudget;

impl MemoryBudget {
    // Keep at least 20% system memory free to reduce swap/pressure risk.
    const MAX_FRACTION_OF_TOTAL: f64 = 0.80;

    /// Check if a model of `model_bytes` can fit in available memory.
    pub fn check(model_bytes: u64) -> anyhow::Result<()> {
        if model_bytes == 0 {
            bail!("model_bytes must be > 0");
        }

        if std::env::var("AX_DISABLE_MEMORY_BUDGET_CHECK").is_ok() {
            return Ok(());
        }

        let total = system_total_memory_bytes()?;
        Self::check_with_total(model_bytes, total)
    }

    fn check_with_total(model_bytes: u64, total_bytes: u64) -> anyhow::Result<()> {
        if total_bytes == 0 {
            bail!("system total memory reported as 0 bytes");
        }
        let allowed = (total_bytes as f64 * Self::MAX_FRACTION_OF_TOTAL) as u64;
        if model_bytes > allowed {
            bail!(
                "model requires {:.2} GiB, exceeds budget {:.2} GiB (80% of {:.2} GiB total)",
                bytes_to_gib(model_bytes),
                bytes_to_gib(allowed),
                bytes_to_gib(total_bytes)
            );
        }
        Ok(())
    }
}

fn system_total_memory_bytes() -> anyhow::Result<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .context("failed to execute sysctl -n hw.memsize")?;
    if !out.status.success() {
        bail!("sysctl -n hw.memsize returned non-zero exit status");
    }
    let s = String::from_utf8(out.stdout).context("sysctl output was not valid UTF-8")?;
    let total = s
        .trim()
        .parse::<u64>()
        .context("failed to parse sysctl hw.memsize output")?;
    Ok(total)
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}

#[cfg(test)]
mod tests {
    use super::MemoryBudget;

    #[test]
    fn rejects_zero_model_size() {
        assert!(MemoryBudget::check(0).is_err());
    }

    #[test]
    fn budget_check_allows_small_model() {
        let total = 64 * 1024 * 1024 * 1024_u64;
        let model = 8 * 1024 * 1024 * 1024_u64;
        assert!(MemoryBudget::check_with_total(model, total).is_ok());
    }

    #[test]
    fn budget_check_rejects_large_model() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let model = 15 * 1024 * 1024 * 1024_u64;
        assert!(MemoryBudget::check_with_total(model, total).is_err());
    }
}
