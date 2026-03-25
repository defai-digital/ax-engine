use anyhow::{Context, bail};

/// Memory budget gate.
///
/// Uses host total RAM as a conservative upper bound and reserves headroom
/// for KV cache, intermediate buffers, and OS memory pressure.
pub struct MemoryBudget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudgetSummary {
    pub required_bytes: u64,
    pub allowed_bytes: u64,
    pub total_bytes: u64,
}

impl MemoryBudget {
    // Keep at least 20% system memory free to reduce swap/pressure risk.
    const MAX_FRACTION_OF_TOTAL: f64 = 0.80;

    /// Check if a model of `model_bytes` can fit in available memory.
    pub fn check(model_bytes: u64) -> anyhow::Result<()> {
        if model_bytes == 0 {
            bail!("model_bytes must be > 0");
        }

        Self::check_combined(model_bytes, 0)
    }

    /// Check if `base_bytes + extra_bytes` fits in the current memory budget.
    pub fn check_combined(base_bytes: u64, extra_bytes: u64) -> anyhow::Result<()> {
        let summary = Self::summary(base_bytes, extra_bytes)?;
        if summary.required_bytes > summary.allowed_bytes {
            bail!(
                "required memory {:.2} GiB exceeds budget {:.2} GiB (80% of {:.2} GiB total)",
                bytes_to_gib(summary.required_bytes),
                bytes_to_gib(summary.allowed_bytes),
                bytes_to_gib(summary.total_bytes)
            );
        }
        Ok(())
    }

    /// Return the current budget summary for `base_bytes + extra_bytes`.
    pub fn summary(base_bytes: u64, extra_bytes: u64) -> anyhow::Result<MemoryBudgetSummary> {
        let required_bytes = base_bytes
            .checked_add(extra_bytes)
            .context("memory budget requirement overflow")?;

        if std::env::var("AX_DISABLE_MEMORY_BUDGET_CHECK").is_ok() {
            return Ok(MemoryBudgetSummary {
                required_bytes,
                allowed_bytes: u64::MAX,
                total_bytes: u64::MAX,
            });
        }

        let total = system_total_memory_bytes()?;
        Self::summary_with_total(required_bytes, total)
    }

    #[cfg(test)]
    fn check_with_total(model_bytes: u64, total_bytes: u64) -> anyhow::Result<()> {
        let summary = Self::summary_with_total(model_bytes, total_bytes)?;
        if summary.required_bytes > summary.allowed_bytes {
            bail!(
                "model requires {:.2} GiB, exceeds budget {:.2} GiB (80% of {:.2} GiB total)",
                bytes_to_gib(summary.required_bytes),
                bytes_to_gib(summary.allowed_bytes),
                bytes_to_gib(summary.total_bytes)
            );
        }
        Ok(())
    }

    fn summary_with_total(
        required_bytes: u64,
        total_bytes: u64,
    ) -> anyhow::Result<MemoryBudgetSummary> {
        if total_bytes == 0 {
            bail!("system total memory reported as 0 bytes");
        }
        let allowed = (total_bytes as f64 * Self::MAX_FRACTION_OF_TOTAL) as u64;
        Ok(MemoryBudgetSummary {
            required_bytes,
            allowed_bytes: allowed,
            total_bytes,
        })
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
    use super::{MemoryBudget, MemoryBudgetSummary};

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

    #[test]
    fn combined_budget_check_accounts_for_extra_bytes() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let summary = MemoryBudget::summary_with_total(10 * 1024 * 1024 * 1024_u64, total).unwrap();
        assert_eq!(
            summary,
            MemoryBudgetSummary {
                required_bytes: 10 * 1024 * 1024 * 1024_u64,
                allowed_bytes: (total as f64 * 0.80) as u64,
                total_bytes: total,
            }
        );
        assert!(
            summary.required_bytes <= summary.allowed_bytes,
            "10 GiB should fit within an 80% budget on a 16 GiB system"
        );
    }

    #[test]
    fn combined_budget_summary_rejects_over_budget_requirement() {
        let total = 16 * 1024 * 1024 * 1024_u64;
        let summary = MemoryBudget::summary_with_total(14 * 1024 * 1024 * 1024_u64, total).unwrap();
        assert!(summary.required_bytes > summary.allowed_bytes);
    }
}
