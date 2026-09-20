// SPDX-License-Identifier: Apache-2.0

//! Bounded first-model admission for the audited Tiel M4 Pro session path.
//! Unknown probes retain legacy Auto. This is neither multi-model admission
//! nor a hard allocator/free-memory guarantee.

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct ResidentBudgetInputs {
    pub audited_export: bool,
    pub required: bool,
    pub cpu_brand: Option<String>,
    pub physical_bytes: Option<u64>,
    pub pressure_level: Option<u32>,
    pub working_set_bytes: Option<u64>,
    pub active_bytes: Option<u64>,
    pub kv_pool_tokens: u64,
    pub prefill_chunk: usize,
    pub footprint_bytes: Option<u64>,
}

pub(crate) fn permits_resident_load(inputs: &ResidentBudgetInputs) -> bool {
    if !inputs.audited_export
        || inputs.required
        || inputs.cpu_brand.as_deref() != Some("Apple M4 Pro")
        || inputs.physical_bytes != Some(64 * GIB)
        || inputs.pressure_level != Some(1)
        || !(1..=16_384).contains(&inputs.kv_pool_tokens)
        || !(1..=2_048).contains(&inputs.prefill_chunk)
    {
        return false;
    }
    let (Some(working_set), Some(active), Some(footprint)) = (
        inputs.working_set_bytes,
        inputs.active_bytes,
        inputs.footprint_bytes,
    ) else {
        return false;
    };
    if working_set == 0 || active > GIB / 2 || footprint == 0 {
        return false;
    }
    // Preserve at least 16 GiB outside this model budget. Metal's recommendation
    // may be lower; it is not a measurement of currently free physical memory.
    let budget = working_set.min(48 * GIB);
    footprint
        .checked_add(active)
        .is_some_and(|projected| projected <= budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admitted() -> ResidentBudgetInputs {
        ResidentBudgetInputs {
            audited_export: true,
            required: false,
            cpu_brand: Some("Apple M4 Pro".to_string()),
            physical_bytes: Some(64 * GIB),
            pressure_level: Some(1),
            working_set_bytes: Some(48 * GIB),
            active_bytes: Some(0),
            kv_pool_tokens: 16_384,
            prefill_chunk: 2_048,
            footprint_bytes: Some(24 * GIB),
        }
    }

    #[test]
    fn bounded_default_session_fits() {
        assert!(permits_resident_load(&admitted()));
        let mut inputs = admitted();
        inputs.active_bytes = Some(GIB / 2);
        assert!(permits_resident_load(&inputs));
    }

    #[test]
    fn only_known_exact_host_and_optional_audited_export() {
        let base = admitted();
        let mut cases = Vec::new();
        let mut changed = base.clone();
        changed.audited_export = false;
        cases.push(changed);
        let mut changed = base.clone();
        changed.required = true;
        cases.push(changed);
        for brand in [
            None,
            Some("Apple M4"),
            Some("Apple M3 Ultra"),
            Some("Apple M5 Max"),
        ] {
            let mut changed = base.clone();
            changed.cpu_brand = brand.map(str::to_string);
            cases.push(changed);
        }
        for physical in [None, Some(48 * GIB), Some(64 * GIB - 1), Some(128 * GIB)] {
            let mut changed = base.clone();
            changed.physical_bytes = physical;
            cases.push(changed);
        }
        for changed in cases {
            assert!(!permits_resident_load(&changed), "{changed:?}");
        }
    }

    #[test]
    fn unknown_or_pressured_allocator_never_relaxes_auto() {
        for pressure in [None, Some(0), Some(2), Some(4)] {
            let mut inputs = admitted();
            inputs.pressure_level = pressure;
            assert!(!permits_resident_load(&inputs));
        }
        for active in [None, Some(GIB / 2 + 1), Some(24 * GIB)] {
            let mut inputs = admitted();
            inputs.active_bytes = active;
            assert!(!permits_resident_load(&inputs));
        }
        for working_set in [None, Some(0), Some(24 * GIB - 1)] {
            let mut inputs = admitted();
            inputs.working_set_bytes = working_set;
            assert!(!permits_resident_load(&inputs));
        }
        for footprint in [None, Some(0), Some(u64::MAX)] {
            let mut inputs = admitted();
            inputs.footprint_bytes = footprint;
            assert!(!permits_resident_load(&inputs));
        }
    }

    #[test]
    fn session_bounds_cannot_be_silently_expanded() {
        for pool in [0, 16_385, u64::MAX] {
            let mut inputs = admitted();
            inputs.kv_pool_tokens = pool;
            assert!(!permits_resident_load(&inputs));
        }
        for chunk in [0, 2_049, usize::MAX] {
            let mut inputs = admitted();
            inputs.prefill_chunk = chunk;
            assert!(!permits_resident_load(&inputs));
        }
    }

    #[test]
    fn lower_metal_budget_and_physical_margin_both_bind() {
        for working_set in [32 * GIB, 48 * GIB, 64 * GIB] {
            let mut inputs = admitted();
            inputs.working_set_bytes = Some(working_set);
            inputs.active_bytes = Some(GIB / 2);
            let exact = working_set.min(48 * GIB) - GIB / 2;
            inputs.footprint_bytes = Some(exact);
            assert!(permits_resident_load(&inputs));
            inputs.footprint_bytes = Some(exact + 1);
            assert!(!permits_resident_load(&inputs));
        }
        let mut inputs = admitted();
        inputs.active_bytes = Some(1);
        inputs.footprint_bytes = Some(u64::MAX);
        assert!(!permits_resident_load(&inputs));
    }
}
