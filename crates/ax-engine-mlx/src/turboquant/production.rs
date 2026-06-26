#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TurboQuantProductionRequirements {
    pub fused_decode_kernel: bool,
    pub runtime_kv_storage: bool,
    pub runner_route_metadata: bool,
    pub long_context_benchmark_artifact: bool,
    pub public_switch_and_docs: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurboQuantProductionBlocker {
    FusedDecodeKernel,
    RuntimeKvStorage,
    RunnerRouteMetadata,
    LongContextBenchmarkArtifact,
    PublicSwitchAndDocs,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurboQuantProductionReadiness {
    pub ready: bool,
    pub blockers: Vec<TurboQuantProductionBlocker>,
}

impl TurboQuantProductionRequirements {
    pub const fn mlx_shadow_fused_kernel() -> Self {
        Self {
            fused_decode_kernel: true,
            runtime_kv_storage: true,
            runner_route_metadata: true,
            long_context_benchmark_artifact: false,
            public_switch_and_docs: true,
        }
    }

    pub fn evaluate(self) -> TurboQuantProductionReadiness {
        let mut blockers = Vec::new();
        if !self.fused_decode_kernel {
            blockers.push(TurboQuantProductionBlocker::FusedDecodeKernel);
        }
        if !self.runtime_kv_storage {
            blockers.push(TurboQuantProductionBlocker::RuntimeKvStorage);
        }
        if !self.runner_route_metadata {
            blockers.push(TurboQuantProductionBlocker::RunnerRouteMetadata);
        }
        if !self.long_context_benchmark_artifact {
            blockers.push(TurboQuantProductionBlocker::LongContextBenchmarkArtifact);
        }
        if !self.public_switch_and_docs {
            blockers.push(TurboQuantProductionBlocker::PublicSwitchAndDocs);
        }

        TurboQuantProductionReadiness {
            ready: blockers.is_empty(),
            blockers,
        }
    }
}

impl TurboQuantProductionReadiness {
    pub fn is_ready(&self) -> bool {
        self.ready
    }
}
