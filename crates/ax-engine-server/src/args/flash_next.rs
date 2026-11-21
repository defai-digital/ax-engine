use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use ax_engine_sdk::{EngineSessionConfig, MlxMtpPolicy, MlxStreamExpertsMode};

use super::{MAX_CONCURRENT_REQUESTS_ENV, MAX_CONCURRENT_REQUESTS_PER_MODEL_ENV, ServerArgs};

const FLASH_NEXT_TOTAL_BLOCKS: u32 = 2048;

#[derive(Clone, Copy)]
pub(crate) struct ModelServingDefaults {
    base_total_blocks: u32,
    standard_host: bool,
    context_override: bool,
    concurrency_override: bool,
}

impl ServerArgs {
    pub(crate) fn model_serving_defaults(
        &self,
        config: &EngineSessionConfig,
    ) -> ModelServingDefaults {
        self.model_serving_defaults_for_host(
            config.kv_config.total_blocks,
            config.resolved_backend.selected_backend.is_mlx() && is_flash_next_standard_host(),
            std::env::var(MAX_CONCURRENT_REQUESTS_ENV).ok().as_deref(),
            std::env::var(MAX_CONCURRENT_REQUESTS_PER_MODEL_ENV)
                .ok()
                .as_deref(),
        )
    }

    fn model_serving_defaults_for_host(
        &self,
        base_total_blocks: u32,
        standard_host: bool,
        process_env: Option<&str>,
        per_model_env: Option<&str>,
    ) -> ModelServingDefaults {
        ModelServingDefaults {
            base_total_blocks,
            standard_host,
            context_override: self.total_blocks.is_some(),
            concurrency_override: self.max_concurrent_requests.is_some()
                || self.max_concurrent_requests_per_model.is_some()
                || concurrency_setting(process_env).is_some()
                || concurrency_setting(per_model_env).is_some(),
        }
    }
}

impl ModelServingDefaults {
    pub(crate) fn apply(
        self,
        config: &mut EngineSessionConfig,
        process_cap: Option<usize>,
        per_model_cap: Option<usize>,
    ) {
        let standard_pack = self.standard_host
            && config.resolved_backend.selected_backend.is_mlx()
            && config
                .mlx_model_artifacts_dir()
                .is_some_and(is_flash_next_pack);
        self.apply_for_pack(config, standard_pack, process_cap, per_model_cap);
        if standard_pack {
            tracing::info!(
                context_tokens = config.kv_config.total_blocks.saturating_mul(config.kv_config.block_size_tokens),
                effective_sequence_cap = ?config.max_concurrent_requests,
                "resolved the Flash Next M5 Max 128 GB serving profile"
            );
        }
    }

    fn apply_for_pack(
        self,
        config: &mut EngineSessionConfig,
        standard_pack: bool,
        process_cap: Option<usize>,
        per_model_cap: Option<usize>,
    ) {
        let standard_cap = standard_pack && !self.concurrency_override;
        config.max_concurrent_requests = if standard_cap {
            Some(1)
        } else {
            process_cap.into_iter().chain(per_model_cap).min()
        };
        config.kv_config.total_blocks = self.base_total_blocks;
        if standard_pack
            && !self.context_override
            && config.kv_config.block_size_tokens == 16
            && config.max_concurrent_requests == Some(1)
            && config.mlx_mtp_policy != MlxMtpPolicy::Required
            && config.mlx_stream_experts != MlxStreamExpertsMode::On
        {
            config.kv_config.total_blocks = FLASH_NEXT_TOTAL_BLOCKS;
        }
    }
}

fn concurrency_setting(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|value| value.trim().parse().ok())
}

fn is_flash_next_standard_host() -> bool {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return false;
    }
    Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.model", "hw.memsize", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .is_some_and(|output| is_standard_host_description(&output))
}

fn is_standard_host_description(output: &str) -> bool {
    output.lines().map(str::trim).eq([
        "Mac17,6",
        "137438953472", // 128 GiB
        "Apple M5 Max",
    ])
}

fn is_flash_next_pack(root: &Path) -> bool {
    // Match the published metadata at ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3.
    // Both an ordinary Hub snapshot and a staged pack preserve these files.
    metadata_matches(
        &root.join("config.json"),
        250_621,
        "fb3c637ce4a240453c454fbc1211fa1e893d0af12df54a9df19c9c017fa59237",
    ) && metadata_matches(
        &root.join("model.safetensors.index.json"),
        290_934,
        "37344288777e7e78b35901cb1d572dad0dc640cb9cedb90024860b932a657696",
    )
}

fn metadata_matches(path: &Path, expected_bytes: u64, expected_sha256: &str) -> bool {
    if !path
        .metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == expected_bytes)
    {
        return false;
    }
    let Ok(file) = File::open(path) else {
        return false;
    };
    let mut bytes = Vec::new();
    file.take(expected_bytes + 1)
        .read_to_end(&mut bytes)
        .is_ok()
        && ax_engine_core::sha256_hex(&bytes) == expected_sha256
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn standard_profile_aligns_the_context_and_slot_budget() -> Result<(), String> {
        let args =
            ServerArgs::try_parse_from(["ax-engine-server"]).map_err(|error| error.to_string())?;
        let mut config = EngineSessionConfig::default();
        let defaults = args.model_serving_defaults_for_host(1024, true, None, Some(" "));
        defaults.apply_for_pack(&mut config, true, None, None);
        assert_eq!(config.max_concurrent_requests, Some(1));
        assert_eq!(config.kv_config.total_blocks, FLASH_NEXT_TOTAL_BLOCKS);
        assert_eq!(config.mlx_mtp_policy, MlxMtpPolicy::Auto);
        assert_eq!(config.mlx_stream_experts, MlxStreamExpertsMode::Auto);
        Ok(())
    }

    #[test]
    fn explicit_block_count_stays_at_its_cli_value() -> Result<(), String> {
        let args = ServerArgs::try_parse_from(["ax-engine-server", "--total-blocks=1024"])
            .map_err(|error| error.to_string())?;
        let mut config = args.session_config()?;
        let defaults = args.model_serving_defaults_for_host(1024, true, None, None);
        defaults.apply_for_pack(&mut config, true, None, None);
        assert_eq!(config.max_concurrent_requests, Some(1));
        assert_eq!(config.kv_config.total_blocks, 1024);
        Ok(())
    }

    #[test]
    fn explicit_caps_take_precedence_and_context_expands_for_a_single_slot() -> Result<(), String> {
        for (flag, process_env, process_cap, per_model_cap, expected_cap, blocks) in [
            (
                "--max-concurrent-requests=0",
                Some("1"),
                None,
                None,
                None,
                1024,
            ),
            (
                "--max-concurrent-requests=1",
                None,
                Some(1),
                None,
                Some(1),
                FLASH_NEXT_TOTAL_BLOCKS,
            ),
            (
                "--max-concurrent-requests=2",
                None,
                Some(2),
                Some(1),
                Some(1),
                FLASH_NEXT_TOTAL_BLOCKS,
            ),
            (
                "--max-concurrent-requests-per-model=1",
                None,
                None,
                Some(1),
                Some(1),
                FLASH_NEXT_TOTAL_BLOCKS,
            ),
            ("--mlx", Some("0"), None, None, None, 1024),
            (
                "--mlx",
                Some(" 1 "),
                Some(1),
                None,
                Some(1),
                FLASH_NEXT_TOTAL_BLOCKS,
            ),
        ] {
            let args = ServerArgs::try_parse_from(["ax-engine-server", flag])
                .map_err(|error| error.to_string())?;
            let mut config = args.session_config()?;
            let defaults = args.model_serving_defaults_for_host(1024, true, process_env, None);
            defaults.apply_for_pack(&mut config, true, process_cap, per_model_cap);
            assert_eq!(config.max_concurrent_requests, expected_cap, "{flag}");
            assert_eq!(config.kv_config.total_blocks, blocks, "{flag}");
        }
        Ok(())
    }

    #[test]
    fn expanded_context_requires_product_geometry_and_direct_resident_policy() -> Result<(), String>
    {
        let args =
            ServerArgs::try_parse_from(["ax-engine-server"]).map_err(|error| error.to_string())?;
        let defaults = args.model_serving_defaults_for_host(1024, true, None, None);
        let mut custom_geometry = EngineSessionConfig::default();
        custom_geometry.kv_config.block_size_tokens = 32;
        for mut config in [
            EngineSessionConfig {
                mlx_mtp_policy: MlxMtpPolicy::Required,
                ..EngineSessionConfig::default()
            },
            EngineSessionConfig {
                mlx_stream_experts: MlxStreamExpertsMode::On,
                ..EngineSessionConfig::default()
            },
            custom_geometry,
        ] {
            defaults.apply_for_pack(&mut config, true, None, None);
            assert_eq!(config.max_concurrent_requests, Some(1));
            assert_eq!(config.kv_config.total_blocks, 1024);
        }
        Ok(())
    }

    #[test]
    fn changing_packs_reapplies_only_the_selected_models_defaults() -> Result<(), String> {
        let args =
            ServerArgs::try_parse_from(["ax-engine-server"]).map_err(|error| error.to_string())?;
        let defaults = args.model_serving_defaults_for_host(1024, true, None, None);
        let mut config = EngineSessionConfig::default();
        for (standard_pack, blocks, cap) in [
            (true, FLASH_NEXT_TOTAL_BLOCKS, Some(1)),
            (false, 1024, None),
            (true, FLASH_NEXT_TOTAL_BLOCKS, Some(1)),
        ] {
            defaults.apply_for_pack(&mut config, standard_pack, None, None);
            assert_eq!(config.kv_config.total_blocks, blocks);
            assert_eq!(config.max_concurrent_requests, cap);
        }
        Ok(())
    }

    #[test]
    fn only_the_qualified_hardware_description_selects_the_profile() {
        assert!(is_standard_host_description(
            "Mac17,6\n137438953472\nApple M5 Max\n"
        ));
        assert!(!is_standard_host_description(
            "Mac17,6\n68719476736\nApple M5 Max\n"
        ));
        assert!(!is_standard_host_description(
            "Mac16,11\n68719476736\nApple M4 Pro\n"
        ));
        assert!(!is_standard_host_description("Mac17,6\n137438953472\n"));
    }
}
