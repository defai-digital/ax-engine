use serde_json::{Value, json};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

#[path = "../tui/mod.rs"]
mod tui;

#[derive(Clone, Copy)]
struct ModelProfile {
    label: &'static str,
    preset: Option<&'static str>,
    repo_id: &'static str,
    aliases: &'static [&'static str],
    downloadable: bool,
    /// Total repo download size summed from Hugging Face file metadata.
    /// A point-in-time estimate for previews and progress totals,
    /// not a contract — repos can republish with different shard sizes.
    approx_size_bytes: Option<u64>,
}

impl ModelProfile {
    /// The download catalog (`download --list`, download options payloads)
    /// intentionally surfaces only the curated AutomatosX Hub organization.
    /// Downloads themselves accept any `downloadable` profile's repo id as
    /// well as explicit Hugging Face repo ids (see `download_repo_id`).
    fn is_downloadable(self) -> bool {
        self.downloadable && self.repo_id.starts_with("AutomatosX/")
    }
}

fn profile_revision(profile: ModelProfile) -> Option<&'static str> {
    match profile.repo_id {
        "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP" => {
            Some("6182ccbc41c7397ff90670f740c6d9eacfa4b09f")
        }
        "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP" => {
            Some("8c37715c7b5f5ebca00eda6f73be47116a3e4ebc")
        }
        "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit" => {
            Some("e932be1b8ab79f5410f607de7eb7312756325fce")
        }
        "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit" => {
            Some("b48b626d9b00e45d6200aa3c15e40cc47d83b7e7")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP" => {
            Some("7e865596cb32bd41b29c7a25c5b66b9c3ea25e5e")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP" => {
            Some("3e290738e96972307c6aeb9934ab170ca0eae1c1")
        }
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP" => {
            Some("952031cbfbb9cf31414a57eeb681c34dc08ec1e9")
        }
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP" => {
            Some("6a4c220734f81112555ee8783d91e0065c54301c")
        }
        "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP" => {
            Some("d2a6ac9d59655f0b86a57a64ed85616d0a10e27e")
        }
        "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP" => {
            Some("7ad79df2b0c272431f3e927b133b7dc3d70872f4")
        }
        "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP" => {
            Some("490b1183ce4505e79334423547422204fb9144d0")
        }
        "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP" => {
            Some("940a60b13e7298140c85d3762492dde6733f8a57")
        }
        "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP" => {
            Some("fdd851347f487c565b067c0593fdb5ac7a3057a2")
        }
        "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP" => {
            Some("7b11bd5179d71a74200fe56075cba5c21212fe6a")
        }
        "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit" => {
            Some("bcfb0b748fc44487c1657fb6ae190592d515398b")
        }
        "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit" => {
            Some("367745bd05b77bf82188f3799677e4beba543e8d")
        }
        "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit" => {
            Some("323a48f2a821f7d0349466095b1b84562d11c9a0")
        }
        "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit" => {
            Some("e52d06296bf133b248a6572561c4f2e150dc3429")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit" => {
            Some("6df63e00b1fa952bffd3b4ad5ecd182f9d48a8a4")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit" => {
            Some("1a54b325bef89b056f8ee9a882452419cceb018e")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit" => {
            Some("36f9d25c4b1ea2282774b9acf84fdad0241a8a54")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP" => {
            Some("4037b7242a4de8deaf71247a685538591cad160a")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4" => {
            Some("4797708af95b9d5cca343d0a4671511fc2765e1a")
        }
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP" => {
            Some("b2c5354f779e430d0c1733143db848a72b71c16e")
        }
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit" => {
            Some("a524f97c81ec82be3eead17aabcf652450d33842")
        }
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit" => {
            Some("29e7bcf5e6ef2471cc3587783713e3631e98b50c")
        }
        "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit" => {
            Some("20f2d2bd0b1055f8ab990e82fa0fc784a9de4c89")
        }
        "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit" => {
            Some("14aee3b601240c5075fc4c84fb6f088400aeeba5")
        }
        "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit" => {
            Some("306f5a9858cadd8e0a6b01201d37ad2d24ddcdd7")
        }
        "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit" => {
            Some("7b2256130cd55ea6b7489817a9a00c46e9874403")
        }
        "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit" => {
            Some("e6cc340b04bfcec57544e462ec756e48dd248cf9")
        }
        "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit" => {
            Some("9ff7a33b034a7e72cdc32a531ed8dd0d07e35116")
        }
        "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit" => {
            Some("41015da430ae62802d9357b0ef31bf46c2b13b58")
        }
        "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit" => {
            Some("93d9991a3636c6c46cb92e711d11f1be5de96b6a")
        }
        "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit" => {
            Some("669dda7a7d78e2fa167d6dae70128f8cf2fe778b")
        }
        "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit" => {
            Some("74cc761a1f6f3e2d0e8bbb4d3d8c15cd17ef221a")
        }
        "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit" => {
            Some("91c20bd52f6c16b6b7e6f6e60b0a859ddd1ad8b0")
        }
        "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit" => {
            Some("f00654783b3e3b2a020a712161eb1ac7861da348")
        }
        "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit" => {
            Some("cb2db117e80571afa466644e91ec39bd528ccf7f")
        }
        "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit" => {
            Some("a4dcc84b9b7318cc206f2b17dbc1555883cf67fd")
        }
        "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit" => {
            Some("e021a6ed572d6d2a99fad028707f09a6b524d7f2")
        }
        "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit" => {
            Some("5608f0c197a7ffcd3366894cce7eb9918b24c8c1")
        }
        "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit" => {
            Some("17e0ce81a7d6aeb6729a0c84b92340e26fbe1a6d")
        }
        "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit" => {
            Some("04be51a3173b94e0a0d859be871cfb7a749405d2")
        }
        "AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-MXFP8" => {
            Some("4d928dce639633f1138113d733dd11c120da87c9")
        }
        "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit" => {
            Some("1c3fb2a006883d88ee0b84a831b480e4a9dc97c6")
        }
        "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit" => {
            Some("d6de0453b22af8bbbcfebbd43326ccea6ed35e64")
        }
        _ => None,
    }
}

fn profile_certification(profile: ModelProfile) -> Option<&'static str> {
    match profile.repo_id {
        "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP"
        | "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit"
        | "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit"
        | "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit"
        | "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit"
        | "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit"
        | "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP"
        | "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP"
        | "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP"
        | "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP"
        | "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP"
        | "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit"
        | "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit"
        | "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit"
        | "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit"
        | "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit"
        | "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit"
        | "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit"
        | "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit"
        | "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit"
        | "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit"
        | "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit"
        | "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit"
        | "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit" => Some("candidate"),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MtpDownloadKind {
    QwenSidecar {
        mtp_source: &'static str,
    },
    GemmaAssistant {
        assistant_repo_id: &'static str,
        target_model_id: &'static str,
        assistant_model_id: &'static str,
        max_depth: u32,
    },
    /// Fallback for models where no MTP sidecar or assistant packager is available.
    #[allow(dead_code)]
    DirectOnly {
        reason: &'static str,
    },
}

#[derive(Clone, Copy, Debug)]
struct MtpDownloadTarget {
    label: &'static str,
    repo_id: &'static str,
    aliases: &'static [&'static str],
    kind: MtpDownloadKind,
}

const MODEL_PROFILES: &[ModelProfile] = &[
    ModelProfile {
        label: "gemma4-12b",
        preset: Some("gemma4-12b"),
        repo_id: "mlx-community/gemma-4-12B-it-4bit",
        aliases: &[
            "gemma4-12b",
            "gemma-4-12b",
            "gemma-4-12b-it",
            "gemma4-12b-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(6_773_372_848),
    },
    ModelProfile {
        label: "gemma4-12b-6bit",
        preset: None,
        repo_id: "mlx-community/gemma-4-12B-it-6bit",
        aliases: &["gemma4-12b-6bit", "gemma-4-12b-6bit", "gemma-4-12b-it-6bit"],
        downloadable: true,
        approx_size_bytes: Some(9_760_954_674),
    },
    ModelProfile {
        label: "gemma4-26b",
        preset: Some("gemma4-26b"),
        repo_id: "mlx-community/gemma-4-26b-a4b-it-4bit",
        aliases: &[
            "gemma4-26b",
            "gemma-4-26b",
            "gemma-4-26b-a4b-it",
            "gemma4-26b-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(15_373_588_575),
    },
    ModelProfile {
        label: "gemma4-31b",
        preset: Some("gemma4-31b"),
        repo_id: "mlx-community/gemma-4-31b-it-4bit",
        aliases: &[
            "gemma4-31b",
            "gemma-4-31b",
            "gemma-4-31b-it",
            "gemma4-31b-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(18_444_421_751),
    },
    ModelProfile {
        label: "glm4.7-flash-4bit",
        preset: Some("glm4.7-flash-4bit"),
        repo_id: "mlx-community/GLM-4.7-Flash-4bit",
        aliases: &[
            "glm4.7-flash-4bit",
            "glm47-flash-4bit",
            "glm4-moe-lite",
            "glm4_moe_lite",
            "glm-4.7-flash-4bit",
            "glm-4-7-flash-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(16_872_850_407),
    },
    ModelProfile {
        label: "qwen3.5-9b",
        preset: Some("qwen3.5-9b"),
        repo_id: "mlx-community/Qwen3.5-9B-MLX-4bit",
        aliases: &[
            "qwen3.5-9b",
            "qwen35-9b",
            "qwen3-5-9b",
            "qwen3.5-9b-4bit",
            "qwen3-5-9b-mlx-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(5_977_074_591),
    },
    ModelProfile {
        label: "qwen3.6-27b",
        preset: Some("qwen3.6-27b"),
        repo_id: "mlx-community/Qwen3.6-27B-4bit",
        aliases: &[
            "qwen3.6-27b",
            "qwen36-27b",
            "qwen3-6-27b",
            "qwen3.6-27b-4bit",
            "qwen36-27b-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(16_081_490_064),
    },
    ModelProfile {
        label: "qwen3.6-27b-5bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-5bit",
        aliases: &["qwen3.6-27b-5bit", "qwen36-27b-5bit", "qwen3-6-27b-5bit"],
        downloadable: true,
        approx_size_bytes: Some(19_443_159_244),
    },
    ModelProfile {
        label: "qwen3.6-27b-6bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-6bit",
        aliases: &["qwen3.6-27b-6bit", "qwen36-27b-6bit", "qwen3-6-27b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(22_804_828_230),
    },
    ModelProfile {
        label: "qwen3.6-27b-8bit",
        preset: None,
        repo_id: "mlx-community/Qwen3.6-27B-8bit",
        aliases: &["qwen3.6-27b-8bit", "qwen36-27b-8bit", "qwen3-6-27b-8bit"],
        downloadable: true,
        approx_size_bytes: Some(29_528_166_726),
    },
    ModelProfile {
        label: "qwen3.6-35b",
        preset: Some("qwen3.6-35b"),
        repo_id: "mlx-community/Qwen3.6-35B-A3B-4bit",
        aliases: &[
            "qwen3.6-35b",
            "qwen36-35b",
            "qwen3-6-35b",
            "qwen3.6-35b-a3b",
            "qwen36-35b-a3b",
        ],
        downloadable: true,
        approx_size_bytes: Some(20_429_169_263),
    },
    // --- AutomatosX packs (https://huggingface.co/AutomatosX) ---
    // AX-branded builds of the primary product families. Chat packs bundle
    // their speculative-decode extras in one repo (Qwen: `mtp.safetensors`
    // sidecar; Gemma: `assistant/` weights + `ax_gemma4_assistant_mtp.json`
    // contract) plus a pre-generated `model-manifest.json`, so a single
    // `ax-engine download <alias>` produces a serve-ready MTP directory —
    // no separate `download-mtp` step. `preset` stays `None` on purpose:
    // these aliases promise the exact AutomatosX repo, so serve resolves
    // through exact Hub snapshot resolution instead of loose hf-cache preset
    // matching that could pick a different org's snapshot. Sizes: summed HF
    // API file metadata, 2026-07-20.
    ModelProfile {
        label: "ax-qwen3.5-9b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.5-9b",
            "ax-qwen35-9b",
            "ax-qwen3.5-9b-optiq-4bit",
            "ax-qwen3.5-9b-mlx-optiq-4bit-mtp",
        ],
        downloadable: true,
        approx_size_bytes: Some(8_355_172_323),
    },
    ModelProfile {
        label: "ax-qwen3.5-9b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.5-9B-MLX-4bit-MTP",
        aliases: &["ax-qwen3.5-9b-4bit", "ax-qwen35-9b-4bit"],
        downloadable: true,
        approx_size_bytes: Some(6_463_848_363),
    },
    ModelProfile {
        label: "ax-qwen3.5-9b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.5-9B-MLX-6bit-MTP",
        aliases: &["ax-qwen3.5-9b-6bit", "ax-qwen35-9b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(8_702_033_839),
    },
    ModelProfile {
        label: "ax-qwen3.6-27b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.6-27b",
            "ax-qwen36-27b",
            "ax-qwen3.6-27b-optiq-4bit",
            "ax-qwen3.6-27b-mlx-optiq-4bit-mtp",
            "qwen3.6-27b:optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(20_239_552_902),
    },
    ModelProfile {
        label: "ax-qwen3.6-27b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-27B-MLX-4bit-MTP",
        aliases: &[
            "ax-qwen3.6-27b-4bit",
            "ax-qwen36-27b-4bit",
            "qwen3.6-27b:uniform-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(16_931_255_394),
    },
    ModelProfile {
        label: "ax-qwen3.6-27b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
        aliases: &[
            "ax-qwen3.6-27b-6bit",
            "ax-qwen36-27b-6bit",
            "qwen3.6-27b:uniform-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(23_654_593_521),
    },
    // AXQ candidates are explicit and pinned. The bare qwen3.6-27b alias must
    // not point here until checkpoint-level quality/runtime/memory gates pass.
    ModelProfile {
        label: "ax-qwen3.6-27b-axq-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-qwen3.6-27b-axq-6bit",
            "ax-qwen3.6-27b-axq",
            "ax-qwen36-27b-axq-6bit",
            "qwen3.6-27b:axq",
            "qwen3.6-27b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(20_857_941_725),
    },
    ModelProfile {
        label: "ax-qwen3.6-27b-axq-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.6-27b-axq-4bit",
            "ax-qwen36-27b-axq-4bit",
            "qwen3.6-27b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(19_399_395_845),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-qwen3.8-27b-axq-6bit",
            "ax-qwen3.8-27b-axq",
            "ax-qwen38-27b-axq-6bit",
            "qwen3.8-27b:axq",
            "qwen3.8-27b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(20_856_327_059),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.8-27b-axq-4bit",
            "ax-qwen38-27b-axq-4bit",
            "qwen3.8-27b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(25_080_915_804),
    },
    ModelProfile {
        label: "ax-qwen3.6-35b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.6-35b",
            "ax-qwen36-35b",
            "ax-qwen3.6-35b-a3b",
            "ax-qwen3.6-35b-optiq-4bit",
            "ax-qwen3.6-35b-a3b-mlx-optiq-4bit-mtp",
        ],
        downloadable: true,
        approx_size_bytes: Some(26_327_314_611),
    },
    ModelProfile {
        label: "ax-qwen3.6-35b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-4bit-MTP",
        aliases: &["ax-qwen3.6-35b-4bit", "ax-qwen36-35b-4bit"],
        downloadable: true,
        approx_size_bytes: Some(22_118_783_082),
    },
    ModelProfile {
        label: "ax-qwen3.6-35b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
        aliases: &["ax-qwen3.6-35b-6bit", "ax-qwen36-35b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(30_778_381_769),
    },
    ModelProfile {
        label: "ax-qwen3.6-35b-axq-6bit",
        preset: Some("qwen3.6-35b"),
        repo_id: "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-qwen3.6-35b-axq-6bit",
            "ax-qwen3.6-35b-axq",
            "ax-qwen36-35b-axq-6bit",
            "qwen3.6-35b:axq",
            "qwen3.6-35b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(51_365_810_695),
    },
    ModelProfile {
        label: "ax-qwen3.6-35b-axq-4bit",
        preset: Some("qwen3.6-35b"),
        repo_id: "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP",
        aliases: &[
            "ax-qwen3.6-35b-axq-4bit",
            "ax-qwen36-35b-axq-4bit",
            "qwen3.6-35b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(43_636_169_195),
    },
    // Qwen3-VL 30B-A3B Instruct AXQ packs: vision MoE (`qwen3_vl_moe`), no MTP.
    // Explicit candidates — development evidence packs, not Tier-1 certified.
    ModelProfile {
        label: "ax-qwen3-vl-30b-a3b-axq-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3-vl-30b-a3b-axq-6bit",
            "ax-qwen3-vl-30b-a3b-axq",
            "ax-qwen3-vl-30b-axq",
            "ax-qwen3-vl-30b-axq-6bit",
            "ax-qwen3-vl-30b-6bit",
            "ax-qwen3-vl-30b",
            "ax-qwen3-vl-30b-a3b",
            "qwen3-vl-30b-a3b:axq",
            "qwen3-vl-30b-a3b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(23_318_717_145),
    },
    ModelProfile {
        label: "ax-qwen3-vl-30b-a3b-axq-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3-vl-30b-a3b-axq-4bit",
            "ax-qwen3-vl-30b-axq-4bit",
            "ax-qwen3-vl-30b-4bit",
            "ax-qwen3-vl-30b-a3b-4bit",
            "qwen3-vl-30b-a3b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(18_891_118_762),
    },
    // Muse-Glimmer 30B AXQ: dense image-text agent, no MTP, development only.
    ModelProfile {
        label: "ax-muse-glimmer-30b-6bit",
        preset: Some("muse-glimmer-30b"),
        repo_id: "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit",
        aliases: &[
            "ax-muse-glimmer-30b-6bit",
            "ax-muse-glimmer-6bit",
            "ax-muse-glimmer-30b-axq-6bit",
            "ax-muse-glimmer-30b-axq",
            "muse-glimmer-30b:axq",
            "muse-glimmer-30b:axq-6bit",
            "muse-glimmer:axq",
            "muse-glimmer:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(30_708_269_056),
    },
    ModelProfile {
        label: "ax-muse-glimmer-30b",
        preset: Some("muse-glimmer-30b"),
        repo_id: "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit",
        aliases: &[
            "ax-muse-glimmer-30b",
            "ax-muse-glimmer",
            "ax-glimmer-30b",
            "ax-muse-glimmer-30b-4bit",
            "ax-muse-glimmer-4bit",
            "ax-muse-glimmer-30b-axq-4bit",
            "muse-glimmer-30b",
            "muse-glimmer",
            "muse-glimmer-30b-4bit",
            "muse-glimmer-30b:axq-4bit",
            "muse-glimmer:axq-4bit",
            "glimmer-30b",
        ],
        downloadable: true,
        approx_size_bytes: Some(22_174_758_085),
    },
    ModelProfile {
        label: "ax-gemma4-12b",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
        aliases: &[
            "ax-gemma4-12b",
            "ax-gemma-4-12b",
            "ax-gemma4-12b-qat-optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(9_306_791_577),
    },
    ModelProfile {
        label: "ax-gemma4-12b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP",
        aliases: &["ax-gemma4-12b-4bit", "ax-gemma4-12b-qat-4bit"],
        downloadable: true,
        approx_size_bytes: Some(11_290_537_445),
    },
    ModelProfile {
        label: "ax-gemma4-12b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP",
        aliases: &["ax-gemma4-12b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(12_260_440_732),
    },
    ModelProfile {
        label: "ax-gemma4-26b",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP",
        aliases: &[
            "ax-gemma4-26b",
            "ax-gemma-4-26b",
            "ax-gemma4-26b-optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(19_679_734_219),
    },
    ModelProfile {
        label: "ax-gemma4-26b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP",
        aliases: &["ax-gemma4-26b-4bit", "ax-gemma4-26b-qat-4bit"],
        downloadable: true,
        approx_size_bytes: Some(15_909_874_490),
    },
    ModelProfile {
        label: "ax-gemma4-26b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP",
        aliases: &["ax-gemma4-26b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(22_685_574_172),
    },
    ModelProfile {
        label: "ax-gemma4-31b",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP",
        aliases: &[
            "ax-gemma4-31b",
            "ax-gemma-4-31b",
            "ax-gemma4-31b-optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(24_519_926_135),
    },
    ModelProfile {
        label: "ax-gemma4-31b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP",
        aliases: &["ax-gemma4-31b-4bit", "ax-gemma4-31b-qat-4bit"],
        downloadable: true,
        approx_size_bytes: Some(29_145_665_562),
    },
    ModelProfile {
        label: "ax-gemma4-31b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP",
        aliases: &["ax-gemma4-31b-6bit"],
        downloadable: true,
        approx_size_bytes: Some(27_091_575_156),
    },
    ModelProfile {
        label: "ax-gemma4-12b-axq-6bit",
        preset: Some("gemma4-12b"),
        repo_id: "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-gemma4-12b-axq-6bit",
            "ax-gemma4-12b-axq",
            "gemma4-12b:axq",
            "gemma4-12b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(18_347_801_283),
    },
    ModelProfile {
        label: "ax-gemma4-12b-axq-4bit",
        preset: Some("gemma4-12b"),
        repo_id: "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
        aliases: &["ax-gemma4-12b-axq-4bit", "gemma4-12b:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(14_938_093_658),
    },
    ModelProfile {
        label: "ax-gemma4-26b-axq-6bit",
        preset: Some("gemma4-26b"),
        repo_id: "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-gemma4-26b-axq-6bit",
            "ax-gemma4-26b-axq",
            "gemma4-26b:axq",
            "gemma4-26b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(20_226_370_940),
    },
    ModelProfile {
        label: "ax-gemma4-26b-axq-4bit",
        preset: Some("gemma4-26b"),
        repo_id: "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP",
        aliases: &["ax-gemma4-26b-axq-4bit", "gemma4-26b:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(16_074_761_316),
    },
    ModelProfile {
        label: "ax-gemma4-31b-axq-6bit",
        preset: Some("gemma4-31b"),
        repo_id: "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP",
        aliases: &[
            "ax-gemma4-31b-axq-6bit",
            "ax-gemma4-31b-axq",
            "gemma4-31b:axq",
            "gemma4-31b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(24_426_134_101),
    },
    ModelProfile {
        label: "ax-gemma4-31b-axq-4bit",
        preset: Some("gemma4-31b"),
        repo_id: "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP",
        aliases: &["ax-gemma4-31b-axq-4bit", "gemma4-31b:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(19_450_776_368),
    },
    ModelProfile {
        label: "ax-qwen3-coder-next",
        preset: Some("qwen3-coder-next"),
        repo_id: "AutomatosX/AX-Qwen3-Coder-Next-MLX-OptiQ-4bit",
        aliases: &[
            "ax-qwen3-coder-next",
            "ax-qwen3-coder",
            "ax-qwen3-coder-next-optiq-4bit",
            "qwen3-coder-next:optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(55139958332),
    },
    ModelProfile {
        label: "ax-qwen3-coder-next-4bit",
        preset: Some("qwen3-coder-next"),
        repo_id: "AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit",
        aliases: &["ax-qwen3-coder-next-4bit", "ax-qwen3-coder-4bit"],
        downloadable: true,
        approx_size_bytes: Some(44_855_983_937),
    },
    ModelProfile {
        label: "ax-qwen3-coder-next-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit",
        aliases: &["ax-qwen3-coder-next-6bit", "ax-qwen3-coder-6bit"],
        downloadable: true,
        approx_size_bytes: Some(64_761_627_099),
    },
    ModelProfile {
        label: "ax-diffusiongemma-26b",
        preset: None,
        repo_id: "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-OptiQ-4bit",
        aliases: &[
            "ax-diffusiongemma-26b",
            "ax-diffusiongemma",
            "ax-diffusiongemma-26b-optiq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(17852322931),
    },
    ModelProfile {
        label: "ax-diffusiongemma-26b-4bit",
        preset: None,
        repo_id: "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
        aliases: &[
            "ax-diffusiongemma-26b-4bit",
            "ax-diffusiongemma-26b-a4b-it-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(16_575_490_802),
    },
    // AutomatosX embedding packs: served natively via /v1/embeddings and
    // co-residable with chat models (multi-model `load_mode=add`).
    ModelProfile {
        label: "ax-embeddinggemma-300m",
        preset: None,
        repo_id: "AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit",
        aliases: &[
            "ax-embeddinggemma-300m",
            "ax-embeddinggemma",
            "ax-embeddinggemma-300m-8bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(366_371_330),
    },
    ModelProfile {
        label: "ax-qwen3-embedding-0.6b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit",
        aliases: &["ax-qwen3-embedding-0.6b", "ax-qwen3-embedding-0.6b-8bit"],
        downloadable: true,
        approx_size_bytes: Some(649_219_787),
    },
    ModelProfile {
        label: "ax-qwen3-embedding-4b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ",
        aliases: &["ax-qwen3-embedding-4b", "ax-qwen3-embedding-4b-dwq"],
        downloadable: true,
        approx_size_bytes: Some(2_278_744_068),
    },
    ModelProfile {
        label: "ax-qwen3-embedding-8b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ",
        aliases: &["ax-qwen3-embedding-8b", "ax-qwen3-embedding-8b-dwq"],
        downloadable: true,
        approx_size_bytes: Some(4_273_261_843),
    },
    // --- Secondary: research / enterprise Llama ---
    ModelProfile {
        label: "llama3.1-8b",
        preset: Some("llama3.1-8b"),
        repo_id: "mlx-community/Llama-3.1-8B-Instruct-4bit",
        aliases: &[
            "llama3.1-8b",
            "llama31-8b",
            "llama-3.1-8b",
            "llama3.1-8b-4bit",
            "llama-3.1-8b-instruct-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(4_534_824_337),
    },
    ModelProfile {
        label: "llama3.3-70b",
        preset: Some("llama3.3-70b"),
        repo_id: "mlx-community/Llama-3.3-70B-Instruct-4bit",
        aliases: &[
            "llama3.3-70b",
            "llama33-70b",
            "llama-3.3-70b",
            "llama3.3-70b-4bit",
            "llama-3.3-70b-instruct-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(39_706_010_909),
    },
    ModelProfile {
        label: "llama4-scout",
        preset: Some("llama4-scout"),
        repo_id: "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        aliases: &[
            "llama4-scout",
            "llama-4-scout",
            "llama4-scout-4bit",
            "llama-4-scout-17b-16e-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(61_143_665_814),
    },
    // --- Secondary: European market Mistral ---
    ModelProfile {
        label: "mistral-small",
        preset: Some("mistral-small"),
        repo_id: "mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
        aliases: &[
            "mistral-small",
            "mistral-small-24b",
            "mistral-small-4bit",
            "mistral-small-24b-4bit",
            "mistral-small-3.1",
        ],
        downloadable: true,
        approx_size_bytes: Some(14_119_058_051),
    },
    ModelProfile {
        label: "ministral-8b",
        preset: Some("ministral-8b"),
        repo_id: "mlx-community/Ministral-8B-Instruct-2410-4bit",
        aliases: &[
            "ministral-8b",
            "ministral",
            "ministral-8b-4bit",
            "ministral-8b-instruct-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(4_543_802_547),
    },
    ModelProfile {
        label: "devstral-small",
        preset: Some("devstral-small"),
        repo_id: "mlx-community/Devstral-Small-2505-4bit",
        aliases: &[
            "devstral-small",
            "devstral",
            "devstral-small-4bit",
            "devstral-small-2505-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(13_277_563_657),
    },
    // --- Secondary: open reasoner GPT-OSS (MXFP4 experts) ---
    ModelProfile {
        label: "gpt-oss-20b",
        preset: Some("gpt-oss-20b"),
        repo_id: "mlx-community/gpt-oss-20b-MXFP4-Q4",
        aliases: &[
            "gpt-oss-20b",
            "gptoss-20b",
            "gpt-oss-20b-4bit",
            "gpt-oss-20b-mxfp4",
            "gpt-oss-20b-mxfp4-q4",
        ],
        downloadable: true,
        approx_size_bytes: Some(11_206_563_096),
    },
    ModelProfile {
        label: "gpt-oss-120b",
        preset: Some("gpt-oss-120b"),
        repo_id: "mlx-community/gpt-oss-120b-MXFP4-Q4",
        aliases: &[
            "gpt-oss-120b",
            "gptoss-120b",
            "gpt-oss-120b-4bit",
            "gpt-oss-120b-mxfp4",
            "gpt-oss-120b-mxfp4-q4",
        ],
        downloadable: true,
        // Prefer 128 GB+ hosts; experts stay MXFP4-packed at runtime.
        approx_size_bytes: Some(62_358_100_309),
    },
    // Gap-fill AutomatosX packs (except DeepSeek): GPT-OSS, VL-8B, Coder AXQ,
    // Ministral-3, Mistral Small, Nemotron, Devstral, Qwen 3.8 extras, OCR, ASR.
    ModelProfile {
        label: "ax-devstral-small",
        preset: Some("devstral-small"),
        repo_id: "AutomatosX/AX-Devstral-Small-2-24B-Instruct-2512-MLX-OptiQ-4bit",
        aliases: &[
            "ax-devstral-small",
            "ax-devstral",
            "ax-devstral-small-2",
            "ax-devstral-small-optiq-4bit",
            "devstral-small-2",
        ],
        downloadable: true,
        approx_size_bytes: Some(17672202750),
    },
    ModelProfile {
        label: "ax-devstral-small-axq-4bit",
        preset: Some("devstral-small"),
        repo_id: "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit",
        aliases: &["ax-devstral-small-axq-4bit", "devstral-small:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(14602976481),
    },
    ModelProfile {
        label: "ax-devstral-small-axq-6bit",
        preset: Some("devstral-small"),
        repo_id: "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit",
        aliases: &[
            "ax-devstral-small-axq-6bit",
            "ax-devstral-small-axq",
            "devstral-small:axq",
            "devstral-small:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(17696941586),
    },
    ModelProfile {
        label: "holo3-35b",
        preset: Some("holo3-35b"),
        repo_id: "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit",
        aliases: &[
            "holo3-35b",
            "holo3-35b-a3b",
            "holo3",
            "holo3-35b-4bit",
            "holo3-35b-a3b-4bit",
            "ax-holo3-35b",
            "ax-holo3",
            "holo3-35b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(24884006585),
    },
    ModelProfile {
        label: "holo3-35b-6bit",
        preset: Some("holo3-35b"),
        repo_id: "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit",
        aliases: &[
            "holo3-35b-6bit",
            "holo3-35b-a3b-6bit",
            "ax-holo3-35b-6bit",
            "holo3-35b:axq",
            "holo3-35b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(30769108800),
    },
    ModelProfile {
        label: "ax-ministral-3-14b-axq-4bit",
        preset: Some("ministral-3-14b"),
        repo_id: "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit",
        aliases: &["ax-ministral-3-14b-axq-4bit", "ministral-3-14b:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(9796902900),
    },
    ModelProfile {
        label: "ax-ministral-3-14b-axq-6bit",
        preset: Some("ministral-3-14b"),
        repo_id: "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit",
        aliases: &[
            "ax-ministral-3-14b-axq-6bit",
            "ax-ministral-3-14b-axq",
            "ministral-3-14b:axq",
            "ministral-3-14b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(10476534273),
    },
    ModelProfile {
        label: "ax-ministral-3-14b",
        preset: Some("ministral-3-14b"),
        repo_id: "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-OptiQ-4bit",
        aliases: &[
            "ax-ministral-3-14b",
            "ax-ministral-3-14b-optiq-4bit",
            "ministral-3-14b",
        ],
        downloadable: true,
        approx_size_bytes: Some(10623200972),
    },
    ModelProfile {
        label: "ax-ministral-3-8b-axq-6bit",
        preset: Some("ministral-3-8b"),
        repo_id: "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit",
        aliases: &[
            "ax-ministral-3-8b-axq-6bit",
            "ax-ministral-3-8b-axq",
            "ministral-3-8b:axq",
            "ministral-3-8b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(6706317552),
    },
    ModelProfile {
        label: "ax-ministral-3-8b",
        preset: Some("ministral-3-8b"),
        repo_id: "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-OptiQ-4bit",
        aliases: &[
            "ax-ministral-3-8b",
            "ax-ministral-3-8b-optiq-4bit",
            "ministral-3-8b",
            "ministral-3",
        ],
        downloadable: true,
        approx_size_bytes: Some(7139470696),
    },
    ModelProfile {
        label: "ax-mistral-small-axq-4bit",
        preset: Some("mistral-small"),
        repo_id: "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit",
        aliases: &[
            "ax-mistral-small-axq-4bit",
            "mistral-small:axq-4bit",
            "ax-mistral-small-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(15475259414),
    },
    ModelProfile {
        label: "ax-mistral-small",
        preset: Some("mistral-small"),
        repo_id: "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit",
        aliases: &[
            "ax-mistral-small",
            "ax-mistral-small-24b",
            "ax-mistral-small-axq",
            "mistral-small:axq",
            "mistral-small:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(18026258863),
    },
    ModelProfile {
        label: "ax-nemotron-3-nano",
        preset: Some("nemotron-3-nano"),
        repo_id: "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit",
        aliases: &[
            "ax-nemotron-3-nano",
            "ax-nemotron-3-nano-30b",
            "nemotron-3-nano",
            "nemotron-3-nano-30b",
            "nemotron-3-nano:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(18968834209),
    },
    ModelProfile {
        label: "ax-nemotron-3-nano-6bit",
        preset: Some("nemotron-3-nano"),
        repo_id: "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit",
        aliases: &[
            "ax-nemotron-3-nano-6bit",
            "ax-nemotron-3-nano-axq",
            "nemotron-3-nano:axq",
            "nemotron-3-nano:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(23669907947),
    },
    ModelProfile {
        label: "ornith-35b",
        preset: Some("ornith-35b"),
        repo_id: "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit",
        aliases: &[
            "ornith-35b",
            "ornith",
            "ornith-1.0-35b",
            "ornith-1.0",
            "ornith-35b-4bit",
            "ornith-1.0-35b-4bit",
            "ax-ornith-35b",
            "ax-ornith",
            "ornith-35b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(21437078089),
    },
    ModelProfile {
        label: "ornith-35b-6bit",
        preset: Some("ornith-35b"),
        repo_id: "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit",
        aliases: &[
            "ornith-35b-6bit",
            "ornith-1.0-35b-6bit",
            "ax-ornith-35b-6bit",
            "ornith-35b:axq",
            "ornith-35b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(26352107653),
    },
    ModelProfile {
        label: "ax-qwen3-asr-1.7b",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3-asr-1.7b",
            "ax-qwen3-asr",
            "qwen3-asr-1.7b",
            "qwen3-asr",
            "qwen3-asr-1.7b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(1765662877),
    },
    ModelProfile {
        label: "ax-qwen3-asr-1.7b-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3-asr-1.7b-6bit",
            "qwen3-asr-1.7b:axq",
            "qwen3-asr-1.7b:axq-6bit",
            "qwen3-asr:axq",
        ],
        downloadable: true,
        approx_size_bytes: Some(2132544013),
    },
    ModelProfile {
        label: "ax-qwen3-coder-next-axq-4bit",
        preset: Some("qwen3-coder-next"),
        repo_id: "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3-coder-next-axq-4bit",
            "ax-qwen3-coder-axq-4bit",
            "qwen3-coder-next:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(47885901311),
    },
    ModelProfile {
        label: "ax-qwen3-coder-next-axq-6bit",
        preset: Some("qwen3-coder-next"),
        repo_id: "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3-coder-next-axq-6bit",
            "ax-qwen3-coder-next-axq",
            "ax-qwen3-coder-axq",
            "qwen3-coder-next:axq",
            "qwen3-coder-next:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(59850809730),
    },
    ModelProfile {
        label: "ax-qwen3-nemotron-32b-genrm",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3-nemotron-32b-genrm",
            "ax-qwen3-nemotron-genrm",
            "qwen3-nemotron-32b-genrm",
            "qwen3-nemotron-32b-genrm:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(19956676570),
    },
    ModelProfile {
        label: "ax-qwen3-nemotron-32b-genrm-6bit",
        preset: None,
        repo_id: "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3-nemotron-32b-genrm-6bit",
            "qwen3-nemotron-32b-genrm:axq",
            "qwen3-nemotron-32b-genrm:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(24584520018),
    },
    ModelProfile {
        label: "ax-qwen3-vl-8b-4bit",
        preset: Some("qwen3-vl-8b"),
        repo_id: "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3-vl-8b-4bit",
            "ax-qwen3-vl-8b-axq-4bit",
            "qwen3-vl-8b:axq-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(6985011416),
    },
    ModelProfile {
        label: "ax-qwen3-vl-8b",
        preset: Some("qwen3-vl-8b"),
        repo_id: "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3-vl-8b",
            "ax-qwen3-vl-8b-axq",
            "ax-qwen3-vl-8b-6bit",
            "qwen3-vl-8b:axq",
            "qwen3-vl-8b:axq-6bit",
            "qwen3-vl-8b-instruct:axq",
        ],
        downloadable: true,
        approx_size_bytes: Some(8782284430),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-4bit-base",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit",
        aliases: &[
            "ax-qwen3.8-27b-axq-4bit-base",
            "qwen3.8-27b:axq-4bit-base",
            "ax-qwen3.8-27b-4bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(17347922966),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-6bit-base",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit",
        aliases: &[
            "ax-qwen3.8-27b-axq-6bit-base",
            "qwen3.8-27b:axq-6bit-base",
            "ax-qwen3.8-27b-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(20539844328),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-8bit-base",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit",
        aliases: &["ax-qwen3.8-27b-axq-8bit-base", "qwen3.8-27b:axq-8bit-base"],
        downloadable: true,
        approx_size_bytes: Some(27379005741),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-8bit",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP",
        aliases: &[
            "ax-qwen3.8-27b-axq-8bit",
            "qwen3.8-27b:axq-8bit",
            "ax-qwen3.8-27b-8bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(30372618029),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-mxfp4-base",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4",
        aliases: &[
            "ax-qwen3.8-27b-axq-mxfp4-base",
            "qwen3.8-27b:axq-mxfp4-base",
        ],
        downloadable: true,
        approx_size_bytes: Some(16586849399),
    },
    ModelProfile {
        label: "ax-qwen3.8-27b-axq-mxfp4",
        preset: Some("qwen3.8-27b"),
        repo_id: "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
        aliases: &[
            "ax-qwen3.8-27b-axq-mxfp4",
            "qwen3.8-27b:axq-mxfp4",
            "ax-qwen3.8-27b-mxfp4",
        ],
        downloadable: true,
        approx_size_bytes: Some(17436265609),
    },
    ModelProfile {
        label: "ax-unlimited-ocr",
        preset: None,
        repo_id: "AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-MXFP8",
        aliases: &[
            "ax-unlimited-ocr",
            "ax-unlimited-ocr-3b",
            "unlimited-ocr",
            "unlimited-ocr-3b",
        ],
        downloadable: true,
        approx_size_bytes: Some(3856472307),
    },
    ModelProfile {
        label: "ax-gpt-oss-120b-axq-6bit",
        preset: Some("gpt-oss-120b"),
        repo_id: "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit",
        aliases: &[
            "ax-gpt-oss-120b-axq-6bit",
            "ax-gpt-oss-120b-axq",
            "gpt-oss-120b:axq",
            "gpt-oss-120b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(96075600563),
    },
    ModelProfile {
        label: "ax-gpt-oss-20b-axq-4bit",
        preset: Some("gpt-oss-20b"),
        repo_id: "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit",
        aliases: &["ax-gpt-oss-20b-axq-4bit", "gpt-oss-20b:axq-4bit"],
        downloadable: true,
        approx_size_bytes: Some(13244904684),
    },
    ModelProfile {
        label: "ax-gpt-oss-20b-axq-6bit",
        preset: Some("gpt-oss-20b"),
        repo_id: "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit",
        aliases: &[
            "ax-gpt-oss-20b-axq-6bit",
            "ax-gpt-oss-20b-axq",
            "gpt-oss-20b:axq",
            "gpt-oss-20b:axq-6bit",
        ],
        downloadable: true,
        approx_size_bytes: Some(15714732576),
    },
];

const MTP_DOWNLOAD_TARGETS: &[MtpDownloadTarget] = &[
    MtpDownloadTarget {
        label: "qwen3.6-27b-6bit",
        repo_id: "mlx-community/Qwen3.6-27B-6bit",
        aliases: &[
            "qwen3.6-27b-6bit",
            "qwen36-27b-6bit",
            "qwen3-6-27b-6bit",
            "qwen3.6-27b",
            "qwen36-27b",
        ],
        kind: MtpDownloadKind::QwenSidecar {
            mtp_source: "Qwen/Qwen3.6-27B",
        },
    },
    MtpDownloadTarget {
        label: "qwen3.6-35b-a3b",
        repo_id: "mlx-community/Qwen3.6-35B-A3B-6bit",
        aliases: &[
            "qwen3.6-35b-a3b",
            "qwen3.6-35b-a3b-6bit",
            "qwen36-35b-a3b",
            "qwen36-35b",
            "qwen3.6-35b",
        ],
        kind: MtpDownloadKind::QwenSidecar {
            mtp_source: "Qwen/Qwen3.6-35B-A3B",
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-12b",
        repo_id: "mlx-community/gemma-4-12B-it-6bit",
        aliases: &[
            "gemma-4-12b",
            "gemma-4-12b-it",
            "gemma-4-12b-6bit",
            "gemma4-12b",
            "gemma4-12b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-6bit",
            target_model_id: "gemma-4-12b-it",
            assistant_model_id: "gemma-4-12b-it-assistant",
            max_depth: 2,
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-12b-4bit",
        repo_id: "mlx-community/gemma-4-12B-it-4bit",
        aliases: &["gemma-4-12b-4bit", "gemma-4-12b-it-4bit", "gemma4-12b-4bit"],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-4bit",
            target_model_id: "gemma-4-12b-it",
            assistant_model_id: "gemma-4-12b-it-assistant",
            max_depth: 2,
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-26b",
        repo_id: "mlx-community/gemma-4-26b-a4b-it-6bit",
        aliases: &[
            "gemma-4-26b",
            "gemma-4-26b-a4b",
            "gemma-4-26b-a4b-it",
            "gemma-4-26b-6bit",
            "gemma4-26b",
            "gemma4-26b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "google/gemma-4-26b-a4b-it-assistant",
            target_model_id: "gemma-4-26b-a4b-it",
            assistant_model_id: "gemma-4-26b-a4b-it-assistant",
            max_depth: 1,
        },
    },
    MtpDownloadTarget {
        label: "gemma-4-31b",
        repo_id: "mlx-community/gemma-4-31b-it-6bit",
        aliases: &[
            "gemma-4-31b",
            "gemma-4-31b-it",
            "gemma-4-31b-6bit",
            "gemma4-31b",
            "gemma4-31b-6bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "google/gemma-4-31b-it-assistant",
            target_model_id: "gemma-4-31b-it",
            assistant_model_id: "gemma-4-31b-it-assistant",
            max_depth: 1,
        },
    },
    // WS-P2: E4B assistant-MTP packaging publication path. E2B still loads
    // from an explicit directory but is not a catalogued download alias.
    MtpDownloadTarget {
        label: "gemma-4-e4b",
        repo_id: "mlx-community/gemma-4-E4B-it-4bit",
        aliases: &[
            "gemma-4-e4b",
            "gemma4-e4b",
            "gemma-4-e4b-it",
            "gemma-4-e4b-4bit",
            "gemma4-e4b-4bit",
        ],
        kind: MtpDownloadKind::GemmaAssistant {
            assistant_repo_id: "mlx-community/gemma-4-E4B-it-assistant-4bit",
            target_model_id: "gemma-4-e4b-it",
            assistant_model_id: "gemma-4-e4b-it-assistant",
            max_depth: 2,
        },
    },
    // WS-P2: Coder-Next MTP publication (Qwen fused sidecar from base).
    MtpDownloadTarget {
        label: "qwen3-coder-next",
        repo_id: "mlx-community/Qwen3-Coder-Next-4bit",
        aliases: &[
            "qwen3-coder-next",
            "qwen3-coder-next-4bit",
            "qwen3_coder_next",
            "coder-next",
        ],
        kind: MtpDownloadKind::QwenSidecar {
            mtp_source: "Qwen/Qwen3-Coder-Next",
        },
    },
];

fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run(args: Vec<OsString>) -> Result<u8, String> {
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return Ok(0);
    }
    match args[0].to_string_lossy().as_ref() {
        "serve" => cmd_serve(&args[1..]),
        "download" => cmd_download(&args[1..]),
        "download-mtp" => cmd_download_mtp(&args[1..]),
        "models" => cmd_models(&args[1..]),
        "doctor" => cmd_doctor(&args[1..]),
        "mtp-capability" => cmd_mtp_capability(&args[1..]),
        "convert-mtplx" => cmd_convert_mtplx(&args[1..]),
        "tui" => tui::cmd_tui(&args[1..]),
        unknown => Err(format!(
            "unknown command: {unknown}\n\nRun `ax-engine --help` for usage."
        )),
    }
}

fn print_usage() {
    println!(
        "Usage:\n  ax-engine serve <model-dir-or-alias> [--host <host>] [--port <port>] [--offline|--local-only] [--download] [--dry-run] [--json] [-- <ax-engine-server args>]\n  ax-engine download [<alias-or-repo-id>] [--dest <path>] [--force|--local-only] [--list] [--json] [--progress-json]\n  ax-engine download-mtp <mtp-target> [--output <dir>] [--force] [--quantize 4|8] [--mtp-depth-max <n>] [--group-size <n>] [--fair-base-only] [--json] [--progress-json]\n  ax-engine models list [--models-dir <path>] [--json]\n  ax-engine models info <alias-or-path> [--json]\n  ax-engine models rm <path> [--dry-run] [--yes] [--json]\n  ax-engine doctor [--json] [--verbose] [--mlx-model-artifacts-dir <path>]\n  ax-engine mtp-capability [--json]\n  ax-engine convert-mtplx <base-model> --mtp-source <repo> [--output <dir>] [--quantize 4|8] [--mtp-depth-max <n>] [--group-size <n>] [--fair-base-only] [--json]\n  ax-engine tui"
    );
}

/// One-line JSON capability contract consumed by AXQuant's
/// `quantize-mtp-sidecar --capability-command` probe. Values derive from the
/// same loader constants `mtp_take_weight` executes against, so the report
/// cannot drift from actual runtime capability.
fn mtp_capability_json() -> serde_json::Value {
    serde_json::json!({
        "ok": true,
        "mtp_enabled": true,
        "layout": ax_engine_mlx::weights::MTP_SIDECAR_QWEN36_LAYOUT,
        "quantized_sidecar": true,
        "supported_bits": ax_engine_mlx::weights::MTP_SIDECAR_SUPPORTED_BITS,
        "packing": "mlx-affine-packed-u32",
        "ax_engine_version": env!("CARGO_PKG_VERSION"),
    })
}

fn cmd_mtp_capability(args: &[OsString]) -> Result<u8, String> {
    for arg in args {
        match arg.to_string_lossy().as_ref() {
            // Output is always one-line JSON; --json is accepted for
            // consistency with the other subcommands.
            "--json" => {}
            "--help" | "-h" => {
                println!(
                    "Usage:\n  ax-engine mtp-capability [--json]\n\nPrints the one-line JSON MTP sidecar capability contract (layout,\nsupported quantization bits, packing) consumed by AXQuant's\n`quantize-mtp-sidecar --capability-command` probe."
                );
                return Ok(0);
            }
            flag => return Err(format!("unknown mtp-capability option: {flag}")),
        }
    }
    println!("{}", mtp_capability_json());
    Ok(0)
}

fn cmd_doctor(args: &[OsString]) -> Result<u8, String> {
    let args = parse_doctor_args(args)?;
    if args.help {
        return Ok(0);
    }
    let mut bench_args = vec![OsString::from("doctor")];
    if args.verbose {
        if args.json {
            bench_args.push(OsString::from("--json"));
        }
        bench_args.extend(args.bench_args);
        return exec_or_status(find_executable("ax-engine-bench"), &bench_args);
    }
    bench_args.push(OsString::from("--json"));
    bench_args.extend(args.bench_args);
    let (code, bench_report, stderr) = run_bench_doctor_json(&bench_args)?;
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    if code != 0 {
        return Ok(code);
    }
    let report = user_doctor_report(&bench_report);
    if args.json {
        print_json(&report)?;
    } else {
        println!("{}", format_user_doctor_report(&report));
    }
    Ok(
        if report.get("result").and_then(Value::as_str) == Some("not_ready") {
            1
        } else {
            0
        },
    )
}

#[derive(Debug)]
struct DoctorArgs {
    json: bool,
    verbose: bool,
    help: bool,
    bench_args: Vec<OsString>,
}

fn parse_doctor_args(args: &[OsString]) -> Result<DoctorArgs, String> {
    let mut json = false;
    let mut verbose = false;
    let mut bench_args = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--json" => json = true,
            "--verbose" => verbose = true,
            "--mlx-model-artifacts-dir" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--mlx-model-artifacts-dir requires a value".to_string())?
                    .clone();
                bench_args.push(OsString::from("--mlx-model-artifacts-dir"));
                bench_args.push(value);
            }
            "--help" | "-h" => {
                println!(
                    "Usage:\n  ax-engine doctor [--json] [--verbose] [--mlx-model-artifacts-dir <path>]\n\nDefault output is an end-user readiness summary. Use --verbose for the detailed ax-engine-bench doctor report."
                );
                return Ok(DoctorArgs {
                    json,
                    verbose,
                    help: true,
                    bench_args,
                });
            }
            flag if flag.starts_with('-') => return Err(format!("unknown doctor option: {flag}")),
            _ => return Err("doctor does not accept positional arguments".into()),
        }
        index += 1;
    }
    Ok(DoctorArgs {
        json,
        verbose,
        help: false,
        bench_args,
    })
}

fn run_bench_doctor_json(args: &[OsString]) -> Result<(u8, Value, String), String> {
    let bench = find_executable("ax-engine-bench");
    let output = Command::new(&bench)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run {}: {err}", bench.display()))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let report = serde_json::from_str::<Value>(stdout.trim()).map_err(|err| {
        format!(
            "ax-engine-bench doctor did not emit valid JSON: {err}\nstdout:\n{}",
            stdout.trim()
        )
    })?;
    Ok((
        output.status.code().unwrap_or(1).try_into().unwrap_or(1),
        report,
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

fn user_doctor_report(bench: &Value) -> Value {
    let server = probe_binary("ax-engine-server");
    let bench_bin = probe_binary("ax-engine-bench");
    let host_system = host_system_summary();
    let bench_status = value_str(bench, &["status"]).unwrap_or("unknown");
    let mlx_ready = value_bool(bench, &["mlx_runtime_ready"]).unwrap_or(false);
    let model_status = value_str(bench, &["model_artifacts", "status"]).unwrap_or("unknown");
    let model_selected = value_bool(bench, &["model_artifacts", "selected"]).unwrap_or(false);
    let model_path = value_str(bench, &["model_artifacts", "path"]);
    let issues = bench
        .get("issues")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let model_issues = bench
        .get("model_artifacts")
        .and_then(|value| value.get("issues"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let result = if !server.available || !bench_bin.available || bench_status == "not_ready" {
        "not_ready"
    } else if bench_status == "bringup_only" {
        "degraded"
    } else {
        "ready"
    };

    let mut next_actions = Vec::new();
    if !server.available {
        next_actions.push("Reinstall ax-engine so ax-engine-server is on PATH.".to_string());
    } else if !bench_bin.available {
        next_actions.push("Reinstall ax-engine so ax-engine-bench is on PATH.".to_string());
    } else if !mlx_ready {
        next_actions.push("Fix the host or Metal runtime issues listed below.".to_string());
    } else if model_status == "not_ready" {
        if let Some(path) = model_path {
            next_actions.push(format!("ax-engine-bench generate-manifest {path} --json"));
            next_actions.push(format!("ax-engine doctor --mlx-model-artifacts-dir {path}"));
        } else {
            next_actions
                .push("Pass --mlx-model-artifacts-dir <model-dir> to inspect a model.".to_string());
        }
    } else if model_selected {
        if let Some(path) = model_path {
            next_actions.push(format!("ax-engine serve {path} --port 31418"));
        } else {
            next_actions.push("ax-engine serve <model-dir> --port 31418".to_string());
        }
    } else {
        next_actions.push("ax-engine serve qwen36-35b --port 31418".to_string());
        next_actions.push("ax-engine models list".to_string());
    }

    json!({
        "schema_version": "ax.engine.doctor.v1",
        "result": result,
        "ready_for": ready_for(result, model_status),
        "install": {
            "version": env!("CARGO_PKG_VERSION"),
            "mode": value_str(bench, &["workflow", "mode"]).unwrap_or("unknown"),
            "cwd": value_str(bench, &["workflow", "cwd"]).unwrap_or("unknown"),
        },
        "host": host_system,
        "checks": [
            check("server_binary", server.available, server.detail),
            check("bench_binary", bench_bin.available, bench_bin.detail),
            check("host", value_bool(bench, &["host", "supported_mlx_runtime"]).unwrap_or(false), host_detail(bench)),
            check("metal_toolchain", metal_check_pass(bench), metal_detail(bench)),
            check("mlx_runtime", mlx_ready, bench_status.to_string()),
            json!({
                "id": "model",
                "status": model_status,
                "selected": model_selected,
                "path": model_path,
            }),
        ],
        "issues": issues,
        "model_issues": model_issues,
        "next_actions": next_actions,
        "details_command": "ax-engine-bench doctor",
        "source": {
            "schema_version": value_str(bench, &["schema_version"]).unwrap_or("unknown"),
            "status": bench_status,
            "details_command": "ax-engine-bench doctor --json",
        },
    })
}

#[derive(Debug)]
struct BinaryProbe {
    available: bool,
    detail: String,
}

fn probe_binary(name: &str) -> BinaryProbe {
    let bin = find_executable(name);
    match Command::new(&bin)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => BinaryProbe {
            available: true,
            detail: format!("{} ok", bin.display()),
        },
        Ok(status) => BinaryProbe {
            available: false,
            detail: format!("{} exited with status {}", bin.display(), status),
        },
        Err(err) => BinaryProbe {
            available: false,
            detail: format!("{}: {err}", bin.display()),
        },
    }
}

fn host_system_summary() -> Value {
    let os = value_or_unknown(env::consts::OS);
    let arch = value_or_unknown(env::consts::ARCH);
    let hardware_profile = command_stdout("system_profiler", &["SPHardwareDataType"]);
    let os_version = detect_os_version();
    let os_build = detect_os_build();
    let ram_bytes =
        detect_memory_bytes().or_else(|| hardware_profile.as_deref().and_then(parse_memory_bytes));
    let cpu_cores = detect_cpu_cores(hardware_profile.as_deref());
    json!({
        "os": os,
        "arch": arch,
        "os_version": os_version,
        "os_build": os_build,
        "ram_bytes": ram_bytes,
        "ram_gib": ram_bytes.map(bytes_to_gib),
        "cpu_cores": cpu_cores,
        "gpu_cores": detect_gpu_cores(),
    })
}

fn value_or_unknown(value: &str) -> &str {
    if value.is_empty() { "unknown" } else { value }
}

fn detect_os_version() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-productVersion"]),
        _ => None,
    }
}

fn detect_os_build() -> Option<String> {
    match env::consts::OS {
        "macos" => command_stdout("sw_vers", &["-buildVersion"]),
        _ => None,
    }
}

fn detect_memory_bytes() -> Option<u64> {
    match env::consts::OS {
        "macos" => command_stdout("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| value.parse::<u64>().ok()),
        _ => None,
    }
}

fn detect_cpu_cores(hardware_profile: Option<&str>) -> Value {
    let physical = command_stdout("sysctl", &["-n", "hw.physicalcpu"])
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| hardware_profile.and_then(parse_physical_cpu_cores));
    let logical = command_stdout("sysctl", &["-n", "hw.logicalcpu"])
        .and_then(|value| value.parse::<u64>().ok());
    let mut performance = None;
    let mut efficiency = None;
    let mut types = serde_json::Map::new();

    for level in ["0", "1", "2", "3"] {
        let name_key = format!("hw.perflevel{level}.name");
        let cpu_key = format!("hw.perflevel{level}.physicalcpu");
        let Some(name) = command_stdout("sysctl", &["-n", &name_key]) else {
            continue;
        };
        let cores =
            command_stdout("sysctl", &["-n", &cpu_key]).and_then(|value| value.parse::<u64>().ok());
        let normalized = name.to_ascii_lowercase();
        if normalized.contains("performance") {
            performance = cores;
        } else if normalized.contains("efficiency") {
            efficiency = cores;
        }
        if let Some(cores) = cores {
            types.insert(normalized.replace(' ', "_"), json!(cores));
        }
    }

    let summary = hardware_profile.and_then(parse_cpu_core_summary);
    if types.is_empty()
        && let Some(summary) = summary.as_deref()
    {
        for (label, cores) in parse_cpu_core_types(summary) {
            let normalized = label.to_ascii_lowercase().replace(' ', "_");
            if normalized.contains("performance") && performance.is_none() {
                performance = Some(cores);
            } else if normalized.contains("efficiency") && efficiency.is_none() {
                efficiency = Some(cores);
            }
            types.insert(normalized, json!(cores));
        }
    }

    json!({
        "physical": physical,
        "logical": logical,
        "performance": performance,
        "efficiency": efficiency,
        "summary": summary,
        "types": types,
    })
}

fn parse_memory_bytes(output: &str) -> Option<u64> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(value) = trimmed.strip_prefix("Memory:") else {
            continue;
        };
        let mut parts = value.split_whitespace();
        let amount = parts.next()?.parse::<u64>().ok()?;
        let unit = parts.next()?.to_ascii_lowercase();
        return match unit.as_str() {
            "gb" | "gib" => amount.checked_mul(1024 * 1024 * 1024),
            "mb" | "mib" => amount.checked_mul(1024 * 1024),
            _ => None,
        };
    }
    None
}

fn parse_physical_cpu_cores(output: &str) -> Option<u64> {
    let summary = parse_cpu_core_summary(output)?;
    summary
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<u64>().ok())
}

fn parse_cpu_core_summary(output: &str) -> Option<String> {
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Total Number of Cores:") {
            return Some(value.trim().to_string());
        }
    }
    None
}

fn parse_cpu_core_types(summary: &str) -> Vec<(String, u64)> {
    let Some(start) = summary.find('(') else {
        return Vec::new();
    };
    let Some(end) = summary[start + 1..].find(')') else {
        return Vec::new();
    };
    let inside = &summary[start + 1..start + 1 + end];
    inside
        .split(" and ")
        .filter_map(|part| {
            let mut words = part.split_whitespace();
            let cores = words.next()?.parse::<u64>().ok()?;
            let label = words.collect::<Vec<_>>().join(" ");
            if label.is_empty() {
                None
            } else {
                Some((label, cores))
            }
        })
        .collect()
}

fn detect_gpu_cores() -> Option<u64> {
    let output = command_stdout("system_profiler", &["SPDisplaysDataType"])?;
    for line in output.lines() {
        let trimmed = line.trim();
        if let Some(value) = trimmed.strip_prefix("Total Number of Cores:") {
            return value.trim().parse::<u64>().ok();
        }
    }
    None
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_string())
}

fn bytes_to_gib(bytes: u64) -> u64 {
    bytes / (1024 * 1024 * 1024)
}

fn ready_for(result: &str, model_status: &str) -> Vec<&'static str> {
    if result == "not_ready" {
        Vec::new()
    } else if model_status == "ready" {
        vec!["serve", "python_sdk", "model_checks"]
    } else {
        vec!["serve", "python_sdk"]
    }
}

fn check(id: &str, pass: bool, detail: String) -> Value {
    json!({
        "id": id,
        "status": if pass { "pass" } else { "fail" },
        "detail": detail,
    })
}

fn host_detail(report: &Value) -> String {
    format!(
        "{} ({}/{})",
        value_str(report, &["host", "detected_soc"]).unwrap_or("unknown Apple Silicon"),
        value_str(report, &["host", "os"]).unwrap_or("unknown"),
        value_str(report, &["host", "arch"]).unwrap_or("unknown")
    )
}

fn metal_detail(report: &Value) -> String {
    if value_bool(report, &["metal_toolchain", "fully_available"]).unwrap_or(false) {
        "Metal compiler and metallib available".to_string()
    } else if value_str(report, &["runtime_assets", "status"]) == Some("ready") {
        "Bundled runtime assets available; Metal compiler only needed for kernel rebuilds"
            .to_string()
    } else {
        "Metal compiler or metallib missing".to_string()
    }
}

fn metal_check_pass(report: &Value) -> bool {
    value_bool(report, &["metal_toolchain", "fully_available"]).unwrap_or(false)
        || value_str(report, &["runtime_assets", "status"]) == Some("ready")
}

fn value_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

fn value_str<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    value_at(value, path)?.as_str()
}

fn value_bool(value: &Value, path: &[&str]) -> Option<bool> {
    value_at(value, path)?.as_bool()
}

fn format_user_doctor_report(report: &Value) -> String {
    let mut lines = vec![
        "AX Engine doctor".to_string(),
        String::new(),
        format!(
            "Result: {}",
            report
                .get("result")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .replace('_', " ")
        ),
        String::new(),
        "Install:".to_string(),
        format!(
            "  version: {}",
            value_str(report, &["install", "version"]).unwrap_or("unknown")
        ),
        format!(
            "  mode: {}",
            value_str(report, &["install", "mode"]).unwrap_or("unknown")
        ),
        format!(
            "  host: {} {} ({})",
            value_str(report, &["host", "os"]).unwrap_or("unknown"),
            value_str(report, &["host", "os_version"]).unwrap_or("unknown"),
            value_str(report, &["host", "arch"]).unwrap_or("unknown")
        ),
        format!(
            "  RAM: {}",
            report
                .get("host")
                .and_then(|host| host.get("ram_gib"))
                .and_then(Value::as_u64)
                .map(|gib| format!("{gib} GiB"))
                .unwrap_or_else(|| "unknown".to_string())
        ),
        format!(
            "  CPU cores: {}",
            format_cpu_cores(report.get("host").and_then(|host| host.get("cpu_cores")))
        ),
        format!(
            "  GPU cores: {}",
            report
                .get("host")
                .and_then(|host| host.get("gpu_cores"))
                .and_then(Value::as_u64)
                .map(|cores| cores.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ),
        String::new(),
        "Checks:".to_string(),
    ];
    if let Some(checks) = report.get("checks").and_then(Value::as_array) {
        for check in checks {
            let id = check.get("id").and_then(Value::as_str).unwrap_or("unknown");
            let status = check
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let detail = check.get("detail").and_then(Value::as_str);
            if let Some(detail) = detail {
                lines.push(format!("  {id}: {status} - {detail}"));
            } else {
                let selected = check
                    .get("selected")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let path = check.get("path").and_then(Value::as_str).unwrap_or("none");
                lines.push(format!(
                    "  {id}: {status} (selected: {selected}, path: {path})"
                ));
            }
        }
    }
    lines.push(String::new());
    lines.push("Issues:".to_string());
    append_string_array(&mut lines, report.get("issues").and_then(Value::as_array));
    lines.push(String::new());
    lines.push("Model issues:".to_string());
    append_string_array(
        &mut lines,
        report.get("model_issues").and_then(Value::as_array),
    );
    lines.push(String::new());
    lines.push("Next:".to_string());
    append_string_array(
        &mut lines,
        report.get("next_actions").and_then(Value::as_array),
    );
    lines.push(String::new());
    lines.push(format!(
        "More details: {}",
        report
            .get("details_command")
            .and_then(Value::as_str)
            .unwrap_or("ax-engine-bench doctor")
    ));
    lines.join("\n")
}

fn format_cpu_cores(cpu_cores: Option<&Value>) -> String {
    let Some(cpu_cores) = cpu_cores else {
        return "unknown".to_string();
    };
    if let Some(summary) = cpu_cores.get("summary").and_then(Value::as_str) {
        return summary.to_string();
    }
    let physical = cpu_cores.get("physical").and_then(Value::as_u64);
    let logical = cpu_cores.get("logical").and_then(Value::as_u64);
    let performance = cpu_cores.get("performance").and_then(Value::as_u64);
    let efficiency = cpu_cores.get("efficiency").and_then(Value::as_u64);
    match (physical, logical, performance, efficiency) {
        (Some(physical), Some(logical), Some(performance), Some(efficiency)) => {
            format!("{physical} physical / {logical} logical ({performance}P+{efficiency}E)")
        }
        (Some(physical), Some(logical), _, _) => format!("{physical} physical / {logical} logical"),
        (Some(physical), _, _, _) => format!("{physical} physical"),
        _ => "unknown".to_string(),
    }
}

fn append_string_array(lines: &mut Vec<String>, values: Option<&Vec<Value>>) {
    let Some(values) = values else {
        lines.push("  none".to_string());
        return;
    };
    if values.is_empty() {
        lines.push("  none".to_string());
    } else {
        for value in values {
            if let Some(text) = value.as_str() {
                lines.push(format!("  {text}"));
            }
        }
    }
}

#[derive(Debug)]
struct ServeArgs {
    model: OsString,
    host: String,
    port: String,
    hf_cache_root: Option<OsString>,
    download: bool,
    offline: bool,
    dry_run: bool,
    json: bool,
    passthrough: Vec<OsString>,
}

fn looks_like_repo_reference(value: &str) -> bool {
    value.contains('/') || value.contains("huggingface.co") || value.contains("hf.co")
}

fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut current = Vec::with_capacity(right.len() + 1);
        current.push(left_index + 1);
        for (right_index, right_character) in right.iter().enumerate() {
            let insertion = current[right_index] + 1;
            let deletion = previous[right_index + 1] + 1;
            let substitution =
                previous[right_index] + usize::from(left_character != *right_character);
            current.push(insertion.min(deletion).min(substitution));
        }
        previous = current;
    }
    previous[right.len()]
}

fn unknown_serve_target_message(target: &str) -> String {
    let normalized = normalize_alias(target);
    let threshold = normalized.chars().count().div_ceil(4).max(2);
    let mut scored = MODEL_PROFILES
        .iter()
        .flat_map(|profile| profile.aliases.iter().copied())
        .map(|alias| (edit_distance(&normalized, &normalize_alias(alias)), alias))
        .filter(|(distance, _)| *distance <= threshold)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(right.1)));
    scored.dedup_by(|left, right| left.1 == right.1);
    let suggestions = scored
        .into_iter()
        .take(3)
        .map(|(_, alias)| format!("{alias:?}"))
        .collect::<Vec<_>>();
    let suggestion = if suggestions.is_empty() {
        String::new()
    } else {
        format!("; did you mean {}?", suggestions.join(", "))
    };
    format!(
        "unknown model alias or missing local directory: {target:?}{suggestion}; pass a local \
         directory, a Hugging Face owner/repo reference, or run `ax-engine download --list`"
    )
}

fn snapshot_has_complete_weights(snapshot: &Path) -> bool {
    let index_path = snapshot.join("model.safetensors.index.json");
    if index_path.is_file() {
        let Ok(bytes) = fs::read(index_path) else {
            return false;
        };
        let Ok(payload) = serde_json::from_slice::<Value>(&bytes) else {
            return false;
        };
        let Some(weight_map) = payload.get("weight_map").and_then(Value::as_object) else {
            return false;
        };
        if weight_map.is_empty() {
            return false;
        }
        return weight_map.values().all(|value| {
            let Some(shard) = value.as_str() else {
                return false;
            };
            let relative = Path::new(shard);
            !shard.is_empty()
                && relative
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                && snapshot.join(relative).is_file()
        });
    }
    fs::read_dir(snapshot).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            entry
                .path()
                .extension()
                .and_then(OsStr::to_str)
                .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        })
    })
}

fn cached_snapshot_path(
    repo_id: &str,
    revision: Option<&str>,
    cache_root: &Path,
) -> Option<PathBuf> {
    let repo_cache = cache_root.join(format!("models--{}", repo_id.replace('/', "--")));
    let snapshots = repo_cache.join("snapshots");
    let mut candidates = Vec::new();
    if let Some(revision) = revision {
        let ref_path = repo_cache.join("refs").join(revision);
        if let Ok(resolved) = fs::read_to_string(ref_path) {
            let resolved = resolved.trim();
            if !resolved.is_empty() {
                candidates.push(snapshots.join(resolved));
            }
        }
        candidates.push(snapshots.join(revision));
    } else {
        if let Ok(resolved) = fs::read_to_string(repo_cache.join("refs").join("main")) {
            let resolved = resolved.trim();
            if !resolved.is_empty() {
                candidates.push(snapshots.join(resolved));
            }
        }
        if let Ok(entries) = fs::read_dir(&snapshots) {
            let mut by_modified = entries
                .flatten()
                .filter_map(|entry| {
                    let path = entry.path();
                    let modified = entry.metadata().ok()?.modified().ok()?;
                    path.is_dir().then_some((modified, path))
                })
                .collect::<Vec<_>>();
            by_modified.sort_by_key(|(modified, _)| *modified);
            candidates.extend(by_modified.into_iter().rev().map(|(_, path)| path));
        }
    }

    candidates.into_iter().find_map(|path| {
        if path.is_symlink() || !path.join("config.json").is_file() {
            return None;
        }
        snapshot_has_complete_weights(&path).then(|| absolute_path(&path))
    })
}

fn preview_snapshot_path(repo_id: &str, revision: Option<&str>, cache_root: &Path) -> PathBuf {
    if let Some(path) = cached_snapshot_path(repo_id, revision, cache_root) {
        return path;
    }
    if let Some(revision) = revision
        && revision.len() >= 40
        && revision
            .chars()
            .all(|character| character.is_ascii_hexdigit())
    {
        return cache_root
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots")
            .join(revision);
    }
    PathBuf::from(format!("<resolved-hf-snapshot:{repo_id}>"))
}

fn cmd_serve(args: &[OsString]) -> Result<u8, String> {
    let args = parse_serve_args(args)?;
    let server = find_executable("ax-engine-server");
    let target = args.model.to_string_lossy();
    let target_path = expand_home(&target);
    let mut argv = vec![
        OsString::from("--host"),
        OsString::from(&args.host),
        OsString::from("--port"),
        OsString::from(&args.port),
    ];

    let resolved = if target_path.exists() {
        let model = absolute_path(&target_path);
        argv.extend([
            OsString::from("--mlx"),
            OsString::from("--mlx-model-artifacts-dir"),
            model.clone().into_os_string(),
        ]);
        json!({
            "kind": "local_dir",
            "model": model.to_string_lossy(),
        })
    } else {
        let profile = profile_for_model(&target);
        if profile.is_none() && !looks_like_repo_reference(&target) {
            return Err(unknown_serve_target_message(&target));
        }
        let (repo_id, profile, revision) = download_repo_id(&target, profile)?;
        let preset = profile.and_then(|profile| profile.preset);
        if args.dry_run {
            let cache_root = args
                .hf_cache_root
                .as_ref()
                .map(|root| expand_home(&root.to_string_lossy()))
                .unwrap_or_else(default_hf_cache_root);
            let cached = cached_snapshot_path(&repo_id, revision.as_deref(), &cache_root).is_some();
            let model_dir = preview_snapshot_path(&repo_id, revision.as_deref(), &cache_root);
            argv.push(OsString::from("--mlx"));
            if let Some(preset) = preset {
                argv.extend([OsString::from("--preset"), OsString::from(preset)]);
            }
            argv.extend([
                OsString::from("--mlx-model-artifacts-dir"),
                model_dir.clone().into_os_string(),
            ]);
            json!({
                "kind": "model_resolution_plan",
                "model": target.as_ref(),
                "repo_id": repo_id,
                "revision": revision,
                "preset": preset,
                "certification": profile.and_then(profile_certification),
                "path": model_dir.to_string_lossy(),
                "resolution": "local_cache_then_download",
                "download": {
                    "required": !cached,
                    "offline": args.offline,
                    "compatibility_flag": args.download,
                    "dry_run": true,
                },
            })
        } else {
            let (code, mut summary, stderr) = run_download_summary(
                &target,
                None,
                false,
                profile,
                DownloadProgress::Bar,
                args.offline,
                args.hf_cache_root.as_deref(),
            )?;
            if code != 0 || summary.get("status").and_then(Value::as_str) != Some("ready") {
                if !stderr.is_empty() {
                    eprint!("{stderr}");
                }
                if !summary.is_null() {
                    print_download_summary(&summary);
                }
                let hint = if args.offline {
                    "disable --offline or download the pinned snapshot first".to_string()
                } else {
                    format!("run: ax-engine download {target}")
                };
                return Err(format!(
                    "model resolution did not produce ready AX artifacts; {hint}"
                ));
            }
            let Some(dest) = summary.get("dest").and_then(Value::as_str) else {
                return Err("download helper returned ready status without a dest".into());
            };
            let model_dir = absolute_path(&expand_home(dest));
            argv.push(OsString::from("--mlx"));
            if let Some(preset) = preset {
                argv.extend([OsString::from("--preset"), OsString::from(preset)]);
                summary["preset"] = json!(preset);
            }
            argv.extend([
                OsString::from("--mlx-model-artifacts-dir"),
                model_dir.clone().into_os_string(),
            ]);
            json!({
                "kind": "resolved_snapshot",
                "model": target.as_ref(),
                "repo_id": summary.get("repo_id").cloned().unwrap_or_else(|| json!(repo_id)),
                "revision": summary.get("revision").cloned().unwrap_or_else(|| json!(revision)),
                "certification": profile.and_then(profile_certification),
                "path": model_dir.to_string_lossy(),
                "preset": preset,
                "resolution": "local_cache_then_download",
                "download": {
                    "status": summary.get("status").cloned().unwrap_or(Value::Null),
                    "manifest_present": summary.get("manifest_present").cloned().unwrap_or(Value::Null),
                    "offline": args.offline,
                    "compatibility_flag": args.download,
                },
            })
        }
    };

    argv.extend(args.passthrough);
    let server_argv = std::iter::once(server.as_os_str().to_string_lossy().to_string())
        .chain(argv.iter().map(|arg| arg.to_string_lossy().to_string()))
        .collect::<Vec<_>>();
    let plan = json!({
        "schema_version": "ax.local_serve_plan.v1",
        "command": "serve",
        "input": target.as_ref(),
        "resolved": resolved,
        "server": {
            "url": format!("http://{}:{}", args.host, args.port),
            "argv": server_argv,
        },
    });

    if args.json {
        print_json(&plan)?;
    } else {
        println!("AX Engine server: http://{}:{}", args.host, args.port);
        println!("Command:");
        println!("  {}", server_argv.join(" "));
    }
    if args.dry_run {
        Ok(0)
    } else {
        exec_or_status(server, &argv)
    }
}

fn parse_serve_args(args: &[OsString]) -> Result<ServeArgs, String> {
    let mut before_separator = Vec::new();
    let mut passthrough = Vec::new();
    let mut after_separator = false;
    for arg in args {
        if !after_separator && arg == "--" {
            after_separator = true;
            continue;
        }
        if after_separator {
            passthrough.push(arg.clone());
        } else {
            before_separator.push(arg.clone());
        }
    }

    let mut model = None;
    let mut host = "127.0.0.1".to_string();
    let mut port = "31418".to_string();
    let mut hf_cache_root = None;
    let mut download = false;
    let mut offline = false;
    let mut dry_run = false;
    let mut json = false;
    let mut index = 0;
    while index < before_separator.len() {
        let arg = before_separator[index].to_string_lossy();
        match arg.as_ref() {
            "--host" => {
                index += 1;
                host = require_value(&before_separator, index, "--host")?;
            }
            "--port" => {
                index += 1;
                port = require_value(&before_separator, index, "--port")?;
            }
            "--hf-cache-root" => {
                index += 1;
                hf_cache_root = Some(
                    before_separator
                        .get(index)
                        .ok_or_else(|| "--hf-cache-root requires a value".to_string())?
                        .clone(),
                );
            }
            "--download" => download = true,
            "--offline" | "--local-only" => offline = true,
            "--dry-run" => dry_run = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => return Err(format!("unknown serve option: {flag}")),
            _ => {
                if model.replace(before_separator[index].clone()).is_some() {
                    return Err("serve accepts exactly one model argument".into());
                }
            }
        }
        index += 1;
    }
    let model = model.ok_or_else(|| "serve requires a model directory or alias".to_string())?;
    Ok(ServeArgs {
        model,
        host,
        port,
        hf_cache_root,
        download,
        offline,
        dry_run,
        json,
        passthrough,
    })
}

fn cmd_models(args: &[OsString]) -> Result<u8, String> {
    let Some(command) = args.first() else {
        return Err(models_usage());
    };
    match command.to_string_lossy().as_ref() {
        "list" => cmd_models_list(&args[1..]),
        "info" => cmd_models_info(&args[1..]),
        "rm" => cmd_models_rm(&args[1..]),
        "--help" | "-h" => {
            println!("{}", models_usage());
            Ok(0)
        }
        unknown => Err(format!(
            "unknown models command: {unknown}\n\n{}",
            models_usage()
        )),
    }
}

fn models_usage() -> String {
    "Usage:\n  ax-engine models list [--models-dir <path>] [--json]\n  ax-engine models info <alias-or-path> [--json]\n  ax-engine models rm <path> [--dry-run] [--yes] [--json]".to_string()
}

fn cmd_models_list(args: &[OsString]) -> Result<u8, String> {
    let mut models_dir = env::var_os("AX_ENGINE_MODELS_DIR").map(PathBuf::from);
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--models-dir" => {
                index += 1;
                models_dir = Some(expand_home(&require_value(args, index, "--models-dir")?));
            }
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models list option: {flag}"));
            }
            _ => return Err("models list does not accept positional arguments".into()),
        }
        index += 1;
    }

    let payload = models_list_payload(models_dir.as_deref());
    if json_output {
        print_json(&payload)?;
    } else {
        println!("{}", format_models_list(&payload));
    }
    Ok(0)
}

fn cmd_models_info(args: &[OsString]) -> Result<u8, String> {
    let mut target = None;
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models info option: {flag}"));
            }
            _ => {
                if target.replace(arg.to_string()).is_some() {
                    return Err("models info accepts exactly one alias or path".into());
                }
            }
        }
        index += 1;
    }
    let target = target.ok_or_else(|| "models info requires an alias or path".to_string())?;
    let payload = model_info_payload(&target)?;
    if json_output {
        print_json(&payload)?;
    } else {
        println!("{}", format_model_info(&payload));
    }
    Ok(0)
}

fn cmd_models_rm(args: &[OsString]) -> Result<u8, String> {
    let mut target = None;
    let mut dry_run = false;
    let mut yes = false;
    let mut json_output = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--dry-run" => dry_run = true,
            "--yes" => yes = true,
            "--json" => json_output = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown models rm option: {flag}"));
            }
            _ => {
                if target.replace(arg.to_string()).is_some() {
                    return Err("models rm accepts exactly one path".into());
                }
            }
        }
        index += 1;
    }
    let target = target.ok_or_else(|| "models rm requires a local model path".to_string())?;
    if profile_for_model(&target).is_some() {
        return Err(
            "models rm refuses aliases; pass an explicit local model directory path".into(),
        );
    }
    let path = absolute_path(&expand_home(&target));
    let effective_dry_run = dry_run || !yes;
    let report = validate_model_rm_target(&path, effective_dry_run)?;
    if !effective_dry_run {
        fs::remove_dir_all(&path)
            .map_err(|err| format!("failed to remove {}: {err}", path.display()))?;
    }
    let payload = json!({
        "schema_version": "ax.models_rm.v1",
        "command": "models rm",
        "path": path.to_string_lossy(),
        "dry_run": effective_dry_run,
        "removed": !effective_dry_run,
        "safety": report,
    });
    if json_output {
        print_json(&payload)?;
    } else if !effective_dry_run {
        println!("Removed {}", path.display());
    } else {
        println!("Dry run: would remove {}", path.display());
        println!("Pass --yes to remove this local artifact directory.");
    }
    Ok(0)
}

fn models_list_payload(models_dir: Option<&Path>) -> Value {
    json!({
        "schema_version": "ax.models_list.v1",
        "supported_aliases": MODEL_PROFILES.iter().map(model_profile_payload).collect::<Vec<_>>(),
        "local_artifacts": models_dir.map(local_model_artifacts_payload).unwrap_or_else(|| {
            json!({
                "source": "not_selected",
                "env": "AX_ENGINE_MODELS_DIR",
                "items": [],
            })
        }),
    })
}

fn model_profile_payload(profile: &ModelProfile) -> Value {
    json!({
        "kind": "supported_alias",
        "label": profile.label,
        "repo_id": profile.repo_id,
        "revision": profile_revision(*profile),
        "certification": profile_certification(*profile),
        "preset": profile.preset,
        "downloadable": profile.is_downloadable(),
        "aliases": profile.aliases,
    })
}

fn local_model_artifacts_payload(root: &Path) -> Value {
    let root = absolute_path(root);
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir(&root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir()
                && let Some(item) = local_model_artifact_payload(&path)
            {
                items.push(item);
            }
        }
    }
    json!({
        "source": "models_dir",
        "path": root.to_string_lossy(),
        "items": items,
    })
}

fn local_model_artifact_payload(path: &Path) -> Option<Value> {
    let manifest_present = path.join("model-manifest.json").is_file();
    let config_present = path.join("config.json").is_file();
    if !manifest_present && !config_present {
        return None;
    }
    Some(json!({
        "kind": "local_artifact",
        "path": absolute_path(path).to_string_lossy(),
        "manifest_present": manifest_present,
        "config_present": config_present,
    }))
}

fn model_info_payload(target: &str) -> Result<Value, String> {
    if let Some(profile) = profile_for_model(target) {
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "supported_alias",
            "profile": model_profile_payload(&profile),
        }));
    }
    let path = expand_home(target);
    if path.exists() {
        let path = absolute_path(&path);
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "local_artifact",
            "path": path.to_string_lossy(),
            "manifest_present": path.join("model-manifest.json").is_file(),
            "config_present": path.join("config.json").is_file(),
            "hf_cache_path": is_hf_cache_path(&path),
        }));
    }
    if target.contains('/') {
        return Ok(json!({
            "schema_version": "ax.models_info.v1",
            "query": target,
            "kind": "repo_id",
            "repo_id": target,
            "managed_alias": false,
        }));
    }
    Err(format!(
        "unknown model alias or missing local path: {target:?}; run `ax-engine models list`"
    ))
}

fn validate_model_rm_target(path: &Path, dry_run: bool) -> Result<Value, String> {
    if !path.exists() {
        return Err(format!(
            "models rm target does not exist: {}",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "models rm target is not a directory: {}",
            path.display()
        ));
    }
    if is_hf_cache_path(path) {
        return Err(format!(
            "models rm refuses Hugging Face cache paths; remove cache entries with huggingface-cli instead: {}",
            path.display()
        ));
    }
    if path.parent().is_none() {
        return Err("models rm refuses filesystem root".into());
    }
    let manifest_present = path.join("model-manifest.json").is_file();
    let config_present = path.join("config.json").is_file();
    if !manifest_present && !config_present {
        return Err(format!(
            "models rm target does not look like an AX/MLX artifact directory: {}",
            path.display()
        ));
    }
    Ok(json!({
        "dry_run": dry_run,
        "manifest_present": manifest_present,
        "config_present": config_present,
        "hf_cache_path": false,
    }))
}

fn is_hf_cache_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.contains("/huggingface/hub/")
        || text.contains("/.cache/huggingface/")
        || path.components().any(|component| {
            component
                .as_os_str()
                .to_string_lossy()
                .starts_with("models--")
        })
}

fn format_models_list(payload: &Value) -> String {
    let mut lines = vec!["Supported aliases:".to_string()];
    if let Some(targets) = payload.get("supported_aliases").and_then(Value::as_array) {
        for target in targets {
            lines.push(format!(
                "  - {} -> {}",
                target
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                target
                    .get("repo_id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
            ));
        }
    }
    lines.push("Local artifacts:".into());
    let local = &payload["local_artifacts"];
    if local.get("source").and_then(Value::as_str) == Some("not_selected") {
        lines.push("  - set AX_ENGINE_MODELS_DIR or pass --models-dir".into());
    } else if let Some(items) = local.get("items").and_then(Value::as_array) {
        if items.is_empty() {
            lines.push("  - none found".into());
        } else {
            for item in items {
                lines.push(format!(
                    "  - {}",
                    item.get("path")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ));
            }
        }
    }
    lines.join("\n")
}

fn format_model_info(payload: &Value) -> String {
    match payload.get("kind").and_then(Value::as_str) {
        Some("supported_alias") => {
            let profile = &payload["profile"];
            format!(
                "Supported alias: {}\nRepo: {}\nPreset: {}",
                profile
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                profile
                    .get("repo_id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown"),
                profile
                    .get("preset")
                    .and_then(Value::as_str)
                    .unwrap_or("none")
            )
        }
        Some("local_artifact") => format!(
            "Local artifact: {}\nmodel-manifest.json: {}\nconfig.json: {}\nHF cache path: {}",
            payload
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            payload
                .get("manifest_present")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            payload
                .get("config_present")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            payload
                .get("hf_cache_path")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        ),
        Some("repo_id") => format!(
            "Repo id: {}\nManaged alias: false",
            payload
                .get("repo_id")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        ),
        _ => "Unknown model".to_string(),
    }
}

#[derive(Debug)]
struct DownloadArgs {
    model: Option<String>,
    dest: Option<String>,
    force: bool,
    local_only: bool,
    list: bool,
    json: bool,
    progress: bool,
}

#[derive(Debug)]
struct DownloadMtpArgs {
    model: String,
    output: Option<String>,
    force: bool,
    quantize: Option<String>,
    mtp_depth_max: Option<String>,
    group_size: String,
    fair_base_only: bool,
    json: bool,
    progress: bool,
}

#[derive(Clone, Copy)]
enum DownloadProgress {
    Quiet,
    Json,
    Bar,
}

fn cmd_download(args: &[OsString]) -> Result<u8, String> {
    let progress_requested = args.iter().any(|arg| arg == OsStr::new("--progress-json"));
    let args = match parse_download_args(args) {
        Ok(args) => args,
        Err(error) if progress_requested => {
            eprintln!("{error}");
            print_download_progress_terminal(&download_error_summary(None, &error))?;
            return Ok(2);
        }
        Err(error) => return Err(error),
    };
    match run_download(&args) {
        Ok(code) => Ok(code),
        Err(error) if args.progress => {
            eprintln!("{error}");
            print_download_progress_terminal(&download_error_summary(
                args.model.as_deref(),
                &error,
            ))?;
            Ok(2)
        }
        Err(error) => Err(error),
    }
}

fn run_download(args: &DownloadArgs) -> Result<u8, String> {
    if args.force && args.local_only {
        return Err("download --force cannot be combined with --local-only".into());
    }
    if args.list {
        if args.progress {
            return Err("download --progress-json cannot be combined with --list".into());
        }
        if args.json {
            print_json(&download_options_payload())?;
        } else {
            println!("{}", format_download_options());
        }
        return Ok(0);
    }
    let Some(model) = args.model.as_deref() else {
        if args.progress {
            return Err(
                "download --progress-json requires a model alias or Hugging Face repo id".into(),
            );
        }
        if args.json {
            print_json(&download_options_payload())?;
        } else {
            println!("missing model alias or repo id\n");
            println!("{}", format_download_options());
        }
        return Ok(2);
    };

    ensure_download_python_deps()?;
    let profile = profile_for_model(model);
    let (code, summary, stderr) = run_download_summary(
        model,
        args.dest.as_deref(),
        args.force,
        profile,
        if args.progress {
            DownloadProgress::Json
        } else {
            DownloadProgress::Quiet
        },
        args.local_only,
        None,
    )?;
    if args.progress {
        if !stderr.is_empty() {
            eprint!("{stderr}");
        }
        if summary.is_null() {
            return Err("download helper did not emit an ax.download_model.v1 summary".into());
        }
        // `--progress-json` is an NDJSON contract: live progress events have
        // already been forwarded, and this enriched single-line summary is
        // the one terminal record. This also takes precedence over `--json`
        // so combining the flags cannot append a duplicate pretty document.
        print_download_progress_terminal(&summary)?;
        return Ok(code);
    }
    if args.json {
        if !summary.is_null() {
            print_json(&summary)?;
        }
        if !stderr.is_empty() {
            eprint!("{stderr}");
        }
        return Ok(code);
    }
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    if summary.is_null() {
        return Err("download helper did not emit an ax.download_model.v1 summary".into());
    }
    print_download_summary(&summary);
    Ok(code)
}

fn cmd_download_mtp(args: &[OsString]) -> Result<u8, String> {
    let progress_requested = args.iter().any(|arg| arg == OsStr::new("--progress-json"));
    let args = match parse_download_mtp_args(args) {
        Ok(args) => args,
        Err(error) if progress_requested => {
            eprintln!("{error}");
            print_download_mtp_progress_terminal(&download_mtp_error_summary(
                None,
                "invalid_arguments",
                &error,
            ))?;
            return Ok(2);
        }
        Err(error) => return Err(error),
    };
    match run_download_mtp(&args) {
        Ok(code) => Ok(code),
        Err(error) if args.progress => {
            eprintln!("{error}");
            print_download_mtp_progress_terminal(&download_mtp_error_summary(
                Some(&args.model),
                "error",
                &error,
            ))?;
            Ok(2)
        }
        Err(error) => Err(error),
    }
}

fn run_download_mtp(args: &DownloadMtpArgs) -> Result<u8, String> {
    ensure_download_python_deps()?;
    let target = mtp_download_target_for_model(&args.model)
        .ok_or_else(|| format_unknown_download_mtp_target(&args.model))?;
    let (download_code, download_summary, download_stderr) = run_download_summary(
        target.repo_id,
        None,
        args.force,
        None,
        if args.progress {
            DownloadProgress::Json
        } else {
            DownloadProgress::Quiet
        },
        false,
        None,
    )?;
    if !download_stderr.is_empty() {
        eprint!("{download_stderr}");
    }
    if download_code != 0 || download_summary.get("status").and_then(Value::as_str) != Some("ready")
    {
        let terminal = json!({
            "schema_version": "ax.download_mtp.v1",
            "command": "download-mtp",
            "base_model": args.model,
            "repo_id": target.repo_id,
            "download": download_summary,
            "status": "download_failed",
        });
        if args.progress {
            print_download_mtp_progress_terminal(&terminal)?;
            return Ok(download_code.max(1));
        }
        if args.json && !terminal["download"].is_null() {
            print_json(&terminal)?;
            return Ok(download_code.max(1));
        }
        if !terminal["download"].is_null() {
            print_download_summary(&terminal["download"]);
        }
        return Err(format!(
            "base model download did not produce ready AX artifacts; run: ax-engine download {}",
            args.model
        ));
    }
    let Some(base_dir) = download_summary
        .get("dest")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return Err("download helper returned ready status without a dest".into());
    };
    if !args.json && !args.progress {
        print_download_summary(&download_summary);
    }

    match target.kind {
        MtpDownloadKind::QwenSidecar { mtp_source } => {
            let convert_args = ConvertArgs {
                base_model: base_dir.clone(),
                mtp_source: mtp_source.to_string(),
                output: args.output.clone(),
                quantize: args.quantize.clone(),
                mtp_depth_max: args.mtp_depth_max.clone(),
                group_size: args.group_size.clone(),
                fair_base_only: args.fair_base_only,
                json: args.json,
            };
            run_convert_mtplx(
                &convert_args,
                "download-mtp",
                "ax.download_mtp.v1",
                Some(download_summary),
                args.progress,
            )
        }
        MtpDownloadKind::GemmaAssistant { .. } => {
            run_download_gemma_assistant_mtp(target, args, &base_dir, target.kind, download_summary)
        }
        MtpDownloadKind::DirectOnly { reason } => {
            let terminal = json!({
                "schema_version": "ax.download_mtp.v1",
                "command": "download-mtp",
                "status": "direct_only",
                "base_model": &args.model,
                "repo_id": target.repo_id,
                "output_dir": base_dir,
                "reason": reason,
                "download": download_summary,
            });
            if args.progress {
                print_download_mtp_progress_terminal(&terminal)?;
            } else if args.json {
                print_json(&terminal)?;
            } else {
                println!("MTP status: direct-only");
                println!("{reason}");
                println!("Next:");
                println!("  ax-engine serve {base_dir}");
            }
            Ok(0)
        }
    }
}

fn parse_download_args(args: &[OsString]) -> Result<DownloadArgs, String> {
    let mut model = None;
    let mut dest = None;
    let mut force = false;
    let mut local_only = false;
    let mut list = false;
    let mut json = false;
    let mut progress = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--dest" => {
                index += 1;
                dest = Some(require_value(args, index, "--dest")?);
            }
            "--force" => force = true,
            "--local-only" => local_only = true,
            "--list" => list = true,
            "--json" => json = true,
            "--progress-json" => progress = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown download option: {flag}"));
            }
            _ => {
                if model.replace(arg.to_string()).is_some() {
                    return Err("download accepts at most one model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(DownloadArgs {
        model,
        dest,
        force,
        local_only,
        list,
        json,
        progress,
    })
}

fn parse_download_mtp_args(args: &[OsString]) -> Result<DownloadMtpArgs, String> {
    let mut model = None;
    let mut output = None;
    let mut force = false;
    let mut quantize = None;
    let mut mtp_depth_max = None;
    let mut group_size = "64".to_string();
    let mut fair_base_only = false;
    let mut json = false;
    let mut progress = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--output" => {
                index += 1;
                output = Some(require_value(args, index, "--output")?);
            }
            "--force" => force = true,
            "--quantize" => {
                index += 1;
                let value = require_value(args, index, "--quantize")?;
                if value != "4" && value != "8" {
                    return Err("--quantize must be 4 or 8".into());
                }
                quantize = Some(value);
            }
            "--mtp-depth-max" => {
                index += 1;
                mtp_depth_max = Some(require_value(args, index, "--mtp-depth-max")?);
            }
            "--group-size" => {
                index += 1;
                group_size = require_value(args, index, "--group-size")?;
            }
            "--fair-base-only" => fair_base_only = true,
            "--json" => json = true,
            "--progress-json" => progress = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown download-mtp option: {flag}"));
            }
            _ => {
                if model.replace(arg.to_string()).is_some() {
                    return Err("download-mtp accepts exactly one model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(DownloadMtpArgs {
        model: model.ok_or_else(|| "download-mtp requires a model".to_string())?,
        output,
        force,
        quantize,
        mtp_depth_max,
        group_size,
        fair_base_only,
        json,
        progress,
    })
}

fn run_download_summary(
    model: &str,
    dest: Option<&str>,
    force: bool,
    profile: Option<ModelProfile>,
    progress: DownloadProgress,
    local_only: bool,
    hf_cache_root: Option<&OsStr>,
) -> Result<(u8, Value, String), String> {
    let (repo_id, profile, revision) = download_repo_id(model, profile)?;
    let helper = find_helper(
        "AX_ENGINE_DOWNLOAD_HELPER",
        "ax-engine-download-model.py",
        "download_model.py",
    )?;
    let mut command = Command::new(python());
    command.arg(helper).arg(&repo_id).arg("--json");
    if let Some(revision) = &revision {
        // The resolved revision is already percent-decoded; the helper decodes
        // its --revision once more, so re-escape literal `%` to hand it
        // exactly this value.
        command.arg(helper_value_option(
            "--revision",
            &revision.replace('%', "%25"),
        ));
    }
    match progress {
        DownloadProgress::Quiet => {}
        DownloadProgress::Json => {
            command.arg("--progress-json");
        }
        DownloadProgress::Bar => {
            command.arg("--progress-bar");
        }
    }
    if let Some(dest) = dest {
        command.arg(helper_value_option("--dest", dest));
    }
    if force {
        command.arg("--force");
    }
    if local_only {
        command.arg("--local-only");
    }
    if let Some(root) = hf_cache_root {
        command.env("HF_HUB_CACHE", root);
    }
    let (code, stdout, stderr) = if matches!(progress, DownloadProgress::Json) {
        run_streaming_progress(command)?
    } else {
        let output = command
            .stdout(Stdio::piped())
            .stderr(if matches!(progress, DownloadProgress::Bar) {
                Stdio::inherit()
            } else {
                Stdio::piped()
            })
            .output()
            .map_err(|err| format!("failed to run download helper: {err}"))?;
        (
            output.status.code().unwrap_or(1).try_into().unwrap_or(1),
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    };
    let mut summary = parse_summary_json(&stdout).unwrap_or(Value::Null);
    if let Value::Object(map) = &mut summary {
        map.insert("input".into(), json!(model));
        if let Some(revision) = &revision {
            map.insert("revision".into(), json!(revision));
        }
        if let Some(profile) = profile {
            map.insert("alias".into(), json!(profile.label));
            map.insert(
                "certification".into(),
                json!(profile_certification(profile)),
            );
            if let Some(preset) = profile.preset {
                map.insert("preset".into(), json!(preset));
            }
        }
    }
    Ok((code, summary, stderr))
}

fn helper_value_option(flag: &str, value: &str) -> OsString {
    OsString::from(format!("{flag}={value}"))
}

/// Run the download helper forwarding `{"event":"progress",...}` stdout lines
/// as they arrive (so a parent process observing our stdout sees live phase
/// updates), while buffering the helper's final summary. The caller enriches
/// and emits that summary as the one terminal NDJSON record.
fn run_streaming_progress(mut command: Command) -> Result<(u8, String, String), String> {
    use std::io::{BufRead, BufReader, Write};

    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to run download helper: {err}"))?;
    let stderr_handle = child.stderr.take().map(|pipe| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let mut reader = BufReader::new(pipe);
            let _ = std::io::Read::read_to_string(&mut reader, &mut buf);
            buf
        })
    });
    let mut stdout_text = String::new();
    if let Some(pipe) = child.stdout.take() {
        for line in BufReader::new(pipe).lines().map_while(Result::ok) {
            if line.contains("\"event\"")
                && serde_json::from_str::<Value>(&line)
                    .ok()
                    .and_then(|v| v.get("event").and_then(Value::as_str).map(String::from))
                    .as_deref()
                    == Some("progress")
            {
                println!("{line}");
                let _ = std::io::stdout().flush();
            }
            stdout_text.push_str(&line);
            stdout_text.push('\n');
        }
    }
    let status = child
        .wait()
        .map_err(|err| format!("failed to wait for download helper: {err}"))?;
    let stderr_text = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    Ok((
        status.code().unwrap_or(1).try_into().unwrap_or(1),
        stdout_text,
        stderr_text,
    ))
}

fn run_download_gemma_assistant_mtp(
    target: MtpDownloadTarget,
    args: &DownloadMtpArgs,
    base_dir: &str,
    kind: MtpDownloadKind,
    target_download: Value,
) -> Result<u8, String> {
    let MtpDownloadKind::GemmaAssistant {
        assistant_repo_id,
        target_model_id,
        assistant_model_id,
        max_depth,
    } = kind
    else {
        return Err("internal error: expected Gemma assistant MTP target".into());
    };
    let (assistant_code, assistant_summary, assistant_stderr) = run_download_summary(
        assistant_repo_id,
        None,
        args.force,
        None,
        if args.progress {
            DownloadProgress::Json
        } else {
            DownloadProgress::Quiet
        },
        false,
        None,
    )?;
    if !assistant_stderr.is_empty() {
        eprint!("{assistant_stderr}");
    }
    if !assistant_download_usable(assistant_code, &assistant_summary) {
        let terminal = json!({
            "schema_version": "ax.download_mtp.v1",
            "command": "download-mtp",
            "status": "assistant_download_failed",
            "base_model": &args.model,
            "repo_id": target.repo_id,
            "assistant_repo_id": assistant_repo_id,
            "download": target_download,
            "assistant_download": assistant_summary,
        });
        if args.progress {
            print_download_mtp_progress_terminal(&terminal)?;
            return Ok(assistant_code.max(1));
        }
        if args.json && !terminal["assistant_download"].is_null() {
            print_json(&terminal)?;
            return Ok(assistant_code.max(1));
        }
        if !terminal["assistant_download"].is_null() {
            print_download_summary(&terminal["assistant_download"]);
        }
        return Err(format!(
            "assistant model download did not produce ready AX artifacts; run: ax-engine download {assistant_repo_id}"
        ));
    }
    let Some(assistant_dir) = assistant_summary.get("dest").and_then(Value::as_str) else {
        return Err("assistant download helper returned ready status without a dest".into());
    };
    if !args.json && !args.progress {
        print_download_summary(&assistant_summary);
    }

    let helper = find_helper(
        "AX_ENGINE_PREPARE_GEMMA4_ASSISTANT_MTP_HELPER",
        "ax-engine-prepare-gemma4-assistant-mtp.py",
        "prepare_gemma4_assistant_mtp.py",
    )?;
    let default_depth = max_depth.to_string();
    let depth = args.mtp_depth_max.as_deref().unwrap_or(&default_depth);
    let mut prepare_cmd = Command::new(python());
    prepare_cmd
        .arg(&helper)
        .args(["--target", base_dir, "--assistant", assistant_dir])
        .args(["--target-model-id", target_model_id])
        .args(["--assistant-model-id", assistant_model_id])
        .args(["--max-depth", depth]);
    if let Some(output) = &args.output {
        prepare_cmd.args(["--output", output]);
    } else {
        let output = default_gemma_assistant_mtp_output(target.repo_id);
        prepare_cmd.arg("--output").arg(output);
    }
    let prepare_output = prepare_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run prepare_gemma4_assistant_mtp helper: {err}"))?;
    let prepare_stdout = String::from_utf8_lossy(&prepare_output.stdout).into_owned();
    let prepare_stderr = String::from_utf8_lossy(&prepare_output.stderr).into_owned();
    if !args.json && !args.progress {
        print!("{prepare_stdout}");
        eprint!("{prepare_stderr}");
    } else if args.progress && !prepare_stderr.is_empty() {
        eprint!("{prepare_stderr}");
    }
    if !prepare_output.status.success() {
        if args.json && !args.progress {
            eprint!("{prepare_stderr}");
        }
        let code: u8 = prepare_output
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1);
        if args.progress {
            print_download_mtp_progress_terminal(&json!({
                "schema_version": "ax.download_mtp.v1",
                "command": "download-mtp",
                "status": "prepare_failed",
                "base_model": &args.model,
                "repo_id": target.repo_id,
                "assistant_repo_id": assistant_repo_id,
                "exit_code": code,
                "download": target_download,
                "assistant_download": assistant_summary,
            }))?;
        }
        return Ok(code.max(1));
    }
    let output_dir =
        parse_output_dir(&prepare_stdout, args.output.as_deref()).ok_or_else(|| {
            "prepare_gemma4_assistant_mtp.py succeeded but output dir could not be determined"
                .to_string()
        })?;

    let terminal = json!({
        "schema_version": "ax.download_mtp.v1",
        "command": "download-mtp",
        "status": "ready",
        "kind": "gemma_assistant_mtp",
        "base_model": &args.model,
        "repo_id": target.repo_id,
        "assistant_repo_id": assistant_repo_id,
        "target_model_id": target_model_id,
        "assistant_model_id": assistant_model_id,
        "max_depth": depth.parse::<u32>().unwrap_or(max_depth),
        "output_dir": output_dir,
        "download": target_download,
        "assistant_download": assistant_summary,
    });
    if args.progress {
        print_download_mtp_progress_terminal(&terminal)?;
    } else if args.json {
        print_json(&terminal)?;
    }
    Ok(0)
}

fn assistant_download_usable(code: u8, summary: &Value) -> bool {
    let status = summary.get("status").and_then(Value::as_str);
    if code == 0 && status == Some("ready") {
        return true;
    }
    status == Some("manifest_missing")
        && summary
            .get("config_present")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        && summary
            .get("safetensors_count")
            .and_then(Value::as_u64)
            .unwrap_or(0)
            > 0
}

#[derive(Debug)]
struct ConvertArgs {
    base_model: String,
    mtp_source: String,
    output: Option<String>,
    quantize: Option<String>,
    mtp_depth_max: Option<String>,
    group_size: String,
    fair_base_only: bool,
    json: bool,
}

fn cmd_convert_mtplx(args: &[OsString]) -> Result<u8, String> {
    let args = parse_convert_args(args)?;
    run_convert_mtplx(&args, "convert-mtplx", "ax.convert_mtplx.v1", None, false)
}

fn run_convert_mtplx(
    args: &ConvertArgs,
    command_name: &str,
    schema_version: &str,
    download_summary: Option<Value>,
    progress: bool,
) -> Result<u8, String> {
    let prepare = find_helper(
        "AX_ENGINE_PREPARE_MTP_SIDECAR_HELPER",
        "ax-engine-prepare-mtp-sidecar.py",
        "prepare_mtp_sidecar.py",
    )?;
    let check = find_helper(
        "AX_ENGINE_CHECK_MTP_SIDECAR_HELPER",
        "ax-engine-check-mtp-sidecar-provenance.py",
        "check_mtp_sidecar_provenance.py",
    )?;
    let depth = args
        .mtp_depth_max
        .clone()
        .unwrap_or_else(|| default_mtp_depth_max(&args.base_model, &args.mtp_source).to_string());

    let mut prepare_cmd = Command::new(python());
    prepare_cmd.arg(&prepare).args([
        "--hf-repo",
        &args.mtp_source,
        "--base",
        &args.base_model,
        "--mtp-depth-max",
        &depth,
        "--group-size",
        &args.group_size,
    ]);
    if let Some(output) = &args.output {
        prepare_cmd.args(["--output", output]);
    }
    if let Some(quantize) = &args.quantize {
        prepare_cmd.args(["--quantize", quantize]);
    }
    let prepare_output = prepare_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run prepare_mtp_sidecar helper: {err}"))?;
    let prepare_stdout = String::from_utf8_lossy(&prepare_output.stdout).into_owned();
    let prepare_stderr = String::from_utf8_lossy(&prepare_output.stderr).into_owned();
    if !args.json && !progress {
        print!("{prepare_stdout}");
        eprint!("{prepare_stderr}");
    } else if progress && !prepare_stderr.is_empty() {
        eprint!("{prepare_stderr}");
    }
    if !prepare_output.status.success() {
        if args.json && !progress {
            eprint!("{prepare_stderr}");
        }
        let code: u8 = prepare_output
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1);
        if progress {
            let mut terminal = json!({
                "schema_version": schema_version,
                "command": command_name,
                "status": "prepare_failed",
                "base_model": &args.base_model,
                "mtp_source": &args.mtp_source,
                "exit_code": code,
            });
            if let Some(download_summary) = &download_summary {
                terminal["download"] = download_summary.clone();
            }
            print_download_mtp_progress_terminal(&terminal)?;
        }
        return Ok(code.max(1));
    }
    let output_dir =
        parse_output_dir(&prepare_stdout, args.output.as_deref()).ok_or_else(|| {
            "prepare_mtp_sidecar.py succeeded but output dir could not be determined".to_string()
        })?;

    let mut check_cmd = Command::new(python());
    check_cmd.arg(&check).arg(&output_dir).arg("--json");
    if args.fair_base_only {
        check_cmd.arg("--fair-base-only");
    }
    let provenance = check_cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to run sidecar provenance checker: {err}"))?;
    let provenance_stdout = String::from_utf8_lossy(&provenance.stdout).into_owned();
    let provenance_stderr = String::from_utf8_lossy(&provenance.stderr).into_owned();
    if !args.json && !progress {
        print!("{provenance_stdout}");
        eprint!("{provenance_stderr}");
    } else if progress && !provenance_stderr.is_empty() {
        eprint!("{provenance_stderr}");
    }
    if !provenance.status.success() {
        if args.json && !progress {
            eprint!("{provenance_stderr}");
        }
        let code: u8 = provenance
            .status
            .code()
            .unwrap_or(1)
            .try_into()
            .unwrap_or(1);
        if progress {
            let mut terminal = json!({
                "schema_version": schema_version,
                "command": command_name,
                "status": "provenance_failed",
                "base_model": &args.base_model,
                "mtp_source": &args.mtp_source,
                "output_dir": output_dir,
                "exit_code": code,
            });
            if let Some(download_summary) = &download_summary {
                terminal["download"] = download_summary.clone();
            }
            print_download_mtp_progress_terminal(&terminal)?;
        }
        return Ok(code.max(1));
    }

    if args.json || progress {
        let provenance_summary = serde_json::from_str::<Value>(&provenance_stdout)
            .unwrap_or_else(|_| json!({ "raw": provenance_stdout }));
        let mut summary = json!({
            "schema_version": schema_version,
            "command": command_name,
            "status": "ready",
            "base_model": &args.base_model,
            "mtp_source": &args.mtp_source,
            "mtp_depth_max": depth.parse::<u32>().unwrap_or(1),
            "output_dir": output_dir,
            "provenance": provenance_summary,
        });
        if let Some(download_summary) = download_summary {
            summary["download"] = download_summary;
        }
        if progress {
            print_download_mtp_progress_terminal(&summary)?;
        } else {
            print_json(&summary)?;
        }
    }
    Ok(0)
}

fn parse_convert_args(args: &[OsString]) -> Result<ConvertArgs, String> {
    let mut base_model = None;
    let mut mtp_source = None;
    let mut output = None;
    let mut quantize = None;
    let mut mtp_depth_max = None;
    let mut group_size = "64".to_string();
    let mut fair_base_only = false;
    let mut json = false;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        match arg.as_ref() {
            "--mtp-source" => {
                index += 1;
                mtp_source = Some(require_value(args, index, "--mtp-source")?);
            }
            "--output" => {
                index += 1;
                output = Some(require_value(args, index, "--output")?);
            }
            "--quantize" => {
                index += 1;
                let value = require_value(args, index, "--quantize")?;
                if value != "4" && value != "8" {
                    return Err("--quantize must be 4 or 8".into());
                }
                quantize = Some(value);
            }
            "--mtp-depth-max" => {
                index += 1;
                mtp_depth_max = Some(require_value(args, index, "--mtp-depth-max")?);
            }
            "--group-size" => {
                index += 1;
                group_size = require_value(args, index, "--group-size")?;
            }
            "--fair-base-only" => fair_base_only = true,
            "--json" => json = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unknown convert-mtplx option: {flag}"));
            }
            _ => {
                if base_model.replace(arg.to_string()).is_some() {
                    return Err("convert-mtplx accepts exactly one base model argument".into());
                }
            }
        }
        index += 1;
    }
    Ok(ConvertArgs {
        base_model: base_model.ok_or_else(|| "convert-mtplx requires a base model".to_string())?,
        mtp_source: mtp_source.ok_or_else(|| "convert-mtplx requires --mtp-source".to_string())?,
        output,
        quantize,
        mtp_depth_max,
        group_size,
        fair_base_only,
        json,
    })
}

fn download_repo_id(
    value: &str,
    profile: Option<ModelProfile>,
) -> Result<(String, Option<ModelProfile>, Option<String>), String> {
    if let Some(profile) = profile {
        if !profile.downloadable {
            return Err(format!(
                "{} is not managed by ax-engine download; use an explicit repo id or one of these targets:\n{}",
                profile.label,
                format_download_options()
            ));
        }
        return Ok((
            profile.repo_id.to_string(),
            Some(profile),
            profile_revision(profile).map(str::to_string),
        ));
    }
    if value.contains('/') || value.contains("huggingface.co") || value.contains("hf.co") {
        let repo_ref = ax_engine_core::repo_ref::parse_repo_ref(value)?;
        return Ok((repo_ref.repo_id, None, repo_ref.revision));
    }
    Err(format!(
        "unknown model alias or repo id: {value:?}; pass a Hugging Face repo id, \
         a https://huggingface.co/owner/repo link, or one of these targets:\n{}",
        format_download_options()
    ))
}

fn mtp_download_target_for_model(value: &str) -> Option<MtpDownloadTarget> {
    let normalized = normalize_alias(value);
    MTP_DOWNLOAD_TARGETS.iter().copied().find(|target| {
        target
            .aliases
            .iter()
            .any(|alias| normalize_alias(alias) == normalized)
            || normalize_alias(target.repo_id) == normalized
    })
}

fn format_unknown_download_mtp_target(value: &str) -> String {
    format!(
        "unknown download-mtp target: {value:?}; use one of these targets:\n{}",
        format_download_mtp_targets()
    )
}

fn format_download_mtp_targets() -> String {
    let mut lines = Vec::new();
    for target in MTP_DOWNLOAD_TARGETS {
        let kind = match target.kind {
            MtpDownloadKind::QwenSidecar { .. } => "qwen-sidecar-mtp",
            MtpDownloadKind::GemmaAssistant { .. } => "gemma-assistant-mtp",
            MtpDownloadKind::DirectOnly { .. } => "direct-only",
        };
        lines.push(format!(
            "  - {} -> {} ({kind}; aliases: {})",
            target.label,
            target.repo_id,
            target.aliases.join(", ")
        ));
    }
    lines.join("\n")
}

fn download_options_payload() -> Value {
    json!({
        "schema_version": "ax.download_options.v1",
        "default_destination": {
            "kind": "huggingface_hub_cache",
            "env": ["HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"],
            "dest_semantics": "--dest copies the resolved snapshot to an explicit directory",
        },
        "targets": MODEL_PROFILES.iter().copied().filter(|profile| profile.is_downloadable()).map(|profile| {
            json!({
                "alias": profile.label,
                "repo_id": profile.repo_id,
                "revision": profile_revision(profile),
                "certification": profile_certification(profile),
                "preset": profile.preset,
                "aliases": profile.aliases,
                "mtp_included": profile.repo_id.to_ascii_lowercase().contains("-mtp"),
            })
        }).collect::<Vec<_>>(),
        "examples": [
            "ax-engine download ax-qwen3.5-9b",
            "ax-engine download ax-qwen3.6-35b",
            "ax-engine download ax-gemma4-12b",
            "ax-engine download ax-qwen3-coder-next",
            "ax-engine download ax-embeddinggemma-300m",
            "ax-engine download AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP --json",
            "ax-engine download https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
            "ax-engine download owner/repo@revision",
        ],
    })
}

fn format_download_options() -> String {
    let mut lines = vec![
        "Available AX-ready AutomatosX snapshots".to_string(),
        "(MTP/assistant artifacts are already included where published):".to_string(),
    ];
    for profile in MODEL_PROFILES
        .iter()
        .copied()
        .filter(|profile| profile.is_downloadable())
    {
        let aliases = profile.aliases.join(", ");
        let revision = profile_revision(profile)
            .map(|revision| format!("@{revision}"))
            .unwrap_or_default();
        let certification = profile_certification(profile)
            .map(|status| format!("; {status}"))
            .unwrap_or_default();
        lines.push(format!(
            "  - {} -> {}{} (aliases: {}{})",
            profile.label, profile.repo_id, revision, aliases, certification
        ));
    }
    lines.push("Examples:".into());
    lines.push("  ax-engine download ax-qwen3.5-9b".into());
    lines.push("  ax-engine download ax-qwen3.6-35b".into());
    lines.push("  ax-engine download ax-gemma4-12b".into());
    lines.push("  ax-engine download ax-qwen3-coder-next".into());
    lines.push("  ax-engine download ax-embeddinggemma-300m".into());
    lines.push("  ax-engine download AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP --json".into());
    lines.push(
        "  ax-engine download https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP".into(),
    );
    lines.push("  ax-engine download owner/repo@revision (or /tree/<revision> links)".into());
    lines.join("\n")
}

fn print_download_summary(summary: &Value) {
    let status = summary
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let repo_id = summary
        .get("repo_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let dest = summary.get("dest").and_then(Value::as_str).unwrap_or("");
    println!("AX Engine model: {repo_id}");
    println!("Status: {status}");
    if !dest.is_empty() {
        println!("Path: {dest}");
    }
    if let Some(errors) = summary.get("errors").and_then(Value::as_array) {
        for error in errors {
            if let Some(error) = error.as_str() {
                eprintln!("Error: {error}");
            }
        }
    }
    if status == "ready" && !dest.is_empty() {
        println!("Next:");
        println!("  ax-engine serve {dest}");
    } else if !dest.is_empty() {
        println!("Next:");
        println!("  ax-engine-bench generate-manifest {dest}");
    }
}

fn parse_summary_json(stdout: &str) -> Option<Value> {
    if let Ok(value @ Value::Object(_)) = serde_json::from_str::<Value>(stdout.trim()) {
        return Some(value);
    }
    stdout.lines().rev().find_map(|line| {
        let value = serde_json::from_str::<Value>(line.trim()).ok()?;
        if value.get("schema_version").and_then(Value::as_str) == Some("ax.download_model.v1") {
            Some(value)
        } else {
            None
        }
    })
}

fn profile_for_model(value: &str) -> Option<ModelProfile> {
    let normalized = normalize_alias(value);
    MODEL_PROFILES.iter().copied().find(|profile| {
        profile
            .aliases
            .iter()
            .any(|alias| normalize_alias(alias) == normalized)
    })
}

fn normalize_alias(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('_', "-")
}

fn find_executable(name: &str) -> PathBuf {
    // Optional absolute override used by dev shells and packaging.
    // e.g. AX_ENGINE_SERVER=/path/to/ax-engine-server
    let env_key = match name {
        "ax-engine-server" => Some("AX_ENGINE_SERVER"),
        "ax-engine-bench" => Some("AX_ENGINE_BENCH"),
        _ => None,
    };
    if let Some(key) = env_key
        && let Some(path) = env::var_os(key)
    {
        let path = PathBuf::from(path);
        if path.is_file() {
            return path;
        }
    }

    if let Ok(current) = env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return sibling;
        }
        // `cargo test` places the harness under target/*/deps; look next to
        // the profile root so `ax-engine-server` still resolves when spawning
        // from tests or a deps-adjacent binary.
        if dir.file_name().and_then(|s| s.to_str()) == Some("deps")
            && let Some(profile_dir) = dir.parent()
        {
            let candidate = profile_dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    // Resolve the first absolute hit on PATH instead of returning a bare name
    // (which depends on the child's PATH inheritance and can pick a stale
    // install ahead of a just-built sibling).
    if let Some(path_var) = env::var_os("PATH") {
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }

    PathBuf::from(name)
}

fn find_helper(env_name: &str, installed_name: &str, source_name: &str) -> Result<PathBuf, String> {
    let explicit_repo_root = env::var_os("AX_ENGINE_REPO_ROOT").map(PathBuf::from);
    find_helper_with_repo_root(
        env_name,
        installed_name,
        source_name,
        explicit_repo_root.as_deref(),
    )
}

fn find_helper_with_repo_root(
    env_name: &str,
    installed_name: &str,
    source_name: &str,
    explicit_repo_root: Option<&Path>,
) -> Result<PathBuf, String> {
    if let Some(path) = env::var_os(env_name) {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Ok(current) = env::current_exe()
        && let Some(dir) = current.parent()
    {
        for name in [installed_name, source_name] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if let Some(root) = explicit_repo_root
        && let Some(candidate) = helper_in_repo_root(root, source_name)
    {
        return Ok(candidate);
    }
    if let Some(root) = verified_build_source_repo_root()
        && let Some(candidate) = helper_in_repo_root(&root, source_name)
    {
        return Ok(candidate);
    }
    Err(format!(
        "cannot locate {source_name}. Reinstall ax-engine, set {env_name}, or set \
         AX_ENGINE_REPO_ROOT to a source checkout."
    ))
}

fn helper_in_repo_root(root: &Path, source_name: &str) -> Option<PathBuf> {
    [
        root.join("scripts").join(source_name),
        root.join(source_name),
    ]
    .into_iter()
    .find(|candidate| candidate.is_file())
}

fn verified_build_source_repo_root() -> Option<PathBuf> {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_dir.parent()?.parent()?;
    if root.join("Cargo.toml").is_file()
        && root.join("crates/ax-engine-bench/Cargo.toml").is_file()
        && root.join("scripts").is_dir()
    {
        Some(root.to_path_buf())
    } else {
        None
    }
}

fn python() -> OsString {
    env::var_os("AX_ENGINE_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

/// Fail closed before spawning the download helper when `huggingface_hub` is
/// missing from the Python used by `AX_ENGINE_PYTHON` / `python3`.
fn ensure_download_python_deps() -> Result<(), String> {
    // Unit tests exercise enqueue/UI without a live HF install. Opt back in
    // with AX_ENGINE_REQUIRE_DOWNLOAD_DEPS=1 when testing the preflight itself.
    #[cfg(test)]
    if env::var_os("AX_ENGINE_REQUIRE_DOWNLOAD_DEPS").is_none() {
        return Ok(());
    }

    let py = python();
    let output = Command::new(&py)
        .args(["-c", "import huggingface_hub"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| {
            format!(
                "failed to invoke Python for download preflight ({}): {err}",
                py.to_string_lossy()
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let py_display = py.to_string_lossy();
    Err(format!(
        "huggingface_hub is required for model downloads.\n\
         Install it into the same Python the CLI uses:\n\
           {py_display} -m pip install huggingface_hub\n\
         or:\n\
           {py_display} -m pip install 'ax-engine[download]'\n\
         Optional: set AX_ENGINE_PYTHON to a venv that already has the package."
    ))
}

fn expand_home(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/")
        && let Some(home) = env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(value)
}

fn default_hf_cache_root() -> PathBuf {
    if let Some(root) = env::var_os("HF_HUB_CACHE") {
        return expand_home(&root.to_string_lossy());
    }
    if let Some(home) = env::var_os("HF_HOME") {
        return expand_home(&home.to_string_lossy()).join("hub");
    }
    let cache_home = env::var_os("XDG_CACHE_HOME")
        .map(|value| expand_home(&value.to_string_lossy()))
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from(".cache"));
    cache_home.join("huggingface").join("hub")
}

fn default_gemma_assistant_mtp_output(repo_id: &str) -> PathBuf {
    let leaf = repo_id
        .rsplit('/')
        .next()
        .unwrap_or(repo_id)
        .to_ascii_lowercase();
    default_hf_cache_root()
        .join(format!("models--ax-local--{leaf}-assistant-mtp"))
        .join("snapshots")
        .join("v1")
}

fn absolute_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn require_value(args: &[OsString], index: usize, flag: &str) -> Result<String, String> {
    let value = args
        .get(index)
        .map(|value| value.to_string_lossy().to_string())
        .ok_or_else(|| format!("{flag} requires a value"))?;
    // A missing value must not silently consume the next flag (e.g.
    // `--dest --force` downloading into a directory literally named
    // `--force`). Dash-led paths are still reachable via a `./` prefix.
    if value.starts_with('-') {
        return Err(format!(
            "{flag} requires a value, got option-like {value:?}"
        ));
    }
    Ok(value)
}

fn print_json(value: &Value) -> Result<(), String> {
    let rendered = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON output: {error}"))?;
    println!("{rendered}");
    Ok(())
}

fn render_json_compact(value: &Value) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("failed to serialize JSON output: {error}"))
}

/// One NDJSON terminal record per stream: schema-checked, compact-rendered.
fn render_progress_terminal(
    value: &Value,
    expected_schema: &str,
    label: &str,
) -> Result<String, String> {
    if value.get("schema_version").and_then(Value::as_str) != Some(expected_schema) {
        return Err(format!(
            "{label} progress terminal has an invalid schema_version"
        ));
    }
    render_json_compact(value)
}

fn render_download_progress_terminal(value: &Value) -> Result<String, String> {
    render_progress_terminal(value, "ax.download_model.v1", "download")
}

fn print_download_progress_terminal(value: &Value) -> Result<(), String> {
    println!("{}", render_download_progress_terminal(value)?);
    Ok(())
}

fn download_error_summary(input: Option<&str>, error: &str) -> Value {
    json!({
        "schema_version": "ax.download_model.v1",
        "input": input,
        "repo_id": Value::Null,
        "revision": Value::Null,
        "dest": Value::Null,
        "status": "download_failed",
        "errors": [error],
    })
}

fn render_download_mtp_progress_terminal(value: &Value) -> Result<String, String> {
    render_progress_terminal(value, "ax.download_mtp.v1", "download-mtp")
}

fn print_download_mtp_progress_terminal(value: &Value) -> Result<(), String> {
    println!("{}", render_download_mtp_progress_terminal(value)?);
    Ok(())
}

fn download_mtp_error_summary(base_model: Option<&str>, status: &str, error: &str) -> Value {
    json!({
        "schema_version": "ax.download_mtp.v1",
        "command": "download-mtp",
        "status": status,
        "base_model": base_model,
        "error": error,
    })
}

fn parse_output_dir(stdout: &str, explicit: Option<&str>) -> Option<String> {
    if let Some(explicit) = explicit {
        return Some(
            absolute_path(&expand_home(explicit))
                .to_string_lossy()
                .into(),
        );
    }
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("Output dir:") {
            return Some(rest.trim().to_string());
        }
    }
    let mut saw_sidecar_ready = false;
    for line in stdout.lines() {
        if saw_sidecar_ready {
            let value = line.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
        saw_sidecar_ready = line.trim() == "Sidecar ready at:";
    }
    None
}

fn default_mtp_depth_max(base_model: &str, mtp_source: &str) -> u32 {
    let label = format!("{base_model} {mtp_source}").to_ascii_lowercase();
    if label.contains("qwen3.6-27b") || label.contains("qwen3-6-27b") {
        3
    } else {
        1
    }
}

#[cfg(unix)]
fn exec_or_status(program: PathBuf, args: &[OsString]) -> Result<u8, String> {
    use std::os::unix::process::CommandExt;
    let err = Command::new(&program).args(args).exec();
    Err(format!("failed to exec {}: {err}", program.display()))
}

#[cfg(not(unix))]
fn exec_or_status(program: PathBuf, args: &[OsString]) -> Result<u8, String> {
    let status = Command::new(&program)
        .args(args)
        .status()
        .map_err(|err| format!("failed to run {}: {err}", program.display()))?;
    Ok(status.code().unwrap_or(1).try_into().unwrap_or(1))
}

#[allow(dead_code)]
fn _os_str(value: &str) -> &OsStr {
    OsStr::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-repo contract: AXQuant's `probe_ax_engine_mtp_capability` parses
    /// the last stdout line as JSON and requires exactly these four fields
    /// (extra fields are ignored). Renaming or removing any of them breaks
    /// the `quantize-mtp-sidecar --capability-command` gate.
    #[test]
    fn mtp_capability_contract_fields_are_stable() {
        let value = mtp_capability_json();
        assert_eq!(value["ok"], true);
        assert_eq!(value["mtp_enabled"], true);
        assert_eq!(value["layout"], "ax-engine-qwen36-v1");
        assert!(
            !value["ax_engine_version"]
                .as_str()
                .unwrap_or_default()
                .is_empty()
        );
        let bits: Vec<i64> = value["supported_bits"]
            .as_array()
            .expect("supported_bits array")
            .iter()
            .filter_map(serde_json::Value::as_i64)
            .collect();
        assert_eq!(bits, vec![2, 4, 6, 8, 16]);
        // One-line output: the serialized form must contain no newlines.
        assert!(!value.to_string().contains('\n'));
    }

    const EXPECTED_AUTOMATOSX_REPOS: [&str; 77] = [
        "AutomatosX/AX-Devstral-Small-2-24B-Instruct-2512-MLX-OptiQ-4bit",
        "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-4bit",
        "AutomatosX/AX-Devstral-Small-2505-MLX-AXQ-6bit",
        "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-4bit",
        "AutomatosX/AX-DiffusionGemma-26B-A4B-IT-MLX-OptiQ-4bit",
        "AutomatosX/AX-EmbeddingGemma-300M-MLX-8bit",
        "AutomatosX/AX-Gemma-4-12B-IT-MLX-6bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-4bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-6bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-OptiQ-4bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-26B-A4B-IT-MLX-QAT-4bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-31B-IT-MLX-6bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-31B-IT-MLX-OptiQ-4bit-Assistant-MTP",
        "AutomatosX/AX-Gemma-4-31B-IT-MLX-QAT-4bit-Assistant-MTP",
        "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-4bit",
        "AutomatosX/AX-Holo3-35B-A3B-MLX-AXQ-6bit",
        "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-4bit",
        "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-AXQ-6bit",
        "AutomatosX/AX-Ministral-3-14B-Instruct-2512-MLX-OptiQ-4bit",
        "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-AXQ-6bit",
        "AutomatosX/AX-Ministral-3-8B-Instruct-2512-MLX-OptiQ-4bit",
        "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-4bit",
        "AutomatosX/AX-Mistral-Small-3.1-24B-Instruct-2503-MLX-AXQ-6bit",
        "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit",
        "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit",
        "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-4bit",
        "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit",
        "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-4bit",
        "AutomatosX/AX-Ornith-1.0-35B-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3-ASR-1.7B-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-4bit",
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-6bit",
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3-Coder-Next-MLX-OptiQ-4bit",
        "AutomatosX/AX-Qwen3-Embedding-0.6B-MLX-8bit",
        "AutomatosX/AX-Qwen3-Embedding-4B-MLX-4bit-DWQ",
        "AutomatosX/AX-Qwen3-Embedding-8B-MLX-4bit-DWQ",
        "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3-Nemotron-32B-GenRM-Principle-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3-VL-8B-Instruct-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3.5-9B-MLX-4bit-MTP",
        "AutomatosX/AX-Qwen3.5-9B-MLX-6bit-MTP",
        "AutomatosX/AX-Qwen3.5-9B-MLX-OptiQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-27B-MLX-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
        "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-8bit-MTP",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4",
        "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-MXFP4-MTP",
        "AutomatosX/AX-Unlimited-OCR-3B-MoE-MLX-MXFP8",
        "AutomatosX/AX-gemma-4-12b-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-gemma-4-26b-a4b-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-gemma-4-31b-MLX-AXQ-4bit-MTP",
        "AutomatosX/AX-gemma-4-31b-MLX-AXQ-6bit-MTP",
        "AutomatosX/AX-gpt-oss-120b-MLX-AXQ-6bit",
        "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-4bit",
        "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit",
    ];

    #[test]
    fn download_options_json_matches_contract() {
        let payload = download_options_payload();
        assert_eq!(payload["schema_version"], "ax.download_options.v1");
        let targets = payload["targets"].as_array().unwrap();
        let actual = targets
            .iter()
            .map(|target| target["repo_id"].as_str().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let expected = EXPECTED_AUTOMATOSX_REPOS
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, expected);
        assert_eq!(targets.len(), EXPECTED_AUTOMATOSX_REPOS.len());
        assert!(targets.iter().all(|target| target["alias"] != "gemma4-12b"));
        assert_eq!(
            targets
                .iter()
                .filter(|target| target["mtp_included"] == true)
                .count(),
            32
        );
    }

    #[test]
    fn download_repo_id_accepts_urls_and_revisions() {
        let (repo, profile, rev) = download_repo_id(
            "https://huggingface.co/AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP",
            None,
        )
        .unwrap();
        assert_eq!(repo, "AutomatosX/AX-Qwen3.6-27B-MLX-6bit-MTP");
        assert!(profile.is_none());
        assert_eq!(rev, None);

        let (repo, _, rev) = download_repo_id("owner/repo@v1", None).unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(rev.as_deref(), Some("v1"));

        let (repo, _, rev) = download_repo_id("https://hf.co/owner/repo/tree/main", None).unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(rev.as_deref(), Some("main"));

        let (repo, _, rev) =
            download_repo_id("https://hf.co/owner/repo/tree/feature%2Fdownloads", None).unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(rev.as_deref(), Some("feature/downloads"));

        let (repo, _, rev) = download_repo_id("owner/repo.git@refs/pr/123", None).unwrap();
        assert_eq!(repo, "owner/repo");
        assert_eq!(rev.as_deref(), Some("refs/pr/123"));

        let (repo, _, _) = download_repo_id("owner/repo", None).unwrap();
        assert_eq!(repo, "owner/repo");
    }

    #[test]
    fn download_repo_id_rejects_bad_references() {
        for bad in [
            "noslash",
            "https://example.com/owner/repo",
            "https://huggingface.co/owner",
            "https://huggingface.co/owner/repo/blob/main/model.safetensors",
            "owner/repo/extra/path",
            "owner/repo@../other",
            "owner/re--po",
        ] {
            assert!(download_repo_id(bad, None).is_err(), "{bad:?} must fail");
        }
    }

    #[test]
    fn download_repo_id_accepts_downloadable_alias_outside_automatosx() {
        // Legacy mlx-community aliases are `downloadable` even though the
        // curated catalog only surfaces AutomatosX packs; the download gate
        // must accept them (and any explicit HF repo id).
        let profile = profile_for_model("gemma4-12b").expect("alias should resolve");
        let (repo, resolved, rev) = download_repo_id("gemma4-12b", Some(profile)).unwrap();
        assert_eq!(repo, "mlx-community/gemma-4-12B-it-4bit");
        assert_eq!(resolved.map(|profile| profile.label), Some("gemma4-12b"));
        assert_eq!(rev, None);
    }

    #[test]
    fn compact_progress_summary_is_one_terminal_json_record() {
        let summary = json!({
            "schema_version": "ax.download_model.v1",
            "repo_id": "owner/repo",
            "revision": "feature/downloads",
            "dest": "/tmp/model",
            "status": "ready",
        });
        let rendered = render_json_compact(&summary).unwrap();
        assert!(!rendered.contains('\n'));
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["schema_version"], "ax.download_model.v1");
        assert_eq!(parsed["revision"], "feature/downloads");
    }

    #[test]
    fn download_progress_without_model_is_a_clear_error() {
        let args = parse_download_args(&[OsString::from("--progress-json")]).unwrap();
        let error = run_download(&args).unwrap_err();
        assert_eq!(
            error,
            "download --progress-json requires a model alias or Hugging Face repo id"
        );
        let terminal = download_error_summary(None, &error);
        let rendered = render_download_progress_terminal(&terminal).unwrap();
        assert_eq!(rendered.lines().count(), 1);
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["schema_version"], "ax.download_model.v1");
        assert_eq!(parsed["status"], "download_failed");
        assert_eq!(parsed["errors"][0], error);
    }

    #[test]
    fn download_progress_parse_errors_have_a_terminal_record() {
        let error = parse_download_args(&[
            OsString::from("--progress-json"),
            OsString::from("--unknown"),
        ])
        .unwrap_err();
        let terminal = download_error_summary(None, &error);
        let rendered = render_download_progress_terminal(&terminal).unwrap();
        assert_eq!(rendered.lines().count(), 1);
        let parsed: Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(parsed["schema_version"], "ax.download_model.v1");
        assert_eq!(parsed["status"], "download_failed");
        assert_eq!(parsed["errors"][0], "unknown download option: --unknown");
        assert!(
            render_download_progress_terminal(&json!({
                "schema_version": "ax.download_mtp.v1"
            }))
            .is_err()
        );
    }

    #[test]
    fn download_helper_value_options_preserve_leading_hyphens() {
        assert_eq!(
            helper_value_option("--revision", "-release"),
            OsString::from("--revision=-release")
        );
        assert_eq!(
            helper_value_option("--dest", "-models"),
            OsString::from("--dest=-models")
        );
    }

    #[test]
    fn download_mtp_progress_records_enforce_schema_and_compact_terminal() {
        let failure =
            download_mtp_error_summary(Some("qwen3.6-27b"), "error", "prepare helper unavailable");
        let success = json!({
            "schema_version": "ax.download_mtp.v1",
            "command": "download-mtp",
            "status": "ready",
            "base_model": "qwen3.6-27b",
            "output_dir": "/tmp/qwen-mtp",
        });
        for (summary, expected_status) in [(&failure, "error"), (&success, "ready")] {
            let rendered = render_download_mtp_progress_terminal(summary).unwrap();
            assert_eq!(rendered.lines().count(), 1);
            let parsed: Value = serde_json::from_str(&rendered).unwrap();
            assert_eq!(parsed["schema_version"], "ax.download_mtp.v1");
            assert_eq!(parsed["command"], "download-mtp");
            assert_eq!(parsed["status"], expected_status);
        }
        assert_eq!(failure["base_model"], "qwen3.6-27b");
        assert_eq!(failure["error"], "prepare helper unavailable");
        assert!(
            render_download_mtp_progress_terminal(&json!({
                "schema_version": "ax.download_model.v1"
            }))
            .is_err()
        );
    }

    #[test]
    fn alias_resolution_matches_python_cli_contract() {
        let profile = profile_for_model("qwen36-35b").unwrap();
        assert_eq!(profile.preset, Some("qwen3.6-35b"));
        assert_eq!(profile.repo_id, "mlx-community/Qwen3.6-35B-A3B-4bit");
        let profile = profile_for_model("gemma-4-12b-it").unwrap();
        assert_eq!(profile.preset, Some("gemma4-12b"));
    }

    #[test]
    fn qwen36_axq_candidates_are_explicit_and_revision_pinned() {
        let six = profile_for_model("qwen3.6-27b:axq").unwrap();
        assert_eq!(six.repo_id, "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-6bit-MTP");
        assert_eq!(
            profile_revision(six),
            Some("8c37715c7b5f5ebca00eda6f73be47116a3e4ebc")
        );
        assert_eq!(profile_certification(six), Some("candidate"));

        let four = profile_for_model("qwen3.6-27b:axq-4bit").unwrap();
        assert_eq!(four.repo_id, "AutomatosX/AX-Qwen3.6-27B-MLX-AXQ-4bit-MTP");
        assert_eq!(
            profile_revision(four),
            Some("6182ccbc41c7397ff90670f740c6d9eacfa4b09f")
        );

        let qwen38 = profile_for_model("qwen3.8-27b:axq").unwrap();
        assert_eq!(qwen38.repo_id, "AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP");
        assert_eq!(
            profile_revision(qwen38),
            Some("3e290738e96972307c6aeb9934ab170ca0eae1c1")
        );
        assert_eq!(profile_certification(qwen38), None);

        let vl_six = profile_for_model("qwen3-vl-30b-a3b:axq").unwrap();
        assert_eq!(
            vl_six.repo_id,
            "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-6bit"
        );
        assert_eq!(
            profile_revision(vl_six),
            Some("b48b626d9b00e45d6200aa3c15e40cc47d83b7e7")
        );
        assert_eq!(profile_certification(vl_six), None);

        let vl_four = profile_for_model("ax-qwen3-vl-30b-4bit").unwrap();
        assert_eq!(
            vl_four.repo_id,
            "AutomatosX/AX-Qwen3-VL-30B-A3B-Instruct-MLX-AXQ-4bit"
        );
        assert_eq!(
            profile_revision(vl_four),
            Some("e932be1b8ab79f5410f607de7eb7312756325fce")
        );
        assert_eq!(profile_certification(vl_four), None);

        assert_eq!(
            profile_for_model("gpt-oss-20b:axq").unwrap().repo_id,
            "AutomatosX/AX-gpt-oss-20b-MLX-AXQ-6bit"
        );
        assert_eq!(
            profile_for_model("ax-qwen3-coder-next").unwrap().repo_id,
            "AutomatosX/AX-Qwen3-Coder-Next-MLX-OptiQ-4bit"
        );
        assert_eq!(
            profile_for_model("qwen3-coder-next:axq").unwrap().repo_id,
            "AutomatosX/AX-Qwen3-Coder-Next-MLX-AXQ-6bit"
        );
        assert_eq!(
            profile_for_model("nemotron-3-nano:axq").unwrap().repo_id,
            "AutomatosX/AX-Nemotron-3-Nano-30B-A3B-MLX-AXQ-6bit"
        );

        let default = profile_for_model("qwen3.6-27b").unwrap();
        assert_eq!(default.repo_id, "mlx-community/Qwen3.6-27B-4bit");
        assert_eq!(profile_certification(default), None);

        let qwen35 = profile_for_model("qwen3.6-35b:axq").unwrap();
        assert_eq!(
            qwen35.repo_id,
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-AXQ-6bit-MTP"
        );
        assert_eq!(
            profile_revision(qwen35),
            Some("6a4c220734f81112555ee8783d91e0065c54301c")
        );
        let gemma12 = profile_for_model("gemma4-12b:axq").unwrap();
        assert_eq!(
            gemma12.repo_id,
            "AutomatosX/AX-gemma-4-12b-MLX-AXQ-6bit-MTP"
        );
        assert_eq!(
            profile_for_model("ax-qwen3.6-35b").unwrap().repo_id,
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-OptiQ-4bit-MTP"
        );
        assert_eq!(
            profile_for_model("ax-gemma4-12b").unwrap().repo_id,
            "AutomatosX/AX-Gemma-4-12B-IT-MLX-QAT-OptiQ-4bit-Assistant-MTP"
        );

        let muse_six = profile_for_model("muse-glimmer-30b:axq").unwrap();
        assert_eq!(
            muse_six.repo_id,
            "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-6bit"
        );
        assert_eq!(
            profile_revision(muse_six),
            Some("367745bd05b77bf82188f3799677e4beba543e8d")
        );
        assert_eq!(profile_certification(muse_six), Some("candidate"));
        assert_eq!(muse_six.preset, Some("muse-glimmer-30b"));
        let muse_default = profile_for_model("muse-glimmer-30b").unwrap();
        assert_eq!(
            muse_default.repo_id,
            "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit"
        );
        assert_eq!(
            profile_for_model("ax-muse-glimmer-30b").unwrap().repo_id,
            "AutomatosX/AX-Muse-Glimmer-30B-MLX-AXQ-4bit"
        );
    }

    #[test]
    fn download_mtp_targets_cover_requested_6bit_models() {
        let cases = [
            (
                "qwen3.6-27b-6bit",
                "mlx-community/Qwen3.6-27B-6bit",
                MtpDownloadKind::QwenSidecar {
                    mtp_source: "Qwen/Qwen3.6-27B",
                },
            ),
            (
                "qwen3.6-35b-a3b",
                "mlx-community/Qwen3.6-35B-A3B-6bit",
                MtpDownloadKind::QwenSidecar {
                    mtp_source: "Qwen/Qwen3.6-35B-A3B",
                },
            ),
            (
                "gemma-4-12b",
                "mlx-community/gemma-4-12B-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-6bit",
                    target_model_id: "gemma-4-12b-it",
                    assistant_model_id: "gemma-4-12b-it-assistant",
                    max_depth: 2,
                },
            ),
            (
                "gemma-4-26b",
                "mlx-community/gemma-4-26b-a4b-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "google/gemma-4-26b-a4b-it-assistant",
                    target_model_id: "gemma-4-26b-a4b-it",
                    assistant_model_id: "gemma-4-26b-a4b-it-assistant",
                    max_depth: 1,
                },
            ),
            (
                "gemma-4-31b",
                "mlx-community/gemma-4-31b-it-6bit",
                MtpDownloadKind::GemmaAssistant {
                    assistant_repo_id: "google/gemma-4-31b-it-assistant",
                    target_model_id: "gemma-4-31b-it",
                    assistant_model_id: "gemma-4-31b-it-assistant",
                    max_depth: 1,
                },
            ),
        ];
        for (alias, repo_id, kind) in cases {
            let target = mtp_download_target_for_model(alias).unwrap();
            assert_eq!(target.repo_id, repo_id);
            assert!(target.repo_id.ends_with("6bit"));
            assert_eq!(target.kind, kind);
        }
        // WS-P2: Coder-Next publication path is registered.
        let coder = mtp_download_target_for_model("qwen3-coder-next").unwrap();
        assert_eq!(coder.label, "qwen3-coder-next");
        assert!(matches!(coder.kind, MtpDownloadKind::QwenSidecar { .. }));
        assert!(mtp_download_target_for_model("gemma-4-e2b").is_none());
        let e4b = mtp_download_target_for_model("gemma4-e4b").unwrap();
        assert_eq!(e4b.label, "gemma-4-e4b");
    }

    #[test]
    fn download_mtp_supports_gemma4_12b_4bit_quickstart_target() {
        let target = mtp_download_target_for_model("gemma-4-12b-4bit").unwrap();
        assert_eq!(target.label, "gemma-4-12b-4bit");
        assert_eq!(target.repo_id, "mlx-community/gemma-4-12B-it-4bit");
        assert_eq!(
            target.kind,
            MtpDownloadKind::GemmaAssistant {
                assistant_repo_id: "mlx-community/gemma-4-12B-it-assistant-4bit",
                target_model_id: "gemma-4-12b-it",
                assistant_model_id: "gemma-4-12b-it-assistant",
                max_depth: 2,
            }
        );
        assert_eq!(
            mtp_download_target_for_model("gemma4-12b-4bit")
                .unwrap()
                .label,
            "gemma-4-12b-4bit"
        );
        assert_eq!(
            mtp_download_target_for_model("gemma-4-12b")
                .unwrap()
                .repo_id,
            "mlx-community/gemma-4-12B-it-6bit"
        );
        assert!(
            default_gemma_assistant_mtp_output(target.repo_id)
                .ends_with("models--ax-local--gemma-4-12b-it-4bit-assistant-mtp/snapshots/v1")
        );
    }

    #[test]
    fn parse_download_mtp_args_matches_convert_knobs() {
        let args = parse_download_mtp_args(&[
            OsString::from("qwen36-35b"),
            OsString::from("--output"),
            OsString::from("/tmp/qwen-mtp"),
            OsString::from("--force"),
            OsString::from("--quantize"),
            OsString::from("4"),
            OsString::from("--mtp-depth-max"),
            OsString::from("1"),
            OsString::from("--group-size"),
            OsString::from("128"),
            OsString::from("--fair-base-only"),
            OsString::from("--json"),
            OsString::from("--progress-json"),
        ])
        .unwrap();
        assert_eq!(args.model, "qwen36-35b");
        assert_eq!(args.output.as_deref(), Some("/tmp/qwen-mtp"));
        assert!(args.force);
        assert_eq!(args.quantize.as_deref(), Some("4"));
        assert_eq!(args.mtp_depth_max.as_deref(), Some("1"));
        assert_eq!(args.group_size, "128");
        assert!(args.fair_base_only);
        assert!(args.json);
        assert!(args.progress);
    }

    #[test]
    fn models_info_distinguishes_aliases_from_repo_ids() {
        let alias = model_info_payload("gemma4-12b").unwrap();
        assert_eq!(alias["kind"], "supported_alias");
        assert_eq!(
            alias["profile"]["repo_id"],
            "mlx-community/gemma-4-12B-it-4bit"
        );

        let repo = model_info_payload("mlx-community/custom-model").unwrap();
        assert_eq!(repo["kind"], "repo_id");
        assert_eq!(repo["managed_alias"], false);
    }

    #[test]
    fn models_list_reports_local_artifacts_from_explicit_root() {
        let root = unique_temp_dir("ax-engine-models-list");
        let model_dir = root.join("local-model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model-manifest.json"), "{}").unwrap();

        let payload = models_list_payload(Some(&root));
        let items = payload["local_artifacts"]["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["kind"], "local_artifact");
        assert_eq!(items[0]["manifest_present"], true);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn models_rm_refuses_hugging_face_cache_paths() {
        let root = unique_temp_dir("ax-engine-models-rm");
        let cache_model = root
            .join("huggingface")
            .join("hub")
            .join("models--org--model");
        fs::create_dir_all(&cache_model).unwrap();
        fs::write(cache_model.join("config.json"), "{}").unwrap();

        let error = validate_model_rm_target(&cache_model, true)
            .expect_err("HF cache paths must be removed with cache tooling");
        assert!(error.contains("Hugging Face cache"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn models_rm_allows_dry_run_for_local_artifact_directories() {
        let root = unique_temp_dir("ax-engine-models-rm-local");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), "{}").unwrap();

        let report = validate_model_rm_target(&root, true).unwrap();
        assert_eq!(report["dry_run"], true);
        assert_eq!(report["config_present"], true);
        assert!(root.exists(), "dry-run validation must not remove files");

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parse_output_dir_handles_prepare_messages() {
        assert_eq!(
            parse_output_dir("Sidecar ready at:\n  /tmp/model\n", None).as_deref(),
            Some("/tmp/model")
        );
        assert_eq!(
            parse_output_dir("Output dir: /tmp/other\n", None).as_deref(),
            Some("/tmp/other")
        );
    }

    #[test]
    fn parse_doctor_args_preserves_summary_and_verbose_modes() {
        let args = parse_doctor_args(&[
            OsString::from("--json"),
            OsString::from("--mlx-model-artifacts-dir"),
            OsString::from("/models/gemma4-12b"),
        ])
        .unwrap();
        assert!(args.json);
        assert!(!args.verbose);
        assert!(!args.help);
        assert_eq!(
            args.bench_args,
            vec![
                OsString::from("--mlx-model-artifacts-dir"),
                OsString::from("/models/gemma4-12b")
            ]
        );

        let args = parse_doctor_args(&[OsString::from("--verbose")]).unwrap();
        assert!(!args.json);
        assert!(args.verbose);
    }

    #[test]
    fn serve_defaults_to_inference_port() {
        let args = parse_serve_args(&[OsString::from("qwen36-35b")]).unwrap();

        assert_eq!(args.port, "31418");
    }

    #[test]
    fn serve_accepts_offline_and_local_only_synonyms() {
        for flag in ["--offline", "--local-only"] {
            let args = parse_serve_args(&[OsString::from("qwen3.6-27b:axq"), OsString::from(flag)])
                .unwrap();
            assert!(args.offline);
        }
    }

    #[test]
    fn unknown_serve_target_suggests_close_alias() {
        let message = unknown_serve_target_message("qwen3.6-27b:axqq");
        assert!(message.contains("did you mean"));
        assert!(message.contains("qwen3.6-27b:axq"));
    }

    #[test]
    fn snapshot_cache_requires_every_indexed_weight_shard() {
        let snapshot = unique_temp_dir("ax-engine-partial-snapshot");
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "weight_map": {
                    "model.a": "model-00001-of-00002.safetensors",
                    "model.b": "model-00002-of-00002.safetensors",
                }
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(snapshot.join("model-00001-of-00002.safetensors"), b"first").unwrap();
        assert!(!snapshot_has_complete_weights(&snapshot));

        fs::write(snapshot.join("model-00002-of-00002.safetensors"), b"second").unwrap();
        assert!(snapshot_has_complete_weights(&snapshot));
        let _ = fs::remove_dir_all(snapshot);
    }

    #[test]
    fn user_doctor_text_highlights_status_checks_and_next_steps() {
        let report = json!({
            "result": "ready",
            "install": {"version": "6.4.3", "mode": "installed_tools"},
            "host": {
                "os": "macos",
                "arch": "aarch64",
                "os_version": "15.5",
                "ram_gib": 64,
                "cpu_cores": {
                    "physical": 16,
                    "logical": 16,
                    "performance": 12,
                    "efficiency": 4,
                    "summary": "16 (4 Efficiency and 12 Performance)",
                    "types": {
                        "efficiency": 4,
                        "performance": 12
                    }
                },
                "gpu_cores": 40
            },
            "checks": [
                {"id": "server_binary", "status": "pass", "detail": "ax-engine-server ok"},
                {"id": "model", "status": "not_selected", "selected": false, "path": null}
            ],
            "issues": [],
            "model_issues": [],
            "next_actions": ["ax-engine serve qwen36-35b --port 31418"],
            "details_command": "ax-engine-bench doctor"
        });
        let output = format_user_doctor_report(&report);
        assert!(output.contains("AX Engine doctor"));
        assert!(output.contains("Result: ready"));
        assert!(output.contains("host: macos 15.5 (aarch64)"));
        assert!(output.contains("RAM: 64 GiB"));
        assert!(output.contains("CPU cores: 16 (4 Efficiency and 12 Performance)"));
        assert!(output.contains("GPU cores: 40"));
        assert!(output.contains("server_binary: pass - ax-engine-server ok"));
        assert!(output.contains("model: not_selected"));
        assert!(output.contains("ax-engine serve qwen36-35b --port 31418"));
        assert!(output.contains("More details: ax-engine-bench doctor"));
    }

    #[test]
    fn metal_check_accepts_bundled_runtime_assets_without_developer_toolchain() {
        let report = json!({
            "runtime_assets": {"status": "ready"},
            "metal_toolchain": {"fully_available": false}
        });

        assert!(metal_check_pass(&report));
        assert_eq!(
            metal_detail(&report),
            "Bundled runtime assets available; Metal compiler only needed for kernel rebuilds"
        );
    }

    #[test]
    fn helper_discovery_does_not_execute_a_script_from_the_current_directory() {
        let root = unique_temp_dir("ax-engine-untrusted-helper-cwd");
        let scripts = root.join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let unique = root
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap()
            .replace('-', "_");
        let source_name = format!("{unique}_download_model.py");
        fs::write(
            scripts.join(&source_name),
            "raise SystemExit('untrusted')\n",
        )
        .unwrap();

        let original = env::current_dir().unwrap();
        env::set_current_dir(&root).unwrap();
        let result = find_helper_with_repo_root(
            &format!("AX_ENGINE_{unique}_HELPER"),
            &format!("ax-engine-{unique}-helper"),
            &source_name,
            None,
        );
        env::set_current_dir(original).unwrap();

        assert!(result.is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn helper_discovery_accepts_an_explicit_source_repo_root() {
        let root = unique_temp_dir("ax-engine-explicit-helper-root");
        let scripts = root.join("scripts");
        fs::create_dir_all(&scripts).unwrap();
        let source_name = "explicit-download-model.py";
        let expected = scripts.join(source_name);
        fs::write(&expected, "# explicit trusted helper\n").unwrap();

        let found = find_helper_with_repo_root(
            "AX_ENGINE_TEST_EXPLICIT_HELPER",
            "ax-engine-test-explicit-helper",
            source_name,
            Some(&root),
        )
        .unwrap();

        assert_eq!(found, expected);
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!("{label}-{nanos}"))
    }
}
