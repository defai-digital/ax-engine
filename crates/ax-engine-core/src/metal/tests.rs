use super::*;
use std::ffi::OsString;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

struct Phase1Fixture {
    root: PathBuf,
    build_dir: PathBuf,
}

impl Phase1Fixture {
    fn cleanup(self) {
        let _ = fs::remove_dir_all(self.root);
    }
}

#[test]
fn phase1_optional_constant_covers_every_manifest_optional_kernel() {
    // Forward direction of the wiring-completeness invariant, checked
    // against the real on-disk manifest so the sg-projection regression
    // cannot recur silently.
    let manifest_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../metal/phase1-kernels.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("manifest readable"))
            .expect("manifest parses");
    for kernel in manifest["kernels"].as_array().expect("kernels array") {
        let name = kernel["name"].as_str().expect("kernel name");
        let tier = kernel["tier"].as_str().expect("kernel tier");
        let constant: &[&str] = match tier {
            "required" => PHASE1_REQUIRED_METAL_KERNELS,
            "deferred" => PHASE1_DEFERRED_METAL_KERNELS,
            "optional" => PHASE1_OPTIONAL_METAL_KERNELS,
            other => panic!("unknown tier {other} for kernel {name}"),
        };
        assert!(
            constant.contains(&name),
            "manifest kernel {name} (tier {tier}) is missing from its PHASE1 constant"
        );
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn metal_runtime_bringup_requires_macos_host() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

    let error = MetalRuntimeBringup::from_build_dir(&fixture.build_dir)
        .expect_err("bring-up should reject non-macos hosts");
    let MetalRuntimeError::UnsupportedPlatform { host_os } = error else {
        panic!("expected unsupported platform error");
    };
    assert_eq!(host_os, std::env::consts::OS);

    fixture.cleanup();
}

#[test]
fn ast_signature_validator_accepts_matching_ast() {
    assert_eq!(
        build::validate_kernel_signatures_against_ast(
            &synthetic_projection_ast_json(),
            build::PROJECTION_KERNEL_SIGNATURES,
            build::PROJECTION_PARAM_STRUCT_EXPECTATIONS,
        ),
        Ok(())
    );
}

#[test]
fn ast_signature_validator_accepts_matching_gate_product_ast() {
    assert_eq!(
        build::validate_kernel_signatures_against_ast(
            &synthetic_projection_ast_json(),
            build::GATE_PRODUCT_KERNEL_SIGNATURES,
            build::GATE_PRODUCT_PARAM_STRUCT_EXPECTATIONS,
        ),
        Ok(())
    );
}

#[test]
fn ast_signature_validator_rejects_drifted_gate_product_index() {
    let mut ast: serde_json::Value =
        serde_json::from_str(&synthetic_projection_ast_json()).expect("synthetic AST should parse");
    // Renumber linear_attention_beta_sigmoid_f32's params slot from 2 to 3.
    let kernel = ast["inner"]
        .as_array_mut()
        .expect("inner array")
        .iter_mut()
        .find(|node| node["name"] == "linear_attention_beta_sigmoid_f32")
        .expect("kernel present");
    kernel["inner"][3]["inner"][0]["inner"][0]["value"] = serde_json::json!("3");

    let error = build::validate_kernel_signatures_against_ast(
        &ast.to_string(),
        build::GATE_PRODUCT_KERNEL_SIGNATURES,
        build::GATE_PRODUCT_PARAM_STRUCT_EXPECTATIONS,
    )
    .expect_err("drifted buffer index must be rejected");
    assert!(error.contains("linear_attention_beta_sigmoid_f32"));
    assert!(error.contains("drifted"));
}

#[test]
fn ast_signature_validator_rejects_drifted_buffer_index() {
    let mut ast: serde_json::Value =
        serde_json::from_str(&synthetic_projection_ast_json()).expect("synthetic AST should parse");
    // Renumber decode_logits_projection_f32's params slot from 3 to 4.
    let kernel = ast["inner"]
        .as_array_mut()
        .expect("inner array")
        .iter_mut()
        .find(|node| node["name"] == "decode_logits_projection_f32")
        .expect("kernel present");
    kernel["inner"][4]["inner"][0]["inner"][0]["value"] = serde_json::json!("4");

    let error = build::validate_kernel_signatures_against_ast(
        &ast.to_string(),
        build::PROJECTION_KERNEL_SIGNATURES,
        build::PROJECTION_PARAM_STRUCT_EXPECTATIONS,
    )
    .expect_err("drifted buffer index must be rejected");
    assert!(error.contains("decode_logits_projection_f32"));
    assert!(error.contains("drifted"));
}

#[test]
fn ast_signature_validator_rejects_renamed_struct_field() {
    let renamed = synthetic_projection_ast_json().replace("\"n_rows\"", "\"row_count\"");
    let error = build::validate_kernel_signatures_against_ast(
        &renamed,
        build::PROJECTION_KERNEL_SIGNATURES,
        build::PROJECTION_PARAM_STRUCT_EXPECTATIONS,
    )
    .expect_err("renamed param-struct field must be rejected");
    assert!(error.contains("Q4KMProjectionParams"));
}

#[test]
fn ast_signature_validator_rejects_unrecognized_root() {
    let error = build::validate_kernel_signatures_against_ast(
        r#"{"kind": "SomethingElse", "inner": []}"#,
        build::PROJECTION_KERNEL_SIGNATURES,
        build::PROJECTION_PARAM_STRUCT_EXPECTATIONS,
    )
    .expect_err("unknown AST root must be rejected");
    assert!(error.contains("TranslationUnitDecl"));
}

#[test]
fn ast_signature_validator_rejects_unparseable_dump() {
    let error = build::validate_kernel_signatures_against_ast(
        "not json at all",
        build::PROJECTION_KERNEL_SIGNATURES,
        build::PROJECTION_PARAM_STRUCT_EXPECTATIONS,
    )
    .expect_err("non-JSON dump must be rejected");
    assert!(error.contains("did not parse"));
}

/// Minimal `-ast-dump=json`-shaped document containing exactly the nodes
/// the AST signature gate inspects, generated from the gate's own
/// expectation tables (projection and gate-product kernel signatures and
/// param structs), so fixture and gate cannot drift apart.
fn synthetic_projection_ast_json() -> String {
    let signatures = build::PROJECTION_KERNEL_SIGNATURES
        .iter()
        .chain(build::GATE_PRODUCT_KERNEL_SIGNATURES);
    let kernels = signatures.map(|signature| {
        let mut inner = vec![serde_json::json!({"kind": "MetalKernelAttr"})];
        inner.extend(signature.buffers.iter().map(|(name, index, qual_type)| {
            serde_json::json!({
                "kind": "ParmVarDecl",
                "name": name,
                "type": {"qualType": qual_type},
                "inner": [{
                    "kind": "MetalBufferIndexAttr",
                    "inner": [{"kind": "IntegerLiteral", "value": index.to_string()}],
                }],
            })
        }));
        serde_json::json!({"kind": "FunctionDecl", "name": signature.kernel, "inner": inner})
    });
    let records = build::PROJECTION_PARAM_STRUCT_EXPECTATIONS
        .iter()
        .chain(build::GATE_PRODUCT_PARAM_STRUCT_EXPECTATIONS)
        .map(|(name, fields)| {
            serde_json::json!({
                "kind": "CXXRecordDecl",
                "name": name,
                "inner": fields
                    .iter()
                    .map(|(field, qual_type)| serde_json::json!({
                        "kind": "FieldDecl",
                        "name": field,
                        "type": {"qualType": qual_type},
                    }))
                    .collect::<Vec<_>>(),
            })
        });
    serde_json::json!({
        "kind": "TranslationUnitDecl",
        "inner": kernels.chain(records).collect::<Vec<_>>(),
    })
    .to_string()
}

fn phase1_source_text() -> &'static str {
    r#"
// The parser should ignore commented kernels such as:
// kernel void commented_line_kernel() {}
/*
kernel void commented_block_kernel() {}
*/
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
kernel void kv_scale_update() {}
kernel void gather_embedding_rows_f32() {}
kernel void gather_embedding_rows_f16() {}
kernel void gather_embedding_rows_bf16() {}
kernel void row_scale_f32() {}
kernel void row_vector_scale_f32() {}
kernel void decode_logits_projection_f32() {}
kernel void decode_logits_projection_f16() {}
kernel void decode_logits_projection_bf16() {}
kernel void decode_logits_projection_batched_f32() {}
kernel void decode_logits_projection_batched_f16() {}
kernel void decode_logits_projection_batched_bf16() {}
kernel void logits_argmax_f32() {}
kernel void logits_argmax_batched_f32() {}
kernel void sample_argmax_logprob_f32() {}
kernel void sample_argmax_logprob_batched_f32() {}
kernel void rms_norm_f32() {}
kernel void rms_norm_f16() {}
kernel void rms_norm_bf16() {}
kernel void rms_norm_batched_f32() {}
kernel void rms_norm_batched_f16() {}
kernel void rms_norm_batched_bf16() {}
kernel void ffn_gate_silu_product_f32() {}
kernel void ffn_gate_gelu_approx_product_f32() {}
kernel void apply_rope_f32() {}
kernel void expand_grouped_kv_heads_f32() {}
"#
}

fn phase1_kernel_specs() -> Vec<MetalKernelSpec> {
    vec![
        MetalKernelSpec {
            name: "reshape_and_cache".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "paged KV writes".to_string(),
        },
        MetalKernelSpec {
            name: "paged_decode_attention".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "decode attention".to_string(),
        },
        MetalKernelSpec {
            name: "gather_kv_cache".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "KV gather".to_string(),
        },
        MetalKernelSpec {
            name: "copy_blocks".to_string(),
            tier: MetalKernelTier::Required,
            purpose: "block copy".to_string(),
        },
        MetalKernelSpec {
            name: "swap_blocks".to_string(),
            tier: MetalKernelTier::Deferred,
            purpose: "future block swap".to_string(),
        },
        MetalKernelSpec {
            name: "kv_scale_update".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "quantized KV scaling".to_string(),
        },
        MetalKernelSpec {
            name: "row_scale_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native per-row scalar multiplication".to_string(),
        },
        MetalKernelSpec {
            name: "row_vector_scale_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native per-row elementwise multiplication by a shared scale vector"
                .to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for f32 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for f16 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "gather_embedding_rows_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native embedding row gather for bf16 token embeddings".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "f32 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "f16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "bf16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched f32 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched f16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "decode_logits_projection_batched_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "batched bf16 decode logits projection".to_string(),
        },
        MetalKernelSpec {
            name: "logits_argmax_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native top-1 logits scan".to_string(),
        },
        MetalKernelSpec {
            name: "logits_argmax_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched top-1 logits scan".to_string(),
        },
        MetalKernelSpec {
            name: "sample_argmax_logprob_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native deterministic top-1 sampling with logprob".to_string(),
        },
        MetalKernelSpec {
            name: "sample_argmax_logprob_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched deterministic top-1 sampling with logprob".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native f16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native bf16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched f32 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_f16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched f16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "rms_norm_batched_bf16".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native batched bf16 RMSNorm".to_string(),
        },
        MetalKernelSpec {
            name: "ffn_gate_silu_product_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native SiLU gate-up activation product".to_string(),
        },
        MetalKernelSpec {
            name: "ffn_gate_gelu_approx_product_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native GELU-approx gate-up activation product".to_string(),
        },
        MetalKernelSpec {
            name: "apply_rope_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native Q/K rotary position embedding application".to_string(),
        },
        MetalKernelSpec {
            name: "expand_grouped_kv_heads_f32".to_string(),
            tier: MetalKernelTier::Optional,
            purpose: "native grouped-KV expansion into query-head layout".to_string(),
        },
    ]
}

fn sample_build_doctor(
    metal_toolchain_fully_available: bool,
    bringup_allowed: bool,
) -> MetalBuildDoctorReport {
    MetalBuildDoctorReport {
        status: if metal_toolchain_fully_available && bringup_allowed {
            "ready".to_string()
        } else if bringup_allowed {
            "bringup_only".to_string()
        } else {
            "not_ready".to_string()
        },
        bringup_allowed,
        mlx_runtime_ready: metal_toolchain_fully_available && bringup_allowed,
        metal_toolchain_fully_available,
        host: MetalBuildHostReport {
            os: "macos".to_string(),
            arch: "aarch64".to_string(),
            detected_soc: Some("Apple M4 Max".to_string()),
            supported_mlx_runtime: true,
            unsupported_host_override_active: false,
        },
        metal_toolchain: MetalBuildToolchainReport {
            fully_available: metal_toolchain_fully_available,
            metal: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metal version 36000.4".to_string()),
            },
            metallib: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metallib version 36000.4".to_string()),
            },
            metal_ar: MetalBuildToolStatus {
                available: metal_toolchain_fully_available,
                version: Some("Apple metal-ar version 36000.4".to_string()),
            },
        },
    }
}

fn fake_xcrun_script() -> &'static str {
    r#"#!/bin/sh
set -eu

while [ "$#" -gt 0 ]; do
  if [ "$1" = "--sdk" ]; then
    shift 2
    continue
  fi
  tool="$1"
  shift
  break
done

case "${tool:-}" in
  metal)
    out=""
    src=""
    ast=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        -c|-O3|-Wall|-Wextra)
          shift
          ;;
        -std=*)
          shift
          ;;
        -fsyntax-only|-Xclang)
          shift
          ;;
        -ast-dump=json)
          ast="1"
          shift
          ;;
        *)
          src="$1"
          shift
          ;;
      esac
    done
    if [ "$ast" = "1" ]; then
      cat "${src}.ast.json"
    else
      cp "$src" "$out"
    fi
    ;;
  metal-ar)
    archive=""
    input=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -q)
          shift
          ;;
        *)
          if [ -z "$archive" ]; then
            archive="$1"
          else
            input="$1"
          fi
          shift
          ;;
      esac
    done
    cp "$input" "$archive"
    ;;
  metallib)
    input=""
    out=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        -o)
          out="$2"
          shift 2
          ;;
        *)
          input="$1"
          shift
          ;;
      esac
    done
    cp "$input" "$out"
    ;;
  *)
    echo "unexpected tool" >&2
    exit 64
    ;;
esac
"#
}

fn fake_failing_xcrun_script() -> &'static str {
    r#"#!/bin/sh
set -eu
echo "unexpected xcrun invocation" >&2
exit 99
"#
}

// -------------------------------------------------------------------------
// Tier 1: apply_rms_norm_with_weights_in_place (no Metal device required)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 1: expand_grouped_kv_heads_cpu (no Metal device required)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 2: apply_model_stage_rope_cpu (needs artifacts on disk, no GPU ops)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 2: *_with_path fallback wiring — verify CPU is used when bringup=None
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 3: per_head_rms_norm_with_tally (needs Metal device for buffer)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 3: project_decode_logits_cpu (needs Metal device for buffer)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 3: initial_model_hidden_states_cpu (needs Metal device + workload)
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Tier 4: GPU vs CPU correctness — each test compiles real Metal kernels via
// xcrun (skipped gracefully when xcrun / Metal toolchain is unavailable) and
// then asserts that the GPU path produces the same numerical result as the
// corresponding CPU reference implementation.
// -------------------------------------------------------------------------

// --- GPU test: expand_grouped_kv_heads_f32 --------------------------------

// --- GPU test: ffn_gate_silu_product_f32 ----------------------------------

// --- GPU test: ffn_gate_gelu_approx_product_f32 ---------------------------

// --- GPU test: apply_rope_f32 (NeoX style, Qwen) --------------------------

// --- GPU test: apply_rope_f32 (Interleaved style, Gemma) ------------------

// --- GPU test: rms_norm_batched_f32 (per-head norm) -----------------------

// --- GPU test: decode_projection_q4km correctness -------------------------

// --- GPU test: reshape_and_cache (required kernel) ------------------------

// --- GPU test: paged_decode_attention (required kernel) -------------------

// --- GPU test: gather_kv_cache (required kernel) --------------------------

// --- GPU test: copy_blocks (required kernel) ------------------------------

fn unique_test_dir(label: &str) -> PathBuf {
    static NEXT_SUFFIX: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let suffix = NEXT_SUFFIX.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ax-engine-core-{label}-{}-{nanos}-{suffix}",
        std::process::id()
    ))
}

// --- Integration test: Qwen3.5-9B-Q4_K_M.gguf MLX load ----------------

// -------------------------------------------------------------------------
// Tier 5: CPU-vs-Metal differential property tests. The same computation
// runs on a CPU oracle and the real Metal kernel over randomized inputs;
// outputs must agree within a per-kernel tolerance. One bringup compile is
// shared across all cases of a test, and each test skips gracefully when
// the toolchain, device, or kernel is unavailable — the same policy as the
// Tier 4 fixed-value tests above. Randomized shapes deliberately straddle
// simd-width and thread-execution-width boundaries, the regions fixed-value
// tests tend to miss.
// -------------------------------------------------------------------------

#[test]
fn metal_assets_load_compiled_build_and_resolve_required_kernels() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);

    let assets = MetalKernelAssets::from_build_dir(&fixture.build_dir).expect("assets should load");

    assert_eq!(assets.build_status(), MetalBuildStatus::Compiled);
    assert_eq!(assets.manifest().library_name, PHASE1_METAL_LIBRARY_NAME);
    assert_eq!(
        assets.default_block_size_tokens(),
        PHASE1_DEFAULT_BLOCK_SIZE_TOKENS
    );
    assert_eq!(
        assets.supported_block_size_tokens(),
        PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS
    );
    assert_eq!(
        assets
            .required_kernel("reshape_and_cache")
            .expect("required kernel should resolve")
            .tier,
        MetalKernelTier::Required
    );
    assert_eq!(
        assets.kernel("kv_scale_update").map(|kernel| kernel.tier),
        Some(MetalKernelTier::Optional)
    );
    assert_eq!(
        assets.kernel("swap_blocks").map(|kernel| kernel.tier),
        Some(MetalKernelTier::Deferred)
    );
    assert!(assets.compiled_metallib_path().is_some());
    assert!(
        !assets
            .compiled_metallib_bytes()
            .expect("compiled metallib should load")
            .is_empty()
    );

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_missing_required_kernel() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest
                .kernels
                .retain(|kernel| kernel.name != "copy_blocks");
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject missing required kernel");
    let MetalRuntimeError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest error");
    };
    assert!(message.contains("missing required kernel copy_blocks"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_missing_deferred_kernel() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest
                .kernels
                .retain(|kernel| kernel.name != "swap_blocks");
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject missing deferred kernel inventory");
    let MetalRuntimeError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest error");
    };
    assert!(message.contains("missing deferred kernel swap_blocks"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_manifest_kernel_missing_from_tier_constants() {
    // A manifest kernel that no PHASE1_*_METAL_KERNELS constant lists is
    // compiled-but-dead: the runtime builds its optional inventory from
    // those constants, so the kernel can never resolve. This is the drift
    // class the sg-projection kernels shipped in.
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest.kernels.push(MetalKernelSpec {
                name: "orphaned_kernel_not_in_constants".to_string(),
                tier: MetalKernelTier::Optional,
                purpose: "wiring-completeness negative fixture".to_string(),
            });
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject a manifest kernel absent from tier constants");
    let MetalRuntimeError::InvalidManifest { message } = error else {
        panic!("expected invalid manifest error");
    };
    assert!(message.contains("orphaned_kernel_not_in_constants"));
    assert!(message.contains("not in any PHASE1_"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_non_phase1_block_size_policy() {
    let fixture = write_phase1_fixture(
        MetalBuildStatus::Compiled,
        Some(|manifest: &mut MetalKernelManifest| {
            manifest.supported_block_size_tokens = vec![8, 16];
        }),
    );

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unsupported block size policy");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("supported_block_size_tokens must be multiples of"));

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_skips_when_toolchain_is_unavailable() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(false, false),
        toolchain_path_override: None,
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit skipped report");

    assert_eq!(
        artifacts.build_status(),
        MetalBuildStatus::SkippedToolchainUnavailable
    );
    assert!(artifacts.build_report.compile_commands.is_empty());
    assert!(artifacts.build_report.outputs.air.is_none());
    assert!(artifacts.build_report.outputs.metalar.is_none());
    assert!(artifacts.build_report.outputs.metallib.is_none());
    assert!(artifacts.doctor_path.is_file());
    assert!(artifacts.build_report_path.is_file());
    let summary =
        fs::read_to_string(&artifacts.summary_path).expect("summary file should be readable");
    assert!(summary.contains("skipped_toolchain_unavailable"));

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_preserves_existing_binaries_when_toolchain_is_unavailable() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let air_path = fixture.build_dir.join("ax_phase1_dense_path.air");
    let metalar_path = fixture.build_dir.join("ax_phase1_dense_path.metalar");
    let metallib_path = fixture.build_dir.join("ax_phase1_dense_path.metallib");
    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(false, false),
        toolchain_path_override: None,
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit skipped report");

    assert_eq!(
        artifacts.build_status(),
        MetalBuildStatus::SkippedToolchainUnavailable
    );
    assert!(artifacts.build_report.outputs.air.is_none());
    assert!(artifacts.build_report.outputs.metalar.is_none());
    assert!(artifacts.build_report.outputs.metallib.is_none());
    assert!(air_path.is_file());
    assert!(metalar_path.is_file());
    assert!(metallib_path.is_file());

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_fails_closed_on_source_manifest_drift() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
    fs::write(
        &source_path,
        r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#,
    )
    .expect("drifted source should write");

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(true, true),
        toolchain_path_override: None,
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit failed report");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::FailedCompile);
    assert!(artifacts.build_report.compile_commands.is_empty());
    assert!(
        artifacts
            .build_report
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("kv_scale_update"))
    );
    assert!(artifacts.build_report.outputs.metallib.is_none());

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_compiles_with_fake_xcrun_toolchain() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_xcrun_script()).expect("fake xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake xcrun should be executable");

    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(true, true),
        toolchain_path_override: Some(fake_path),
    };

    let artifacts = build_phase1_kernel_artifacts(&request)
        .expect("builder should compile through fake toolchain");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
    assert!(!artifacts.reused_existing_artifacts());
    // ast-signature gate + air + metalar + metallib
    assert_eq!(artifacts.build_report.compile_commands.len(), 4);
    assert!(
        artifacts
            .build_report
            .outputs
            .air
            .as_deref()
            .is_some_and(Path::is_file)
    );
    assert!(
        artifacts
            .build_report
            .outputs
            .metalar
            .as_deref()
            .is_some_and(Path::is_file)
    );
    assert!(
        artifacts
            .build_report
            .outputs
            .metallib
            .as_deref()
            .is_some_and(Path::is_file)
    );
    assert_eq!(
        artifacts
            .build_report
            .outputs
            .metallib_sha256
            .as_deref()
            .map(str::len),
        Some(64)
    );

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_compiles_without_metal_ar() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_xcrun_script()).expect("fake xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake xcrun should be executable");

    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
    let mut doctor = sample_build_doctor(true, true);
    doctor.metal_toolchain.metal_ar.available = false;
    doctor.metal_toolchain.metal_ar.version = None;

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor,
        toolchain_path_override: Some(fake_path),
    };

    let artifacts = build_phase1_kernel_artifacts(&request)
        .expect("builder should compile directly from AIR when metal-ar is unavailable");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
    // ast-signature gate + air + metallib (no metal-ar)
    assert_eq!(artifacts.build_report.compile_commands.len(), 3);
    assert!(artifacts.build_report.outputs.metalar.is_none());
    assert!(artifacts.build_report.outputs.metalar_sha256.is_none());
    assert!(
        artifacts
            .build_report
            .outputs
            .metallib
            .as_deref()
            .is_some_and(Path::is_file)
    );

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_fails_closed_on_ast_signature_drift() {
    let fixture = write_phase1_fixture(MetalBuildStatus::SkippedToolchainUnavailable, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_xcrun_script()).expect("fake xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake xcrun should be executable");

    // Drop one projection kernel from the AST the fake toolchain serves —
    // models a shader edit that renamed the kernel or removed its
    // `kernel` qualifier without the Rust dispatch table noticing.
    let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
    let ast_path = format!("{}.ast.json", source_path.display());
    let mut ast: serde_json::Value =
        serde_json::from_str(&synthetic_projection_ast_json()).expect("synthetic AST should parse");
    ast["inner"]
        .as_array_mut()
        .expect("synthetic AST inner array")
        .retain(|node| node["name"] != "decode_projection_q4km");
    fs::write(&ast_path, ast.to_string()).expect("tampered AST should write");

    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());
    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: sample_build_doctor(true, true),
        toolchain_path_override: Some(fake_path),
    };

    let artifacts =
        build_phase1_kernel_artifacts(&request).expect("builder should emit failed report");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::FailedCompile);
    assert!(
        artifacts
            .build_report
            .reason
            .as_deref()
            .is_some_and(|reason| reason.contains("metal AST signature gate failed")
                && reason.contains("decode_projection_q4km"))
    );
    assert!(artifacts.build_report.outputs.metallib.is_none());

    fixture.cleanup();
}

#[test]
fn metal_kernel_builder_reuses_valid_compiled_artifacts_without_recompiling() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let bin_dir = fixture.root.join("fake-bin");
    fs::create_dir_all(&bin_dir).expect("fake bin directory should create");
    let fake_xcrun = bin_dir.join("xcrun");
    fs::write(&fake_xcrun, fake_failing_xcrun_script()).expect("fake failing xcrun should write");
    let mut permissions = fs::metadata(&fake_xcrun)
        .expect("fake failing xcrun metadata should load")
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_xcrun, permissions).expect("fake failing xcrun should be executable");

    let mut doctor = sample_build_doctor(true, true);
    doctor.metal_toolchain.metal.version = Some("Apple metal version 99999.1".to_string());

    let original_path = std::env::var_os("PATH");
    let fake_path = prepend_to_path(&bin_dir, original_path.as_ref());

    let request = MetalKernelBuildRequest {
        manifest_path: fixture.root.join("metal/phase1-kernels.json"),
        output_dir: fixture.build_dir.clone(),
        doctor: doctor.clone(),
        toolchain_path_override: Some(fake_path),
    };

    let artifacts = build_phase1_kernel_artifacts(&request)
        .expect("builder should reuse validated compiled artifacts");

    assert_eq!(artifacts.build_status(), MetalBuildStatus::Compiled);
    assert!(artifacts.reused_existing_artifacts());
    assert_eq!(artifacts.build_report.doctor, doctor);
    assert!(
        artifacts
            .build_report
            .outputs
            .metallib
            .as_deref()
            .is_some_and(Path::is_file)
    );

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_unknown_build_status() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unknown build status");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("status unknown is not allowed"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_unknown_build_status_before_manifest_resolution() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Unknown, None);
    let manifest_path = fixture.root.join("metal/phase1-kernels.json");
    fs::remove_file(&manifest_path).expect("manifest file should be removable");

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject unknown status before reading manifest");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("status unknown is not allowed"));

    fixture.cleanup();
}

#[test]
fn metal_assets_reject_source_kernel_manifest_drift() {
    let fixture = write_phase1_fixture(MetalBuildStatus::Compiled, None);
    let source_path = fixture.root.join("metal/kernels/phase1_dense_path.metal");
    let drifted_source = r#"
kernel void reshape_and_cache() {}
kernel void paged_decode_attention() {}
kernel void gather_kv_cache() {}
kernel void copy_blocks() {}
kernel void swap_blocks() {}
"#;
    fs::write(&source_path, drifted_source).expect("source file should be updated");

    let build_report_path = fixture.build_dir.join("build_report.json");
    let mut build_report: MetalBuildReport =
        read_json_file(&build_report_path).expect("build report should load");
    build_report.source_sha256 = sha256_hex(drifted_source.as_bytes());
    write_json_file(&build_report_path, &build_report);

    let error = MetalKernelAssets::from_build_dir(&fixture.build_dir)
        .expect_err("assets should reject source/manifest drift");
    let MetalRuntimeError::InvalidBuildReport { message } = error else {
        panic!("expected invalid build report error");
    };
    assert!(message.contains("source kernel declarations do not match manifest"));
    assert!(message.contains("kv_scale_update"));

    fixture.cleanup();
}

fn write_phase1_fixture(
    status: MetalBuildStatus,
    manifest_edit: Option<fn(&mut MetalKernelManifest)>,
) -> Phase1Fixture {
    let root = unique_test_dir("metal-fixture");
    let metal_dir = root.join("metal");
    let kernels_dir = metal_dir.join("kernels");
    let build_dir = root.join("build").join("metal");
    fs::create_dir_all(&kernels_dir).expect("kernels directory should create");
    fs::create_dir_all(&build_dir).expect("build directory should create");

    let source_path = kernels_dir.join("phase1_dense_path.metal");
    let source_text = phase1_source_text();
    fs::write(&source_path, source_text.as_bytes()).expect("source file should write");
    // The fake xcrun serves `<src>.ast.json` for the AST signature gate;
    // generate it from the same expectation tables the gate validates
    // against, so fixture and gate cannot drift apart.
    fs::write(
        format!("{}.ast.json", source_path.display()),
        synthetic_projection_ast_json(),
    )
    .expect("synthetic AST fixture should write");

    let manifest_path = metal_dir.join("phase1-kernels.json");
    let mut manifest = MetalKernelManifest {
        schema_version: PHASE1_METAL_KERNEL_MANIFEST_SCHEMA_VERSION.to_string(),
        mlx_target: PHASE1_MLX_METAL_TARGET.to_string(),
        metal_language_standard: PHASE1_METAL_LANGUAGE_STANDARD.to_string(),
        library_name: PHASE1_METAL_LIBRARY_NAME.to_string(),
        default_block_size_tokens: PHASE1_DEFAULT_BLOCK_SIZE_TOKENS,
        supported_block_size_tokens: PHASE1_SUPPORTED_BLOCK_SIZE_TOKENS.to_vec(),
        source_file: PathBuf::from("metal/kernels/phase1_dense_path.metal"),
        toolchain_requirements: REQUIRED_TOOLCHAIN_REQUIREMENTS
            .iter()
            .map(|tool| (*tool).to_string())
            .collect(),
        build_gate: PHASE1_METAL_BUILD_GATE.to_string(),
        kernels: phase1_kernel_specs(),
    };
    if let Some(edit) = manifest_edit {
        edit(&mut manifest);
    }
    write_json_file(&manifest_path, &manifest);

    let air_path = build_dir.join("ax_phase1_dense_path.air");
    let metalar_path = build_dir.join("ax_phase1_dense_path.metalar");
    let metallib_path = build_dir.join("ax_phase1_dense_path.metallib");
    let mut outputs = MetalBuildOutputs {
        air: None,
        metalar: None,
        metallib: None,
        air_sha256: None,
        metalar_sha256: None,
        metallib_sha256: None,
    };
    let reason = match status {
        MetalBuildStatus::Compiled => {
            fs::write(&air_path, b"fake-air").expect("air file should write");
            fs::write(&metalar_path, b"fake-metalar").expect("metalar should write");
            fs::write(&metallib_path, b"fake-metallib").expect("metallib should write");
            outputs.air = Some(air_path.clone());
            outputs.metalar = Some(metalar_path.clone());
            outputs.metallib = Some(metallib_path.clone());
            outputs.air_sha256 = Some(sha256_hex(b"fake-air"));
            outputs.metalar_sha256 = Some(sha256_hex(b"fake-metalar"));
            outputs.metallib_sha256 = Some(sha256_hex(b"fake-metallib"));
            None
        }
        MetalBuildStatus::SkippedToolchainUnavailable => {
            Some("Metal toolchain is incomplete on this machine".to_string())
        }
        MetalBuildStatus::SkippedNotReady => {
            Some("AX native bring-up is not allowed on this machine without override".to_string())
        }
        MetalBuildStatus::FailedCompile => Some("command failed with exit code 1".to_string()),
        MetalBuildStatus::Unknown => None,
    };
    let build_report = MetalBuildReport {
        schema_version: PHASE1_METAL_BUILD_REPORT_SCHEMA_VERSION.to_string(),
        manifest_path: manifest_path.clone(),
        source_file: source_path.clone(),
        mlx_target: manifest.mlx_target.clone(),
        metal_language_standard: manifest.metal_language_standard.clone(),
        library_name: manifest.library_name.clone(),
        default_block_size_tokens: manifest.default_block_size_tokens,
        supported_block_size_tokens: manifest.supported_block_size_tokens.clone(),
        toolchain_requirements: manifest.toolchain_requirements.clone(),
        doctor: MetalBuildDoctorReport {
            status: match status {
                MetalBuildStatus::Compiled => "ready".to_string(),
                MetalBuildStatus::SkippedToolchainUnavailable => "not_ready".to_string(),
                MetalBuildStatus::SkippedNotReady => "not_ready".to_string(),
                MetalBuildStatus::FailedCompile => "bringup_only".to_string(),
                MetalBuildStatus::Unknown => "not_ready".to_string(),
            },
            bringup_allowed: matches!(
                status,
                MetalBuildStatus::Compiled | MetalBuildStatus::FailedCompile
            ),
            mlx_runtime_ready: status == MetalBuildStatus::Compiled,
            metal_toolchain_fully_available: !matches!(
                status,
                MetalBuildStatus::SkippedToolchainUnavailable
            ),
            host: MetalBuildHostReport {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                detected_soc: Some("Apple M4 Max".to_string()),
                supported_mlx_runtime: true,
                unsupported_host_override_active: false,
            },
            metal_toolchain: MetalBuildToolchainReport {
                fully_available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                metal: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metal version 36000.4".to_string()),
                },
                metallib: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metallib version 36000.4".to_string()),
                },
                metal_ar: MetalBuildToolStatus {
                    available: !matches!(status, MetalBuildStatus::SkippedToolchainUnavailable),
                    version: Some("Apple metal-ar version 36000.4".to_string()),
                },
            },
        },
        kernels: manifest.kernels.clone(),
        source_sha256: sha256_hex(source_text.as_bytes()),
        outputs,
        compile_commands: if status == MetalBuildStatus::Compiled {
            vec![
                vec!["xcrun".to_string(), "metal".to_string()],
                vec!["xcrun".to_string(), "metal-ar".to_string()],
                vec!["xcrun".to_string(), "metallib".to_string()],
            ]
        } else {
            Vec::new()
        },
        status,
        reason,
    };
    write_json_file(&build_dir.join("build_report.json"), &build_report);

    Phase1Fixture { root, build_dir }
}

fn prepend_to_path(dir: &Path, existing: Option<&OsString>) -> OsString {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(existing) = existing {
        paths.extend(std::env::split_paths(existing));
    }
    std::env::join_paths(paths).expect("PATH should join")
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) {
    let json = serde_json::to_vec_pretty(value).expect("json should serialize");
    fs::write(path, json).expect("json file should write");
}
