from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class FakeNativeSession:
    instances: list["FakeNativeSession"] = []

    def __init__(
        self,
        model_id: str,
        *,
        deterministic: bool = True,
        max_batch_tokens: int = 2048,
        cache_group_id: int = 0,
        block_size_tokens: int = 16,
        total_blocks: int = 1024,
        native_mode: bool = False,
        mlx: bool = False,
        support_tier: str = "compatibility",
        compat_backend: str = "llama_cpp",
        compat_cli_path: str = "llama-cli",
        compat_model_path: str | None = None,
        compat_server_url: str | None = None,
        llama_fallback_cli_path: str = "llama-cli",
        llama_fallback_model_path: str | None = None,
        llama_fallback_server_url: str | None = None,
        native_runtime_artifacts_dir: str | None = None,
        native_model_artifacts_dir: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.native_mode = native_mode
        self.mlx = mlx
        self.support_tier = "native_preview" if native_mode else support_tier
        self.compat_backend = "mlx" if mlx else compat_backend
        self.compat_cli_path = compat_cli_path
        self.compat_model_path = compat_model_path
        self.compat_server_url = compat_server_url
        self.llama_fallback_cli_path = llama_fallback_cli_path
        self.llama_fallback_model_path = llama_fallback_model_path
        self.llama_fallback_server_url = llama_fallback_server_url
        self.native_runtime_artifacts_dir = native_runtime_artifacts_dir
        self.native_model_artifacts_dir = native_model_artifacts_dir
        self.closed = False
        self.cancelled: list[int] = []
        self.generate_calls: list[tuple[list[int], dict[str, object]]] = []
        self.submit_calls: list[tuple[list[int], dict[str, object]]] = []
        self.step_calls = 0
        self.snapshot_calls = 0
        self._snapshot_sequence = [
            {
                "request_id": 11,
                "model_id": model_id,
                "state": "waiting",
                "prompt_tokens": [1, 2, 3],
                "processed_prompt_tokens": 0,
                "output_tokens": [],
                "output_token_logprobs": [],
                "prompt_len": 3,
                "output_len": 0,
                "max_output_tokens": 2,
                "cancel_requested": False,
                "route": {},
            },
            {
                "request_id": 11,
                "model_id": model_id,
                "state": "runnable",
                "prompt_tokens": [1, 2, 3],
                "processed_prompt_tokens": 3,
                "output_tokens": [4],
                "output_token_logprobs": [-0.25],
                "prompt_len": 3,
                "output_len": 1,
                "max_output_tokens": 2,
                "cancel_requested": False,
                "execution_plan_ref": "phase1.qwen3_dense.paged_decode",
                "route": {
                    "execution_plan": "phase1.qwen3_dense.paged_decode",
                    "attention_route": "qwen3_dense_paged_decode",
                    "kv_mode": "paged_metadata",
                    "barrier_mode": "serial",
                },
            },
            {
                "request_id": 11,
                "model_id": model_id,
                "state": "finished",
                "prompt_tokens": [1, 2, 3],
                "processed_prompt_tokens": 3,
                "output_tokens": [4, 5],
                "output_token_logprobs": [-0.25, -0.5],
                "prompt_len": 3,
                "output_len": 2,
                "max_output_tokens": 2,
                "cancel_requested": False,
                "execution_plan_ref": "phase1.qwen3_dense.paged_decode",
                "finish_reason": "max_output_tokens",
                "terminal_stop_reason": "max_output_tokens",
                "route": {
                    "execution_plan": "phase1.qwen3_dense.paged_decode",
                    "attention_route": "qwen3_dense_paged_decode",
                    "kv_mode": "paged_metadata",
                    "barrier_mode": "serial",
                },
            },
        ]
        FakeNativeSession.instances.append(self)

    def close(self) -> None:
        self.closed = True

    def runtime(self) -> dict[str, object]:
        selected_backend = (
            self.compat_backend if self.support_tier == "compatibility" else "ax_native"
        )
        resolution_policy = (
            "allow_compat" if self.support_tier == "compatibility" else "strict_native"
        )
        runtime = {
            "selected_backend": selected_backend,
            "support_tier": self.support_tier,
            "resolution_policy": resolution_policy,
            "capabilities": {
                "text_generation": True,
                "token_streaming": True,
                "deterministic_mode": True,
                "prefix_reuse": True,
                "long_context_validation": "preview",
                "benchmark_metrics": "preview",
            },
        }
        if self.support_tier != "compatibility":
            runtime["native_runtime"] = {
                "runner": "deterministic",
                "artifacts_source": "repo_auto_detect",
            }
            runtime["native_model"] = {
                "artifacts_source": "explicit_config",
                "model_family": "qwen3_dense",
                "tensor_format": "safetensors",
                "layer_count": 36,
                "tensor_count": 512,
                "tie_word_embeddings": False,
                "bindings_prepared": True,
                "buffers_bound": True,
                "buffer_count": 12,
                "buffer_bytes": 4096,
            }
        return runtime

    def generate(self, input_tokens: list[int] | None = None, **kwargs: object) -> dict[str, object]:
        tokens = list(input_tokens or [])
        self.generate_calls.append((tokens, kwargs))
        prompt_text = kwargs.get("input_text")
        result = {
            "request_id": 1,
            "model_id": self.model_id,
            "prompt_tokens": tokens,
            "output_tokens": [4, 5],
            "output_token_logprobs": [-0.25, -0.5],
            "status": "finished",
            "finish_reason": "max_output_tokens",
            "step_count": 3,
            "ttft_step": 2,
            "route": {
                "execution_plan": "phase1.qwen3_dense.paged_decode",
                "attention_route": "qwen3_dense_paged_decode",
                "kv_mode": "paged_metadata",
                "barrier_mode": "serial",
            },
            "runtime": self.runtime(),
        }
        if isinstance(prompt_text, str):
            result["prompt_text"] = prompt_text
            result["output_text"] = f"compat::{prompt_text}"
            result["output_tokens"] = []
        return result

    def stream_generate(
        self, input_tokens: list[int] | None = None, **kwargs: object
    ) -> list[dict[str, object]]:
        tokens = list(input_tokens or [])
        self.generate_calls.append((tokens, kwargs))
        if self.support_tier == "compatibility":
            if self.compat_server_url is None:
                backend_label = {
                    "llama_cpp": "LlamaCpp",
                    "vllm": "Vllm",
                    "mistral_rs": "MistralRs",
                }.get(self.compat_backend, self.compat_backend)
                raise RuntimeError(
                    f"compatibility backend {backend_label} does not support stream_generate in this preview contract"
                )

            prompt_text = kwargs.get("input_text")
            return [
                {
                    "event": "request",
                    "runtime": self.runtime(),
                    "request": {
                        "request_id": 11,
                        "model_id": self.model_id,
                        "state": "waiting",
                        "prompt_tokens": tokens,
                        "processed_prompt_tokens": 0,
                        "output_tokens": [],
                        "output_token_logprobs": [],
                        "prompt_len": len(tokens),
                        "output_len": 0,
                        "max_output_tokens": 2,
                        "cancel_requested": False,
                        "execution_plan_ref": "compatibility.llama_cpp.server_completion_stream",
                        "route": {
                            "execution_plan": "compatibility.llama_cpp.server_completion_stream",
                        },
                    },
                },
                {
                    "event": "step",
                    "request": {
                        "request_id": 11,
                        "model_id": self.model_id,
                        "state": "running",
                        "prompt_tokens": tokens,
                        "processed_prompt_tokens": len(tokens),
                        "output_tokens": [4],
                        "output_token_logprobs": [None],
                        "prompt_len": len(tokens),
                        "output_len": 1,
                        "max_output_tokens": 2,
                        "cancel_requested": False,
                        "execution_plan_ref": "compatibility.llama_cpp.server_completion_stream",
                        "route": {
                            "execution_plan": "compatibility.llama_cpp.server_completion_stream",
                        },
                    },
                    "step": {
                        "step_id": None,
                        "scheduled_requests": 1,
                        "scheduled_tokens": 1,
                        "ttft_events": 1,
                        "prefix_hits": 0,
                        "kv_usage_blocks": 0,
                        "evictions": 0,
                        "cpu_time_us": 0,
                        "runner_time_us": 0,
                    },
                    "delta_tokens": [4],
                    "delta_token_logprobs": [None],
                    "delta_text": "compat",
                },
                {
                    "event": "step",
                    "request": {
                        "request_id": 11,
                        "model_id": self.model_id,
                        "state": "finished",
                        "prompt_tokens": tokens,
                        "processed_prompt_tokens": len(tokens),
                        "output_tokens": [4, 5],
                        "output_token_logprobs": [None, None],
                        "prompt_len": len(tokens),
                        "output_len": 2,
                        "max_output_tokens": 2,
                        "cancel_requested": False,
                        "execution_plan_ref": "compatibility.llama_cpp.server_completion_stream",
                        "finish_reason": "max_output_tokens",
                        "terminal_stop_reason": "max_output_tokens",
                        "route": {
                            "execution_plan": "compatibility.llama_cpp.server_completion_stream",
                        },
                    },
                    "step": {
                        "step_id": None,
                        "scheduled_requests": 1,
                        "scheduled_tokens": 1,
                        "ttft_events": 0,
                        "prefix_hits": 0,
                        "kv_usage_blocks": 0,
                        "evictions": 0,
                        "cpu_time_us": 0,
                        "runner_time_us": 0,
                    },
                    "delta_tokens": [5],
                    "delta_token_logprobs": [None],
                    "delta_text": " stream",
                },
                {
                    "event": "response",
                    "response": {
                        "request_id": 11,
                        "model_id": self.model_id,
                        "prompt_tokens": tokens,
                        "prompt_text": prompt_text if isinstance(prompt_text, str) else None,
                        "output_tokens": [4, 5],
                        "output_token_logprobs": [None, None],
                        "output_text": (
                            f"compat::{prompt_text}" if isinstance(prompt_text, str) else "compat::stream"
                        ),
                        "status": "finished",
                        "finish_reason": "max_output_tokens",
                        "step_count": 2,
                        "ttft_step": 1,
                        "route": {
                            "execution_plan": "compatibility.llama_cpp.server_completion_stream",
                        },
                        "runtime": self.runtime(),
                    },
                },
            ]
        return [
            {
                "event": "request",
                "runtime": self.runtime(),
                "request": self._snapshot_sequence[0],
            },
            {
                "event": "step",
                "request": {
                    "request_id": 11,
                    "model_id": self.model_id,
                    "state": "runnable",
                    "prompt_tokens": [1, 2, 3],
                    "processed_prompt_tokens": 3,
                    "output_tokens": [],
                    "output_token_logprobs": [],
                    "prompt_len": 3,
                    "output_len": 0,
                    "max_output_tokens": 2,
                    "cancel_requested": False,
                    "execution_plan_ref": "phase1.qwen3_dense.dense_prefill",
                    "route": {
                        "execution_plan": "phase1.qwen3_dense.dense_prefill",
                        "attention_route": "qwen3_dense_prefill",
                        "kv_mode": "paged_metadata",
                        "barrier_mode": "serial",
                    },
                },
                "step": {
                    "step_id": 0,
                    "scheduled_requests": 1,
                    "scheduled_tokens": 3,
                    "ttft_events": 0,
                    "prefix_hits": 0,
                    "kv_usage_blocks": 1,
                    "evictions": 0,
                    "cpu_time_us": 21,
                    "runner_time_us": 1,
                    "route": {
                        "execution_plan": "phase1.qwen3_dense.dense_prefill",
                        "attention_route": "qwen3_dense_prefill",
                        "kv_mode": "paged_metadata",
                        "barrier_mode": "serial",
                    },
                },
                "delta_tokens": [],
                "delta_token_logprobs": [],
            },
            {
                "event": "step",
                "request": self._snapshot_sequence[1],
                "step": {
                    "step_id": 1,
                    "scheduled_requests": 1,
                    "scheduled_tokens": 1,
                    "ttft_events": 1,
                    "prefix_hits": 0,
                    "kv_usage_blocks": 1,
                    "evictions": 0,
                    "cpu_time_us": 20,
                    "runner_time_us": 1,
                    "route": {
                        "execution_plan": "phase1.qwen3_dense.paged_decode",
                        "attention_route": "qwen3_dense_paged_decode",
                        "kv_mode": "paged_metadata",
                        "barrier_mode": "serial",
                    },
                    "metal_dispatch": {
                        "command_queue_label": "ax.queue",
                        "command_buffer_label": "ax.buffer",
                        "command_buffer_status": "completed",
                        "runtime_device_name": "Apple M4 Max",
                        "runtime_required_pipeline_count": 4,
                        "runtime_max_thread_execution_width": 64,
                        "runtime_model_conditioned_inputs": True,
                        "runtime_real_model_tensor_inputs": True,
                        "runtime_complete_model_forward_supported": True,
                        "runtime_model_bindings_prepared": True,
                        "runtime_model_buffers_bound": True,
                        "runtime_model_buffer_count": 12,
                        "runtime_model_buffer_bytes": 4096,
                        "runtime_model_family": "qwen3_dense",
                        "execution_direct_decode_token_count": 1,
                        "execution_direct_decode_checksum_lo": 4660,
                        "execution_logits_output_count": 1,
                        "execution_remaining_logits_handle_count": 0,
                        "execution_model_bound_ffn_decode": True,
                        "execution_real_model_forward_completed": True,
                        "execution_prefix_native_dispatch_count": 35,
                        "execution_prefix_cpu_reference_dispatch_count": 1,
                        "execution_qkv_projection_token_count": 72,
                        "execution_layer_continuation_token_count": 37,
                        "execution_logits_projection_token_count": 1,
                        "execution_logits_vocab_scan_row_count": 151936,
                        "binary_archive_state": "loaded",
                        "binary_archive_attached_pipeline_count": 4,
                        "binary_archive_serialized": True,
                        "arena_token_capacity": 8,
                        "arena_slot_capacity": 64,
                        "arena_attention_ref_capacity": 8,
                        "arena_gather_ref_capacity": 8,
                        "arena_gather_output_capacity": 8,
                        "arena_copy_pair_capacity": 4,
                        "arena_sequence_capacity": 4,
                        "arena_reused_existing": True,
                        "arena_grew_existing": False,
                        "kernels": [
                            {
                                "function_name": "reshape_and_cache",
                                "element_count": 32,
                                "threads_per_grid_width": 32,
                                "threads_per_threadgroup_width": 32,
                            }
                        ],
                        "numeric": {
                            "key_cache_checksum": 1,
                            "attention_output_checksum": 2,
                            "gather_output_checksum": 3,
                            "copy_output_checksum": 4,
                            "validation": {
                                "expected_key_cache_checksum": 1,
                                "expected_attention_output_checksum": 2,
                                "expected_gather_output_checksum": 3,
                                "expected_copy_output_checksum": 4,
                                "attention_max_abs_diff_microunits": 0,
                            },
                        },
                    },
                },
                "delta_tokens": [4],
                "delta_token_logprobs": [-0.25],
            },
            {
                "event": "step",
                "request": self._snapshot_sequence[2],
                "step": {
                    "step_id": 2,
                    "scheduled_requests": 1,
                    "scheduled_tokens": 1,
                    "ttft_events": 0,
                    "prefix_hits": 0,
                    "kv_usage_blocks": 0,
                    "evictions": 0,
                    "cpu_time_us": 18,
                    "runner_time_us": 0,
                    "route": {
                        "execution_plan": "phase1.qwen3_dense.paged_decode",
                        "attention_route": "qwen3_dense_paged_decode",
                        "kv_mode": "paged_metadata",
                        "barrier_mode": "serial",
                    },
                },
                "delta_tokens": [5],
                "delta_token_logprobs": [-0.5],
            },
            {
                "event": "response",
                "response": {
                    "request_id": 11,
                    "model_id": self.model_id,
                    "prompt_tokens": [1, 2, 3],
                    "output_tokens": [4, 5],
                    "output_token_logprobs": [-0.25, -0.5],
                    "status": "finished",
                    "finish_reason": "max_output_tokens",
                    "step_count": 3,
                    "ttft_step": 2,
                    "route": {
                        "execution_plan": "phase1.qwen3_dense.paged_decode",
                        "attention_route": "qwen3_dense_paged_decode",
                        "kv_mode": "paged_metadata",
                        "barrier_mode": "serial",
                    },
                    "runtime": self.runtime(),
                },
            },
        ]

    def submit(self, input_tokens: list[int] | None = None, **kwargs: object) -> int:
        self.submit_calls.append((list(input_tokens or []), kwargs))
        return 11

    def step(self) -> dict[str, object]:
        self.step_calls += 1
        if self.step_calls == 1:
            return {
                "step_id": 1,
                "scheduled_requests": 1,
                "scheduled_tokens": 1,
                "ttft_events": 1,
                "prefix_hits": 0,
                "kv_usage_blocks": 1,
                "evictions": 0,
                "cpu_time_us": 20,
                "runner_time_us": 1,
                "route": {
                    "execution_plan": "phase1.qwen3_dense.paged_decode",
                    "attention_route": "qwen3_dense_paged_decode",
                    "kv_mode": "paged_metadata",
                    "barrier_mode": "serial",
                },
                "metal_dispatch": {
                    "command_queue_label": "ax.queue",
                    "command_buffer_label": "ax.buffer",
                    "command_buffer_status": "completed",
                    "runtime_device_name": "Apple M4 Max",
                    "runtime_required_pipeline_count": 4,
                    "runtime_max_thread_execution_width": 64,
                    "runtime_model_conditioned_inputs": True,
                    "runtime_real_model_tensor_inputs": True,
                    "runtime_complete_model_forward_supported": True,
                    "runtime_model_bindings_prepared": True,
                    "runtime_model_buffers_bound": True,
                    "runtime_model_buffer_count": 12,
                    "runtime_model_buffer_bytes": 4096,
                    "runtime_model_family": "qwen3_dense",
                    "execution_direct_decode_token_count": 1,
                    "execution_direct_decode_checksum_lo": 4660,
                    "execution_logits_output_count": 1,
                    "execution_remaining_logits_handle_count": 0,
                    "execution_model_bound_ffn_decode": True,
                    "execution_real_model_forward_completed": True,
                    "execution_prefix_native_dispatch_count": 35,
                    "execution_prefix_cpu_reference_dispatch_count": 1,
                    "execution_qkv_projection_token_count": 72,
                    "execution_layer_continuation_token_count": 37,
                    "execution_logits_projection_token_count": 1,
                    "execution_logits_vocab_scan_row_count": 151936,
                    "binary_archive_state": "loaded",
                    "binary_archive_attached_pipeline_count": 4,
                    "binary_archive_serialized": True,
                    "arena_token_capacity": 8,
                    "arena_slot_capacity": 64,
                    "arena_attention_ref_capacity": 8,
                    "arena_gather_ref_capacity": 8,
                    "arena_gather_output_capacity": 8,
                    "arena_copy_pair_capacity": 4,
                    "arena_sequence_capacity": 4,
                    "arena_reused_existing": True,
                    "arena_grew_existing": False,
                    "kernels": [
                        {
                            "function_name": "reshape_and_cache",
                            "element_count": 32,
                            "threads_per_grid_width": 32,
                            "threads_per_threadgroup_width": 32,
                        }
                    ],
                    "numeric": {
                        "key_cache_checksum": 1,
                        "attention_output_checksum": 2,
                        "gather_output_checksum": 3,
                        "copy_output_checksum": 4,
                        "validation": {
                            "expected_key_cache_checksum": 1,
                            "expected_attention_output_checksum": 2,
                            "expected_gather_output_checksum": 3,
                            "expected_copy_output_checksum": 4,
                            "attention_max_abs_diff_microunits": 0,
                        },
                    },
                },
            }

        return {
            "step_id": 2,
            "scheduled_requests": 1,
            "scheduled_tokens": 1,
            "ttft_events": 0,
            "prefix_hits": 0,
            "kv_usage_blocks": 0,
            "evictions": 0,
            "cpu_time_us": 18,
            "runner_time_us": 0,
            "route": {
                "execution_plan": "phase1.qwen3_dense.paged_decode",
                "attention_route": "qwen3_dense_paged_decode",
                "kv_mode": "paged_metadata",
                "barrier_mode": "serial",
            },
        }

    def snapshot(self, request_id: int) -> dict[str, object] | None:
        if request_id != 11:
            return None
        index = min(self.snapshot_calls, len(self._snapshot_sequence) - 1)
        snapshot = self._snapshot_sequence[index]
        self.snapshot_calls += 1
        return snapshot

    def cancel(self, request_id: int) -> None:
        self.cancelled.append(request_id)


class HungNativeSession(FakeNativeSession):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._snapshot_sequence = [
            {
                "request_id": 11,
                "model_id": self.model_id,
                "state": "runnable",
                "prompt_tokens": [9],
                "processed_prompt_tokens": 1,
                "output_tokens": [],
                "output_token_logprobs": [],
                "prompt_len": 1,
                "output_len": 0,
                "max_output_tokens": 1,
                "cancel_requested": False,
                "route": {},
            }
        ]

    def stream_generate(
        self, input_tokens: list[int] | None = None, **kwargs: object
    ) -> list[dict[str, object]]:
        self.generate_calls.append((list(input_tokens or []), kwargs))
        raise RuntimeError("request 11 did not terminate within 258 steps")


def import_wrapper_module(session_cls: type[FakeNativeSession] = FakeNativeSession) -> types.ModuleType:
    sys.path.insert(0, str(SOURCE_ROOT))
    for name in list(sys.modules):
        if name == "ax_engine" or name.startswith("ax_engine."):
            del sys.modules[name]

    native_module = types.ModuleType("ax_engine._ax_engine")
    native_module.Session = session_cls
    sys.modules["ax_engine._ax_engine"] = native_module
    return importlib.import_module("ax_engine")


class WrapperContractTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeNativeSession.instances.clear()
        self.ax_engine = import_wrapper_module()

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "ax_engine" or name.startswith("ax_engine."):
                del sys.modules[name]
        if str(SOURCE_ROOT) in sys.path:
            sys.path.remove(str(SOURCE_ROOT))

    def test_generate_converts_native_payload_to_dataclass(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
            result = session.generate([1, 2, 3], max_output_tokens=2)

        self.assertEqual(result.request_id, 1)
        self.assertEqual(result.model_id, "qwen3_5_9b_q4")
        self.assertEqual(result.prompt_tokens, [1, 2, 3])
        self.assertEqual(result.output_tokens, [4, 5])
        self.assertEqual(result.output_token_logprobs, [-0.25, -0.5])
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.finish_reason, "max_output_tokens")
        self.assertEqual(result.runtime.support_tier, "native_preview")
        self.assertEqual(result.runtime.host.os, "")
        self.assertFalse(result.runtime.metal_toolchain.fully_available)
        self.assertEqual(result.runtime.native_runtime.runner, "deterministic")
        self.assertEqual(result.runtime.native_model.model_family, "qwen3_dense")
        self.assertTrue(result.runtime.native_model.bindings_prepared)
        self.assertEqual(result.route.execution_plan, "phase1.qwen3_dense.paged_decode")

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])
        self.assertEqual(native.generate_calls[0][1]["max_output_tokens"], 2)
        self.assertTrue(native.closed)

    def test_generate_supports_text_requests_for_compatibility_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_cli_path="/tmp/llama-cli",
            compat_model_path="/tmp/model.gguf",
        ) as session:
            result = session.generate(input_text="hello wrapper", max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.support_tier, "compatibility")
        self.assertEqual(native.compat_cli_path, "/tmp/llama-cli")
        self.assertEqual(native.compat_model_path, "/tmp/model.gguf")
        self.assertEqual(native.generate_calls[0][0], [])
        self.assertEqual(native.generate_calls[0][1]["input_text"], "hello wrapper")
        self.assertEqual(result.prompt_text, "hello wrapper")
        self.assertEqual(result.output_text, "compat::hello wrapper")
        self.assertEqual(result.runtime.selected_backend, "llama_cpp")
        self.assertEqual(result.runtime.support_tier, "compatibility")

    def test_generate_supports_server_backed_compatibility_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_server_url="http://127.0.0.1:8081",
        ) as session:
            result = session.generate([1, 2, 3], max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.support_tier, "compatibility")
        self.assertEqual(native.compat_server_url, "http://127.0.0.1:8081")
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])
        self.assertEqual(result.output_tokens, [4, 5])
        self.assertEqual(result.runtime.selected_backend, "llama_cpp")

    def test_session_forwards_explicit_native_artifact_dirs(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_5_9b_q4",
            native_mode=True,
            native_runtime_artifacts_dir="/tmp/ax-metal",
            native_model_artifacts_dir="/tmp/ax-model",
        ) as session:
            runtime = session.runtime()

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.native_runtime_artifacts_dir, "/tmp/ax-metal")
        self.assertEqual(native.native_model_artifacts_dir, "/tmp/ax-model")
        self.assertEqual(runtime.selected_backend, "ax_native")

    def test_generate_text_convenience_uses_input_text_contract(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_cli_path="/tmp/llama-cli",
            compat_model_path="/tmp/model.gguf",
        ) as session:
            result = session.generate_text("hello text helper", max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.generate_calls[0][0], [])
        self.assertEqual(native.generate_calls[0][1]["input_text"], "hello text helper")
        self.assertEqual(result.output_text, "compat::hello text helper")

    def test_chat_convenience_flattens_messages_to_prompt(self) -> None:
        messages = [
            self.ax_engine.ChatMessage(role="system", content="You are AX"),
            {"role": "user", "content": "Say hello"},
        ]

        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_cli_path="/tmp/llama-cli",
            compat_model_path="/tmp/model.gguf",
        ) as session:
            result = session.chat(messages, max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "system: You are AX\nuser: Say hello\nassistant:",
        )
        self.assertEqual(
            result.prompt_text,
            "system: You are AX\nuser: Say hello\nassistant:",
        )

    def test_submit_chat_convenience_reuses_text_prompt_path(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_server_url="http://127.0.0.1:8081",
        ) as session:
            request_id = session.submit_chat(
                [{"role": "user", "content": "queue this"}],
                max_output_tokens=2,
            )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(request_id, 11)
        self.assertEqual(
            native.submit_calls[0][1]["input_text"],
            "user: queue this\nassistant:",
        )

    def test_stepwise_controls_convert_native_payloads(self) -> None:
        session = self.ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True)

        request_id = session.submit([1, 2, 3], max_output_tokens=2)
        initial = session.snapshot(request_id)
        step = session.step()
        session.cancel(request_id)

        self.assertEqual(request_id, 11)
        self.assertEqual(initial.state, "waiting")
        self.assertEqual(initial.prompt_tokens, [1, 2, 3])
        self.assertEqual(initial.output_token_logprobs, [])
        self.assertEqual(step.scheduled_requests, 1)
        self.assertEqual(step.ttft_events, 1)
        self.assertEqual(step.route.execution_plan, "phase1.qwen3_dense.paged_decode")
        self.assertEqual(step.metal_dispatch.runtime_model_family, "qwen3_dense")
        self.assertTrue(step.metal_dispatch.runtime_real_model_tensor_inputs)
        self.assertTrue(step.metal_dispatch.runtime_complete_model_forward_supported)
        self.assertEqual(step.metal_dispatch.execution_direct_decode_token_count, 1)
        self.assertEqual(step.metal_dispatch.execution_logits_output_count, 1)
        self.assertEqual(step.metal_dispatch.execution_remaining_logits_handle_count, 0)
        self.assertTrue(step.metal_dispatch.execution_model_bound_ffn_decode)
        self.assertTrue(step.metal_dispatch.execution_real_model_forward_completed)
        self.assertEqual(step.metal_dispatch.execution_prefix_native_dispatch_count, 35)
        self.assertEqual(step.metal_dispatch.execution_prefix_cpu_reference_dispatch_count, 1)
        self.assertEqual(step.metal_dispatch.execution_qkv_projection_token_count, 72)
        self.assertEqual(step.metal_dispatch.execution_layer_continuation_token_count, 37)
        self.assertEqual(step.metal_dispatch.execution_logits_projection_token_count, 1)
        self.assertEqual(
            step.metal_dispatch.execution_logits_vocab_scan_row_count, 151936
        )
        self.assertEqual(step.metal_dispatch.runtime_model_buffer_count, 12)
        self.assertEqual(
            step.metal_dispatch.numeric.validation.attention_max_abs_diff_microunits, 0
        )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.submit_calls[0][0], [1, 2, 3])
        self.assertEqual(native.submit_calls[0][1]["max_output_tokens"], 2)
        self.assertEqual(native.cancelled, [11])

    def test_stream_generate_emits_request_step_and_response_events(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
            events = list(session.stream_generate([1, 2, 3], max_output_tokens=2))

        self.assertEqual(
            [event.event for event in events],
            ["request", "step", "step", "step", "response"],
        )
        self.assertEqual(events[0].request.state, "waiting")
        self.assertEqual(events[0].runtime.support_tier, "native_preview")
        self.assertEqual(events[1].delta_tokens, [])
        self.assertEqual(events[1].delta_token_logprobs, [])
        self.assertEqual(events[1].step.ttft_events, 0)
        self.assertEqual(events[1].step.route.execution_plan, "phase1.qwen3_dense.dense_prefill")
        self.assertEqual(events[2].delta_tokens, [4])
        self.assertEqual(events[2].delta_token_logprobs, [-0.25])
        self.assertEqual(events[2].step.ttft_events, 1)
        self.assertTrue(events[2].step.metal_dispatch.runtime_model_conditioned_inputs)
        self.assertTrue(events[2].step.metal_dispatch.runtime_complete_model_forward_supported)
        self.assertEqual(events[2].step.metal_dispatch.runtime_model_family, "qwen3_dense")
        self.assertEqual(events[2].step.metal_dispatch.execution_direct_decode_token_count, 1)
        self.assertEqual(events[2].step.metal_dispatch.execution_logits_output_count, 1)
        self.assertTrue(events[2].step.metal_dispatch.execution_real_model_forward_completed)
        self.assertEqual(events[2].step.metal_dispatch.execution_prefix_native_dispatch_count, 35)
        self.assertEqual(
            events[2].step.metal_dispatch.execution_prefix_cpu_reference_dispatch_count, 1
        )
        self.assertEqual(
            events[2].step.metal_dispatch.execution_qkv_projection_token_count, 72
        )
        self.assertEqual(
            events[2].step.metal_dispatch.execution_layer_continuation_token_count, 37
        )
        self.assertEqual(
            events[2].step.metal_dispatch.execution_logits_projection_token_count, 1
        )
        self.assertEqual(
            events[2].step.metal_dispatch.execution_logits_vocab_scan_row_count, 151936
        )
        self.assertEqual(events[3].delta_tokens, [5])
        self.assertEqual(events[3].delta_token_logprobs, [-0.5])
        self.assertEqual(events[3].request.state, "finished")
        self.assertEqual(events[3].request.finish_reason, "max_output_tokens")
        self.assertEqual(events[3].request.terminal_stop_reason, "max_output_tokens")
        self.assertEqual(events[4].response.output_tokens, [4, 5])
        self.assertEqual(events[4].response.output_token_logprobs, [-0.25, -0.5])
        self.assertEqual(events[4].response.step_count, 3)
        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])
        self.assertEqual(native.generate_calls[0][1]["max_output_tokens"], 2)

    def test_stream_generate_raises_when_request_never_terminates(self) -> None:
        self.ax_engine = import_wrapper_module(HungNativeSession)

        with self.ax_engine.Session(model_id="qwen3_5_9b_q4", native_mode=True) as session:
            with self.assertRaisesRegex(
                RuntimeError,
                r"request 11 did not terminate within 258 steps",
            ):
                list(session.stream_generate([9], max_output_tokens=1))

    def test_stream_generate_supports_server_backed_compatibility_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_server_url="http://127.0.0.1:8081",
        ) as session:
            events = list(session.stream_generate([1, 2, 3], max_output_tokens=2))

        self.assertEqual(
            [event.event for event in events],
            ["request", "step", "step", "response"],
        )
        self.assertEqual(events[0].runtime.support_tier, "compatibility")
        self.assertEqual(events[1].request.state, "running")
        self.assertEqual(events[1].delta_tokens, [4])
        self.assertEqual(events[1].delta_token_logprobs, [None])
        self.assertEqual(events[1].delta_text, "compat")
        self.assertEqual(events[2].request.state, "finished")
        self.assertEqual(events[2].request.finish_reason, "max_output_tokens")
        self.assertEqual(events[2].request.terminal_stop_reason, "max_output_tokens")
        self.assertEqual(events[2].delta_token_logprobs, [None])
        self.assertEqual(events[2].delta_text, " stream")
        self.assertEqual(events[3].response.output_tokens, [4, 5])
        self.assertEqual(events[3].response.output_token_logprobs, [None, None])
        self.assertEqual(
            events[3].response.route.execution_plan,
            "compatibility.llama_cpp.server_completion_stream",
        )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.compat_server_url, "http://127.0.0.1:8081")
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])

    def test_stream_text_convenience_uses_input_text_contract(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_server_url="http://127.0.0.1:8081",
        ) as session:
            events = list(session.stream_text("hello streamed text", max_output_tokens=2))

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "hello streamed text",
        )
        self.assertEqual(events[-1].response.prompt_text, "hello streamed text")

    def test_stream_chat_convenience_flattens_messages_to_prompt(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="compatibility",
            compat_server_url="http://127.0.0.1:8081",
        ) as session:
            events = list(
                session.stream_chat(
                    [{"role": "user", "content": "hello chat helper"}],
                    max_output_tokens=2,
                )
            )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "user: hello chat helper\nassistant:",
        )
        self.assertEqual(
            events[-1].response.prompt_text,
            "user: hello chat helper\nassistant:",
        )

    def test_chat_convenience_rejects_empty_messages(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_dense") as session:
            with self.assertRaisesRegex(ValueError, "chat requires at least one message"):
                session.chat([], max_output_tokens=2)


if __name__ == "__main__":
    unittest.main()
