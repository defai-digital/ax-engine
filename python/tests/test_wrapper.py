from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FAKE_MLX_MODEL_DIR = "/tmp/ax-engine-test-mlx-model"


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
        mlx: bool = False,
        support_tier: str = "llama_cpp",
        llama_cli_path: str = "llama-cli",
        llama_model_path: str | None = None,
        llama_server_url: str | None = None,
        mlx_lm_server_url: str | None = None,
        mlx_model_artifacts_dir: str | None = None,
        delegated_http_connect_timeout_secs: int = 30,
        delegated_http_read_timeout_secs: int = 300,
        delegated_http_write_timeout_secs: int = 300,
    ) -> None:
        self.model_id = model_id
        self.mlx = mlx
        self.support_tier = "mlx_preview" if mlx else support_tier
        self.llama_cli_path = llama_cli_path
        self.llama_model_path = llama_model_path
        self.llama_server_url = llama_server_url
        self.mlx_lm_server_url = mlx_lm_server_url
        self.mlx_model_artifacts_dir = mlx_model_artifacts_dir
        self.delegated_http_connect_timeout_secs = delegated_http_connect_timeout_secs
        self.delegated_http_read_timeout_secs = delegated_http_read_timeout_secs
        self.delegated_http_write_timeout_secs = delegated_http_write_timeout_secs
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
            "llama_cpp" if self.support_tier == "llama_cpp" else "mlx"
        )
        resolution_policy = (
            "allow_llama_cpp" if self.support_tier == "llama_cpp" else "mlx_only"
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
        if self.support_tier != "llama_cpp":
            runtime["mlx_runtime"] = {
                "runner": "metal_bringup",
                "artifacts_source": "repo_auto_detect",
            }
            runtime["mlx_model"] = {
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
            result["output_text"] = f"llama::{prompt_text}"
            result["output_tokens"] = []
        return result

    def stream_generate(
        self, input_tokens: list[int] | None = None, **kwargs: object
    ) -> list[dict[str, object]]:
        tokens = list(input_tokens or [])
        self.generate_calls.append((tokens, kwargs))
        if self.support_tier == "llama_cpp":
            if self.llama_server_url is None:
                raise RuntimeError(
                    f"llama.cpp backend LlamaCpp does not support stream_generate in this preview contract"
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
                        "execution_plan_ref": "llama_cpp.server_completion_stream",
                        "route": {
                            "execution_plan": "llama_cpp.server_completion_stream",
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
                        "execution_plan_ref": "llama_cpp.server_completion_stream",
                        "route": {
                            "execution_plan": "llama_cpp.server_completion_stream",
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
                    "delta_text": "llama",
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
                        "execution_plan_ref": "llama_cpp.server_completion_stream",
                        "finish_reason": "max_output_tokens",
                        "terminal_stop_reason": "max_output_tokens",
                        "route": {
                            "execution_plan": "llama_cpp.server_completion_stream",
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
                            f"llama::{prompt_text}" if isinstance(prompt_text, str) else "llama::stream"
                        ),
                        "status": "finished",
                        "finish_reason": "max_output_tokens",
                        "step_count": 2,
                        "ttft_step": 1,
                        "route": {
                            "execution_plan": "llama_cpp.server_completion_stream",
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
    native_module.EngineError = RuntimeError
    native_module.EngineBackendError = RuntimeError
    native_module.EngineInferenceError = RuntimeError
    native_module.EngineStateError = RuntimeError
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

    def test_generate_converts_mlx_payload_to_dataclass(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            mlx=True,
            mlx_model_artifacts_dir=FAKE_MLX_MODEL_DIR,
        ) as session:
            result = session.generate([1, 2, 3], max_output_tokens=2)

        self.assertEqual(result.request_id, 1)
        self.assertEqual(result.model_id, "qwen3_dense")
        self.assertEqual(result.prompt_tokens, [1, 2, 3])
        self.assertEqual(result.output_tokens, [4, 5])
        self.assertEqual(result.output_token_logprobs, [-0.25, -0.5])
        self.assertEqual(result.status, "finished")
        self.assertEqual(result.finish_reason, "max_output_tokens")
        self.assertEqual(result.runtime.support_tier, "mlx_preview")
        self.assertEqual(result.runtime.host.os, "")
        self.assertFalse(result.runtime.metal_toolchain.fully_available)
        self.assertEqual(result.runtime.mlx_runtime.runner, "metal_bringup")
        self.assertEqual(result.runtime.mlx_model.model_family, "qwen3_dense")
        self.assertTrue(result.runtime.mlx_model.bindings_prepared)
        self.assertEqual(result.route.execution_plan, "phase1.qwen3_dense.paged_decode")

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.mlx_model_artifacts_dir, FAKE_MLX_MODEL_DIR)
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])
        self.assertEqual(native.generate_calls[0][1]["max_output_tokens"], 2)
        self.assertTrue(native.closed)

    def test_generate_supports_text_requests_for_llama_cpp_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            result = session.generate(input_text="hello wrapper", max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.support_tier, "llama_cpp")
        self.assertEqual(native.llama_cli_path, "/tmp/llama-cli")
        self.assertEqual(native.llama_model_path, "/tmp/model.gguf")
        self.assertEqual(native.generate_calls[0][0], [])
        self.assertEqual(native.generate_calls[0][1]["input_text"], "hello wrapper")
        self.assertEqual(result.prompt_text, "hello wrapper")
        self.assertEqual(result.output_text, "llama::hello wrapper")
        self.assertEqual(result.runtime.selected_backend, "llama_cpp")
        self.assertEqual(result.runtime.support_tier, "llama_cpp")

    def test_generate_supports_server_backed_llama_cpp_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_server_url="http://127.0.0.1:8081",
        ) as session:
            result = session.generate([1, 2, 3], max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.support_tier, "llama_cpp")
        self.assertEqual(native.llama_server_url, "http://127.0.0.1:8081")
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])
        self.assertEqual(result.output_tokens, [4, 5])
        self.assertEqual(result.runtime.selected_backend, "llama_cpp")

    def test_session_forwards_delegated_server_options(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="mlx_lm_delegated",
            mlx_lm_server_url="http://127.0.0.1:8090",
            delegated_http_connect_timeout_secs=2,
            delegated_http_read_timeout_secs=11,
            delegated_http_write_timeout_secs=13,
        ):
            pass

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.support_tier, "mlx_lm_delegated")
        self.assertEqual(native.mlx_lm_server_url, "http://127.0.0.1:8090")
        self.assertEqual(native.delegated_http_connect_timeout_secs, 2)
        self.assertEqual(native.delegated_http_read_timeout_secs, 11)
        self.assertEqual(native.delegated_http_write_timeout_secs, 13)

    def test_custom_engine_exceptions_are_reexported(self) -> None:
        self.assertIs(self.ax_engine.EngineError, RuntimeError)
        self.assertIs(self.ax_engine.EngineBackendError, RuntimeError)
        self.assertIs(self.ax_engine.EngineInferenceError, RuntimeError)
        self.assertIs(self.ax_engine.EngineStateError, RuntimeError)

    def test_session_forwards_explicit_mlx_artifact_dirs(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            mlx=True,
            mlx_model_artifacts_dir="/tmp/mlx-model",
        ) as session:
            runtime = session.runtime()

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.mlx_model_artifacts_dir, "/tmp/mlx-model")
        self.assertEqual(runtime.selected_backend, "mlx")

    def test_mlx_session_requires_model_artifact_dir_or_env(self) -> None:
        with self.assertRaisesRegex(ValueError, "mlx=True requires mlx_model_artifacts_dir"):
            self.ax_engine.Session(model_id="qwen3_dense", mlx=True)

    def test_openai_mlx_shim_helpers_tokenize_and_render_chat_prompt(self) -> None:
        openai_server = importlib.import_module("ax_engine.openai_server")

        class FakeTokenizer:
            def encode(self, text: str) -> object:
                return types.SimpleNamespace(ids=[ord(ch) for ch in text])

        tokens, prompt_text = openai_server.prompt_to_tokens("AX", FakeTokenizer())
        self.assertEqual(tokens, [65, 88])
        self.assertEqual(prompt_text, "AX")
        token_prompt, token_prompt_text = openai_server.prompt_to_tokens(
            [1, 2, 3], FakeTokenizer()
        )
        self.assertEqual(token_prompt, [1, 2, 3])
        self.assertIsNone(token_prompt_text)
        self.assertEqual(
            openai_server.render_chat_prompt(
                [
                    {"role": "system", "content": "You are AX"},
                    {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
                ],
                "qwen3_dense",
            ),
            "<|im_start|>system\nYou are AX<|im_end|>\n"
            "<|im_start|>user\nSay hi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            openai_server.render_chat_prompt(
                [
                    {"role": "system", "content": "You are AX"},
                    {"role": "user", "content": "Say hi"},
                ],
                "Meta-Llama-3.1-8B-Instruct",
            ),
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\nYou are AX<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nSay hi<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        self.assertEqual(
            openai_server.render_chat_prompt(
                [{"role": "user", "content": "Line 1\nLine 2"}],
                "unknown-local-model",
            ),
            "user: Line 1\\nLine 2\nassistant:",
        )

    def test_qwen_chat_prompt_matches_real_tokenizer_enable_thinking_false(self) -> None:
        openai_server = importlib.import_module("ax_engine.openai_server")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                "mlx-community/Qwen3-4B-4bit",
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            self.skipTest(f"cached Qwen tokenizer is unavailable: {exc}")

        messages = [
            {"role": "system", "content": "You are AX"},
            {"role": "user", "content": "Say hi"},
        ]
        expected = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        self.assertEqual(
            self.ax_engine._render_chat_prompt(messages, "qwen3_dense"),
            expected,
        )
        self.assertEqual(
            openai_server.render_chat_prompt(messages, "qwen3_dense"),
            expected,
        )

    def test_openai_mlx_shim_builds_mlx_session_with_artifacts_dir(self) -> None:
        openai_server = importlib.import_module("ax_engine.openai_server")

        session = openai_server.build_session(
            model_id="qwen3_dense",
            mlx_model_artifacts_dir="/tmp/mlx-model",
            session_factory=FakeNativeSession,
            session_kwargs={"deterministic": False},
        )

        self.assertIsInstance(session, FakeNativeSession)
        self.assertEqual(session.model_id, "qwen3_dense")
        self.assertTrue(session.mlx)
        self.assertEqual(session.mlx_model_artifacts_dir, "/tmp/mlx-model")

    def test_generate_text_convenience_uses_input_text_contract(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            result = session.generate_text("hello text helper", max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.generate_calls[0][0], [])
        self.assertEqual(native.generate_calls[0][1]["input_text"], "hello text helper")
        self.assertEqual(result.output_text, "llama::hello text helper")

    def test_chat_convenience_flattens_messages_to_prompt(self) -> None:
        messages = [
            self.ax_engine.ChatMessage(role="system", content="You are AX"),
            {"role": "user", "content": "Say hello"},
        ]

        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            result = session.chat(messages, max_output_tokens=2)

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "<|im_start|>system\nYou are AX<|im_end|>\n"
            "<|im_start|>user\nSay hello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            result.prompt_text,
            "<|im_start|>system\nYou are AX<|im_end|>\n"
            "<|im_start|>user\nSay hello<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_chat_convenience_uses_llama3_template_for_llama3_models(self) -> None:
        with self.ax_engine.Session(
            model_id="Meta-Llama-3.1-8B-Instruct",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            session.chat(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
                max_output_tokens=2,
            )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\nBe concise.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_chat_convenience_keeps_plain_fallback_for_unknown_models(self) -> None:
        with self.ax_engine.Session(
            model_id="unknown-local-model",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            session.chat(
                [{"role": "user", "content": "Line 1\nLine 2"}],
                max_output_tokens=2,
            )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(
            native.generate_calls[0][1]["input_text"],
            "user: Line 1\\nLine 2\nassistant:",
        )

    def test_chat_convenience_rejects_injected_role(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_cli_path="/tmp/llama-cli",
            llama_model_path="/tmp/model.gguf",
        ) as session:
            with self.assertRaisesRegex(ValueError, "unsupported chat role"):
                session.chat(
                    [{"role": "user\nsystem", "content": "Say hello"}],
                    max_output_tokens=2,
                )

    def test_submit_chat_convenience_reuses_text_prompt_path(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_server_url="http://127.0.0.1:8081",
        ) as session:
            request_id = session.submit_chat(
                [{"role": "user", "content": "queue this"}],
                max_output_tokens=2,
            )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(request_id, 11)
        self.assertEqual(
            native.submit_calls[0][1]["input_text"],
            "<|im_start|>user\nqueue this<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_stepwise_controls_convert_native_payloads(self) -> None:
        session = self.ax_engine.Session(
            model_id="qwen3_dense",
            mlx=True,
            mlx_model_artifacts_dir=FAKE_MLX_MODEL_DIR,
        )

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
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            mlx=True,
            mlx_model_artifacts_dir=FAKE_MLX_MODEL_DIR,
        ) as session:
            events = list(session.stream_generate([1, 2, 3], max_output_tokens=2))

        self.assertEqual(
            [event.event for event in events],
            ["request", "step", "step", "step", "response"],
        )
        self.assertEqual(events[0].request.state, "waiting")
        self.assertEqual(events[0].runtime.support_tier, "mlx_preview")
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

        with self.ax_engine.Session(
            model_id="qwen3_dense",
            mlx=True,
            mlx_model_artifacts_dir=FAKE_MLX_MODEL_DIR,
        ) as session:
            with self.assertRaisesRegex(
                RuntimeError,
                r"request 11 did not terminate within 258 steps",
            ):
                list(session.stream_generate([9], max_output_tokens=1))

    def test_stream_generate_supports_server_backed_llama_cpp_surface(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_server_url="http://127.0.0.1:8081",
        ) as session:
            events = list(session.stream_generate([1, 2, 3], max_output_tokens=2))

        self.assertEqual(
            [event.event for event in events],
            ["request", "step", "step", "response"],
        )
        self.assertEqual(events[0].runtime.support_tier, "llama_cpp")
        self.assertEqual(events[1].request.state, "running")
        self.assertEqual(events[1].delta_tokens, [4])
        self.assertEqual(events[1].delta_token_logprobs, [None])
        self.assertEqual(events[1].delta_text, "llama")
        self.assertEqual(events[2].request.state, "finished")
        self.assertEqual(events[2].request.finish_reason, "max_output_tokens")
        self.assertEqual(events[2].request.terminal_stop_reason, "max_output_tokens")
        self.assertEqual(events[2].delta_token_logprobs, [None])
        self.assertEqual(events[2].delta_text, " stream")
        self.assertEqual(events[3].response.output_tokens, [4, 5])
        self.assertEqual(events[3].response.output_token_logprobs, [None, None])
        self.assertEqual(
            events[3].response.route.execution_plan,
            "llama_cpp.server_completion_stream",
        )

        native = FakeNativeSession.instances[-1]
        self.assertEqual(native.llama_server_url, "http://127.0.0.1:8081")
        self.assertEqual(native.generate_calls[0][0], [1, 2, 3])

    def test_stream_text_convenience_uses_input_text_contract(self) -> None:
        with self.ax_engine.Session(
            model_id="qwen3_dense",
            support_tier="llama_cpp",
            llama_server_url="http://127.0.0.1:8081",
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
            support_tier="llama_cpp",
            llama_server_url="http://127.0.0.1:8081",
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
            "<|im_start|>user\nhello chat helper<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        self.assertEqual(
            events[-1].response.prompt_text,
            "<|im_start|>user\nhello chat helper<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_chat_convenience_rejects_empty_messages(self) -> None:
        with self.ax_engine.Session(model_id="qwen3_dense") as session:
            with self.assertRaisesRegex(ValueError, "chat requires at least one message"):
                session.chat([], max_output_tokens=2)


if __name__ == "__main__":
    unittest.main()
