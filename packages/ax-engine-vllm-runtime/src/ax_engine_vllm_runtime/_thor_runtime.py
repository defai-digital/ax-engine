"""Internal vLLM 0.25.1 SM110 fallbacks installed by the general plugin.

This module deliberately imports version-pinned vLLM internals.  It is loaded
only after :mod:`ax_engine_vllm_runtime.thor_compat` has verified native Linux/aarch64,
CUDA capability 11.0, and the exact supported vLLM version.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels import linear as linear_kernels
from vllm.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
    CompressedTensorsWNA16MoEMethod as _UpstreamWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    WNA16_SUPPORTED_BITS,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    unpack_quantized_values_into_int32,
)
from vllm.platforms import PlatformEnum, current_platform
from vllm.scalar_type import scalar_types

logger = logging.getLogger(__name__)

_INSTALLED = False
_THOR_UNQUANTIZED_BACKEND = "ax_engine_vllm_sm110_native_torch"


def _validate_sm110_native_torch_moe_config(moe: FusedMoEConfig) -> None:
    capability = current_platform.get_device_capability()
    parallel = moe.moe_parallel_config
    if not current_platform.is_cuda() or capability != (11, 0):
        raise ValueError("native Torch MoE fallback is restricted to CUDA SM110")
    if (
        parallel.tp_size != 1
        or parallel.dp_size != 1
        or parallel.ep_size != 1
        or parallel.pcp_size != 1
        or parallel.sp_size != 1
        or parallel.use_ep
        or parallel.enable_eplb
        or moe.is_lora_enabled
    ):
        raise ValueError("SM110 native Torch MoE supports single-device inference only")
    if moe.activation != MoEActivation.SILU or not moe.is_act_and_mul or moe.has_bias:
        raise ValueError("SM110 native Torch MoE requires unbiased SiLU-gated experts")
    if moe.swiglu_limit is not None:
        raise ValueError("SM110 native Torch MoE does not support clamped SwiGLU")


def _native_torch_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: MoEActivation,
    apply_router_weight_on_input: bool,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate selected experts with native BF16/FP16 PyTorch operations."""
    if activation != MoEActivation.SILU:
        raise RuntimeError("SM110 native Torch MoE only supports SiLU gating")
    if apply_router_weight_on_input:
        raise RuntimeError("SM110 native Torch MoE applies router weights on output")
    if expert_map is not None:
        raise RuntimeError("SM110 native Torch MoE does not support expert mapping")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[0] != w2.shape[0]:
        raise RuntimeError("invalid native Torch MoE weight shapes")
    if global_num_experts != w1.shape[0]:
        raise RuntimeError("SM110 native Torch MoE requires every expert locally")
    if w1.dtype not in (torch.bfloat16, torch.float16) or w2.dtype != w1.dtype:
        raise RuntimeError("native Torch MoE requires matching BF16/FP16 weights")
    if hidden_states.dtype != w1.dtype:
        raise RuntimeError("activation and native Torch expert dtypes must match")
    if w1.shape[1] % 2 or w1.shape[2] != hidden_states.shape[-1]:
        raise RuntimeError("invalid native Torch gate/up projection shape")
    if w2.shape[1] != hidden_states.shape[-1] or w2.shape[2] * 2 != w1.shape[1]:
        raise RuntimeError("invalid native Torch down projection shape")

    original_shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    ids = topk_ids.reshape(hidden.shape[0], -1)
    weights = topk_weights.reshape(hidden.shape[0], -1).to(hidden.dtype)
    if ids.shape != weights.shape:
        raise RuntimeError("top-k ids and weights have incompatible shapes")
    selected = [int(value) for value in torch.unique(ids).tolist()]
    if any(value < 0 or value >= w1.shape[0] for value in selected):
        raise RuntimeError("router selected an out-of-range expert")

    output = torch.zeros_like(hidden)
    for expert_id in selected:
        token_ids, slot_ids = torch.where(ids == expert_id)
        expert_input = hidden[token_ids]
        gate_up = F.linear(expert_input, w1[expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2[expert_id])
        route_weight = weights[token_ids, slot_ids].unsqueeze(-1)
        output.index_add_(0, token_ids, expert_output * route_weight)
    return output.reshape(original_shape)


class _NativeTorchPrepareFinalize:
    @staticmethod
    def topk_indices_dtype() -> torch.dtype | None:
        return None


class _NativeTorchMoEKernel:
    """Minimal facade for vLLM's modular MoE runner."""

    is_monolithic = False
    can_overlap_shared_experts = False

    def __init__(self, moe_config: FusedMoEConfig) -> None:
        _validate_sm110_native_torch_moe_config(moe_config)
        self.moe_config = moe_config
        self.prepare_finalize = _NativeTorchPrepareFinalize()

    @staticmethod
    def output_is_reduced() -> bool:
        return False

    def apply(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        apply_router_weight_on_input: bool,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        shared_experts: Any = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The modular runner combines shared experts separately because this
        # kernel explicitly declares that it cannot overlap them.
        del shared_experts, shared_experts_input
        return _native_torch_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )


class _ThorWNA16LinearKernel(MPLinearKernel):
    """Load-time W4A16-to-BF16 expansion for dense SM110 linears."""

    @classmethod
    def get_min_capability(cls) -> int:
        return 110

    @classmethod
    def can_implement(cls, config: MPLinearLayerConfig) -> tuple[bool, str | None]:
        capability = current_platform.get_device_capability()
        if not current_platform.is_cuda() or capability != (11, 0):
            return False, "fallback is restricted to CUDA capability 11.0"
        if config.weight_type != scalar_types.uint4b8:
            return False, "only symmetric uint4b8 weights are supported"
        if config.act_type not in (torch.bfloat16, torch.float16):
            return False, "only BF16/FP16 activations are supported"
        if config.group_size != 128:
            return False, "only group size 128 is release-validated"
        if config.zero_points:
            return False, "explicit zero points are not supported"
        if config.has_g_idx:
            return False, "activation ordering is not supported"
        if config.partition_weight_shape[0] % config.group_size:
            return False, "input partition must be divisible by group size"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        packed = getattr(layer, self.w_q_name)
        scales = getattr(layer, self.w_s_name)
        if packed.dtype != torch.int32 or packed.ndim != 2:
            raise ValueError("expected a 2-D int32 compressed-tensors weight")
        if scales.dtype not in (torch.bfloat16, torch.float16) or scales.ndim != 2:
            raise ValueError("expected a 2-D BF16/FP16 group scale")

        output_size, packed_input_size = packed.shape
        input_size = packed_input_size * 8
        expected_scales = (output_size, input_size // self.config.group_size)
        if tuple(scales.shape) != expected_scales:
            raise ValueError(
                f"invalid W4A16 scale shape {tuple(scales.shape)}; expected {expected_scales}"
            )
        unpacked = unpack_quantized_values_into_int32(packed, scalar_types.uint4b8, packed_dim=1)
        expanded_scales = scales.repeat_interleave(self.config.group_size, dim=1)
        dequantized = (
            (unpacked - scalar_types.uint4b8.bias).to(scales.dtype) * expanded_scales
        ).contiguous()
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(dequantized, requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = getattr(layer, self.w_q_name)
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError("SM110 W4A16 fallback weight was not dequantized")
        return F.linear(x, weight, bias)


class _ThorWNA16MoEMethod(_UpstreamWNA16MoEMethod):
    """Load-time W4A16 expansion plus native routed-expert evaluation."""

    def __init__(
        self,
        weight_quant: Any,
        input_quant: Any,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ) -> None:
        _validate_sm110_native_torch_moe_config(moe)
        if (
            weight_quant.num_bits != 4
            or weight_quant.type != QuantizationType.INT
            or weight_quant.strategy != QuantizationStrategy.GROUP
            or weight_quant.group_size != 128
            or not weight_quant.symmetric
            or weight_quant.actorder is not None
            or input_quant is not None
        ):
            raise ValueError("SM110 fallback requires symmetric W4A16 group128 without actorder")
        super().__init__(weight_quant, input_quant, moe, layer_name)

    @staticmethod
    def _dequantize_checkpoint_tensor(
        packed: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
        name: str,
    ) -> torch.Tensor:
        if packed.dtype != torch.int32 or packed.ndim != 3:
            raise ValueError(f"{name} must be a 3-D int32 packed tensor")
        if scales.dtype not in (torch.bfloat16, torch.float16) or scales.ndim != 3:
            raise ValueError(f"{name} scales must be 3-D BF16/FP16")
        unsigned = (
            unpack_quantized_values_into_int32(packed, scalar_types.uint4b8, packed_dim=1)
            .transpose(1, 2)
            .contiguous()
        )
        transposed_scales = scales.transpose(1, 2).contiguous()
        expected = (*unsigned.shape[:-1], unsigned.shape[-1] // group_size)
        if tuple(transposed_scales.shape) != expected:
            raise ValueError(
                f"invalid {name} scale shape {tuple(transposed_scales.shape)}; expected {expected}"
            )
        expanded_scales = transposed_scales.repeat_interleave(group_size, dim=-1)
        return (
            (unsigned - scalar_types.uint4b8.bias).to(scales.dtype) * expanded_scales
        ).contiguous()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight_packed = torch.nn.Parameter(
            self._dequantize_checkpoint_tensor(
                layer.w13_weight_packed,
                layer.w13_weight_scale,
                self.group_size,
                "w13",
            ),
            requires_grad=False,
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            self._dequantize_checkpoint_tensor(
                layer.w2_weight_packed,
                layer.w2_weight_scale,
                self.group_size,
                "w2",
            ),
            requires_grad=False,
        )
        self.moe_kernel = _NativeTorchMoEKernel(self.moe)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        del layer
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.moe_kernel is None:
            raise RuntimeError("SM110 W4A16 MoE weights were not processed")
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    @property
    def supports_eplb(self) -> bool:
        return False


# vLLM 0.25.1 branches on this exact class name while loading packed expert
# tensors.  Preserve the upstream name without modifying vLLM's source tree.
_ThorWNA16MoEMethod.__name__ = "CompressedTensorsWNA16MoEMethod"


def _install_router_dtype_patch() -> None:
    from vllm.transformers_utils.configs.unlimited_ocr import UnlimitedOCRConfig

    if getattr(UnlimitedOCRConfig.__init__, "_ax_engine_vllm_thor_patch", False):
        return
    original_init = UnlimitedOCRConfig.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.text_config.moe_router_dtype = "float32"

    patched_init._ax_engine_vllm_thor_patch = True  # type: ignore[attr-defined]
    UnlimitedOCRConfig.__init__ = patched_init


def _install_unquantized_moe_patch() -> None:
    from vllm.model_executor.layers.fused_moe import (
        unquantized_fused_moe_method as method_module,
    )
    from vllm.model_executor.layers.fused_moe.oracle import unquantized as oracle

    if getattr(oracle.select_unquantized_moe_backend, "_ax_engine_vllm_thor_patch", False):
        return
    original_select = oracle.select_unquantized_moe_backend

    def patched_select(moe_config: FusedMoEConfig):
        if moe_config.moe_backend == "emulation":
            _validate_sm110_native_torch_moe_config(moe_config)
            logger.info("AX Engine vLLM runtime enabled native Torch SM110 unquantized MoE")
            return _THOR_UNQUANTIZED_BACKEND, None
        return original_select(moe_config)

    patched_select._ax_engine_vllm_thor_patch = True  # type: ignore[attr-defined]
    oracle.select_unquantized_moe_backend = patched_select
    method_module.select_unquantized_moe_backend = patched_select

    original_is_monolithic = UnquantizedFusedMoEMethod.is_monolithic.fget
    original_supports_eplb = UnquantizedFusedMoEMethod.supports_eplb.fget
    original_process = UnquantizedFusedMoEMethod.process_weights_after_loading
    if original_is_monolithic is None or original_supports_eplb is None:
        raise RuntimeError("vLLM unquantized MoE properties are incompatible")

    def is_monolithic(self: UnquantizedFusedMoEMethod) -> bool:
        if self.unquantized_backend == _THOR_UNQUANTIZED_BACKEND:
            return False
        return original_is_monolithic(self)

    def supports_eplb(self: UnquantizedFusedMoEMethod) -> bool:
        if self.unquantized_backend == _THOR_UNQUANTIZED_BACKEND:
            return False
        return original_supports_eplb(self)

    def process_weights_after_loading(
        self: UnquantizedFusedMoEMethod, layer: RoutedExperts
    ) -> None:
        if self.unquantized_backend != _THOR_UNQUANTIZED_BACKEND:
            return original_process(self, layer)
        FusedMoEMethodBase.process_weights_after_loading(self, layer)
        layer.w13_weight.data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w2_weight.data = self._maybe_pad_weight(layer.w2_weight.data)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = _NativeTorchMoEKernel(self.moe)

    UnquantizedFusedMoEMethod.is_monolithic = property(is_monolithic)
    UnquantizedFusedMoEMethod.supports_eplb = property(supports_eplb)
    UnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading


def _install_wna16_patches() -> None:
    linear_kernels._LINEAR_BACKEND_KERNEL_MAP["emulation"].add(  # type: ignore[attr-defined]
        _ThorWNA16LinearKernel
    )
    cuda_kernels = linear_kernels._POSSIBLE_KERNELS[PlatformEnum.CUDA]  # type: ignore[attr-defined]
    if _ThorWNA16LinearKernel not in cuda_kernels:
        cuda_kernels.append(_ThorWNA16LinearKernel)

    if getattr(CompressedTensorsMoEMethod.get_moe_method, "_ax_engine_vllm_thor_patch", False):
        return
    original_get_moe_method = CompressedTensorsMoEMethod.get_moe_method

    def patched_get_moe_method(
        quant_config: Any,
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        if get_current_vllm_config().kernel_config.moe_backend == "emulation":
            quant_config._add_fused_moe_to_target_scheme_map()
            unfused_names = [
                layer_name + projection
                for projection in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
            ]
            schemes = [quant_config.get_scheme_dict(layer, name) for name in unfused_names]
            scheme = schemes[-1]
            if not all(candidate == scheme for candidate in schemes):
                raise ValueError("All MoE projections must use one quantization scheme")
            if scheme is not None:
                weight_quant = scheme.get("weights")
                input_quant = scheme.get("input_activations")
                compression_format = scheme.get("format")
                if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
                    if (
                        weight_quant.num_bits not in WNA16_SUPPORTED_BITS
                        or compression_format != CompressionFormat.pack_quantized.value
                    ):
                        raise ValueError("SM110 MoE requires pack-quantized W4A16 weights")
                    logger.info("AX Engine vLLM runtime enabled load-time W4A16 SM110 MoE fallback")
                    return _ThorWNA16MoEMethod(
                        weight_quant,
                        input_quant,
                        layer.moe_config,
                        layer_name,
                    )
        return original_get_moe_method(quant_config, layer, layer_name)

    patched_get_moe_method._ax_engine_vllm_thor_patch = True  # type: ignore[attr-defined]
    CompressedTensorsMoEMethod.get_moe_method = staticmethod(patched_get_moe_method)


def install() -> None:
    """Install all exact vLLM 0.25.1 SM110 patches once per process."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not current_platform.is_cuda() or current_platform.get_device_capability() != (
        11,
        0,
    ):
        raise RuntimeError("AX Engine vLLM runtime Thor runtime was loaded outside CUDA SM110")
    _install_router_dtype_patch()
    _install_unquantized_moe_patch()
    _install_wna16_patches()
    _INSTALLED = True
    logger.warning(
        "AX Engine vLLM runtime installed vLLM 0.25.1 SM110 native-op and W4A16 compatibility"
    )
