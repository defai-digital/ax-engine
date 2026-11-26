"""Bar colors for AX Engine charts.

Stock SGLang, TensorRT, vLLM, and AX Engine use the same light, standard,
and dark shades as the ax-engine-sglang README charts. The standard AX green
matches the README banner. Peers that do not appear on those charts have
their own shades, chosen so they stay apart from the four shared families.
"""

# AX Engine. Standard green is the banner mark.
AX_LIGHT = "#91d7b7"
AX = "#37b67c"
AX_MID = "#1f8a56"
AX_DARK = "#0b5d36"
AX_DEEPER = "#063320"

# Stock SGLang.
SGLANG_LIGHT = "#eea36c"
SGLANG = "#e36a12"
SGLANG_DARK = "#a34108"

# TensorRT.
TRT_LIGHT = "#c5cad3"
TRT = "#8b939e"
TRT_DARK = "#3e4650"

# vLLM.
VLLM_LIGHT = "#89b1e9"
VLLM = "#3b7ddb"
VLLM_DARK = "#1a4e9a"

# mlx-lm. Amber, kept off the SGLang orange.
MLX_LIGHT = "#f3d48a"
MLX = "#d4a017"
MLX_DARK = "#8a6808"

# llama.cpp. Violet, kept off the SGLang orange.
LLAMA_LIGHT = "#cbbbe8"
LLAMA = "#7c5cbf"
LLAMA_DARK = "#4c3480"

# MTPLX.
MTPLX_LIGHT = "#f3b4cb"
MTPLX = "#d4537e"
MTPLX_DARK = "#8e2348"

# oMLX.
OMLX_LIGHT = "#9ad8d8"
OMLX = "#1aa6a6"
OMLX_DARK = "#0d5c5c"

# oMLX Lightning. A darker teal, not vLLM blue.
LIGHTNING_LIGHT = "#99d5cf"
LIGHTNING = "#0f766e"
LIGHTNING_DARK = "#134e4a"

# Splash.
SPLASH_LIGHT = "#c4bef3"
SPLASH = "#5b4fcf"
SPLASH_DARK = "#322a86"

# TensorFold. Brown, not TensorRT grey.
FOLD_LIGHT = "#e0c7ae"
FOLD = "#a67c52"
FOLD_DARK = "#6e4b2a"

# ds4.
DS4_LIGHT = "#c5cedb"
DS4 = "#5c6b8a"
DS4_DARK = "#2e3848"

# A non-engine reference mark, such as a published bandwidth ceiling.
REFERENCE = "#9ca3af"
REFERENCE_DARK = "#374151"

# Engine keys used by the M5 README charts.
M5_ENGINE_COLORS = {
    "ax-engine": AX,
    "splash": SPLASH,
    "omlx": OMLX,
    "mlx-lm": MLX,
    "mtplx": MTPLX,
    "tensorfold": FOLD,
    "ds4": DS4,
}
