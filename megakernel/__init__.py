"""
Megakernel — 将多个独立的计算阶段融合进一个"巨型核函数"。

本包用 PyTorch 模拟 GPU megakernel 的核心思想，
提供 multi-kernel (分阶段) 与 megakernel (全融合) 的对比实现。
"""

from .multi_kernel import (
    multi_kernel_attention,
    multi_kernel_transformer_block,
    multi_kernel_layernorm_ffn,
)
from .megakernel_fused import (
    megakernel_attention,
    megakernel_transformer_block,
    megakernel_layernorm_ffn,
)
from .persistent_megakernel import (
    persistent_megakernel_attention,
)
from .triton_megakernel import (
    triton_megakernel_attention,
    auto_megakernel_attention,
    HAS_TRITON,
)

__all__ = [
    "multi_kernel_attention",
    "multi_kernel_transformer_block",
    "multi_kernel_layernorm_ffn",
    "megakernel_attention",
    "megakernel_transformer_block",
    "megakernel_layernorm_ffn",
    "persistent_megakernel_attention",
    "triton_megakernel_attention",
    "auto_megakernel_attention",
    "HAS_TRITON",
]
