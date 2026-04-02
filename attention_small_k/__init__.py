"""
Attention optimizations for small K (head dimension).

When K is very small (e.g., 8, 16, 32), standard attention becomes memory-bandwidth
bound rather than compute bound. This package implements 6 optimization strategies
targeting this regime.
"""

from .baseline import standard_attention
from .opt1_vectorized import vectorized_small_k_attention
from .opt2_fused import fused_qk_softmax_v_attention
from .opt3_register_tiled import register_tiled_attention
from .opt4_unrolled import unrolled_attention
from .opt5_quantized import quantized_small_k_attention
from .opt6_blocked_seq import blocked_seq_attention
from .opt7_k_packed import kpadded_attention, head_grouped_attention, head_grouped_attention_ref

__all__ = [
    "standard_attention",
    "vectorized_small_k_attention",
    "fused_qk_softmax_v_attention",
    "register_tiled_attention",
    "unrolled_attention",
    "quantized_small_k_attention",
    "blocked_seq_attention",
    "kpadded_attention",
    "head_grouped_attention",
    "head_grouped_attention_ref",
]
