"""
Baseline standard attention — reference implementation for correctness comparison.

Materialises the full (S_q x S_k) attention matrix.
Complexity: O(B * H * S_q * S_k * K) compute, O(B * H * S_q * S_k) memory.
"""

import math
import torch


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Standard scaled dot-product attention.

    Args:
        q: [B, H, S_q, K]
        k: [B, H, S_k, K]
        v: [B, H, S_k, K]
        causal: apply causal (lower-triangular) mask.

    Returns:
        [B, H, S_q, K]
    """
    K = q.size(-1)
    scale = 1.0 / math.sqrt(K)

    # [B, H, S_q, S_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        # row i can attend to columns 0..i (for equal-length Q/K)
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    # Replace NaN from all-masked rows with 0
    attn = attn.nan_to_num(0.0)

    return torch.matmul(attn, v)
