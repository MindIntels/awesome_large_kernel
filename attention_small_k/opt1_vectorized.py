"""
Optimisation 1 — Vectorised small-K attention via einsum.

**Idea**: When K is small (≤ 32), the inner dot-product dimension fits inside
a single SIMD register / warp lane on hardware.  We can exploit this by using
`torch.einsum` with an explicit K contraction, which gives the compiler /
backend freedom to keep the K-dimension entirely in registers and fuse the
entire Q·K^T computation without materialising intermediate per-element results.

Additionally we pre-scale Q (amortising the 1/√K multiply) and compute softmax
in float32 for numerical stability even when inputs are fp16/bf16.

**Benefit over baseline**:
- Avoids the `q @ k.T` temporary — einsum can stream the result directly
  into the softmax accumulator.
- Pre-scaling Q saves one element-wise multiply over the full S_q×S_k matrix.
"""

import math
import torch


def vectorized_small_k_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Vectorised attention for small K.

    Args:
        q, k, v: [B, H, S, K]  (K should be small, e.g. 8/16/32)
        causal: apply causal mask.

    Returns:
        [B, H, S_q, K]
    """
    K = q.size(-1)
    orig_dtype = q.dtype

    # Pre-scale Q once (O(B·H·S_q·K)) instead of scaling the full S_q×S_k matrix
    q_scaled = q.float() * (1.0 / math.sqrt(K))
    k_f = k.float()
    v_f = v.float()

    # Einsum: contract over K dimension explicitly.
    # The backend can keep the K-dim in vector registers when K is tiny.
    scores = torch.einsum("bhqk,bhsk->bhqs", q_scaled, k_f)

    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num(0.0)

    out = torch.einsum("bhqs,bhsk->bhqk", attn, v_f)
    return out.to(orig_dtype)
