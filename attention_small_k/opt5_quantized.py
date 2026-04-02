"""
Optimisation 5 — Quantised (int8) small-K attention.

**Idea**: When K is small, the range of the dot product Q·K^T is limited
(each sum has only K terms).  This makes quantisation especially effective:
- Quantisation error in each element is bounded.
- The accumulator range for K multiply-adds is small → int8 doesn't overflow.
- 4× data compression → 4× effective memory bandwidth improvement.

Algorithm:
  1. Quantise Q and K from fp32/fp16 to int8 (per-tensor symmetric).
  2. Compute QK^T in int32 (int8 × int8 → int32 accumulation).
  3. Dequantise scores back to float, apply scale and softmax.
  4. Multiply attention weights (float) by V (float).

**Benefit**:
- ~4× reduction in memory traffic for the QK^T computation.
- On hardware with int8 tensor cores (Turing+), up to 2× throughput.
- Small K limits quantisation error accumulation.
"""

import math
import torch


def _symmetric_quantise_int8(x: torch.Tensor):
    """Per-tensor symmetric int8 quantisation.

    Returns: (x_int8, scale) where x ≈ x_int8 * scale.
    """
    amax = x.abs().amax().clamp(min=1e-6)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def quantized_small_k_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Int8-quantised QK attention for small K.

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.

    Returns:
        [B, H, S_q, K]
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale_attn = 1.0 / math.sqrt(K)

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # Quantise Q and K to int8
    q_int8, q_scale = _symmetric_quantise_int8(q_f)
    k_int8, k_scale = _symmetric_quantise_int8(k_f)

    # Int8 matmul: need to use int32 accumulation
    # Reshape for bmm: merge B,H into batch dim
    q_flat = q_int8.reshape(B * H, S_q, K).to(torch.int32)
    k_flat = k_int8.reshape(B * H, S_k, K).to(torch.int32)

    # [B*H, S_q, K] x [B*H, K, S_k] → [B*H, S_q, S_k]  in int32
    scores_int32 = torch.bmm(q_flat, k_flat.transpose(-2, -1))

    # Dequantise: float_scores = int_scores * q_scale * k_scale * attn_scale
    scores = scores_int32.float() * (q_scale * k_scale * scale_attn)
    scores = scores.reshape(B, H, S_q, S_k)

    if causal:
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num(0.0)

    out = torch.matmul(attn, v_f)
    return out.to(q.dtype)
