"""
Multi-Kernel baseline — 每一步计算单独调度一个 kernel。

在 GPU 上，这意味着每步之间都有一次 kernel launch + 全局内存写回/读回。
在 Python 里，每个 torch op 都会产生一个中间 tensor (materialized)，
模拟了 multi-kernel 的行为：

  Step 1: Q_proj = X @ W_q           → 写回 global memory
  Step 2: K_proj = X @ W_k           → 写回
  Step 3: scores = Q @ K^T * scale   → 写回 S_q×S_k 矩阵
  Step 4: mask                       → 读+写 scores
  Step 5: attn = softmax(scores)     → 读+写
  Step 6: out = attn @ V             → 写回
  ...

每一步都是独立 kernel → N 次 launch 开销 + 中间矩阵反复访存。
"""

import math
import torch
import torch.nn.functional as F


# =====================================================================
#  Multi-kernel attention (6 个独立 kernel)
# =====================================================================

def multi_kernel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Multi-kernel attention: 每步单独一个 kernel。

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.

    Returns:
        [B, H, S_q, K]

    Simulated kernel launches:
        K1: scale Q             → intermediate tensor (write to DRAM)
        K2: matmul Q @ K^T      → S_q×S_k score matrix (write to DRAM)
        K3: causal mask          → read + write score matrix
        K4: softmax              → read + write score matrix
        K5: matmul attn @ V     → write output
    """
    K_dim = q.size(-1)

    # --- Kernel 1: scale ---
    scale = 1.0 / math.sqrt(K_dim)
    q_scaled = q * scale  # materialised intermediate

    # --- Kernel 2: QK^T ---
    scores = torch.matmul(q_scaled, k.transpose(-2, -1))  # materialised [B,H,S_q,S_k]

    # --- Kernel 3: causal mask ---
    if causal:
        S_q, S_k = scores.size(-2), scores.size(-1)
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))  # materialised copy

    # --- Kernel 4: softmax ---
    attn = torch.softmax(scores, dim=-1)  # materialised [B,H,S_q,S_k]
    attn = attn.nan_to_num(0.0)

    # --- Kernel 5: attn @ V ---
    out = torch.matmul(attn, v)  # materialised [B,H,S_q,K]

    return out


# =====================================================================
#  Multi-kernel LayerNorm + FFN (4 个独立 kernel)
# =====================================================================

def multi_kernel_layernorm_ffn(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-kernel LayerNorm → FFN(GeLU) → Linear.

    Args:
        x: [B, S, D]
        ln_weight, ln_bias: [D]
        w1: [D, D_ff],  w2: [D_ff, D]
        b1: [D_ff] optional, b2: [D] optional

    Returns:
        [B, S, D]

    Simulated kernels:
        K1: LayerNorm           → write normalised x
        K2: Linear (up-proj)    → write hidden
        K3: GeLU activation     → read + write hidden
        K4: Linear (down-proj)  → write output
    """
    # --- Kernel 1: LayerNorm ---
    x_norm = F.layer_norm(x, (x.size(-1),), ln_weight, ln_bias)

    # --- Kernel 2: up-projection ---
    hidden = F.linear(x_norm, w1.T, b1)  # [B, S, D_ff]

    # --- Kernel 3: GeLU ---
    hidden = F.gelu(hidden)  # read + write

    # --- Kernel 4: down-projection ---
    out = F.linear(hidden, w2.T, b2)  # [B, S, D]

    return out


# =====================================================================
#  Multi-kernel Transformer Block (attention + FFN with residuals)
# =====================================================================

def multi_kernel_transformer_block(
    x: torch.Tensor,
    w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, w_o: torch.Tensor,
    ln1_w: torch.Tensor, ln1_b: torch.Tensor,
    w_ff1: torch.Tensor, w_ff2: torch.Tensor,
    ln2_w: torch.Tensor, ln2_b: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Multi-kernel transformer block: ~12+ separate kernels.

    Args:
        x: [B, S, D]
        w_q/k/v/o: [D, D]  attention weights
        ln1_w/b, ln2_w/b: [D]  layer norm params
        w_ff1: [D, D_ff],  w_ff2: [D_ff, D]
        num_heads: H
        causal: causal mask.

    Returns:
        [B, S, D]
    """
    B, S, D = x.shape
    K = D // num_heads

    # --- Kernel 1: LayerNorm 1 ---
    x_norm = F.layer_norm(x, (D,), ln1_w, ln1_b)

    # --- Kernels 2-4: Q/K/V projections (3 separate matmuls) ---
    q = F.linear(x_norm, w_q.T)  # [B, S, D]
    k = F.linear(x_norm, w_k.T)
    v = F.linear(x_norm, w_v.T)

    # --- Kernel 5: reshape ---
    q = q.view(B, S, num_heads, K).transpose(1, 2)  # [B, H, S, K]
    k = k.view(B, S, num_heads, K).transpose(1, 2)
    v = v.view(B, S, num_heads, K).transpose(1, 2)

    # --- Kernels 6-10: attention (scale, QK^T, mask, softmax, attn@V) ---
    attn_out = multi_kernel_attention(q, k, v, causal=causal)

    # --- Kernel 11: reshape + output projection ---
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    attn_out = F.linear(attn_out, w_o.T)

    # --- Kernel 12: residual add ---
    x = x + attn_out

    # --- Kernels 13-16: LayerNorm 2 + FFN ---
    ffn_out = multi_kernel_layernorm_ffn(x, ln2_w, ln2_b, w_ff1, w_ff2)

    # --- Kernel 17: residual add ---
    x = x + ffn_out

    return x
