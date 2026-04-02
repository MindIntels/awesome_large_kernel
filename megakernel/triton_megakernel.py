"""
Triton Megakernel Attention — 真正的 GPU 单 kernel 融合实现。

============================================================================
  与 Python 模拟版的本质区别
============================================================================

Python 模拟版 (megakernel_fused.py):
  - 每个 torch.matmul / torch.exp 仍然是一次独立的 CUDA kernel launch
  - Python for-loop 在 CPU 上顺序执行
  - 中间 tensor 仍在 GPU global memory
  →不是真正的 megakernel，只是算法演示

本 Triton 版本:
  - 整个 attention 编译为 **一个 GPU kernel**
  - 所有阶段 (QK^T + scale + mask + online-softmax + V) 在一次 launch 内完成
  - 中间数据 (scores tile, softmax 状态) 完全在 SRAM (shared memory / registers) 中
  - 只有输入读取 + 最终输出写入经过 global memory
  → 真正的 megakernel，获得真实的性能提升

需要: pip install triton  (仅支持 NVIDIA GPU)
============================================================================
"""

from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# =====================================================================
#  Triton kernel: fused attention (QK^T + scale + causal mask +
#                                  online softmax + attn @ V)
# =====================================================================

if HAS_TRITON:

    @triton.jit
    def _fused_attention_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qs, stride_qk,
        stride_kb, stride_kh, stride_ks, stride_kk,
        stride_vb, stride_vh, stride_vs, stride_vk,
        stride_ob, stride_oh, stride_os, stride_ok,
        S_q: tl.constexpr, S_k: tl.constexpr, K_dim: tl.constexpr,
        scale: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Single Triton kernel: the entire attention for one (batch, head, q-tile).

        Grid: (num_q_tiles, B * H)
        Each program instance handles one Q-tile × all KV-tiles.
        """
        # ── program IDs → (q_tile, batch*head) ──
        pid_q = tl.program_id(0)
        pid_bh = tl.program_id(1)

        # ── offset into (batch, head) ──
        # Q/K/V/Out 都是 [B, H, S, K] 连续布局
        q_offset = pid_bh * stride_qh  # 因为 B*H 已展平, stride_qh = S*K
        k_offset = pid_bh * stride_kh
        v_offset = pid_bh * stride_vh
        o_offset = pid_bh * stride_oh

        # ── Q tile 范围 ──
        q_start = pid_q * BLOCK_Q
        offs_q = q_start + tl.arange(0, BLOCK_Q)      # [BLOCK_Q]
        offs_k_dim = tl.arange(0, K_dim)               # [K_dim]

        # ── 加载 Q tile 到 SRAM (registers) ── 只读一次
        # Q[b, h, q_start:q_start+BLOCK_Q, :K_dim]
        q_ptrs = Q_ptr + q_offset + offs_q[:, None] * stride_qs + offs_k_dim[None, :] * stride_qk
        q_mask = offs_q[:, None] < S_q
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_Q, K_dim]
        q_tile = q_tile * scale  # pre-scale, 在 register 中完成

        # ── online softmax 状态 — 完全在 registers 中 ──
        m_i = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)  # running max
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)                # running sum-of-exp
        o_i = tl.zeros([BLOCK_Q, K_dim], dtype=tl.float32)         # running output

        # ── 流式扫描所有 KV tiles ──
        for k_start in range(0, S_k, BLOCK_K):
            offs_kv = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

            # ── 加载 K tile ──
            k_ptrs = K_ptr + k_offset + offs_kv[:, None] * stride_ks + offs_k_dim[None, :] * stride_kk
            k_mask = offs_kv[:, None] < S_k
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K, K_dim]

            # ── QK^T (tile): [BLOCK_Q, K_dim] x [K_dim, BLOCK_K] → [BLOCK_Q, BLOCK_K] ──
            s_ij = tl.dot(q_tile, tl.trans(k_tile))  # fused matmul, 在 SRAM 中

            # ── causal mask: 原地在 registers 中 ──
            if IS_CAUSAL:
                causal_mask = offs_q[:, None] < offs_kv[None, :]
                s_ij = tl.where(causal_mask, float("-inf"), s_ij)

            # ── out-of-bounds mask ──
            oob_mask = offs_kv[None, :] >= S_k
            s_ij = tl.where(oob_mask, float("-inf"), s_ij)

            # ── online softmax update — 全在 registers 中 ──
            m_ij = tl.max(s_ij, axis=1)                # [BLOCK_Q]
            m_new = tl.maximum(m_i, m_ij)              # [BLOCK_Q]

            # correction factor for old accumulators
            alpha = tl.exp(m_i - m_new)                # [BLOCK_Q]
            # new block weights
            p_ij = tl.exp(s_ij - m_new[:, None])       # [BLOCK_Q, BLOCK_K]

            l_i = l_i * alpha + tl.sum(p_ij, axis=1)  # [BLOCK_Q]
            o_i = o_i * alpha[:, None]                 # rescale old output

            # ── 加载 V tile ──
            v_ptrs = V_ptr + v_offset + offs_kv[:, None] * stride_vs + offs_k_dim[None, :] * stride_vk
            v_tile = tl.load(v_ptrs, mask=offs_kv[:, None] < S_k, other=0.0)

            # ── attn @ V (tile): [BLOCK_Q, BLOCK_K] x [BLOCK_K, K_dim] → [BLOCK_Q, K_dim] ──
            o_i += tl.dot(p_ij.to(v_tile.dtype), v_tile)

            m_i = m_new
            # ── s_ij, p_ij 在此被丢弃 — 不写回 global memory ──

        # ── 最终归一化 ──
        o_i = o_i / l_i[:, None]

        # ── 写回 global memory — 整个 kernel 中唯一的输出写 ──
        o_ptrs = Out_ptr + o_offset + offs_q[:, None] * stride_os + offs_k_dim[None, :] * stride_ok
        o_mask = offs_q[:, None] < S_q
        tl.store(o_ptrs, o_i.to(Out_ptr.dtype.element_ty), mask=o_mask)


# =====================================================================
#  Python wrapper
# =====================================================================

def triton_megakernel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_q: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Triton fused megakernel attention — 真正的单 kernel GPU 实现.

    Args:
        q, k, v: [B, H, S, K] — 必须在 CUDA device 上，float16 或 float32
        causal: causal mask.
        block_q, block_k: tile sizes (must be power of 2).

    Returns:
        [B, H, S_q, K]

    Raises:
        RuntimeError: 如果 Triton 不可用或输入不在 GPU 上。
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. Install with: pip install triton\n"
            "Triton requires an NVIDIA GPU with CUDA support."
        )
    if not q.is_cuda:
        raise RuntimeError(
            "triton_megakernel_attention requires CUDA tensors. "
            "Use megakernel_attention() for CPU."
        )

    B, H, S_q, K_dim = q.shape
    S_k = k.size(2)

    # K_dim must be power of 2 for tl.dot; pad if needed
    K_padded = triton.next_power_of_2(K_dim) if K_dim & (K_dim - 1) else K_dim
    if K_padded != K_dim:
        q = torch.nn.functional.pad(q, (0, K_padded - K_dim))
        k = torch.nn.functional.pad(k, (0, K_padded - K_dim))
        v = torch.nn.functional.pad(v, (0, K_padded - K_dim))
        K_dim = K_padded

    # Ensure contiguous
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # Output
    out = torch.empty_like(q)

    # Grid: (num_q_tiles, B * H)
    num_q_tiles = math.ceil(S_q / block_q)
    grid = (num_q_tiles, B * H)

    scale = 1.0 / math.sqrt(K_dim)

    _fused_attention_kernel[grid](
        q, k, v, out,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S_q=S_q, S_k=S_k, K_dim=K_dim,
        scale=scale,
        IS_CAUSAL=causal,
        BLOCK_Q=block_q, BLOCK_K=block_k,
    )

    # Trim padding
    if out.size(-1) != q.size(-1):
        out = out[..., :q.size(-1)]

    return out


# =====================================================================
#  Fallback: 自动选择 Triton (GPU) 或 Python (CPU)
# =====================================================================

def auto_megakernel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_size: int = 64,
) -> torch.Tensor:
    """自动选择最优实现: GPU 上用 Triton 真 megakernel，CPU 上用 Python 模拟。

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        block_size: tile size.

    Returns:
        [B, H, S_q, K]
    """
    if q.is_cuda and HAS_TRITON:
        return triton_megakernel_attention(q, k, v, causal=causal,
                                           block_q=block_size, block_k=block_size)
    else:
        # Fallback to Python simulation
        from .megakernel_fused import megakernel_attention
        return megakernel_attention(q, k, v, causal=causal, block_size=block_size)
