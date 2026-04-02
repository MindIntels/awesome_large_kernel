"""
Megakernel (Fused) — 将多阶段计算合并进单个"巨型核函数"。

============================================================================
  Megakernel 核心原理
============================================================================

传统 GPU 编程模型：一个 op 对应一个 kernel launch。
  Transformer block = LayerNorm kernel → Q-proj kernel → K-proj kernel
    → V-proj kernel → scale kernel → QK^T kernel → mask kernel
    → softmax kernel → attn@V kernel → O-proj kernel → add kernel
    → LayerNorm kernel → FFN-up kernel → GeLU kernel → FFN-down kernel
    → add kernel
  = 15+ 次 kernel launch

每次 kernel launch 的代价：
  1. **Launch overhead**: CPU→GPU 调度 3-10 μs / 次
  2. **Global memory round-trip**: 中间结果必须写回 DRAM (bandwidth: ~1 TB/s)
     再由下一个 kernel 从 DRAM 读回 — 对 memory-bound 操作是致命瓶颈
  3. **SM idle gap**: 两个 kernel 之间 SM 有空闲等待期 (pipeline bubble)
  4. **寄存器/共享内存失效**: 每个 kernel 结束时局部存储全部丢失

Megakernel 解决方案 — 把所有阶段融合为一个 kernel：
  ┌─────────────────────────────────────────────┐
  │  Single Megakernel Launch                    │
  │                                              │
  │  for each tile / work unit:                  │
  │    ① LayerNorm (registers)                   │
  │    ② QKV projection (shared memory)          │
  │    ③ QK^T + scale (registers)                │
  │    ④ Causal mask (registers, branch)         │
  │    ⑤ Online softmax (registers, streaming)   │
  │    ⑥ Attn @ V (registers → shared mem)       │
  │    ⑦ O projection (shared mem → registers)   │
  │    ⑧ Residual add                            │
  │    ⑨ LayerNorm 2 (registers)                 │
  │    ⑩ FFN up + GeLU + down (shared mem)       │
  │    ⑪ Residual add → write final result       │
  │  end for                                     │
  │                                              │
  │  Only ONE global memory write at the end     │
  └─────────────────────────────────────────────┘

关键优势：
  A. 消除 kernel launch 开销 (15次 → 1次)
  B. 中间结果驻留 registers / shared memory，不写回 DRAM
  C. 编译器可跨阶段优化 (寄存器分配、指令调度)
  D. 持续占据 SM，无 pipeline bubble
  E. 总 DRAM 访问量 ≈ 只读输入 + 只写输出

典型应用：
  - FlashAttention: 融合 QK^T + softmax + attn@V
  - FlashDecoding: 融合 tiled attention 的 reduce
  - FasterTransformer: 融合整个 transformer block
  - TensorRT-LLM: 全层融合 megakernel - Triton: 用户自定义融合程度的 megakernel

============================================================================
"""

import math
import torch
import torch.nn.functional as F


# =====================================================================
#  Megakernel Attention: 全融合 (scale + QK^T + mask + softmax + V)
# =====================================================================

def megakernel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_size: int = 64,
) -> torch.Tensor:
    """Megakernel attention: 全部阶段在一个"kernel"内完成。

    对比 multi_kernel_attention 的 5 个 kernel:
      这里用 online softmax + tiling 实现单遍扫描。
      中间的 S_q×S_k score 矩阵 **从不完整物化**（只有 tile 大小的局部块）。

    模拟的硬件行为：
      - Q tile 常驻 "寄存器" (Python 变量，不写回全局 tensor)
      - KV tile 流式经过 "共享内存"
      - softmax 状态 (m, l) 在 "寄存器" 中维护
      - 只在最后写一次输出到 "全局内存"

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        block_size: tile size.

    Returns:
        [B, H, S_q, K]
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale = 1.0 / math.sqrt(K)
    device = q.device

    # ────────── "Kernel launch" — 只有这一次 ──────────

    # 工作在 float32 (模拟 register 中的累加精度)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # Pre-scale Q — 在 "寄存器" 中完成，不写回 DRAM
    # (multi-kernel 版本会 materialise 一个 q_scaled tensor)
    q_f = q_f * scale

    # 输出 buffer — 模拟最终写回全局内存的唯一位置
    output = torch.zeros(B, H, S_q, K, device=device)

    Br = block_size
    Bc = block_size
    n_br = math.ceil(S_q / Br)
    n_bc = math.ceil(S_k / Bc)

    for i in range(n_br):
        i0, i1 = i * Br, min((i + 1) * Br, S_q)
        bq = i1 - i0

        # ── Q tile 加载到 "寄存器文件" ──
        qi = q_f[:, :, i0:i1, :]  # [B, H, bq, K]  — 这是 register-resident

        # Online softmax 状态 — 全在 "寄存器" 中
        m_i = torch.full((B, H, bq, 1), float("-inf"), device=device)
        l_i = torch.zeros((B, H, bq, 1), device=device)
        o_i = torch.zeros((B, H, bq, K), device=device)

        for j in range(n_bc):
            j0, j1 = j * Bc, min((j + 1) * Bc, S_k)

            # ── KV tile 流入 "共享内存" ──
            kj = k_f[:, :, j0:j1, :]  # [B, H, bc, K]
            vj = v_f[:, :, j0:j1, :]

            # ── 阶段 1: QK^T (tile) — "寄存器 → 寄存器" ──
            s_ij = torch.matmul(qi, kj.transpose(-2, -1))  # [B,H,bq,bc]
            # 注意: s_ij 是 tile 大小 (bq × bc)，不是完整 S_q × S_k

            # ── 阶段 2: causal mask — 原地在 "寄存器" 中 ──
            if causal:
                qi_idx = torch.arange(i0, i1, device=device).unsqueeze(1)
                kj_idx = torch.arange(j0, j1, device=device).unsqueeze(0)
                mask = kj_idx > qi_idx
                s_ij.masked_fill_(mask, float("-inf"))
                if mask.all():
                    continue  # 整个 tile 被 mask → 跳过 (causal early-exit)

            # ── 阶段 3: online softmax update — "寄存器 → 寄存器" ──
            m_new = torch.maximum(m_i, s_ij.max(dim=-1, keepdim=True).values)
            exp_diff = torch.exp(m_i - m_new)
            p_ij = torch.exp(s_ij - m_new)

            l_i = l_i * exp_diff + p_ij.sum(dim=-1, keepdim=True)
            o_i = o_i * exp_diff + torch.matmul(p_ij, vj)
            m_i = m_new
            # ── 至此 s_ij, p_ij 可以被丢弃 (不写回 DRAM) ──

        # ── 最终归一化 + 写回 "全局内存" — 整个循环中唯一的输出写 ──
        output[:, :, i0:i1, :] = o_i / l_i.clamp(min=1e-6)

    # ────────── "Kernel 结束" ──────────

    return output.to(q.dtype)


# =====================================================================
#  Megakernel LayerNorm + FFN: 融合 4 个操作
# =====================================================================

def megakernel_layernorm_ffn(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Megakernel fused LayerNorm → FFN(GeLU) → Linear.

    Multi-kernel 版本 materialise 了 3 个中间 tensor:
      x_norm [B,S,D], hidden [B,S,D_ff], gelu_out [B,S,D_ff]

    Megakernel 版本按 chunk 处理，每个 chunk 内：
      LayerNorm → up-proj → GeLU → down-proj 在 "寄存器/共享内存" 中流水线执行。
      chunk 间独立 → 模拟 persistent thread block 的 tile 处理。

    Args:
        x: [B, S, D]
        chunk_size: 每次处理的 S 维 tile 大小

    Returns:
        [B, S, D]
    """
    B, S, D = x.shape
    device = x.device
    output = torch.zeros_like(x)

    n_chunks = math.ceil(S / chunk_size)

    for c in range(n_chunks):
        s0 = c * chunk_size
        s1 = min(s0 + chunk_size, S)

        # ── "寄存器": 取出 x 的一个 chunk ──
        x_chunk = x[:, s0:s1, :].float()  # [B, cs, D]

        # ── 阶段 1: LayerNorm (在 "寄存器" 中) ──
        mean = x_chunk.mean(dim=-1, keepdim=True)
        var = x_chunk.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_chunk - mean) / torch.sqrt(var + 1e-5)
        x_norm = x_norm * ln_weight.float() + ln_bias.float()
        # x_chunk 可以丢弃 — 不写回 DRAM

        # ── 阶段 2: up-projection (在 "共享内存" 中) ──
        hidden = x_norm @ w1.float()  # [B, cs, D_ff]
        if b1 is not None:
            hidden = hidden + b1.float()
        # x_norm 可以丢弃

        # ── 阶段 3: GeLU (原地，"寄存器" 中) ──
        hidden = hidden * 0.5 * (1.0 + torch.erf(hidden / math.sqrt(2.0)))

        # ── 阶段 4: down-projection (写回 "全局内存") ──
        out_chunk = hidden @ w2.float()  # [B, cs, D]
        if b2 is not None:
            out_chunk = out_chunk + b2.float()

        output[:, s0:s1, :] = out_chunk.to(x.dtype)

    return output


# =====================================================================
#  Megakernel Transformer Block: 全融合
# =====================================================================

def megakernel_transformer_block(
    x: torch.Tensor,
    w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor, w_o: torch.Tensor,
    ln1_w: torch.Tensor, ln1_b: torch.Tensor,
    w_ff1: torch.Tensor, w_ff2: torch.Tensor,
    ln2_w: torch.Tensor, ln2_b: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """Megakernel transformer block: 将整个 block 融合为少量阶段。

    相比 multi_kernel_transformer_block 的 15+ kernel launch，这里实现：

      Phase 1 (fused): LayerNorm + QKV projection + reshape  (1 pass over x)
      Phase 2 (fused): Attention megakernel (QK^T + mask + softmax + V)
      Phase 3 (fused): O-proj + residual add  (1 write)
      Phase 4 (fused): LayerNorm + FFN-up + GeLU + FFN-down + residual add

    总计: 4 个 "mega phase" vs 15+ 个独立 kernel。

    Args:
        x: [B, S, D]
        w_q/k/v/o: [D, D]
        ln1_w/b, ln2_w/b: [D]
        w_ff1: [D, D_ff], w_ff2: [D_ff, D]
        num_heads: H
        causal: causal mask.

    Returns:
        [B, S, D]
    """
    B, S, D = x.shape
    K = D // num_heads

    # ════════════ Phase 1: fused LN + QKV proj ════════════
    # In a real megakernel, LN and QKV proj are fused:
    # each thread block computes LN for its tile, then immediately
    # multiplies by W_q/W_k/W_v without writing the normalised x to DRAM.

    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    var = x_f.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x_f - mean) / torch.sqrt(var + 1e-5)
    x_norm = x_norm * ln1_w.float() + ln1_b.float()

    # QKV projection — immediately from normalised x (still "in register")
    q = (x_norm @ w_q.float()).view(B, S, num_heads, K).transpose(1, 2)
    k = (x_norm @ w_k.float()).view(B, S, num_heads, K).transpose(1, 2)
    v = (x_norm @ w_v.float()).view(B, S, num_heads, K).transpose(1, 2)
    # x_norm is discarded — never written to DRAM

    # ════════════ Phase 2: fused attention megakernel ════════════
    attn_out = megakernel_attention(q, k, v, causal=causal)

    # ════════════ Phase 3: fused O-proj + residual ════════════
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    x = x + (attn_out.float() @ w_o.float()).to(x.dtype)

    # ════════════ Phase 4: fused LN + FFN + residual ════════════
    ffn_out = megakernel_layernorm_ffn(x, ln2_w, ln2_b, w_ff1, w_ff2)
    x = x + ffn_out

    return x
