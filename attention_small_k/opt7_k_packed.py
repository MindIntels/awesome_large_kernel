"""
Optimisation 7 — K-Packing: 将小K变成大K以提升访存效率

============================================================================
  可行性分析："小K → 大K" 的三条路径
============================================================================

当 K 非常小时（如 K=8），QK^T 的 dot product 只有 K=8 次乘加，算术强度极低，
访存开销（写/读 S_q×S_k score 矩阵）远大于计算开销。

"把小K变成大K"有三种思路，可行性各异：

【路径 A】多头K维拼接 — ❌ 不直接可行
  将 G 个 head 的 K 拼在一起：Q_packed=[S_q, G*K], K_packed=[S_k, G*K]
  做一个大 matmul → scores = Q_packed @ K_packed^T = Σ_g (Q_g · K_g^T)
  问题：这把 G 个 head 的 score 加在了一起！softmax 需要每个 head 独立运算，
  直接拼接会破坏正确性。

【路径 B】K-Padding 对齐 — ✅ 完全可行
  将 K 补零到 SIMD/cache-line 对齐的大小（如 K=12→16, K=8→32）。
  补零不影响 dot product 结果，但显著改善：
  - SIMD 利用率：向量指令满载运行，无需 mask
  - Cache-line 对齐：每行 Q/K 完整占满 cache line，无浪费
  - 内存合并访问：GPU 上 coalesced access

【路径 C】Head-Group 合并 — ✅ 可行（改变了注意力语义）
  将 G 个相邻 head 合并成 1 个宽 head（K_new = G * K_old，H_new = H / G）。
  每个"宽 head"在更大的 K 空间里做 attention，算术强度提高 G 倍。
  这本质上就是 GQA（Grouped Query Attention）/ 宽头注意力的思想。
  注意：这改变了模型的注意力语义（G个head共享score），需要配合训练使用。

本文件实现了路径 B 和路径 C 两种方案。

============================================================================
"""

import math
import torch


# =====================================================================
#  方案 B: K-Padding — 补零到对齐大小
# =====================================================================

def _pad_k_dim(t: torch.Tensor, target_k: int) -> torch.Tensor:
    """将 tensor 最后一维从 K 补零到 target_k。"""
    K = t.size(-1)
    if K >= target_k:
        return t
    pad_size = target_k - K
    return torch.nn.functional.pad(t, (0, pad_size), value=0.0)


def _next_aligned(k: int, alignment: int) -> int:
    """将 k 上取整到 alignment 的倍数。"""
    return ((k + alignment - 1) // alignment) * alignment


def kpadded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    alignment: int = 32,
) -> torch.Tensor:
    """K-Padded Attention: 将小K补零到对齐大小，提升SIMD/cache效率。

    核心思路：
      - K=8 → pad to 32 (4× larger), SIMD 利用率从 25% → 100%
      - K=16 → pad to 32 (2× larger)
      - K=32 → no padding needed
      - 补零不改变 dot product 结果，因为 0 × anything = 0

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        alignment: 目标对齐大小 (32 = AVX-256 一条指令处理 8 个 float32)

    Returns:
        [B, H, S_q, K]  (裁剪回原始K)
    """
    B, H, S_q, K_orig = q.shape
    S_k = k.size(2)

    K_padded = _next_aligned(K_orig, alignment)

    # 如果已经对齐，无需 padding
    if K_padded == K_orig:
        # 走高效路径：合并 (B,H) → 大 batch bmm
        return _batched_bmm_attention(q, k, v, causal)

    # Pad Q, K, V 的 K 维到对齐大小
    # 补零不影响 dot product：score[i,j] = Σ_{d=0}^{K_orig-1} Q[i,d]*K[j,d] + Σ_{d=K_orig}^{K_pad-1} 0*0
    q_pad = _pad_k_dim(q.float(), K_padded)  # [B, H, S_q, K_padded]
    k_pad = _pad_k_dim(k.float(), K_padded)  # [B, H, S_k, K_padded]
    v_pad = _pad_k_dim(v.float(), K_padded)  # [B, H, S_k, K_padded]

    scale = 1.0 / math.sqrt(K_orig)  # 注意：scale 用原始 K，不是 padded K

    # 合并 (B, H) → 大 batch bmm，让 BLAS 更好调度
    q_flat = (q_pad * scale).reshape(B * H, S_q, K_padded).contiguous()
    k_flat = k_pad.reshape(B * H, S_k, K_padded).contiguous()
    v_flat = v_pad.reshape(B * H, S_k, K_padded).contiguous()

    # 现在 K_padded 是对齐的，内层 dot product 可满载 SIMD
    scores = torch.bmm(q_flat, k_flat.transpose(-2, -1))  # [B*H, S_q, S_k]

    if causal:
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num(0.0)

    out = torch.bmm(attn, v_flat)  # [B*H, S_q, K_padded]
    out = out.reshape(B, H, S_q, K_padded)

    # 裁剪回原始 K 维度
    return out[:, :, :, :K_orig].to(q.dtype)


def _batched_bmm_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """Head-batch fused attention: 合并 (B,H) → 大 batch bmm。

    当 B*H 很大时，一次 bmm 调用能更好地利用 BLAS 并行性，
    避免 per-head 独立 matmul 的 kernel launch overhead。
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale = 1.0 / math.sqrt(K)

    q_flat = (q.float() * scale).reshape(B * H, S_q, K).contiguous()
    k_flat = k.float().reshape(B * H, S_k, K).contiguous()
    v_flat = v.float().reshape(B * H, S_k, K).contiguous()

    scores = torch.bmm(q_flat, k_flat.transpose(-2, -1))

    if causal:
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num(0.0)

    out = torch.bmm(attn, v_flat)
    return out.reshape(B, H, S_q, K).to(q.dtype)


# =====================================================================
#  方案 C: Head-Group 合并 — 多头 → 宽头
# =====================================================================

def head_grouped_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_size: int = 4,
    causal: bool = False,
) -> torch.Tensor:
    """Head-Grouped Attention: 将 G 个相邻 head 合并成 1 个宽 head。

    核心思路：
      - 原始: H=16 heads, K=8  → 16 个 [S,8]@[8,S] matmul (K太小，算术强度低)
      - 合并: H=4 groups, K_eff=32 → 4 个 [S,32]@[32,S] matmul (K变大，算术强度4x)

    语义变化说明：
      合并后同组 head 共享同一个 attention score 矩阵。这等价于：
      - 每个 group 内的 G 个 head 看到相同的 attention pattern
      - 但输出的 value 仍然是 G 个独立 head 的加权组合
      实际上类似于 Multi-Query Attention (MQA) 的反向：Q/K 共享但 V 独立。

    注意：此方案改变了注意力语义，需要配合训练使用，不是 drop-in 替换。
    这里为了展示可行性，同时对V也做了grouping。对比测试使用独立的 ref 函数。

    Args:
        q, k, v: [B, H, S, K]  (H 必须能被 group_size 整除)
        group_size: 合并的 head 数 G
        causal: causal mask.

    Returns:
        [B, H, S_q, K]  (shape 与输入一致)
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    G = group_size

    assert H % G == 0, f"H={H} must be divisible by group_size={G}"
    H_new = H // G
    K_eff = K * G  # 有效 K 维度变大了 G 倍

    scale = 1.0 / math.sqrt(K_eff)

    # 合并 G 个相邻 head 的 K 维：
    # [B, H, S, K] → [B, H/G, G, S, K] → [B, H/G, S, G*K]
    q_grouped = q.float().reshape(B, H_new, G, S_q, K).transpose(2, 3).reshape(B, H_new, S_q, K_eff)
    k_grouped = k.float().reshape(B, H_new, G, S_k, K).transpose(2, 3).reshape(B, H_new, S_k, K_eff)
    v_grouped = v.float().reshape(B, H_new, G, S_k, K).transpose(2, 3).reshape(B, H_new, S_k, K_eff)

    # 现在 K_eff 足够大，matmul 效率高
    scores = torch.matmul(q_grouped, k_grouped.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(
            torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = attn.nan_to_num(0.0)

    # [B, H_new, S_q, K_eff]
    out_grouped = torch.matmul(attn, v_grouped)

    # 还原成 [B, H, S_q, K]
    out = out_grouped.reshape(B, H_new, S_q, G, K).transpose(2, 3).reshape(B, H, S_q, K)

    return out.to(q.dtype)


def head_grouped_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group_size: int = 4,
    causal: bool = False,
) -> torch.Tensor:
    """Head-Grouped Attention 的参考实现（用标准attention模拟）。

    对每组 G 个 head：先拼 K 维成宽 head，再做标准 attention。
    结果应与 head_grouped_attention 完全一致。
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    G = group_size
    H_new = H // G
    K_eff = K * G
    scale = 1.0 / math.sqrt(K_eff)

    output = torch.zeros(B, H, S_q, K, device=q.device, dtype=torch.float32)

    for g in range(H_new):
        # 取出 G 个 head，拼 K 维
        h_start = g * G
        h_end = h_start + G
        # [B, G, S, K] → [B, S, G*K]
        q_g = q[:, h_start:h_end, :, :].float().permute(0, 2, 1, 3).reshape(B, S_q, K_eff)
        k_g = k[:, h_start:h_end, :, :].float().permute(0, 2, 1, 3).reshape(B, S_k, K_eff)
        v_g = v[:, h_start:h_end, :, :].float().permute(0, 2, 1, 3).reshape(B, S_k, K_eff)

        scores = torch.matmul(q_g, k_g.transpose(-2, -1)) * scale  # [B, S_q, S_k]

        if causal:
            mask = torch.triu(
                torch.ones(S_q, S_k, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out_g = torch.matmul(attn, v_g)  # [B, S_q, G*K]

        # 还原 [B, S_q, G*K] → [B, G, S_q, K]
        out_g = out_g.reshape(B, S_q, G, K).permute(0, 2, 1, 3)
        output[:, h_start:h_end, :, :] = out_g

    return output.to(q.dtype)
