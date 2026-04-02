"""
Persistent Megakernel Attention — 模拟 "persistent thread" 执行模型。

============================================================================
  Persistent Thread / Persistent Kernel 原理
============================================================================

传统 GPU kernel：每个 thread block 处理一个 tile，处理完就退出。
  → 下一个 kernel launch 才能开始新任务
  → thread block 启动/退出开销 × N_tiles

Persistent kernel：thread block **不退出**，而是在一个循环中持续领取新任务：
  while (tile = atomicAdd(&global_counter, 1)) < total_tiles:
      process(tile)

好处：
  1. 只有一次 kernel launch（消除 launch overhead）
  2. Thread block 的共享内存 / 寄存器在整个生命周期内有效（跨 tile 复用）
  3. 对硬件 SM 数量自适应：恰好让 N_sm 个 block 持续满载
  4. 天然支持 **动态负载均衡**：快的 block 自动领取更多 tile
  5. 可以在 tile 之间共享只读数据（如 K/V cache in decoding）

本模块使用 Python 模拟 persistent thread 的行为：
  - "thread block" = 一个迭代步
  - "共享内存" = Python 变量（不写入 output tensor 直到最后）
  - "atomic counter" = for-loop index
  - "跨 tile 共享" = 同一个 KV cache reference
============================================================================
"""

import math
import torch


def persistent_megakernel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    num_sms: int = 4,
    block_q: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Persistent megakernel attention with simulated SM scheduling.

    Simulates how a persistent kernel maps work to a fixed number of SMs:
      1. Total tiles = n_q_tiles
      2. Each "SM" (simulated) processes tiles in round-robin order
      3. KV data is "pinned in L2/shared memory" and re-read (not re-loaded
         from DRAM) across Q-tile iterations on the same SM
      4. Only one "kernel launch"

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        num_sms: simulated number of SMs (persistent threads).
        block_q: Q tile size per SM.
        block_k: KV tile size.

    Returns:
        [B, H, S_q, K]
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale = 1.0 / math.sqrt(K)
    device = q.device

    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()

    output = torch.zeros(B, H, S_q, K, device=device)

    n_q_tiles = math.ceil(S_q / block_q)
    n_k_tiles = math.ceil(S_k / block_k)

    # ────────── Persistent kernel launch (只有一次) ──────────

    # Simulate round-robin tile assignment to SMs:
    # SM 0 gets tiles [0, num_sms, 2*num_sms, ...]
    # SM 1 gets tiles [1, 1+num_sms, 1+2*num_sms, ...]
    # This simulates atomic counter-based work stealing.

    for sm_id in range(min(num_sms, n_q_tiles)):
        # ── 该 SM 的 "寄存器文件" 和 "共享内存" 在整个生命周期有效 ──
        # KV tiles 可以被视为 "pinned in L2" — 多个 Q tile 复用同一份 KV

        # SM round-robin: this SM processes tiles sm_id, sm_id+num_sms, ...
        tile_idx = sm_id
        while tile_idx < n_q_tiles:
            i0 = tile_idx * block_q
            i1 = min(i0 + block_q, S_q)
            bq = i1 - i0

            # ── 加载 Q tile 到 "寄存器" ──
            qi = q_f[:, :, i0:i1, :]

            # ── Online softmax 状态 — "寄存器" 变量 ──
            m_i = torch.full((B, H, bq, 1), float("-inf"), device=device)
            l_i = torch.zeros((B, H, bq, 1), device=device)
            o_i = torch.zeros((B, H, bq, K), device=device)

            # ── 流式扫描所有 KV tiles ──
            for j in range(n_k_tiles):
                j0 = j * block_k
                j1 = min(j0 + block_k, S_k)

                # ── KV tile 从 "L2 cache / 共享内存" 读取 ──
                # (在持一个 persistent kernel中，同一 SM 的不同 Q tile 轮次
                #  可以复用 L2 中的 KV 数据 — 不需要重新从 DRAM 加载)
                kj = k_f[:, :, j0:j1, :]
                vj = v_f[:, :, j0:j1, :]

                # ── fused: QK^T + mask + online softmax ──
                s_ij = torch.matmul(qi, kj.transpose(-2, -1))

                if causal:
                    qi_idx = torch.arange(i0, i1, device=device).unsqueeze(1)
                    kj_idx = torch.arange(j0, j1, device=device).unsqueeze(0)
                    mask = kj_idx > qi_idx
                    s_ij.masked_fill_(mask, float("-inf"))
                    if mask.all():
                        continue

                m_new = torch.maximum(m_i, s_ij.max(dim=-1, keepdim=True).values)
                exp_diff = torch.exp(m_i - m_new)
                p_ij = torch.exp(s_ij - m_new)

                l_i = l_i * exp_diff + p_ij.sum(dim=-1, keepdim=True)
                o_i = o_i * exp_diff + torch.matmul(p_ij, vj)
                m_i = m_new

            # ── 写回 "全局内存" — SM 生命周期中的唯一写操作 ──
            output[:, :, i0:i1, :] = o_i / l_i.clamp(min=1e-6)

            # ── "Atomic counter increment": 领取下一个 tile ──
            tile_idx += num_sms

    # ────────── Persistent kernel 结束 ──────────

    return output.to(q.dtype)
