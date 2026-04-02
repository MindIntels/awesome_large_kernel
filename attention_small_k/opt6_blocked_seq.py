"""
Optimisation 6 — Blocked-sequence attention with KV-cache-friendly layout.

**Idea**: When K is small, each KV row is tiny (K floats ≈ 32–128 bytes).
This means that an enormous number of KV rows fit in L1/L2 cache simultaneously.
We exploit this by processing large contiguous **blocks of KV rows** while
iterating over Q in the outer loop, maximising temporal reuse of KV data.

The key difference from standard tiling:
  - Standard tiling: tile_q and tile_k are roughly equal.
  - Here: tile_k is **very large** (since each row is small) while tile_q
    is moderate.  This ensures that each KV block is loaded from DRAM
    only once and reused across all Q tiles.

Additionally, we transpose-pack K into [B, H, K, S_k] layout once upfront,
so the inner dot product is a contiguous memory access along S_k, avoiding
strided loads.

**Benefit**:
- Near-optimal DRAM→cache traffic: each KV element loaded exactly once.
- Contiguous K-transpose layout eliminates strided access in the inner loop.
- For K=16, S_k=4096: KV block = 2×4096×16×4 = 512 KB, fits in L2.
"""

import math
import torch


def blocked_seq_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    tile_q: int = 32,
    tile_k: int = 512,
) -> torch.Tensor:
    """Blocked-sequence attention with transposed K layout.

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        tile_q: Q tile size.
        tile_k: KV tile size (large — exploits small K for cache residency).

    Returns:
        [B, H, S_q, K]
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale = 1.0 / math.sqrt(K)
    device = q.device

    q_f = q.float() * scale
    v_f = v.float()

    # Pre-transpose K: [B, H, S_k, K] → [B, H, K, S_k] (contiguous)
    kt = k.float().transpose(-2, -1).contiguous()

    output = torch.zeros(B, H, S_q, K, device=device)

    n_tq = math.ceil(S_q / tile_q)

    for tq in range(n_tq):
        i0 = tq * tile_q
        i1 = min(i0 + tile_q, S_q)
        qi = q_f[:, :, i0:i1, :]  # [B, H, Bq, K]
        bq = i1 - i0

        # Online-softmax state
        m_i = torch.full((B, H, bq, 1), float("-inf"), device=device)
        l_i = torch.zeros((B, H, bq, 1), device=device)
        o_i = torch.zeros((B, H, bq, K), device=device)

        n_tk = math.ceil(S_k / tile_k)
        for tk in range(n_tk):
            j0 = tk * tile_k
            j1 = min(j0 + tile_k, S_k)

            # QK^T via pre-transposed K: [B,H,Bq,K] x [B,H,K,Bk] → [B,H,Bq,Bk]
            s_ij = torch.matmul(qi, kt[:, :, :, j0:j1])
            vj = v_f[:, :, j0:j1, :]

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

        output[:, :, i0:i1, :] = o_i / l_i.clamp(min=1e-6)

    return output.to(q.dtype)
