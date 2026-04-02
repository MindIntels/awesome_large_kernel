"""
Optimisation 3 — Register-tiled attention for small K.

**Idea**: When K is tiny, each Q/K/V row occupies very few elements.  We can
tile along **both** the S_q and S_k dimensions and keep multiple Q-rows *and*
KV-rows resident simultaneously, maximising data reuse from fast memory
(cache / registers).

The key insight is that the "register pressure" per row is proportional to K,
so when K is small we can afford a much larger tile in the sequence dimensions,
which amortises the cost of loading KV data across more Q rows.

Tile parameters:
  - B_q: number of Q rows per tile (large, since K is small)
  - B_k: number of KV rows per tile

Algorithm:
  For each (B_q × B_k) tile:
    1. Load Q_tile [B_q, K], K_tile [B_k, K], V_tile [B_k, K] into fast mem.
    2. Compute partial scores [B_q, B_k] — tiny matmul when K is small.
    3. Update online-softmax accumulators for those B_q rows.

**Benefit**:
- Cache efficiency: Q_tile stays resident while streaming KV tiles through.
- For K=16, B_q=128: each Q-tile is only 128×16=2048 elements ≈ 8 KB.
- Minimal inner-loop overhead → high ALU utilisation.
"""

import math
import torch


def register_tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    tile_q: int = 128,
    tile_k: int = 64,
) -> torch.Tensor:
    """Register-tiled attention — large Q-tiles enabled by small K.

    Args:
        q, k, v: [B, H, S, K]
        causal: causal mask.
        tile_q: Q-tile size (can be large since K is small).
        tile_k: KV-tile size.

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

    n_tq = math.ceil(S_q / tile_q)

    for tq in range(n_tq):
        i0 = tq * tile_q
        i1 = min(i0 + tile_q, S_q)
        qi = q_f[:, :, i0:i1, :]  # [B, H, tile_q', K]  — stays "in register"

        # Per Q-tile online-softmax state
        bq = i1 - i0
        m_i = torch.full((B, H, bq, 1), float("-inf"), device=device)
        l_i = torch.zeros((B, H, bq, 1), device=device)
        o_i = torch.zeros((B, H, bq, K), device=device)

        n_tk = math.ceil(S_k / tile_k)
        for tk in range(n_tk):
            j0 = tk * tile_k
            j1 = min(j0 + tile_k, S_k)

            kj = k_f[:, :, j0:j1, :]  # [B, H, tile_k', K]
            vj = v_f[:, :, j0:j1, :]

            # Tiny matmul: [B, H, tile_q', K] x [B, H, K, tile_k'] → [B, H, tile_q', tile_k']
            s_ij = torch.matmul(qi, kj.transpose(-2, -1))

            if causal:
                qi_idx = torch.arange(i0, i1, device=device).unsqueeze(1)
                kj_idx = torch.arange(j0, j1, device=device).unsqueeze(0)
                mask = kj_idx > qi_idx
                s_ij.masked_fill_(mask, float("-inf"))

                # Early exit: if entire tile is masked, skip
                if mask.all():
                    continue

            # Online softmax update
            m_new = torch.maximum(m_i, s_ij.max(dim=-1, keepdim=True).values)
            exp_diff = torch.exp(m_i - m_new)
            p_ij = torch.exp(s_ij - m_new)

            l_i = l_i * exp_diff + p_ij.sum(dim=-1, keepdim=True)
            o_i = o_i * exp_diff + torch.matmul(p_ij, vj)
            m_i = m_new

        output[:, :, i0:i1, :] = o_i / l_i.clamp(min=1e-6)

    return output.to(q.dtype)
