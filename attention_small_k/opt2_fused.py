"""
Optimisation 2 — Fused QK·softmax·V in a single pass (online softmax).

**Idea**: For small K, materialising the full S_q×S_k score matrix is wasteful
because each element is only K multiplies — the memory traffic to write and
re-read that matrix dominates.  We fuse the three stages (QK, softmax, V-mul)
into a single streaming pass over K-blocks using the **online softmax** trick
(Milakov & Gimelshein 2018).

Algorithm per query row q_i:
  1. Stream through all key rows k_j in blocks.
  2. For each block, compute dot(q_i, k_j) (cheap when K is small).
  3. Update running max, running sum-of-exp, and running weighted output O_i
     using the online-softmax correction factor.
  4. After all blocks: O_i /= running_sum → done, no S_q×S_k matrix ever stored.

**Benefit**:
- Memory: O(B·H·S_q·K) instead of O(B·H·S_q·S_k) — huge win for long sequences.
- Compute: same FLOPs, but much less memory traffic when K is small.
"""

import math
import torch


def fused_qk_softmax_v_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_size: int = 64,
) -> torch.Tensor:
    """Fused QK·softmax·V with online softmax — no S_q×S_k materialisation.

    Args:
        q, k, v: [B, H, S_q/S_k, K]
        causal: causal mask.
        block_size: number of K-rows processed per inner-loop step.

    Returns:
        [B, H, S_q, K]
    """
    B, H, S_q, K = q.shape
    S_k = k.size(2)
    scale = 1.0 / math.sqrt(K)
    device = q.device

    # Work in float32 for stability
    q_f = q.float() * scale
    k_f = k.float()
    v_f = v.float()

    # Online softmax accumulators — per query position
    # m: running max  [B, H, S_q, 1]
    # l: running sum-of-exp  [B, H, S_q, 1]
    # o: running weighted output  [B, H, S_q, K]
    m = torch.full((B, H, S_q, 1), float("-inf"), device=device)
    l = torch.zeros((B, H, S_q, 1), device=device)
    o = torch.zeros((B, H, S_q, K), device=device)

    n_blocks = math.ceil(S_k / block_size)

    for j in range(n_blocks):
        j_start = j * block_size
        j_end = min(j_start + block_size, S_k)

        # [B, H, S_q, block] — dot product
        kj = k_f[:, :, j_start:j_end, :]          # [B, H, block, K]
        vj = v_f[:, :, j_start:j_end, :]          # [B, H, block, K]
        sij = torch.matmul(q_f, kj.transpose(-2, -1))  # [B, H, S_q, block]

        if causal:
            # query row i can attend to key col c if c <= i
            qi_idx = torch.arange(S_q, device=device).unsqueeze(1)       # [S_q, 1]
            kj_idx = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # [1, block]
            mask = kj_idx > qi_idx  # [S_q, block]
            sij.masked_fill_(mask, float("-inf"))

        # Online softmax update
        m_new = torch.maximum(m, sij.max(dim=-1, keepdim=True).values)
        # Correction factor for old accumulators
        exp_diff = torch.exp(m - m_new)
        # New block weights
        p_ij = torch.exp(sij - m_new)

        l = l * exp_diff + p_ij.sum(dim=-1, keepdim=True)
        o = o * exp_diff + torch.matmul(p_ij, vj)
        m = m_new

    # Final normalisation
    o = o / l.clamp(min=1e-6)
    return o.to(q.dtype)
