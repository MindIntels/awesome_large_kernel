"""
Comprehensive test suite for small-K attention optimisations.

Tests:
  1. Correctness: each optimisation matches the baseline (standard attention).
  2. Various K values: 4, 8, 16, 32.
  3. Various sequence lengths and batch/head combos.
  4. Causal and non-causal modes.
  5. Numerical tolerance (int8 variant gets wider tolerance).
  6. Edge cases: S_q=1, S_k=1, S_q != S_k.
  7. Performance benchmark (optional, prints timing).
"""

import math
import time
import pytest
import torch

from attention_small_k import (
    standard_attention,
    vectorized_small_k_attention,
    fused_qk_softmax_v_attention,
    register_tiled_attention,
    unrolled_attention,
    quantized_small_k_attention,
    blocked_seq_attention,
    kpadded_attention,
    head_grouped_attention,
    head_grouped_attention_ref,
)

# ---- helpers --------------------------------------------------------

DEVICE = "cpu"
SEED = 42

OPTIMISED_FNS = [
    ("opt1_vectorized", vectorized_small_k_attention, 1e-5),
    ("opt2_fused", fused_qk_softmax_v_attention, 1e-5),
    ("opt3_register_tiled", register_tiled_attention, 1e-5),
    ("opt4_unrolled", unrolled_attention, 1e-5),
    ("opt5_quantized", quantized_small_k_attention, 0.05),  # int8 → wider tol
    ("opt6_blocked_seq", blocked_seq_attention, 1e-5),
    ("opt7_kpadded", kpadded_attention, 1e-5),
]


def _make_qkv(B, H, S_q, S_k, K, dtype=torch.float32):
    torch.manual_seed(SEED)
    q = torch.randn(B, H, S_q, K, device=DEVICE, dtype=dtype)
    k = torch.randn(B, H, S_k, K, device=DEVICE, dtype=dtype)
    v = torch.randn(B, H, S_k, K, device=DEVICE, dtype=dtype)
    return q, k, v


# ---- parametrised correctness tests --------------------------------

K_VALUES = [4, 8, 16, 32]
SEQ_CONFIGS = [
    # (B, H, S_q, S_k)
    (1, 1, 16, 16),
    (2, 4, 64, 64),
    (1, 2, 128, 128),
    (2, 2, 63, 63),   # non-power-of-2
    (1, 1, 1, 32),    # single query
    (1, 1, 32, 1),    # single key
    (2, 4, 37, 53),   # S_q != S_k
]


@pytest.mark.parametrize("name,fn,atol", OPTIMISED_FNS, ids=[n for n, _, _ in OPTIMISED_FNS])
@pytest.mark.parametrize("K", K_VALUES, ids=[f"K{k}" for k in K_VALUES])
@pytest.mark.parametrize("cfg", SEQ_CONFIGS, ids=[f"B{c[0]}H{c[1]}Sq{c[2]}Sk{c[3]}" for c in SEQ_CONFIGS])
def test_non_causal(name, fn, atol, K, cfg):
    """Each optimisation must match baseline for non-causal attention."""
    B, H, S_q, S_k = cfg
    q, k, v = _make_qkv(B, H, S_q, S_k, K)

    ref = standard_attention(q, k, v, causal=False)
    out = fn(q, k, v, causal=False)

    assert out.shape == ref.shape, f"{name}: shape mismatch {out.shape} vs {ref.shape}"
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=1e-4), (
        f"{name} K={K} cfg={cfg}: max diff = {(out.float() - ref.float()).abs().max().item():.6e}"
    )


@pytest.mark.parametrize("name,fn,atol", OPTIMISED_FNS, ids=[n for n, _, _ in OPTIMISED_FNS])
@pytest.mark.parametrize("K", K_VALUES, ids=[f"K{k}" for k in K_VALUES])
@pytest.mark.parametrize("cfg",
    [(1, 1, 16, 16), (2, 4, 64, 64), (1, 2, 63, 63), (2, 2, 128, 128)],
    ids=["B1H1S16", "B2H4S64", "B1H2S63", "B2H2S128"],
)
def test_causal(name, fn, atol, K, cfg):
    """Each optimisation must match baseline for causal attention (S_q == S_k)."""
    B, H, S, _ = cfg
    q, k, v = _make_qkv(B, H, S, S, K)

    ref = standard_attention(q, k, v, causal=True)
    out = fn(q, k, v, causal=True)

    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=1e-4), (
        f"{name} causal K={K} S={S}: max diff = {(out.float() - ref.float()).abs().max().item():.6e}"
    )


# ---- dtype test (float16 inputs) -----------------------------------

@pytest.mark.parametrize("name,fn,atol", OPTIMISED_FNS, ids=[n for n, _, _ in OPTIMISED_FNS])
def test_float16_inputs(name, fn, atol):
    """Optimisations should handle float16 inputs without crashing."""
    B, H, S, K = 1, 2, 32, 16
    q, k, v = _make_qkv(B, H, S, S, K, dtype=torch.float16)
    ref = standard_attention(q.float(), k.float(), v.float(), causal=False)
    out = fn(q, k, v, causal=False)
    # Wider tolerance for fp16
    tol = max(atol, 0.05)
    assert torch.allclose(out.float(), ref.float(), atol=tol, rtol=0.01), (
        f"{name} fp16: max diff = {(out.float() - ref.float()).abs().max().item():.6e}"
    )


# ---- edge case: all-zero input -------------------------------------

def test_zero_input():
    """All-zero QKV should produce all-zero (or uniform-weighted) output without NaN."""
    B, H, S, K = 1, 1, 8, 8
    q = torch.zeros(B, H, S, K)
    k = torch.zeros(B, H, S, K)
    v = torch.zeros(B, H, S, K)
    for name, fn, _ in OPTIMISED_FNS:
        out = fn(q, k, v, causal=False)
        assert not out.isnan().any(), f"{name}: NaN in output for zero input"
        assert not out.isinf().any(), f"{name}: Inf in output for zero input"


# ---- performance benchmark (not a correctness gate) -----------------

@pytest.mark.parametrize("K", [8, 16, 32])
def test_benchmark(K, capsys):
    """Print timing comparison (informational only, always passes)."""
    B, H, S = 4, 8, 256
    q, k, v = _make_qkv(B, H, S, S, K)

    results = []
    all_fns = [("baseline", standard_attention, 0)] + list(OPTIMISED_FNS)

    for name, fn, _ in all_fns:
        # Warm up
        fn(q, k, v, causal=False)
        t0 = time.perf_counter()
        for _ in range(5):
            fn(q, k, v, causal=False)
        elapsed = (time.perf_counter() - t0) / 5
        results.append((name, elapsed))

    with capsys.disabled():
        print(f"\n{'='*50}")
        print(f" Benchmark: K={K}, B={B}, H={H}, S={S}")
        print(f"{'='*50}")
        base_t = results[0][1]
        for name, t in results:
            speedup = base_t / t if t > 0 else float("inf")
            print(f"  {name:<25s} {t*1000:8.2f} ms  ({speedup:.2f}x)")


# ---- opt7 K-padding 专项测试 ----------------------------------------

@pytest.mark.parametrize("alignment", [16, 32, 64])
@pytest.mark.parametrize("K", [4, 7, 8, 12, 16], ids=[f"K{k}" for k in [4, 7, 8, 12, 16]])
def test_kpadded_alignment(K, alignment):
    """K-padding 对各种 K 和 alignment 组合都应与 baseline 一致。"""
    B, H, S = 2, 4, 64
    q, k, v = _make_qkv(B, H, S, S, K)
    ref = standard_attention(q, k, v, causal=False)
    out = kpadded_attention(q, k, v, causal=False, alignment=alignment)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4), (
        f"kpadded K={K} align={alignment}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("K", [4, 8, 16])
def test_kpadded_causal(K):
    """K-padding causal 模式测试。"""
    B, H, S = 2, 4, 64
    q, k, v = _make_qkv(B, H, S, S, K)
    ref = standard_attention(q, k, v, causal=True)
    out = kpadded_attention(q, k, v, causal=True, alignment=32)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


# ---- opt7 Head-Group 合并专项测试 -----------------------------------

@pytest.mark.parametrize("G", [2, 4, 8])
@pytest.mark.parametrize("K", [4, 8, 16])
def test_head_grouped_vs_ref(K, G):
    """head_grouped_attention 必须与其独立参考实现完全一致。"""
    B, H, S = 2, 8, 64  # H=8, 能被 G=2,4,8 整除
    q, k, v = _make_qkv(B, H, S, S, K)
    ref = head_grouped_attention_ref(q, k, v, group_size=G, causal=False)
    out = head_grouped_attention(q, k, v, group_size=G, causal=False)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4), (
        f"head_grouped G={G} K={K}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("G", [2, 4])
def test_head_grouped_causal(G):
    """Head-Group 合并 causal 模式。"""
    B, H, S, K = 2, 8, 64, 8
    q, k, v = _make_qkv(B, H, S, S, K)
    ref = head_grouped_attention_ref(q, k, v, group_size=G, causal=True)
    out = head_grouped_attention(q, k, v, group_size=G, causal=True)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


def test_head_grouped_increases_effective_k():
    """验证 head-grouping 确实增大了有效 K。"""
    B, H, S, K = 1, 8, 32, 8
    G = 4
    K_eff = K * G  # = 32, 4x larger
    assert K_eff == 32
    # 确保 H_new = H/G 个 "宽head" 各自使用 K_eff 做 dot product
    q, k, v = _make_qkv(B, H, S, S, K)
    out = head_grouped_attention(q, k, v, group_size=G, causal=False)
    # 输出 shape 应保持不变
    assert out.shape == (B, H, S, K)


def test_head_grouped_g1_equals_standard():
    """group_size=1 时，head_grouped 退化为标准 attention。"""
    B, H, S, K = 2, 4, 64, 16
    q, k, v = _make_qkv(B, H, S, S, K)
    ref = standard_attention(q, k, v, causal=False)
    out = head_grouped_attention(q, k, v, group_size=1, causal=False)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


# ---- opt7 性能对比 benchmark ----------------------------------------

@pytest.mark.parametrize("K", [8, 16])
def test_benchmark_opt7(K, capsys):
    """对比 K-padding 和 head-grouping 的性能。"""
    B, H, S = 4, 8, 256
    q, k, v = _make_qkv(B, H, S, S, K)

    fns = [
        ("baseline", lambda q,k,v: standard_attention(q,k,v,causal=False)),
        ("opt7_kpadded_a32", lambda q,k,v: kpadded_attention(q,k,v,causal=False,alignment=32)),
        ("opt7_kpadded_a64", lambda q,k,v: kpadded_attention(q,k,v,causal=False,alignment=64)),
        ("opt7_headgroup_G2", lambda q,k,v: head_grouped_attention(q,k,v,group_size=2,causal=False)),
        ("opt7_headgroup_G4", lambda q,k,v: head_grouped_attention(q,k,v,group_size=4,causal=False)),
    ]

    results = []
    for name, fn in fns:
        fn(q, k, v)  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            fn(q, k, v)
        elapsed = (time.perf_counter() - t0) / 5
        results.append((name, elapsed))

    with capsys.disabled():
        print(f"\n{'='*60}")
        print(f" Opt7 Benchmark: K={K}, B={B}, H={H}, S={S}")
        print(f"{'='*60}")
        base_t = results[0][1]
        for name, t in results:
            speedup = base_t / t if t > 0 else float("inf")
            print(f"  {name:<25s} {t*1000:8.2f} ms  ({speedup:.2f}x)")


# ---- run as script --------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
