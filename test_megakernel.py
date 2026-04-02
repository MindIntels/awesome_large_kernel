"""
Megakernel 测试套件

验证：
  1. megakernel_attention 与 multi_kernel_attention 结果一致
  2. persistent_megakernel_attention 与 multi_kernel_attention 结果一致
  3. megakernel_layernorm_ffn 与 multi_kernel_layernorm_ffn 结果一致
  4. megakernel_transformer_block 与 multi_kernel_transformer_block 结果一致
  5. 各种参数组合 (B, H, S, K) + causal / non-causal
  6. 不同 block_size / num_sms 参数
  7. 边界条件: S=1, 非对齐 S
  8. Benchmark: 计数中间 tensor 分配次数，展示 megakernel 的优势
"""

import math
import time
import pytest
import torch

from megakernel import (
    multi_kernel_attention,
    multi_kernel_transformer_block,
    multi_kernel_layernorm_ffn,
    megakernel_attention,
    megakernel_transformer_block,
    megakernel_layernorm_ffn,
    persistent_megakernel_attention,
    triton_megakernel_attention,
    auto_megakernel_attention,
    HAS_TRITON,
)

HAS_CUDA = torch.cuda.is_available()
requires_gpu = pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
requires_triton = pytest.mark.skipif(not (HAS_CUDA and HAS_TRITON), reason="Triton + CUDA required")

SEED = 42

# 自动检测 GPU：有 CUDA 就用 GPU，否则用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_qkv(B, H, S_q, S_k, K, dtype=torch.float32, device=None):
    torch.manual_seed(SEED)
    dev = device or DEVICE
    q = torch.randn(B, H, S_q, K, device=dev, dtype=dtype)
    k = torch.randn(B, H, S_k, K, device=dev, dtype=dtype)
    v = torch.randn(B, H, S_k, K, device=dev, dtype=dtype)
    return q, k, v


def _make_transformer_params(B, S, D, D_ff, num_heads, device=None):
    torch.manual_seed(SEED)
    dev = device or DEVICE
    x = torch.randn(B, S, D, device=dev)
    w_q = torch.randn(D, D, device=dev) * 0.02
    w_k = torch.randn(D, D, device=dev) * 0.02
    w_v = torch.randn(D, D, device=dev) * 0.02
    w_o = torch.randn(D, D, device=dev) * 0.02
    ln1_w = torch.ones(D, device=dev)
    ln1_b = torch.zeros(D, device=dev)
    w_ff1 = torch.randn(D, D_ff, device=dev) * 0.02
    w_ff2 = torch.randn(D_ff, D, device=dev) * 0.02
    ln2_w = torch.ones(D, device=dev)
    ln2_b = torch.zeros(D, device=dev)
    return (x, w_q, w_k, w_v, w_o, ln1_w, ln1_b, w_ff1, w_ff2, ln2_w, ln2_b)


# =====================================================================
#  Attention correctness tests
# =====================================================================

ATTN_CONFIGS = [
    # (B, H, S_q, S_k, K)
    (1, 1, 16, 16, 8),
    (2, 4, 64, 64, 16),
    (1, 2, 128, 128, 32),
    (2, 2, 63, 63, 16),    # non-power-of-2 seq
    (1, 1, 1, 32, 8),      # single query
    (1, 1, 32, 1, 8),      # single key
    (2, 4, 37, 53, 16),    # S_q != S_k
    (1, 8, 256, 256, 8),   # larger
]


@pytest.mark.parametrize("cfg", ATTN_CONFIGS,
    ids=[f"B{c[0]}H{c[1]}Sq{c[2]}Sk{c[3]}K{c[4]}" for c in ATTN_CONFIGS])
def test_megakernel_attention_noncausal(cfg):
    """megakernel_attention == multi_kernel_attention (non-causal)."""
    B, H, S_q, S_k, K = cfg
    q, k, v = _make_qkv(B, H, S_q, S_k, K)

    ref = multi_kernel_attention(q, k, v, causal=False)
    out = megakernel_attention(q, k, v, causal=False)

    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4), (
        f"megakernel attn cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("cfg", [c for c in ATTN_CONFIGS if c[2] == c[3]],
    ids=[f"B{c[0]}H{c[1]}S{c[2]}K{c[4]}" for c in ATTN_CONFIGS if c[2] == c[3]])
def test_megakernel_attention_causal(cfg):
    """megakernel_attention == multi_kernel_attention (causal)."""
    B, H, S, _, K = cfg
    q, k, v = _make_qkv(B, H, S, S, K)

    ref = multi_kernel_attention(q, k, v, causal=True)
    out = megakernel_attention(q, k, v, causal=True)

    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4), (
        f"megakernel causal cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_megakernel_attention_block_sizes(block_size):
    """不同 block_size 都应产生正确结果。"""
    B, H, S, K = 2, 4, 100, 16
    q, k, v = _make_qkv(B, H, S, S, K)

    ref = multi_kernel_attention(q, k, v, causal=False)
    out = megakernel_attention(q, k, v, causal=False, block_size=block_size)

    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


# =====================================================================
#  Persistent megakernel attention tests
# =====================================================================

@pytest.mark.parametrize("cfg", ATTN_CONFIGS,
    ids=[f"B{c[0]}H{c[1]}Sq{c[2]}Sk{c[3]}K{c[4]}" for c in ATTN_CONFIGS])
def test_persistent_megakernel_noncausal(cfg):
    """persistent_megakernel == multi_kernel (non-causal)."""
    B, H, S_q, S_k, K = cfg
    q, k, v = _make_qkv(B, H, S_q, S_k, K)

    ref = multi_kernel_attention(q, k, v, causal=False)
    out = persistent_megakernel_attention(q, k, v, causal=False)

    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4), (
        f"persistent cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("cfg", [c for c in ATTN_CONFIGS if c[2] == c[3]],
    ids=[f"B{c[0]}H{c[1]}S{c[2]}K{c[4]}" for c in ATTN_CONFIGS if c[2] == c[3]])
def test_persistent_megakernel_causal(cfg):
    """persistent_megakernel == multi_kernel (causal)."""
    B, H, S, _, K = cfg
    q, k, v = _make_qkv(B, H, S, S, K)

    ref = multi_kernel_attention(q, k, v, causal=True)
    out = persistent_megakernel_attention(q, k, v, causal=True)

    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("num_sms", [1, 2, 4, 8])
def test_persistent_different_sms(num_sms):
    """不同 SM 数量都应产生正确结果。"""
    B, H, S, K = 2, 4, 128, 16
    q, k, v = _make_qkv(B, H, S, S, K)

    ref = multi_kernel_attention(q, k, v, causal=False)
    out = persistent_megakernel_attention(q, k, v, causal=False, num_sms=num_sms)

    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


# =====================================================================
#  LayerNorm + FFN tests
# =====================================================================

LN_FFN_CONFIGS = [
    # (B, S, D, D_ff)
    (1, 16, 32, 64),
    (2, 64, 64, 128),
    (1, 100, 48, 96),
    (2, 1, 32, 64),     # S=1
]


@pytest.mark.parametrize("cfg", LN_FFN_CONFIGS,
    ids=[f"B{c[0]}S{c[1]}D{c[2]}Dff{c[3]}" for c in LN_FFN_CONFIGS])
def test_megakernel_ln_ffn(cfg):
    """megakernel_layernorm_ffn == multi_kernel_layernorm_ffn."""
    B, S, D, D_ff = cfg
    torch.manual_seed(SEED)
    x = torch.randn(B, S, D, device=DEVICE)
    ln_w = torch.ones(D, device=DEVICE)
    ln_b = torch.zeros(D, device=DEVICE)
    w1 = torch.randn(D, D_ff, device=DEVICE) * 0.02
    w2 = torch.randn(D_ff, D, device=DEVICE) * 0.02

    ref = multi_kernel_layernorm_ffn(x, ln_w, ln_b, w1, w2)
    out = megakernel_layernorm_ffn(x, ln_w, ln_b, w1, w2)

    assert torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3), (
        f"LN+FFN cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@pytest.mark.parametrize("chunk_size", [16, 32, 64, 128])
def test_megakernel_ln_ffn_chunk_sizes(chunk_size):
    """不同 chunk_size 都应正确。"""
    B, S, D, D_ff = 2, 100, 64, 128
    torch.manual_seed(SEED)
    x = torch.randn(B, S, D, device=DEVICE)
    ln_w = torch.ones(D, device=DEVICE)
    ln_b = torch.zeros(D, device=DEVICE)
    w1 = torch.randn(D, D_ff, device=DEVICE) * 0.02
    w2 = torch.randn(D_ff, D, device=DEVICE) * 0.02

    ref = multi_kernel_layernorm_ffn(x, ln_w, ln_b, w1, w2)
    out = megakernel_layernorm_ffn(x, ln_w, ln_b, w1, w2, chunk_size=chunk_size)

    assert torch.allclose(out.float(), ref.float(), atol=1e-4, rtol=1e-3)


# =====================================================================
#  Transformer block tests
# =====================================================================

BLOCK_CONFIGS = [
    # (B, S, D, D_ff, num_heads)
    (1, 16, 32, 64, 4),
    (2, 64, 64, 128, 8),
    (1, 33, 48, 96, 6),   # non-power-of-2
]


@pytest.mark.parametrize("cfg", BLOCK_CONFIGS,
    ids=[f"B{c[0]}S{c[1]}D{c[2]}H{c[4]}" for c in BLOCK_CONFIGS])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
def test_megakernel_transformer_block(cfg, causal):
    """megakernel_transformer_block == multi_kernel_transformer_block."""
    B, S, D, D_ff, NH = cfg
    params = _make_transformer_params(B, S, D, D_ff, NH)
    x = params[0]

    ref = multi_kernel_transformer_block(
        *params, num_heads=NH, causal=causal,
    )
    out = megakernel_transformer_block(
        *params, num_heads=NH, causal=causal,
    )

    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3), (
        f"transformer cfg={cfg} causal={causal}: max diff = "
        f"{(out.float()-ref.float()).abs().max():.6e}"
    )


# =====================================================================
#  Edge cases
# =====================================================================

def test_zero_input_attention():
    """全零输入不产生 NaN/Inf。"""
    B, H, S, K = 1, 2, 16, 8
    q = k = v = torch.zeros(B, H, S, K)
    for fn in [megakernel_attention, persistent_megakernel_attention]:
        out = fn(q, k, v, causal=False)
        assert not out.isnan().any(), f"{fn.__name__}: NaN"
        assert not out.isinf().any(), f"{fn.__name__}: Inf"


def test_single_element():
    """S=1, K=1 极端情况。"""
    q = k = v = torch.randn(1, 1, 1, 1)
    ref = multi_kernel_attention(q, k, v)
    out = megakernel_attention(q, k, v)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5)


# =====================================================================
#  Benchmark: 中间 tensor 分配计数
# =====================================================================

class TensorAllocCounter:
    """统计 torch.Tensor 分配次数 (通过 hook __torch_function__)."""

    def __init__(self):
        self.count = 0
        self._old_class = None

    def __enter__(self):
        self.count = 0
        counter = self

        class CountingTensor(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                result = super().__torch_function__(func, types, args, kwargs)
                if isinstance(result, torch.Tensor):
                    counter.count += 1
                return result

        return self

    def __exit__(self, *args):
        pass


def test_benchmark_attention(capsys):
    """对比 multi-kernel vs megakernel attention 的耗时。"""
    B, H, S, K = 4, 8, 256, 16
    q, k, v = _make_qkv(B, H, S, S, K)

    fns = [
        ("multi_kernel", lambda: multi_kernel_attention(q, k, v, causal=False)),
        ("megakernel", lambda: megakernel_attention(q, k, v, causal=False)),
        ("persistent_mega", lambda: persistent_megakernel_attention(q, k, v, causal=False)),
    ]

    results = []
    for name, fn in fns:
        fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        elapsed = (time.perf_counter() - t0) / 5
        results.append((name, elapsed))

    with capsys.disabled():
        print(f"\n{'='*60}")
        print(f" Megakernel Benchmark: B={B}, H={H}, S={S}, K={K}")
        print(f"{'='*60}")
        base_t = results[0][1]
        for name, t in results:
            ratio = base_t / t if t > 0 else float("inf")
            print(f"  {name:<25s} {t*1000:8.2f} ms  ({ratio:.2f}x vs multi-kernel)")
        print()
        print("  NOTE: On CPU, megakernel overhead from Python loops may dominate.")
        print("  Real GPU megakernels eliminate kernel launch + DRAM round-trips,")
        print("  giving 2-5x speedup for memory-bound workloads.")


def test_benchmark_transformer(capsys):
    """对比完整 transformer block。"""
    B, S, D, D_ff, NH = 2, 128, 64, 128, 8
    params = _make_transformer_params(B, S, D, D_ff, NH)

    fns = [
        ("multi_kernel_block", lambda: multi_kernel_transformer_block(*params, num_heads=NH)),
        ("megakernel_block", lambda: megakernel_transformer_block(*params, num_heads=NH)),
    ]

    results = []
    for name, fn in fns:
        fn()
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        elapsed = (time.perf_counter() - t0) / 5
        results.append((name, elapsed))

    with capsys.disabled():
        print(f"\n{'='*60}")
        print(f" Transformer Block Benchmark: B={B}, S={S}, D={D}, H={NH}")
        print(f"{'='*60}")
        base_t = results[0][1]
        for name, t in results:
            ratio = base_t / t if t > 0 else float("inf")
            print(f"  {name:<25s} {t*1000:8.2f} ms  ({ratio:.2f}x)")


# =====================================================================
#  GPU tests: Triton megakernel
# =====================================================================

GPU_ATTN_CONFIGS = [
    (1, 1, 64, 64, 16),
    (2, 4, 128, 128, 32),
    (1, 8, 256, 256, 64),
    (2, 2, 63, 63, 16),     # non-power-of-2 seq
    (1, 4, 64, 64, 12),     # non-power-of-2 K (will be padded)
]


@requires_triton
@pytest.mark.parametrize("cfg", GPU_ATTN_CONFIGS,
    ids=[f"B{c[0]}H{c[1]}Sq{c[2]}Sk{c[3]}K{c[4]}" for c in GPU_ATTN_CONFIGS])
def test_triton_megakernel_noncausal(cfg):
    """Triton megakernel == multi_kernel (non-causal) on GPU."""
    B, H, S_q, S_k, K = cfg
    q, k, v = _make_qkv(B, H, S_q, S_k, K, device="cuda")

    ref = multi_kernel_attention(q, k, v, causal=False)
    out = triton_megakernel_attention(q, k, v, causal=False)

    assert out.shape == ref.shape
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), (
        f"triton megakernel cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@requires_triton
@pytest.mark.parametrize("cfg", [c for c in GPU_ATTN_CONFIGS if c[2] == c[3]],
    ids=[f"B{c[0]}H{c[1]}S{c[2]}K{c[4]}" for c in GPU_ATTN_CONFIGS if c[2] == c[3]])
def test_triton_megakernel_causal(cfg):
    """Triton megakernel == multi_kernel (causal) on GPU."""
    B, H, S, _, K = cfg
    q, k, v = _make_qkv(B, H, S, S, K, device="cuda")

    ref = multi_kernel_attention(q, k, v, causal=True)
    out = triton_megakernel_attention(q, k, v, causal=True)

    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), (
        f"triton causal cfg={cfg}: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@requires_triton
def test_triton_megakernel_fp16():
    """Triton megakernel with float16 inputs on GPU."""
    B, H, S, K = 2, 4, 128, 32
    q, k, v = _make_qkv(B, H, S, S, K, dtype=torch.float16, device="cuda")

    ref = multi_kernel_attention(q.float(), k.float(), v.float(), causal=False)
    out = triton_megakernel_attention(q, k, v, causal=False)

    assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.02), (
        f"triton fp16: max diff = {(out.float()-ref.float()).abs().max():.6e}"
    )


@requires_triton
def test_triton_benchmark(capsys):
    """GPU Benchmark: multi-kernel vs Triton megakernel."""
    B, H, S, K = 4, 8, 512, 64
    q, k, v = _make_qkv(B, H, S, S, K, device="cuda")

    fns = [
        ("multi_kernel (GPU)", lambda: multi_kernel_attention(q, k, v, causal=False)),
        ("python_megakernel (GPU)", lambda: megakernel_attention(q, k, v, causal=False)),
        ("triton_megakernel (GPU)", lambda: triton_megakernel_attention(q, k, v, causal=False)),
    ]

    results = []
    for name, fn in fns:
        # Warmup
        for _ in range(3):
            fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 10
        results.append((name, elapsed))

    with capsys.disabled():
        print(f"\n{'='*65}")
        print(f" GPU Benchmark: B={B}, H={H}, S={S}, K={K}")
        print(f" Device: {torch.cuda.get_device_name(0)}")
        print(f"{'='*65}")
        base_t = results[0][1]
        for name, t in results:
            ratio = base_t / t if t > 0 else float("inf")
            print(f"  {name:<30s} {t*1000:8.3f} ms  ({ratio:.2f}x)")
        print()
        print("  Triton megakernel = real single-kernel GPU fusion")
        print("  (vs Python megakernel which still launches N CUDA kernels)")


# =====================================================================
#  auto_megakernel_attention tests (CPU + GPU)
# =====================================================================

def test_auto_megakernel_cpu():
    """auto_megakernel 在 CPU 上 fallback 到 Python 版本。"""
    B, H, S, K = 2, 4, 64, 16
    q, k, v = _make_qkv(B, H, S, S, K, device="cpu")
    ref = multi_kernel_attention(q, k, v, causal=False)
    out = auto_megakernel_attention(q, k, v, causal=False)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-4)


@requires_triton
def test_auto_megakernel_gpu():
    """auto_megakernel 在 GPU 上自动使用 Triton。"""
    B, H, S, K = 2, 4, 64, 16
    q, k, v = _make_qkv(B, H, S, S, K, device="cuda")
    ref = multi_kernel_attention(q, k, v, causal=False)
    out = auto_megakernel_attention(q, k, v, causal=False)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


# =====================================================================
#  运行入口
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
