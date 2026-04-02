# Attention Optimisations for Small K (Head Dimension)

当 attention 的 head dimension **K 非常小**（如 4、8、16、32）时，计算特征从**计算瓶颈**转变为**访存瓶颈**。本项目实现了 6 种针对此场景的优化手段。

## 问题分析

标准 attention: `O = softmax(Q·K^T / √K) · V`

| 指标 | K=128 (常规) | K=16 (小K) |
|------|-------------|-----------|
| QK^T 每元素 FLOPs | 256 | 32 |
| 算术强度 (FLOPs/Byte) | 高 | **极低** |
| 瓶颈 | 计算 | **访存带宽** |
| S_q×S_k矩阵写回开销 | 可容忍 | **主导开销** |

## 6 种优化手段

### 1. Vectorised Small-K (`opt1_vectorized.py`)
**思路**: 当 K ≤ SIMD 宽度时，dot product 可在单条向量指令内完成。
- 使用 `einsum` 显式指定 K 维度收缩，让后端将 K 维保持在寄存器中
- **预缩放 Q**（一次 O(S_q·K) 乘法）代替缩放整个 S_q×S_k 矩阵
- 强制 float32 计算保证数值稳定

### 2. Fused QK·Softmax·V — Online Softmax (`opt2_fused.py`)
**思路**: 小 K 时，S_q×S_k 注意力矩阵的写回/读回开销远大于 QK dot 计算本身。通过 **online softmax** 将三阶段融合为单遍扫描，**不再物化整个注意力矩阵**。
- 内存: O(B·H·S_q·K) → 消除 O(B·H·S_q·S_k) 中间矩阵
- 对长序列效果显著

### 3. Register-Tiled Attention (`opt3_register_tiled.py`)
**思路**: K 小 → 每行 Q/K/V 只占几十字节 → 可同时在 cache 中驻留大量行。
- **超大 Q tile**（如 tile_q=128），因为 128×16=2048 元素仅 8KB
- Q tile 常驻 L1，KV tiles 流式经过 → Q 数据完全复用
- 配合 online softmax 避免中间矩阵

### 4. Loop-Unrolled Dot Product (`opt4_unrolled.py`)  
**思路**: K 已知且很小时，dot product 内层循环只有 K 次迭代，全部展开。
- 按 4 元素一组展开（模拟 SIMD FMA4）
- 消除循环控制开销，启用指令级并行
- 可被 `torch.compile` 进一步融合为单 kernel

### 5. Int8 Quantised QK Attention (`opt5_quantized.py`)
**思路**: 小 K 限制了 dot product 累加范围，量化误差有界。
- Q/K 量化为 int8，QK^T 用 int32 累加
- 4× 数据压缩 → 4× 有效访存带宽
- 在有 int8 tensor core 的硬件上额外获得 2× 算力

### 6. Blocked-Sequence with Transposed K (`opt6_blocked_seq.py`)
**思路**: K 小 → 每行 KV 极小 → 巨量 KV 行可同时驻留 L2 cache。
- **超大 KV tile**（如 tile_k=512），Q tile 适中
- K 预转置为 [B,H,K,S_k] 连续布局，消除 dot product 中的 strided 访问
- 每个 KV 元素仅从 DRAM 加载一次

## 目录结构

```
larger_kernel/
├── attention_small_k/
│   ├── __init__.py
│   ├── baseline.py              # 标准 attention (参考实现)
│   ├── opt1_vectorized.py       # 优化1: 向量化小K
│   ├── opt2_fused.py            # 优化2: 融合QK+softmax+V
│   ├── opt3_register_tiled.py   # 优化3: 寄存器分块
│   ├── opt4_unrolled.py         # 优化4: 循环展开
│   ├── opt5_quantized.py        # 优化5: Int8量化
│   └── opt6_blocked_seq.py      # 优化6: 序列分块+K转置
├── megakernel/
│   ├── __init__.py              # 统一导出所有实现
│   ├── multi_kernel.py          # 基线: 每步独立 kernel launch
│   ├── megakernel_fused.py      # 融合 megakernel (online softmax + tiling)
│   ├── persistent_megakernel.py # Persistent thread 模型
│   └── triton_megakernel.py     # Triton 真实 GPU 单 kernel 融合
├── test_attention_small_k.py    # Small-K attention 测试套件
├── test_megakernel.py           # Megakernel 测试套件
└── README.md
```

## 运行测试

```bash
cd larger_kernel

# Small-K attention 测试
python -m pytest test_attention_small_k.py -v --tb=short

# Megakernel 测试
python -m pytest test_megakernel.py -v --tb=short

# Benchmark (带输出)
python -m pytest test_attention_small_k.py -v -k "benchmark" -s
python -m pytest test_megakernel.py -v -k "benchmark" -s
```

---

# Megakernel: 巨型核函数融合

## 1. 核心问题: 为什么需要 Megakernel?

传统 GPU 编程模型中, 一个 op 对应一个 kernel launch. 一个 Transformer block 需要 **15+ 次 kernel launch**:

```
LayerNorm → Q_proj → K_proj → V_proj → scale → QK^T → mask → softmax → attn@V → O_proj → add → LayerNorm → FFN_up → GeLU → FFN_down → add
    K1        K2       K3       K4      K5      K6     K7      K8        K9       K10     K11     K12         K13     K14      K15      K16
```

**每次 kernel launch 的代价:**

| 开销来源 | 量级 | 说明 |
|---------|------|------|
| Launch overhead | 3-10 μs/次 | CPU→GPU 调度, 小 tensor 时占比巨大 |
| Global memory 往返 | ~1 TB/s HBM | 中间结果写回 DRAM → 下一个 kernel 再读回 |
| SM idle gap | 数 μs | 两个 kernel 之间 SM 有等待期 (pipeline bubble) |
| 寄存器/共享内存失效 | — | 每个 kernel 结束时局部存储全部丢失 |

**对 memory-bound 操作 (如 Attention), 反复读写中间矩阵是致命瓶颈.**

## 2. Megakernel 解决方案

将所有阶段融合为一个 kernel, 中间结果驻留 registers/shared memory:

```
┌──────────────────────────────────────────────────────────────┐
│  Single Megakernel Launch                                     │
│                                                               │
│  for each tile / work unit:                                   │
│    ① LayerNorm               (registers)                      │
│    ② QKV projection          (shared memory)                  │
│    ③ QK^T + scale            (registers)                      │
│    ④ Causal mask             (registers, branch)              │
│    ⑤ Online softmax          (registers, streaming)           │
│    ⑥ Attn @ V                (registers → shared mem)         │
│    ⑦ O projection            (shared mem → registers)         │
│    ⑧ Residual add                                             │
│    ⑨ LayerNorm 2             (registers)                      │
│    ⑩ FFN up + GeLU + down    (shared mem)                     │
│    ⑪ Residual add → 写出最终结果                                │
│  end for                                                      │
│                                                               │
│  只有一次全局内存写入!                                           │
└──────────────────────────────────────────────────────────────┘
```

**关键优势:**

| 优势 | Multi-Kernel | Megakernel |
|------|-------------|------------|
| Kernel launch 次数 | 15+ | **1** |
| 中间结果存储 | DRAM (HBM) | **Registers / SRAM** |
| DRAM 访问 | 读输入 + 写/读中间 × N + 写输出 | **只读输入 + 只写输出** |
| SM 利用率 | 有 bubble | **持续满载** |
| 跨阶段编译器优化 | 不可能 | **可以** (寄存器分配、指令调度) |

## 3. 四种实现对比

### 3.1 Multi-Kernel Baseline (`multi_kernel.py`)

**原理**: 标准分步执行, 每步一个独立 kernel, 中间结果写回全局内存.

```python
# 每步生成一个中间 tensor (写回 DRAM)
q_scaled = q * scale          # Kernel 1: scale → 写回
scores = q_scaled @ k.T       # Kernel 2: matmul → 写回 S_q×S_k
scores = mask(scores)          # Kernel 3: mask → 读写
attn = softmax(scores)         # Kernel 4: softmax → 读写
output = attn @ v              # Kernel 5: matmul → 写回
```

**问题**: 5 个 kernel = 5 次 launch + 完整 S_q×S_k 矩阵反复读写.
对于 S=2048, H=32: score 矩阵 = 2048×2048×32 = 512MB, 每步都要读写!

---

### 3.2 Megakernel Fused (`megakernel_fused.py`)

**原理**: Online Softmax + Tiling, 所有阶段在一个 "kernel" 内完成. **S_q×S_k 矩阵从不完整物化.**

```
对每个 Q-tile (block_size 行):
  ┌─── Q tile 加载到 "寄存器文件", 整个内循环中常驻 ───┐
  │                                                    │
  │  对每个 KV-tile:                                    │
  │    1. KV tile 流入 "共享内存"                        │
  │    2. s_ij = Q_tile @ K_tile^T  (tile 大小, 在寄存器中) │
  │    3. Causal mask (原地, 在寄存器中)                  │
  │    4. Online softmax 更新:                          │
  │         m_new = max(m_old, rowmax(s_ij))           │
  │         rescale = exp(m_old - m_new)               │
  │         l = l * rescale + sum(exp(s_ij - m_new))   │
  │         O = O * rescale + exp(s_ij - m_new) @ V    │
  │    5. 丢弃 s_ij (不写回 DRAM!)                      │
  │                                                    │
  └────────────────────────────────────────────────────┘
  最终: O = O / l → 写回全局内存 (唯一一次输出写)
```

**Online Softmax 详解**: 无需预先知道全局 max, 边扫描边更新:

$$m^{(j+1)} = \max(m^{(j)},\ \max_{c}(s_{ij}))$$

$$l^{(j+1)} = l^{(j)} \cdot e^{m^{(j)} - m^{(j+1)}} + \sum_c e^{s_{ij,c} - m^{(j+1)}}$$

$$O^{(j+1)} = O^{(j)} \cdot e^{m^{(j)} - m^{(j+1)}} + e^{(s_{ij} - m^{(j+1)})} \cdot V_j$$

最终: $O = O^{(\text{last})} / l^{(\text{last})}$

---

### 3.3 Persistent Megakernel (`persistent_megakernel.py`)

**原理**: 模拟 **Persistent Thread** 执行模型 — thread block 不退出, 在循环中持续领取新 tile.

```
传统 kernel:
  Block 0 → tile 0 → 退出
  Block 1 → tile 1 → 退出     → 新一轮 launch
  ...                           → launch overhead × N

Persistent kernel:
  SM 0: while (tile = atomic_next()) < total:
            process(tile)        → 一次 launch, 持续执行
  SM 1: while (tile = atomic_next()) < total:
            process(tile)        → 跨 tile 复用 shared memory / registers
  ...
```

**优势:**
1. 只有 1 次 kernel launch (消除全部 launch overhead)
2. Shared memory / 寄存器在整个生命周期内有效 (跨 tile 复用 KV cache)
3. 对硬件 SM 数量自适应: N_sm 个 block 持续满载
4. **动态负载均衡**: 快的 block 自动领取更多 tile (atomic counter)
5. KV cache 可以 "钉在 L2" 供所有 Q tile 复用

```python
# Persistent kernel 伪代码
tile_counter = 0  # global atomic counter

def persistent_kernel():
    while True:
        tile_id = atomicAdd(tile_counter, 1)
        if tile_id >= total_tiles:
            break
        b, h, q_tile = decode(tile_id)  # 解码 tile 坐标
        # 处理这个 tile (online softmax attention)
        for kv_tile in all_kv_tiles:
            # KV 在 shared memory 中, 跨 Q tile 复用
            compute_attention_tile(q_tile, kv_tile)
        write_output(q_tile)
```

---

### 3.4 Triton Megakernel (`triton_megakernel.py`)

**原理**: **真正的 GPU 单 kernel** — 整个 attention 编译为一个 GPU kernel.

| 对比项 | Python 模拟版 | Triton 版 |
|-------|--------------|-----------|
| Kernel launch | 每个 torch op 仍是独立 launch | **真正 1 次 launch** |
| 中间数据 | GPU global memory | **SRAM (shared mem + registers)** |
| 循环 | CPU Python for-loop | **GPU 上的 tile loop** |
| 性能 | 仅验证算法正确性 | **真实 2-5x 加速** |

```
Triton kernel 编译结果:
  Grid: (num_q_tiles, B × H)
  每个 program instance 处理:  一个 Q-tile × 所有 KV-tiles

  ┌────────────────────────────────────────────┐
  │  q_tile = load(Q[q_start:q_end])  → SRAM   │
  │  m = -inf, l = 0, O = 0                    │
  │                                             │
  │  for kv_tile in range(0, S_k, BLOCK_K):     │
  │    k_tile = load(K[kv_start:kv_end])        │
  │    v_tile = load(V[kv_start:kv_end])        │
  │    scores = q_tile @ k_tile^T  (全在 SRAM)  │
  │    [causal mask, if needed]                 │
  │    online softmax update (m, l, O)          │
  │  end for                                    │
  │                                             │
  │  O = O / l                                  │
  │  store(Out[q_start:q_end], O)  → 唯一写出   │
  └────────────────────────────────────────────┘
```

需要 GPU + Triton: `pip install triton`

## 4. 内存访问量对比

以 B=1, H=1, S=2048, K=64 为例:

| 实现 | 中间矩阵 (S×S) | DRAM 读取 | DRAM 写入 | 总访存 |
|------|----------------|----------|----------|--------|
| Multi-Kernel | 完整 2048×2048 = **16MB** | 16MB×3(读score) + QKV | 16MB×3(写score) + output | **~112 MB** |
| Megakernel | 仅 tile 大小 64×64 = **16KB** (SRAM) | QKV ≈ 1.5MB | output ≈ 0.5MB | **~2 MB** |
| **减少** | — | — | — | **~56×** |

## 5. test_megakernel.py 测试内容

### 测试策略

| 验证目标 | 方法 |
|---------|------|
| 数值正确性 | megakernel vs multi_kernel 结果一致 (atol=1e-5) |
| Persistent 正确性 | persistent_megakernel vs multi_kernel 一致 |
| LayerNorm+FFN | megakernel_layernorm_ffn vs multi_kernel 一致 |
| Transformer Block | 完整 block 融合 vs 分步执行一致 |
| 参数鲁棒性 | 多种 block_size (16/32/64/128), 多种 num_sms (1/2/4/8) |
| 边界条件 | S=1, K=1, 非对齐序列长度, 全零输入 |
| Causal mask | 有/无 causal mask 对比 |
| Triton GPU | 真实 GPU 上 Triton kernel 正确性 + FP16 |
| Benchmark | 多种实现的耗时对比 |

### 测试配置矩阵

```python
ATTN_CONFIGS = [
    (B=1, H=1, Sq=16,  Sk=16,  K=8),    # 最小 case
    (B=2, H=4, Sq=64,  Sk=64,  K=16),   # 典型 small-K
    (B=1, H=2, Sq=128, Sk=128, K=32),   # 中等规模
    (B=2, H=2, Sq=63,  Sk=63,  K=16),   # 非 2 幂序列长度
    (B=1, H=1, Sq=1,   Sk=32,  K=8),    # 单 query (解码场景)
    (B=1, H=1, Sq=32,  Sk=1,   K=8),    # 单 key
    (B=2, H=4, Sq=37,  Sk=53,  K=16),   # Sq ≠ Sk
    (B=1, H=8, Sq=256, Sk=256, K=8),    # 更大规模
]
```

### 测试分类

**Attention 正确性** (CPU):
- `test_megakernel_attention_noncausal` — 8 种参数 × non-causal
- `test_megakernel_attention_causal` — causal mask 正确性
- `test_megakernel_attention_block_sizes` — block_size = 16/32/64/128

**Persistent Megakernel** (CPU):
- `test_persistent_megakernel_noncausal` — 8 种参数
- `test_persistent_megakernel_causal` — causal mask
- `test_persistent_different_sms` — num_sms = 1/2/4/8

**LayerNorm + FFN 融合** (CPU):
- `test_megakernel_ln_ffn` — 4 种 (B,S,D,D_ff) 配置
- `test_megakernel_ln_ffn_chunk_sizes` — chunk_size = 16/32/64/128

**完整 Transformer Block** (CPU):
- `test_megakernel_transformer_block` — 3 种配置 × causal/non-causal

**边界条件**:
- `test_zero_input_attention` — 全零输入不产生 NaN/Inf
- `test_single_element` — S=1, K=1 极端情况

**Triton GPU 测试** (需要 CUDA + Triton):
- `test_triton_megakernel_noncausal` — 5 种 GPU 配置
- `test_triton_megakernel_causal` — causal mask
- `test_triton_megakernel_fp16` — FP16 精度
- `test_auto_megakernel_cpu/gpu` — 自动 dispatch

**Benchmark**:
- `test_benchmark_attention` — multi vs mega vs persistent 耗时
- `test_benchmark_transformer` — 完整 block 耗时
- `test_triton_benchmark` — GPU 上真实耗时
