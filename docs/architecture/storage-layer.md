# 03 — 存储层

存储层是"彻底存算分离"的根基。它托管两类数据：**权重**与 **KV cache**，并提供分层缓存。

**存储池是长期存续、模型无关的独立基础设施**：同一池同时承载多个 `(model_id, revision)` 的权重与 KV，可对接任意模型；模型上下线与池生命周期解耦。池提供按模型的空间分配/扩缩容、GC、碎片整理能力（详见 [`kv-cache-pool.md`](kv-cache-pool.md) 的"多模型生命周期 / 空间分配与扩缩容 / GC / 碎片整理"节）。

## 数据模型

### 权重 (Weights)
- 不可变，按 `(model_id, revision)` 寻址。
- 粒度：按 layer / tensor 分片存储，支持并行加载与按需加载（lazy layer load）。
- 格式：原始 fp16/bf16，可选量化副本 (int8/int4/fp8)。量化版本作为独立 artifact，不在线反量化。

### KV cache
- 可变、有生命周期。按 `(model_id, layer, block_index, token_range)` 寻址。
- 粒度策略（待定）：
  - **per-block**（如 vLLM 的 block）：复用友好，元数据多。
  - **per-layer**：传输批量好，复用粒度粗。
  - **per-sequence**：最简单，复用差。
  - 倾向 **per-block + 前缀树（prefix tree）** 索引，支持前缀复用。
- 持久化：热数据在 RAM/NVMe，冷数据可落对象存储（长上下文会话、共享系统提示词的 KV）。

## 分层缓存

| 层级 | 介质 | 容量 | 延迟 | 驱逐策略 | 一致性 |
|------|------|------|------|----------|--------|
| L0 | GPU HBM | 极小 | ~ns | 调度驱逐 | 节点本地，无跨节点一致 |
| L1 | 主机 RAM (per-node) | 小 | ~μs | LRU | 节点本地 |
| L2 | 本地 NVMe (per-node) | 中 | ~10μs | LRU + TTL | 节点本地 |
| L3 | 远端内存池 (RDMA) | 大 | ~10-100μs | 全局 LRU | 强一致元数据 |
| L4 | 对象存储 (SSOT) | 无限 | ~ms | 永久（带版本） | 唯一权威 |

**读取路径**：L0 → L1 → L2 → L3 → L4，逐层回填上层。
**写入路径**：Prefill 产出 → L0 → 异步写 L1/L3 → 按热度决定是否落 L4。

## KV Pool 架构（详见 05）

存储层的 L3（远端内存池）是 KV Pool 的物理载体，由一组 KV Node 组成：
- 每个 KV Node 贡献主机 RAM + NVMe。
- 通过 RDMA 暴露 KV block 的读写。
- 元数据（哪些 block 在哪个 KV Node）由控制面维护。

## 一致性模型

- 权重：不可变，无一致性问题，靠缓存失效（revision 变更）。
- KV cache：**写一次读多次**。Prefill 写入后不可变（针对该 token range）；后续 token 的 KV 是新 block。因此不需要多写者并发控制，只需单写者屏障。
- 故障恢复：KV block 写入 L3/L4 后才认为 Prefill 完成；崩溃可从最近的 KV checkpoint 续推。

## 开放问题

- KV block 的压缩：PagedAttention 之外，是否有跨 block 的低秩 / 量化压缩降低传输量？
- 远端内存池与对象存储之间的分级策略：冷热判定阈值如何动态调整？
- 多租户隔离：不同租户的 KV 如何隔离 + 共享（公共前缀可共享，私有部分隔离）？
