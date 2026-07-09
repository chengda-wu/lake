# 05 — KV Cache 池

KV Cache Pool 是本系统区别于传统推理引擎的核心组件。它把 KV cache 从"GPU 的私有副产品"提升为"全局可寻址、可复用、可迁移的分布式资源"。

## 设计目标

1. **前缀复用**：多个请求共享公共前缀（system prompt、few-shot、共享上下文）的 KV，避免重复计算。
2. **跨节点迁移**：Decode 节点缩容/故障时，KV 可迁移到其他节点续推。
3. **解耦计算与存储**：Prefill 节点不关心 Decode 在哪，只把 KV 写入 Pool；Decode 节点从 Pool 拉。
4. **可控成本**：热 KV 在 RAM，冷 KV 落 NVMe / 对象存储。

## 数据结构

### Block 寻址
```
KVBlockID = (model_id, layer_idx, block_hash)
# block_hash 由该 block 内 token ids 的内容哈希得到，保证相同内容 → 相同 KV
```
内容寻址（content-addressed）使前缀复用天然成立：相同前缀 → 相同 block hash → 命中同一 KV。

### 前缀树索引
维护一棵全局 radix tree，节点 = block hash，路径 = token 序列。给定 prompt，沿树匹配最长公共前缀，即可确定可复用的 KV block 范围。

```
root
├─ [sys_prompt_hash] → KV blocks [0..k]
   ├─ [fewshot_hash] → KV blocks [k..m]
   │   └─ [user_query_A_hash] → ...
   └─ [user_query_B_hash] → ...
```

## 物理布局

KV Pool 由 N 个 KV Node 组成，每个 node 贡献 RAM + NVMe。block 按 hash 分片到 node：
```
node_id = hash(KVBlockID) % N
```
- 写：Prefill 节点通过 RDMA write 把 block 推到目标 KV Node。
- 读：Decode 节点通过 RDMA read 拉取，或由 Transfer Bus prefetch。

## 传输协议

- **控制平面**：block 元数据（位置、引用计数、热度）→ etcd，强一致。
- **数据平面**：block 字节 → RDMA，最终一致，best-effort。
- **分块流水线**：大 block 按 sub-block 分块传输，与计算重叠。Prefill 第 i+1 层时，第 i 层 KV 已在传输。

## 引用计数与驱逐

- 每个 block 维护引用计数（被多少在途请求使用）。
- 引用为 0 的 block 进入 LRU 候选，按热度与容量阈值驱逐。
- 被驱逐但仍在对象存储（L4）有副本的 block，可按需回填。
- **公共前缀** block（高复用）给予高权重，不易驱逐。

## 故障恢复

- KV block 写入 Pool（L3）即视为 Prefill 的持久点。
- Decode 节点崩溃：控制面检测 → 把该 sequence 路由到另一 Decode 节点 → 从 Pool 拉取已有 KV → 续推。丢失的仅是最后一次增量写回之后的少量 token。
- 增量写回频率（每 N 步）权衡：N 小 → 恢复快、写放大大；N 大 → 恢复慢、写放大小。

## 开放问题

- 内容寻址的哈希碰撞与安全（是否需要加盐区分租户）。
- RDMA 不可用时退化到 TCP，带宽-延迟模型如何变化。
- KV Pool 自身的弹性：KV Node 扩缩时 block 重分布的成本（一致性哈希缓解）。
- 跨模型版本（revision）的 KV 复用：通常不可复用，如何快速失效旧 block。
