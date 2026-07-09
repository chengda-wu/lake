# 05 — KV Cache 池

KV Cache Pool 是本系统区别于传统推理引擎的核心组件。它把 KV cache 从"GPU 的私有副产品"提升为"全局可寻址、可复用、可迁移的分布式资源"。在 lake 的**彻底存算分离**下更进一步：连 GPU HBM / 主机 RAM / 本地 NVMe 都不归计算节点私有，而是存储池统一管理的物理载体（L0–L2）。计算节点不持有任何内存，所有 KV 位置（含"哪段 KV 在哪个节点 HBM"）均为存储池权威元数据。

## 设计目标

1. **前缀复用**：多个请求共享公共前缀（system prompt、few-shot、共享上下文）的 KV，避免重复计算。
2. **跨节点迁移**：Decode 节点缩容/故障时，KV 可迁移到其他节点续推。
3. **解耦计算与存储**：Prefill 节点不关心 Decode 在哪，只把 KV 写入 Pool；Decode 节点从 Pool 拉。计算节点不拥有本地内存——HBM 放置亦由存储池管理，"本地命中"是存储池放置决策的结果（支撑 D-direct）。
4. **可控成本**：热 KV 在 RAM，冷 KV 落 NVMe / 对象存储。
5. **模型无关、长期存续**：Pool 是独立运维的基础设施，生命周期与模型解耦——模型上下线不影响 Pool 存续；同一 Pool 同时承载多个 `(model_id, revision)` 的 KV，可对接任意模型。

## 数据结构

### Block 寻址
```
KVBlockID = (model_id, layer_idx, block_hash)
# block_hash 由该 block 内 token ids 的内容哈希得到，保证相同内容 → 相同 KV
```
内容寻址（content-addressed）使前缀复用天然成立：相同前缀 → 相同 block hash → 命中同一 KV。

**模型无关**：`model_id` 是寻址命名空间的一部分，Pool 不解释张量布局（层数、头维、dtype），按**不透明字节块**存取。不同模型的 block 共存于同一 Pool，互不干扰；接入新模型只需注册 `model_id`，无需新建池。

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

## 多模型生命周期

Pool 长期存续，模型在其上动态注册/注销，二者生命周期解耦：

- **模型注册**：登记 `model_id`（+ revision、层数、block 规格、配额），分配初始空间。无需新建池或迁移已有数据。
- **模型下线**：级联删除该 `model_id` 的所有 block（KV + 元数据 + radix 索引子树），释放配额归还池空闲池。进行中的请求由 F4 处理完再最终清理。
- **revision 更新**：新 revision 视为新 `model_id` 命名空间（内容寻址下 KV 通常不可跨 revision 复用）；旧 revision 按失效策略（引用归零 + TTL）逐步淘汰，与下线同理但可保留过渡期。
- **池存续**：模型全下线后 Pool 仍运行，等待接入新模型。Pool 自身的重启不丢对象存储（L4 SSOT）中的持久副本。

## 空间分配与扩缩容

Pool 总容量 = 各 KV Node 贡献的 RAM+NVMe 之和。空间在模型间按**配额**分配：

- **配额模型**：每模型设软配额（常态上限）与硬配额（绝对上限）。
  - 软配额内：自由写入。
  - 超软配额：按该模型 LRU 淘汰冷块腾位（模型内自管理）。
  - 闲时借用：模型可借用池全局空闲空间（best-effort，遇压力可被回收）。
  - 触硬配额：Pool 返回写入背压信号，向上传播（请求级 shedding 仍归 gateway，见 [`../features/slo.md`](../features/slo.md)）。
- **配额弹性**：按模型负载/命中率动态调整各模型配额权重（调度器决策，控制面下发）。
- **池扩缩容**：
  - 扩容：加入 KV Node，按**一致性哈希**重分布——仅迁移落在新节点区间的 block，最小化迁移量。
  - 缩容：Drain 目标 Node（其 block 迁出至他处或下沉 L4），再下线。
  - 迁移与数据面并发：迁移为后台低优先级任务，可暂停让路高峰。

## GC（垃圾回收）

回收无效/不可达 block，回收空间：

- **冷块回收**：引用0 + 冷（LRU 末尾）→ 淘汰（与"引用计数与驱逐"一致）。
- **孤儿块**：Prefill 崩溃残留的部分写入 block（无完整引用）→ 由写入屏障标记未完成，TTL 后回收。
- **模型下线/旧 revision**：级联删除（见上"多模型生命周期"）。
- **元数据一致性**：GC 以控制面元数据为权威，block 字节删除前确认元数据已无引用；崩溃恢复时扫描 reconcile 孤儿块。
- **节流**：GC 后台运行，受限于带宽/IO 预算，不阻塞数据面读写。

## 碎片整理

长期写入/删除/迁移会导致两类碎片，需后台压实：

- **逻辑碎片**：同一序列的 block 散落多个 KV Node → Decode 读取扇出大、传输慢。
  - 整理：把**热点序列**的 block 迁移到少数节点共置（colocate），降低读扇出。热度由 radix 索引 + 访问频次判定。
- **物理碎片**：NVMe/RAM 空闲空间页级零散 → 写入放大、分配失败。
  - 整理：后台压实（compaction），合并空闲页。
- **权衡与节流**：整理消耗带宽与 CPU，必须节流并与低峰重叠；可暂停、可恢复；目标整理开销 < 总带宽的 X%（P7 校准）。

## 故障恢复

- KV block 写入 Pool（L3）即视为 Prefill 的持久点。
- Decode 节点崩溃：存储池检测 → 把该 sequence 路由到另一 Decode 节点 → 由存储池把已有 KV 放置到新节点 HBM → 续推。丢失的仅是最后一次增量写回（落 L3+）之后的少量 token；原节点 HBM 中的放置副本随节点销毁而失效（本就是存储池的易失副本，非私有状态）。
- 增量写回频率（每 N 步）权衡：N 小 → 恢复快、写放大大；N 大 → 恢复慢、写放大小。

## 开放问题

- 内容寻址的哈希碰撞与安全（是否需要加盐区分租户）。
- RDMA 不可用时退化到 TCP，带宽-延迟模型如何变化。
- 多模型配额的公平性：高负载模型挤占他人时的仲裁策略与抢占回收代价。
- GC/碎片整理与数据面竞争的隔离机制（带宽预留 vs. 优先级抢占）。
- 碎片整理的触发判定：基于扇出阈值、碎片率，还是周期性？
