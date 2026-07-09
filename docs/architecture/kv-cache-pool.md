# 05 — KV Cache 池

KV Cache Pool 把 KV cache 从"附属于产生它的 GPU"提升为全局可寻址、可复用、可迁移的分布式资源。在彻底存算分离下,连 HBM/RAM/NVMe 都不归计算节点私有,而是存储池统一管理的物理载体(L0–L2)。所有 KV 位置(含"哪段 KV 在哪个节点 HBM")均为存储池权威元数据。

## 设计目标

1. **前缀复用**:多请求共享公共前缀(system prompt、few-shot、共享上下文)的 KV,避免重复计算。
2. **跨节点迁移**:节点缩容/故障时,KV 迁移到他处续推。
3. **解耦计算与存储**:Prefill 不关心 Decode 在哪,只把 KV 写入 Pool;Decode 从 Pool 拉。HBM 放置亦由存储池管理,本地命中是放置决策的结果(支撑 D-direct)。
4. **可控成本**:热 KV 在 RAM,冷 KV 落 NVMe / 对象存储。
5. **模型无关、长期存续**:池生命周期与模型解耦,同一池同时承载多个 `(model_id, revision)` 的 KV。

## 数据结构

### Block 寻址
```
KVBlockID = (model_id, layer_idx, block_hash)
# block_hash 由该 block 内 token ids 内容哈希得到,相同内容 → 相同 KV
```
内容寻址使前缀复用天然成立:相同前缀 → 相同 block hash → 命中同一 KV。

**模型无关**:`model_id` 是寻址命名空间,Pool 不解释张量布局(层数、头维、dtype),按不透明字节块存取。接入新模型只需注册 `model_id`,无需新建池。

### 前缀树索引

radix tree 归存储池,按 `model_id` 分命名空间。节点 = block hash,路径 = token 序列;给定 prompt 沿树匹配最长公共前缀,确定可复用 KV block 范围。Router 一次查询同时拿到可复用 block 列表与各自位置(前缀复用 + 本地命中判定)。

```
root
├─ [sys_prompt_hash] → KV blocks [0..k]
   ├─ [fewshot_hash] → KV blocks [k..m]
   │   └─ [user_query_A_hash] → ...
   └─ [user_query_B_hash] → ...
```

## 物理布局

L3 由 N 个 KV Node 组成,每个贡献 RAM + NVMe,block 按 hash 分片:
```
node_id = hash(KVBlockID) % N
```
- 写:Prefill 节点通过 RDMA write 推到目标 KV Node。
- 读:Decode 节点通过 RDMA read 拉取,或由 Transfer Bus prefetch。

## 传输协议

- **控制平面**:block 元数据(位置、引用计数、热度)→ etcd,强一致。
- **数据平面**:block 字节 → RDMA,最终一致,best-effort。
- **分块流水线**:大 block 按 sub-block 分块传输,与计算重叠——Prefill 第 i+1 层时,第 i 层 KV 已在传输。

## 引用计数与驱逐

- 每个 block 维护引用计数(在途请求数)。引用 >0 → 冻结迁移/驱逐(正确性约束)。
- 引用为 0 的 block 进冷热排序,按**热度分**(f(频次, recency),LFU-Aging)与容量阈值驱逐/下沉。
- 公共前缀 block 给予前缀亲和加权保护,驱逐时不易被选。
- 层间副本/移动、promotion/demotion、主动迁移见 [`storage-layer.md`](storage-layer.md) "冷热与生命周期管理";本节驱逐是其中一环。
- 被驱逐但 L4 仍有副本的 block,可按需回填。

## 多模型生命周期

池长期存续,模型在其上动态注册/注销,二者解耦:

- **注册**:登记 `model_id`(+ revision、层数、block 规格、配额),分配初始空间。无需新建池或迁移已有数据。
- **下线**:级联删除该 `model_id` 的所有 block(KV + 元数据 + radix 子树),配额归还空闲池。进行中请求由 F4 处理完再清理。
- **revision 更新**:新 revision 视为新命名空间(内容寻址下 KV 通常不可跨 revision 复用);旧 revision 按失效策略(引用归零 + TTL)逐步淘汰。
- **池存续**:模型全下线后池仍运行。池重启不丢 L4 中的持久副本。

## 空间分配与扩缩容

总容量 = 各 KV Node 的 RAM+NVMe 之和,按**配额**在模型间分配:

- 每模型设软配额(常态上限)与硬配额(绝对上限)。
  - 软配额内自由写入;超软配额按该模型 LRU 淘汰冷块腾位。
  - 闲时借用池全局空闲空间(best-effort,遇压力可回收)。
  - 触硬配额 → 返回写入背压信号向上传播(请求级 shedding 仍归 gateway,见 [`../features/slo.md`](../features/slo.md))。
- 配额权重按模型负载/命中率动态调整(调度器决策,控制面下发)。
- **扩容**:加入 KV Node,按一致性哈希重分布,仅迁移落在新节点区间的 block。
- **缩容**:Drain 目标 Node(block 迁出或下沉 L4)再下线。
- 迁移为后台低优先级任务,可暂停让路高峰。

## GC

回收无效/不可达 block:

- **冷块回收**:引用 0 + 冷(LRU 末尾)→ 淘汰。
- **孤儿块**:Prefill 崩溃残留的部分写入 block → 写入屏障标记未完成,TTL 后回收。
- **模型下线/旧 revision**:级联删除。
- **元数据一致性**:以控制面元数据为权威,block 字节删除前确认元数据已无引用;崩溃恢复扫描 reconcile 孤儿块。
- **节流**:后台运行,受带宽/IO 预算限制,不阻塞数据面。

## 碎片整理

长期写入/删除/迁移导致两类碎片:

- **逻辑碎片**:同一序列 block 散落多 KV Node → Decode 读扇出大、传输慢。整理:把热点序列 block 迁到少数节点共置,降读扇出(热度由 radix + 访问频次判定)。
- **物理碎片**:NVMe/RAM 空闲页零散 → 写入放大、分配失败。整理:后台压实合并空闲页。
- 节流:消耗带宽与 CPU,须节流并与低峰重叠,可暂停可恢复;目标开销 < 总带宽 X%(P7 校准)。

## 故障恢复

- KV block 写入 L3 即视为 Prefill 持久点。
- Decode 节点崩溃:存储池检测 → 把该 sequence 路由到新节点 → 由存储池把已有 KV 放置到新节点 HBM → 续推。丢失的仅是最后一次增量写回(落 L3+)之后的少量 token;原节点 HBM 副本随销毁失效(本就是易失副本,非私有状态)。
- 增量写回频率(每 N 步):N 小 → 恢复快、写放大大;N 大 → 恢复慢、写放大小。另有前缀生长诉求,见 [`execution-modes.md`](execution-modes.md)。

## 开放问题

- 内容寻址哈希碰撞与安全(是否加盐区分租户)。
- RDMA 不可用时退化 TCP,带宽-延迟模型如何变化。
- 多模型配额公平性:高负载模型挤占他人时的仲裁与抢占回收代价。
- GC/碎片整理与数据面竞争的隔离(带宽预留 vs 优先级抢占)。
- 碎片整理触发判定:扇出阈值、碎片率,还是周期性?
