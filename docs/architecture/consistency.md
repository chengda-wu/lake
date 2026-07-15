# 07 — 一致性与故障模型

本文把散见于 [`storage-layer.md`](storage-layer.md) "一致性模型"、[`kv-cache-pool.md`](kv-cache-pool.md) "引用计数与驱逐"/"写回与生命周期"/"故障恢复"、[`scheduling.md`](scheduling.md) "调度器一致性"、[`data-flow.md`](data-flow.md) §5 的内容，收敛成一篇**自洽的一致性与故障模型**。目标：任一 KV block 的"谁写、谁读、何时一致、崩了丢什么、怎么恢复"在此有唯一答案。

## 1. 一致性分层：控制面强一致 / 数据面最终一致

| 面 | 内容 | 一致性 | 机制 |
|----|------|--------|------|
| 控制面 | KV block 位置元数据、radix 前缀树、节点拓扑、配额、引用汇总 | **强一致** | etcd（SSOT 位置视图） |
| 数据面 | KV 字节、权重字节、跨节点传输、本地命中视图镜像 | **最终一致** | RDMA/TCP 直传 + 推送刷新 |

**为何分两面**：hot step loop 每步都要定位/组装 block table，不可能每步 RPC 强一致查控制面（撑不住 5ms 预算）。故**正确性地基**（block 在哪、是否可驱逐、radix 结构）归控制面强一致；**性能路径**（字节搬运、本地决策）归数据面最终一致，读镜像 + miss 兜底。

- **控制面是权威**：位置视图、radix、引用汇总的唯一真相在 etcd。任何本地副本（agent 视图镜像、Router 命中视图镜像）都是它的缓存。
- **数据面读镜像**：agent / Router 各持本地镜像（零 RPC 决策），由控制面**推送**刷新（etcd watch / gRPC stream），触发 = 位置视图权威变更（放置 / 驱逐覆写 / 迁移 / 满块注册）。详见 [`scheduling.md`](scheduling.md) §1 前缀解析。
- **陈旧只损性能不损正确性**：镜像滞后导致误判本地命中 → agent pull 向控制面确认 → miss 则从池（L1/L2，未命中退 L3）回填（多一跳）。热点前缀变动少、基本不陈旧；陈旧风险集中在冷门 block，miss 代价也小。

> **参考对照**：LMCache 一致性节明示"无全局强一致"，靠 controller ZMQ 消息 + 心跳 + 序列号 + `RWLockWithTimeout`、full-sync best-effort（`sharing-and-backends.md`）；Mooncake 元数据全在 leader 内存、etcd 仅存 OpLog（`mooncake/kv-store.md` MasterService）。lake 比两者都强：位置视图本身进 etcd 强一致，而非靠 best-effort 复制。

## 2. 写一次读多次（KV immutability）

KV cache 是**写一次读多次**的数据：

- **单个 token range 的 KV 不可变**：某 token 位置的 KV 一经 prefill/decode 产出写入 block，永不被原地改写。后续 token 的 KV 是**新 block**，不存在多写者并发改同一字节。
- **写者单一**：满块路（block 填满 → 池算哈希 → 注册 radix → 写回 L2，NVMe F4 恢复点）与尾块路（请求结束写一次、整块覆盖）都由池单写者执行，只需**单写者屏障**，无需多写者并发控制。对齐 vLLM `ExternalBlockHash`（只对完整 block 算哈希）。注册 radix 与 L2 写回非原子——注册后到 L2 durable 之间靠 writeback ref 防 block 被驱逐（见 §3），避免 radix 指向"已驱逐但未落盘"的悬空态。
- **读者多个**：同一 block 可被多节点同时读——本地命中复用、跨节点直传源（D→P / PD 正向）、回填。读不阻塞写回、不阻塞驱逐（驱逐前先确认无在途 ref 且 L2 已 durable，见 §3）。

> **参考对照**：Mooncake `PutStart`→`PutEnd` 两阶段写 + `Get` 只读 `COMPLETE` 状态副本（`master_service.cpp`），对象写后 immutable、`Get` "not necessarily the latest"。lake 的"写一次读多次"是其 immutable 语义的更彻底版：block 一旦注册 radix 即永久可读，无"latest"歧义（radix 命中的是确定的前缀链哈希）。

## 3. 引用计数分两级（B1 闭环）

ref 是"正确性地基"——决定 step 期间冻结、可驱逐性、GC 真删。分两级，**解耦"每 step 高频"与"低频全局"**，避免 per-step 强一致撑不住预算：

### 第一级：本地引用计数（池本地 agent，请求级，高频）

同 vLLM `block_pool.py::free_blocks`/`touch`、sglang `radix_cache.py::inc_lock_ref`/`dec_lock_ref` 的机制——归属从引擎进程改为**池的本地 agent**（存算分离、block 归池权威、多引擎共享，ref 不能放某引擎进程内）。引擎只通过 read set/write set 间接表达引用，不持计数。

- **请求引用**：block 进入某请求 read set 时 +1。减点只在**请求结束且无续推引用**时（attention 每步读全部 KV，前缀 block 全程 in-flight，不能中途早减）。F4 续推时 ref 不归零而是**转移到新请求**，避免被淘汰。
- **在途传输引用**：跨实例传输发起时源 block +1（源端冻结，防 RDMA 半传被覆写致损坏）；完成 -1。D→P 子情况 A 的 L0→L0 直传同样有在途 ref（不因"零存储读取"豁免，见 [`data-flow.md`](data-flow.md) §3.4）。
- **写回引用（writeback ref，防悬空 radix）**：满块注册 radix 后到 L2 写回 durable 之间，block 持 writeback ref，**不可驱逐**（驱逐前必先确认 L2 已 durable）。这保证 radix 已发布的 block 恒有后盾——要么 L0 活着（请求/在途 ref 钉住），要么已落 L2，不存在"radix 有、durable 无"的悬空态。**请求结束是写回屏障**：释放请求引用前，flush+ack 该请求所有已注册 block 的 L2 写回，writeback ref 随 durable ack 归零。
  - **参考对照**：SGLang `write_back` 模式驱逐即回写（`hiradix_cache.py::_evict_write_back`），TreeNode 的 `host_value` 即"evicted 但 backuped"——节点驱逐时先备份到 L2 再摘 L1，恒带 durable 后盾。lake 因 HBM 归池、ref 跨请求/跨节点，驱逐不再是唯一回写点（写回频率 N 可提前批写），改用 **writeback ref + 请求结束屏障**实现等价不变量。
  - **与 F4 风险窗口的关系**：writeback ref 只阻**正常路径驱逐**；F4（worker 崩溃）是节点销毁不是驱逐，L0 随故障丢失，未 durable 的 block 仍按"丢最后一次写回 L2 之后的少量 token"重算（见 §4）。即未 durable 的 block 只能经 F4 路径丢失，正常路径不会让它在 radix 里裸奔。
- **ref 归 0 ≠ 删内存，而是"可驱逐候选"**：对齐 vLLM `free_block_queue` / sglang `evictable_size_`——归 0 后 block 还在 HBM，可被前缀命中复用、可作传输源。真正释放 slot 只在 L0 容量不足**驱逐覆写**时。
- **归 0 不摘位置视图**：未被驱逐覆写则位置视图仍记"X 在该节点 L0"→ 仍可命中、仍可直传（D-direct / D→P 直传的命中来源）。只有驱逐覆写才摘视图。
- **step 期间冻结是引用计数的自然结果**：请求在跑 → ref>0 → 副本不被驱逐。无需额外 fence。

### 第二级：全局引用汇总（控制面，最终一致，低频）

汇总各请求 / 在途引用，供 **tier up/down**（冷热下沉/提升前看是否还有请求在用）与 **GC 真删**（引用归 0 且冷）。**不进 hot step loop**。

### 为什么驱逐不是问题

L0/L1 是**副本**（见 [`storage-layer.md`](storage-layer.md) "层间副本 vs 移动"）：驱逐 L0 只丢自己这份缓存，别的节点读自己的副本不受影响，L2/L3 还在可回填。故不需要"谁还在用"来阻止驱逐——**全局 ref 只用于 tier/GC，不用于阻止 L0 副本驱逐**。这是与"每层强一致计数"的根本差异：副本可随便丢，权威不可丢。唯一例外是上面 §3 的 **writeback ref**：它是本地 ref（非全局 ref），且只在"L2 未 durable"时拦驱逐——拦的正是此处"L2/L3 还在可回填"前提尚未成立的那一刻；L2 一旦 durable，驱逐即恢复自由。故 writeback ref 与"全局 ref 不阻驱逐"不冲突。

> **参考对照**：Mooncake per-object lease（TTL 5s，lease 期内免 Remove/Evict）是其"引用保护"机制，过期即失效（`ObjectMetadata::GrantLease`/`IsLeaseExpired`）。lake 用显式两级 ref 替代 lease：本地 ref 精确到请求级而非定时，无需续约、无过期误删风险；全局 ref 用于 GC 而非阻塞驱逐。

## 4. 持久语义分层与风险窗口

层=介质，持久性分级——不同层抗的故障不同：

| 层 | 介质 | 角色 | 抗的故障 | 丢失条件 |
|----|------|------|----------|----------|
| L0 | HBM | 缓存副本（易失） | 无（丢了回填） | 随节点销毁失效 |
| L1 | DRAM | 缓存副本（易失） | 无（丢了回填） | 断电失效 |
| L2 | NVMe | **F4 恢复点**（持久） | **NPU/进程级故障** | 整机级故障（连本机 NVMe 一起没）且未落 L3 |
| L3 | 对象存储 | **SSOT 永久权威** | **整机级/池级失败** | 几乎不丢（L3 缺失才视为 block 不存在） |

**关键点**：L2 价值 = NVMe 持久（断电不丢）+ NPU 故障不烧 NVMe。NPU/进程级故障只销毁 worker 的 L0/L1（易失副本），不波及 NVMe——无论该 block 被池放在本机还是远端 KV Node 的 NVMe，都存活。F4 续推从 L2 读：池放本机就本地 NVMe 读（零网络），放远端就网络读远端 NVMe。整机级故障（连本机 NVMe 一起没）才退 L3（SSOT）。即 L2 的恢复能力**与位置无关**，只取决于"是否已写回 L2 durable"。

### 风险窗口（分两级）

```
block 在 L0 产出 ── 每 N 步写回 L2(NVMe) ──> 落 L3(冷下沉,非每步)
   NPU 故障丢: 两次写回 L2 之间的增量(尾巴)
   整机故障丢: 自上次冷下沉 L3 之后的增量(大窗口,但罕见)
```

- **NPU/进程级故障（常见）**：worker 崩溃，L0/L1 随销毁失效；NVMe（L2）不波及（block 无论在本机还是远端 NVMe 均存活）→ 从 L2 续推，丢失**最后一次写回 L2 之后的少量 token**（满块路 + 尾块路）。
- **整机级故障（罕见）**：连本机 NVMe 一起没 → 退 L3（SSOT），丢失**自上次冷下沉 L3 之后的增量**。冷下沉 L3 频率由冷热生命周期决定，非每步；热工作集常驻 L1/L2。
- **写回频率 N 的权衡**：N 小（满块即写 L2）→ NPU 故障风险窗口短、丢得少、写放大大；N 大 → 反之。另有前缀生长诉求（多轮 agent 要快）反向压 N 小。N 留 P7 校准。
- **风险窗口的本质**：F4 续推能恢复到"最后一次写回 L2 的点"，之后到崩溃的增量 token 丢失。这是用"NVMe 作恢复点"换弹性的固有代价，非 bug。整机级故障退 SSOT 是兜底，代价是更大的增量窗口、但概率远低于 NPU 级故障。

> **参考对照**：SGLang `--hicache-write-policy` `write_back`/`write_through`（`storage-backends.md`）正是同一权衡——write-back 省带宽但有丢窗口，write-through 无窗口但写放大。lake 选 write-back（L2 NVMe 恢复点）+ 尾块路请求结束兜底，NPU 故障风险窗口收窄到"两次写回之间的增量"。差异：SGLang L1/L2 实例私有、崩溃即丢；我们 L2 是池化的 NVMe 副本，NPU 崩溃只丢该 worker 的 L0/L1，NVMe 副本（无论本机远端）存活。

## 5. 故障恢复（F4）

执行失败（节点故障 / 超时）→ 触发 F4 → Router 依最新集群状态**重跑模式选择**（纯函数，不设降级阶梯）→ 池把该 sequence 已有 KV 放置到新节点 HBM → 续推。

### worker 崩溃续推

1. 存储池检测节点失败（心跳超时）。
2. 把该 sequence 路由到新节点（Router 重跑选路）。
3. 池把已有 KV（**L2 F4 恢复点**，从 L2 读——本机 NVMe 或远端 KV Node NVMe，按池放置位置）放置到新节点 HBM。
4. **ref 从原请求转移到新请求**（避免被冷热淘汰，见 §3）。
5. 原节点 HBM/DRAM 副本随销毁失效（本就是易失副本，非私有状态）；原节点 NVMe 若未被整机级故障波及则仍存活、可作回填源。
6. 续推从"最后一次写回 L2 的点"开始，丢失之后的少量 token（§4 NPU 级风险窗口）。

### 断点 KV 也丢失

若该 sequence 的断点 KV 也未写回 L2（请求刚起、尚未触发满块写回 / 尾块写回）→ 退化为**从 prompt 重算**（[`features/features.md`](../features/features.md) F4 失败语义）。仅损失"最后增量窗口"的 token，不丢请求。

### Drain / 主动下线（非故障）

节点进入 Drain 时，agent 先把"还被远端引用的 block"（在途传输 ref>0 或被其他节点 read set 引用）落一份到 L2（NVMe，F4 恢复点，抗 worker 销毁即可），再下线——避免销毁后远端拉取落空。这是"默认直传 + Drain 推 L2"在弹性侧的落点（[`kv-cache-pool.md`](kv-cache-pool.md) "故障恢复"）。

### 池级失败 / SSOT 恢复

- **L3 = SSOT**：池重启 / 控制面 etcd 重建后，从对象存储恢复持久副本，不丢数据（[`features/features.md`](../features/features.md) F11）。
- **元数据重建**：位置视图在 etcd（强一致），池重启从 etcd 恢复位置视图 + 从 L3 回填字节。
- **L3 缺失才视为 block 不存在**：逐级下查到 L3 仍无 → 该 block 不存在（功能退化，非数据损坏）。

## 6. GC 与孤儿块 reconcile

回收无效 / 不可达 block（[`kv-cache-pool.md`](kv-cache-pool.md) "GC"）：

- **冷块回收**：引用 0 + 冷（LRU 末尾）→ 淘汰。
- **孤儿块**：Prefill 崩溃残留的部分写入 block → 写入屏障标记未完成，TTL 后回收。对齐 Mooncake zombie 清理（`put_start_discard_timeout` 30s 无 `PutEnd` → 抢占释放）。
- **元数据一致性**：以控制面元数据为权威，**block 字节删除前确认元数据已无引用**；崩溃恢复扫描 reconcile 孤儿块。
- **节流**：后台运行，受带宽 / IO 预算限制（<10%），不阻塞数据面。

> **参考对照**：Mooncake `ClearInvalidHandles`（`client_live_ttl_sec` 10s 过期移除死 client 副本）+ zombie 清理是同类机制。lake 加一层"元数据权威先于字节删除"的顺序约束——先摘位置视图、确认全局 ref=0、再删字节，避免"字节先删、引用还在"的悬空访问。

## 7. 一致性速查

| 关注点 | 一致性 | 权威 | 见 |
|--------|--------|------|-----|
| block 位置 / radix / 引用汇总 | 强一致 | etcd | §1 |
| 本地命中视图镜像 / agent 视图 | 最终一致 | etcd 推送 | §1、[`scheduling.md`](scheduling.md) §1 |
| KV 字节（单 token range） | 写一次读多次（immutable） | 产出即定 | §2 |
| ref（本地） | 强一致（agent 内） | 池 agent | §3 |
| ref（全局） | 最终一致 | etcd 汇总 | §3 |
| L0/L1 副本 | 易失，可丢 | L2/L3 回填 | §3、§4 |
| L2 恢复点 | 抗 NPU/进程级故障 | NVMe（持久） | §4、§5 |
| L3 SSOT | 永久权威 | 对象存储 | §4、§5 |
| 字节删除 | 元数据先于字节 | 控制面 | §6 |

## 8. 开放问题

- 写回频率 N（风险窗口 vs 写放大 vs 前缀生长时效）待 P7。
- 满块写回频率（满一个写 vs 攒几个）待 P7。
- 反向回传 radix 生长时效上限（满块注册进 radix 的滞后，影响 D→P 子情况 A 成立，见 [`data-flow.md`](data-flow.md) §3.4）。
- 推送刷新的延迟与带宽（etcd watch vs gRPC stream 取舍）待 P7。
- 池级失败下 L3 回填的冷启动时延（见 [`../features/slo.md`](../features/slo.md) 冷启动）。
