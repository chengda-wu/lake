# 10 — 控制面与通信选型(草案,对应 #3 #4)

> **状态:P2 草案 / 讨论中**。本文借 [Dynamo](../research/dynamo/overview.md) 源码对照,给 [#3](https://github.com/chengda-wu/lake/issues/3)(架构图+通信边)与 [#4](https://github.com/chengda-wu/lake/issues/4)(Router 持镜像方式)的待讨论项提供输入。定稿后并入 `overview.md`。已定边界见 #3,本文不重复,只讲 dynamo 对照与 lake 取舍。

## 定位差异:集中权威 vs P2P 涌现

| | lake(已定方向) | Dynamo |
|---|---|---|
| 位置视图权威 | Rust 存储控制面**独立进程** + etcd SSOT,强一致 | **无独立控制面进程**;etcd 只做注册表,KV 事件走 NATS/ZMQ,最终一致 |
| 存储状态归属 | HBM 归池,`BlockManager<L0..L3>` 全四层 | G1(HBM)engine 外部拥有,无 `BlockManager<G1>`;G2/G3 在 `InstanceLeader` |
| 协调模型 | 集中权威 + etcd(Raft 在 etcd 内) | per-instance `InstanceLeader` **P2P**(velo RPC + onboard session),非单 leader、非 Raft 副本 |

**结论**:dynamo 是"多发布者 + 事件总线 + P2P 涌现"的去中心模型;lake 是"单权威 + 强一致 + 集中控制面"的更彻底模型。dynamo 的 P2P leader 满足不了 lake"位置视图强一致"要求——lake 应**明确拒绝** per-instance P2P,选集中控制面 + etcd。这不是缺陷,dynamo 不需要强一致(KV 归 engine,错了回退预测式),lake 需要(HBM 归池,位置错=拉错节点)。

## 进程边界(对应 #3 待讨论 #1 #2)

Dynamo 部署拓扑(`components/src/dynamo/router/CLAUDE.md` "Frontend/Router Boundary"):

1. **集成 Rust frontend**:`frontend → in-process Rust KvRouter → worker`,**frontend+router 同进程,无 router RPC hop**(默认)。
2. **Python processor 内嵌**:仍内嵌 Rust router。
3. **standalone router service**:`frontend → router RPC → worker`,router 独立进程(可选,横向扩展时)。
4. **scheduler**:`LocalScheduler`/`SchedulerQueue`/`DefaultWorkerSelector` 在 kv-router crate 内,**无独立 Scheduler 进程**——调度是 router 内逻辑。

**对 lake 的输入**:
- **Router+Scheduler 不必拆进程**:dynamo 把调度内嵌 router,lake 可同——Scheduler 作 Router 内逻辑,同进程同机直接调用,省 gRPC。拆独立进程只在调度变重/要独立扩缩时才有必要(先不拆)。
- **Gateway+Router 同机可同进程**:dynamo 默认 frontend+router 同进程省一跳。lake Gateway 待定自研/外部(见 #3 待讨论 #5);若自研且与 Router 同机,可同进程。但 lake Router=Go、dynamo router=Rust,"内嵌"机制不同(lake 是 Go 同进程调用,非 PyO3)。
- **存储控制面是独立进程**:dynamo 没有这个(lake 独有),因为 lake 把存储权威从 worker 剥离了。这是 lake 比 dynamo 多出来的一个进程。

## 通信边对照(对应 #3 通信表)

Dynamo 把通信拆**三层正交**(`lib/runtime/src/`),正好印证 lake 通信表的多后端分工:

| 通信需求 | Dynamo | lake 对应边 |
|---|---|---|
| 注册/发现(低频强一致) | etcd `Discovery` trait(`discovery/mod.rs:781`),`v1/<类别>/<ns>/<component>/...` 前缀 | 边4/5/6 控制面元数据;lake 一套 etcd + key space 隔离同此模式 |
| 高频事件流(best-effort) | `EventTransportTx/Rx`(NATS/ZMQ,`transports/event_plane/transport.rs:28`) | 边4/5 位置视图推送(候选);见下 #4 |
| 请求面 RPC | TCP(`transports/tcp.rs`) | 边2/3/6/11 gRPC |
| 大块数据 | NIXL RDMA(`kvbm-physical` TransferManager) | 边8/9 RDMA 旁路 |
| 对象存储 | S3/Azure(`object/mod.rs` `ObjectBlockOps`) | 边10 S3 API |
| 同进程 | in-process trait 调用 | 边7 FFI(PyO3) |

**关键印证**:
- **etcd key space 隔离**:`v1/instances/{ns}/{component}/{endpoint}/{id}`(`discovery/kv_store.rs:21-23,55`)正是 lake `/lake/storage/*` `/lake/scheduling/*` 隔离的范式。bucket = key path prefix,按前缀分层 watch。
- **worker 存活自动收敛**:etcd lease 绑定实例生命周期(`transports/etcd/lease.rs:17`)——lease TTL 到 → etcd 自动删 key → watch 推 Removed。lake worker 崩溃后位置视图权威收敛**无需额外心跳服务**,直接复用此机制。
- **权威 vs 事件流分离**:dynamo 明确 etcd 管低频注册、NATS/ZMQ 管高频事件(`distributed.rs:635` ZMQ 是所有 discovery 后端默认事件面,NATS opt-in;`distributed.rs:483` NATS 不可用即跳过=best-effort)。→ 见下"tension"。

### ⚠️ 待讨论的 tension:位置视图写 etcd 的频率

lake 现定"位置视图进 etcd 强一致"。但位置变更含**满块注册**(每次 prefill 新块都触发),是高频写。dynamo 专门为避开 etcd 高频写,把 KV 事件放 NATS/ZMQ(`distributed.rs:483` 注释 "expected in approximate mode")。

两种取舍(不推翻已定原则,仅列供讨论):
- **(a) 维持 etcd 强一致**:位置视图(含满块注册)全进 etcd,Router/agent watch。简单、强一致,但 etcd 承高频写/watch 放大。lake 规模若扛得住,可接受。
- **(b) 分层**:etcd 存低频权威(节点/模型/配额/revision + 位置视图**低频 checkpoint**),高频位置**变更**走事件流(存储控制面发布 → Router/agent 订阅),控制面内存持权威聚合。强一致权威仍在控制面,etcd 降频。≈ dynamo 的拆法。

→ 留 #3 讨论项。默认走 (a)(守已定原则),(b) 作 P7 校准时的备选。

## Router 持位置视图镜像(对应 #4)

Dynamo router 维护**本地 radix 树副本**(每 router 实例一份),更新路径(`lib/kv-router/src/services/indexer/listener.rs`):
- **事件订阅为主**:ZMQ SUB 收 worker 发的 `KvEventBatch` → `apply_live_batch`(`listener.rs:320`)→ `indexer.apply_event_routed` 更新本地 radix 树。事件源 = worker(engine 持有 KV,worker 自己知道)。
- **路由决策预测回填为辅**:`apply_routing_decision_with_prune_tracking`(`kv_indexer.rs:41`)——选路后乐观预测选中 worker 会缓存这些 block,后续 prune 校正。
- **一致性机制**:`EventEnvelope{publisher_id, sequence, ...}`(`event_plane/traits.rs:13`)+ `DeduplicatingStream` LRU 去重(按 `(publisher_id, sequence)`,`mod.rs:218`)+ **gap 检测 + replay**(`listener.rs:291 handle_gap → replay_gap`,DEALER socket 请求重放缺失序号)。最终一致。

### 对 #4 三方案的输入

| #4 方案 | Dynamo 对应 | 输入 |
|---|---|---|
| 方案1 FFI 嵌 Rust agent | 无(集成拓扑里 router 是 Rust 同进程,但非跨语言 FFI) | lake Router=Go,dynamo 的 Rust 同进程不直接适用 |
| 方案2 直连 etcd watch | `Discovery::list_and_watch`(`kv_store.rs:578`) | dynamo 用它做**注册/发现**(低频),**不**用它跟 KV 位置(高频走事件流)。印证"etcd watch 适合低频,不适合高频 KV 位置" |
| 方案3 gRPC stream 推送 | 无(dynamo 用 pub/sub 而非 gRPC stream 推位置) | 但方向一致:权威源推 → 订阅者本地副本 |

**关键结论**:
1. **传输选择跟发布者拓扑走**。dynamo 用 pub/sub(NATS/ZMQ)是因为**多发布者**(每个 worker 发自己的 KV 事件);lake 是**单权威**(存储控制面),gRPC stream 直推更自然,**不必引入 NATS/ZMQ 总线**。→ dynamo 不构成反对方案3 的证据,反而支持"从权威源解耦推流"方向。
2. **粒度 = 增量事件 + 序号 + gap replay(非快照)**。直接答 #4 待定 #2:增量变更批 + 每发布者序号 + 缺口重放 + LRU 去重。`DeduplicatingStream` 的 `(publisher_id, sequence)` 去重 key 可直接借鉴。
3. **agent 与 Router 同协议**。dynamo router 和 worker 引擎都连同一事件流(不同订阅者)。→ 答 #4 待定 #3:lake 边4(Router)与边5(agent)用**同一套推送协议**,不同订阅者(对应 #3 待讨论 #6 = 是,同协议)。
4. **陈旧只损性能**。dynamo gap/replay 保证最终一致,期间 router 误判 → 回退确认 → 回填,正合 lake `consistency.md` §1"陈旧只损性能不损正确性"。gap/replay 是该容忍的**机制补全**,不是额外复杂度。

## 待讨论项的 dynamo 输入汇总

| # | #3 待讨论项 | dynamo 输入 | lake 倾向 |
|---|---|---|---|
| 1 | Gateway/Router/Scheduler 拆几个进程 | frontend+router 默认同进程;scheduler 内嵌 router | Gateway 待定;Router+Scheduler 同进程(调度作 Router 内逻辑) |
| 2 | Scheduler 独立进程? | `LocalScheduler` 在 kv-router 内,无独立 Scheduler | 不拆,作 Router 内逻辑 |
| 3 | KV Node 有 agent? | dynamo 无 KV Node 概念(远端=对象存储+NIXL);`TransferManager`+`export/import_metadata` 是 RDMA 注册参考 | KV Node 跑 agent(复用计算节点 agent 代码),做 RDMA 注册/读写服务 |
| 4 | 存储控制面单 leader/多副本? | per-instance `InstanceLeader` P2P,非单 leader/非 Raft | lake 拒绝 P2P(不满足强一致);单 leader + etcd(Raft 在 etcd),多副本待 P7 |
| 5 | Gateway 自研/外部? | dynamo frontend 自研(OpenAI HTTP)或 k8s Gateway API+EPP | 待定;过载 shedding 属外部控制面职责(CLAUDE.md 第3条) |
| 6 | 边4=边5? | router 与 worker 同事件流不同订阅者 | 同协议、不同订阅者(是) |

## 参考

- [Dynamo 总览](../research/dynamo/overview.md)
- [`consistency.md`](consistency.md) §1(陈旧只损性能不损正确性)、§3-4(位置视图权威/持久性)
- [`scheduling.md`](scheduling.md) §1(Router 本地命中视图镜像,B3 闭环)
- [`topology.md`](topology.md)(双网络/RDMA 退化)
- [#3](https://github.com/chengda-wu/lake/issues/3) / [#4](https://github.com/chengda-wu/lake/issues/4)
