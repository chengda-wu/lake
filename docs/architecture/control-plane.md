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
4. **请求级选路(Dynamo 叫 scheduler)**:`LocalScheduler`/`SchedulerQueue`/`DefaultWorkerSelector` 在 kv-router crate 内,做"请求送去哪个 worker"(overlap 量化 + logit 打分 + 排队),**无独立进程**——是 router 内逻辑。注意:这是**请求级选路**,不是 vLLM 那种 engine 内 batch 调度(见下术语澄清)。

> **术语澄清(lake 约定,与 reviewer 对齐)**:lake 两个"调度"同名不同层,必须分清——
> - **Router**(集群级,大写):决定请求**去哪**(模式 + 节点),`f(请求,集群状态)→(模式,节点)`,集群级少数实例。集群级的调度逻辑(池间/弹性那一档)归 Router,**lake 集群级调度就叫 Router**。
> - **scheduler**(计算节点级,小写):每个计算节点/每 engine 一个,控制计算流程:continuous batching、KV block 分配、抢占、queue 顺序。**lake 计算节点的调度就叫 scheduler**(对应 vLLM `vllm/v1/core/sched/scheduler.py::Scheduler`)。
>
> 两者非二选一,都必然存在。Dynamo 的 `LocalScheduler` 是**请求级**(对应 lake Router),**不**对应 lake 的节点级 scheduler——别被"scheduler"同名误导。vLLM 的 `Scheduler` 是 engine 内(对应 lake 节点级 scheduler)。

**对 lake 的输入**:
- **集群级调度逻辑归 Router,不拆独立进程**:Dynamo 的请求级选路(`LocalScheduler`)内嵌 router 进程,lake 可同——集群级调度(池间/弹性)作 Router 内逻辑,同进程同机直接调用,省 gRPC。拆独立进程只在调度变重/要独立扩缩时才有必要(先不拆)。注:这里"调度"=集群级=Router,**不含**节点级 scheduler(那个在计算节点上,per-engine,跟 Router 无关)。
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

### ⚠️ tension:位置视图写 etcd 的频率(已倾向 b)

lake 原定"位置视图进 etcd 强一致"。但位置变更含**满块注册**(每次 prefill 新块都触发),是高频写。dynamo 专门为避开 etcd 高频写,把 KV 事件放 NATS/ZMQ(`distributed.rs:483` 注释 "expected in approximate mode")。

两种取舍:
- **(a) 维持 etcd 强一致**:位置视图(含满块注册)全进 etcd。简单强一致,但 etcd 承高频写/watch 放大(Raft 复制 + watch fan-out,突发扎堆易拖慢)。**不合理**——etcd 不该扛这写量。
- **(b) 分层(倾向)**:强一致权威在**存储控制面进程内存**(单写者线性一致,不进 etcd);etcd 降频只存低频 checkpoint(节点/模型/配额/revision + 位置快照),供控制面崩溃重建;高频位置变更在控制面内存聚合 → gRPC stream 推 Router/agent。

**结论:走 (b)。** 满块注册写控制面内存 + 上报,**不写 etcd**。dynamo 的实证(KV 事件踢出 etcd)是反证——dynamo 强一致位置在每实例内存(`InstanceLeader` 持 G2/G3 manager),根本不进外部存储。lake 不能完全照搬(KV 归池要全局权威,不能每实例自治),但强一致权威放控制面内存、etcd 只做持久后盾这点一致。详见下"一致性理论对照"节。

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

## 一致性理论对照(为什么 b 合理、a 不合理)

前面"满块注册高频写 etcd 不合理""不做强一致还能全局管理吗"的根,是**一致性强度分级**。lake 要的不是单一强一致,是**组合一致**。下面用多核/DSM/disaggregated memory 的成熟理论对照,把 lake 的位置一致性定位清楚。

### 一致性谱系:lake 要哪档

| 强度 | 术语 | 含义 | lake 对应 |
|---|---|---|---|
| 最强 | 线性一致(linearizability) | 写后立即可见,全局单序 | 搬 KV 那一刻查权威 |
| | 顺序一致(sequential) | 全局同序,不必即时 | — |
| | 因果一致(causal) | 有因果的操作保序 | 满块注册→可被引用 |
| | 释放一致(release) | 同步点刷一致,普通操作松 | **请求结束=写回屏障** |
| | 最终一致(eventual) | 不再写后收敛 | Router 选路镜像 / GC 视图 |

**关键:全局 KV 管理要的是"不丢 + 最终收敛",不是线性一致;自由流动要的是"搬时查准"(同步 RPC),不是全局实时推送。** 强一致只在"搬 KV 时查权威"那一刻需要,而那是查询不是高频写。

### lake = 三范式的组合

lake 的位置一致性不是发明,是三个成熟范式叠加:

```
lake 位置一致性 = 目录一致性(directory coherence)   ← 多核 scale 版 MESI
               + 释放一致(release consistency)       ← DSM 经典
               + flat disaggregated memory           ← 当代内存池化
```

| 范式 | 解决什么 | lake 对应 |
|---|---|---|
| **目录一致性** | 多副本位置管理,scale 到集群(不靠总线广播) | 控制面=目录,持"block 在哪些节点有副本";写时**定向失效**(不广播) |
| **释放一致** | 普通操作松,同步点刷一致 | 满块注册可松,**请求结束 = release 屏障**(flush+ack) |
| **flat memory** | 软件显式管分层,local cache + far authority,miss 拉远端 | L0(local HBM)↔ L1/L2/L3(池),搬 KV = 显式 pull |

### 范式 1:目录一致性(lake 位置管理的骨架)

**MESI 解决多核 cache 一致性**:多 cache 副本 + 主存,写一个要让其他副本失效。但 MESI 靠 **bus snooping(总线广播)**,scale 到几十核就吃力;集群级无总线,改用 **directory(目录)**——目录记录每个 block 在哪些 cache 有副本,写时查目录**定向失效**,不广播。

```
多核 MESI (snoop, 不 scale):           lake (directory, scale):
                                       
  Core0 cache ─┐                         Router 镜像 ─┐
  Core1 cache ─┼─ bus 广播失效            agent 镜像 ──┼─→ 控制面(目录)
  Core2 cache ─┘ (谁改了谁喊)            Router 镜像 ─┘   查目录→定向失效
                                                        (不广播)
```

**lake 的目录 = 存储控制面**(持位置视图 = 哪个 block 在哪些节点的 L0/L1/L2)。block 被驱逐/迁移时,控制面**查目录知道谁持镜像 → 定向发失效**(摘视图),不是全集群广播。这正是 `scheduling.md` §1"驱逐覆写才摘视图并推送"的语义——**lake 已在用 write-invalidate**。

**MESI 状态机 → lake block 位置状态**(借鉴分类,不照搬协议):

| MESI | 含义 | lake block 状态 |
|---|---|---|
| M (Modified) | 独占且已改 | 独占放某节点 L0,刚写未写回 |
| E (Exclusive) | 独占未改 | 独占放某节点 L0(单副本) |
| S (Shared) | 多副本共享 | 多节点 L0/L1 有缓存副本 |
| I (Invalid) | 无效 | 已驱逐,镜像摘除 |

**为什么不照搬 MESI**:MESI 是硬件级纳秒强一致 + snoop 总线;lake 是软件级、集群规模、只要 release 一致——借**状态分类 + write-invalidate 语义**,不借 snoop 总线与硬件强一致。

### 范式 2:释放一致(请求结束 = release 屏障)

**Release consistency(DSM 经典,TreadMarks/Munin)**:普通读写可松,**只在同步点(release/acquire)保证一致**。进入/离开临界区才 flush。

**lake 直接对应**:`consistency.md` §3"请求结束是写回屏障"——满块注册(请求执行中)可松(晚几毫秒被 Router 看见无所谓),**请求结束那刻 = release**:flush 全部满块 + ack,保证此后该请求的 KV 对全局可见。

**例子**(一条请求 R 的生命周期):

```
R 执行中(prefill 产出 block B1..Bn):
  t1  B1 满块 → 注册 → 控制面内存(可松:Router 此刻没看到 B1,无所谓)
  t2  B2 满块 → 注册 → 控制面内存(同上)
  ...
  —— 此时一致性"松":B1..Bn 在控制面,Router 镜像可能还没更新 ——

t_end  R 请求结束 → RELEASE 屏障:
       flush(B1..Bn 全部 durable 写回 L2) + ack
       —— 一致性"刷":此后 B1..Bn 对全局可见、可被引用、GC 可管 ——

t_end+  别的请求 R' 要复用 B1..Bn 前缀:
       Router 镜像已收敛(或 miss 回填) → 命中 → D-direct
```

**为什么这合理**:满块注册高频,但**不必即时全局可见**——同一请求的 block 在请求结束前不会被别人引用(因果上,别的请求复用前缀发生在 R 结束后)。release 屏障卡在"可能被引用"的边界,既不阻塞高频注册,又保证正确性。这是 release consistency 的精确语义。

### 范式 3:flat disaggregated memory(L0↔池 的分层)

**Flat memory(软件管分层)**:local cache + far authority,miss 时显式拉远端。对比 **coherent**(硬件一致,如 CXL.cache)——lake 选 flat(软件显式管,不依赖硬件 coherence)。

```
flat disaggregated memory:           lake 分层:

  local cache ──miss──→ far memory     L0(本机 HBM)──miss──→ L1/L2/L3(池)
     ↑ 拉远端,显式 pull                    ↑ 控制面查位置 → RDMA pull
     ↑ 软件管放置/驱逐                       ↑ 池管放置/驱逐
```

**搬 KV = flat memory 的显式 pull**:agent 要远端 block → 同步查控制面(权威,准)→ 拿到位置 → RDMA 拉。这一查是**线性一致查询**(那一刻要准),但是**查询不是高频写**——无写瓶颈,只有 ms 级查询延迟(搬 KV 本就 ms 级,可接受)。

**例子**(D→P 回传 KV):

```
P 节点要复用 D 节点产出的前缀 block B(已在池):
  1. P 的 agent 同步查控制面:"B 在哪?"        ← 线性一致查询(强)
  2. 控制面查目录:"B 在 D 节点 L0"            ← 权威,准
  3. P 的 agent → RDMA pull B 从 D            ← flat memory 显式 pull
  4. B 入 P 的 L0,控制面更新目录(S 副本)     ← 定向失效给持镜像者
```

第 1 步强一致(查权威),但不写、不广播、不进 etcd 高频路径——**这是"自由流动"不需要全局强一致推送"的落点**。

### 三档一致性汇总(lake 实际方案)

| 用途 | 一致性档 | 机制 | 瓶颈 |
|---|---|---|---|
| 搬 KV(查 block 在哪) | **线性一致查询** | 同步查控制面目录(非推送) | 无写瓶颈,ms 查询延迟 |
| 全局 GC/配额/迁移 | **不丢 + 最终收敛** | agent 上报带 ack+序号,可攒批 | 可控(攒批降频) |
| Router 选路镜像 | **最终一致** | gRPC stream 推送,增量+gap replay | 可松(错了 miss 回填) |
| 满块注册写权威 | **release 一致** | 写控制面内存,请求结束 flush | 不进 etcd,无 Raft 放大 |

**对照之前讨论**:
- "满块注册高频写 etcd 不合理" → 走 b:写控制面内存(release 一致),不写 etcd。✓
- "不做强一致能全局管理吗" → 能:全局管理要"不丢+最终收敛"(不是线性一致),靠 ack+攒批。✓
- "自由流动呢" → 搬时同步查权威(线性一致查询,非推送)。✓
- "瓶颈" → 控制面聚合(fan-out)+ agent 上报(攒批+ack),不卡在 etcd。✓

### dynamo 在这框架里的位置

dynamo 没有上面三档的完整组合,因为它**不需要全局管理**(KV 归 engine,无全局 GC/配额/迁移权威):

| | dynamo | lake |
|---|---|---|
| 目录一致性 | 无全局目录(router radix 副本 + 每实例本地) | **有**(控制面=集群目录) |
| 释放一致 | 无(事件流 best-effort,无 release 屏障) | **有**(请求结束=写回屏障) |
| flat memory | 部分(`find_matches` 搬时查本地权威) | **完整**(L0↔L1↔L2↔L3) |
| 一致性强度 | 最终一致(best-effort,可丢) | release 一致(不丢)+ 线性查询(搬时) |

dynamo 最接近的是 `find_matches`(`lib/kvbm-engine/src/leader/mod.rs:31`)——搬 KV 时查本地强一致视图,即 flat memory 的 pull。但缺目录(无全局副本失效)和 release 屏障(无写回屏障),因为它的"全局管理"由 engine 各自负责、不做集群级权威。

## 待讨论项的 dynamo 输入汇总

| # | #3 待讨论项 | dynamo 输入 | lake 倾向 |
|---|---|---|---|
| 1 | Gateway/Router/Scheduler 拆几个进程 | frontend+router 默认同进程;请求级选路(`LocalScheduler`)内嵌 router | Gateway 待定;**集群级调度归 Router、同进程**(节点级 scheduler 不在此列,每计算节点一个) |
| 2 | Scheduler 独立进程? | `LocalScheduler`(请求级)在 kv-router 内,无独立进程 | 集群级调度不拆(归 Router 内逻辑);**节点级 scheduler(每计算节点一个,vLLM 式)另论,跟 Router 无关** |
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
