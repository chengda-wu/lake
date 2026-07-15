# 00 — 路线图与执行计划

本文件是 lake 仓库的**主线计划**，列出依次要做的事情。每完成一个里程碑更新本文档状态。

> 总立地：探索并验证一套**彻底的存算分离推理系统**。所有有状态物（权重、KV cache、调度队列）从算力路径剥离，算力节点可随时销毁/拉起。

## 阶段总览

```
P0  特性设计         → docs/ 里把"要做什么"定义清楚
P1  架构设计         → 围绕特性设计把"怎么搭"定下来
P2  模块划分 + 技术选型 → 各模块语言/框架/接口边界
P3  最小可运行骨架   → 跨语言跑通一条请求（mock 模型）
P4  KV Pool 原型     → 内容寻址 + 前缀复用 + 分层缓存
P5  存算分离验证     → Prefill/Decode 物理隔离 + KV 迁移
P6  弹性与调度       → 无状态路由器 + 秒级扩缩容
P7  性能建模与验证   → 量化各假设，回填设计
```

---

## P0 — 特性设计（当前首要）

**目标**：把系统"要交付哪些能力"讲清楚，作为架构设计的输入。不谈实现。

产出文档（`docs/features/`）：
- [x] [`features.md`](features/features.md) 特性清单：按 Must / Should / Could 分级（每条含输入/输出/失败语义）
  - 设计前提：**三种执行模式**（PD 分离 / 混部 / D-direct），Router 按存储池本地命中、prompt 规模、传输成本逐请求选路；详见 features.md "执行模式"节
  - [x] F1 KV cache 池化与前缀复用（内容寻址、radix tree）
  - [x] F2 混合执行模式（PD 分离 / 混部 / D-direct，含模式选择）
  - [x] F3 分层缓存（HBM→RAM→NVMe→远端内存池→对象存储，**全部由存储池统一管理**，计算节点不拥有本地内存）
  - [x] F4 故障恢复（基于 KV Pool 续推）
  - [x] F5 无状态路由
  - [x] F11 多模型存储池与生命周期管理（长期存续/模型无关/配额扩缩/GC/碎片整理）
  - [x] F6 投机解码（draft / target 分离）
  - [x] F7 秒级弹性扩缩容
  - [x] F8 多租户隔离与共享前缀
  - [x] F9 模型版本/热更新（Could）
  - [x] F10 跨机房（Could）
- [x] [`slo.md`](features/slo.md) SLO 与衡量指标（TTFT / ITL P50/P99 / 吞吐 / 命中率 / 冷启动时延，初版 draft）
- [x] [`nonfunctional.md`](features/nonfunctional.md) 非功能需求（可观测性、安全、成本、部署、可维护性、可测试性）

**完成判据**：每条特性有明确的输入/输出/失败语义 ✅；SLO 数值化 ✅；与 [`goals.md`](features/goals.md) 对齐且无矛盾 ✅。
**P0 状态：done 2026-07-09**

---

## P1 — 架构设计（围绕特性设计）

**目标**：基于 P0 特性，定下数据流、组件边界、一致性模型、故障域。

产出文档（`docs/architecture/`）：
- [x] 更新 [`overview.md`](architecture/overview.md)：纳入混合执行模式与 KV 流转视角，替代刚性 P→D；去 ⚠️
- [x] [`architecture/execution-modes.md`](architecture/execution-modes.md) 以 KV 为中心的执行模式与 KV 流转时序（本地完成 / 跨节点传输含正向产出与反向回传）；失败处理统一归 F4 重路由，不设独立降级阶梯
- [x] [`architecture/data-flow.md`](architecture/data-flow.md) 请求生命周期详图（含 F4 故障分支、模式选择决策树 mermaid、三模式执行段、ready/done 双 fence 一步契约）
- [ ] `architecture/consistency.md` 一致性与故障模型（KV 写一次读多次、控制面强一致/数据面最终一致、崩溃恢复点）
- [ ] `architecture/topology.md` 部署拓扑（单机房/跨机房、网络 fabric 假设、RDMA 可用性退化）

**完成判据**：任一特性的"数据从哪来、写到哪、谁来调度、失败怎么办"都可在此找到答案。

### P1 已定决策摘要（跨轮固化）

- **彻底存算分离**：L0–L4 全归存储池统一管理，计算节点不拥有任何内存；APC 概念删除，"本地命中"= 存储池放置决策的结果。
- **radix tree 归存储池**，按 `model_id` 分命名空间；Router 一次查询拿前缀复用 + 本地命中，守 5ms 模式选择预算。
- **放置与 batch 职责边界（方案 Z）**：存储池按热度主动预放置 KV 到 HBM + 发布位置视图；调度器读视图组 batch（本地命中优先→D-direct，缺失补拉），不反向指挥放置。单向耦合。
- **冷热与生命周期**：L0/L1 做副本、L2/L3/L4 间按移动、L4 永久权威；冷热按"引用数>0 冻结 + 热度分(LFU-Aging) + 前缀亲和"；迁移主动为主 + 被动兜底；迁移/GC/碎片整理共享后台带宽池（<10%）。
- **执行模式时序**（存储池视角不区分 P/D）：时序一本地完成（D-direct/混部共用，入口由本地命中定 prefill 工作量）；时序二跨节点传输——正向（产出→消费，服务本次）+ 反向（消费→池，D 延伸 KV 回传增强未来前缀，agent 多轮核心）。
- **decode 增量写回双重目的**：容错 + 前缀生长。频率 N 策略留开放。
- **HBM 池化下的入图与 KV 管理（Q1/Q2，本轮定）**：
  - **Q1 入图**：固定基址 KV arena（不上 VA，分配给模型后不扩缩容/不跨模型回收物理页）；入图三约束（静态输入 buffer / 固定 KV 基址 / 固定地址 block table）；decode 走 graph、prefill 走 eager；block table 由**本地 agent（in-process，持本地视图镜像）组装**，非全局池每步 RPC 推表（守 5ms）；**ready/done 双 fence 一步契约**（池发 ready→引擎 replay→引擎发 done→池解冻/写回/注册 radix/驱逐），引擎零分层逻辑；正确性地基 = in-flight 跨层冻结（ref>0 的 block step 期间物理映射冻结）。详见 [`architecture/compute-layer.md`](architecture/compute-layer.md) "HBM 池化下的入图与 KV 管理"。
  - **Q2 KV 管理**：block 对引擎**纯寻址单位**（连 block table 索引填充都归池，引擎只 replay 读，不感知满块）；写回两路——**满块路**（填满→池算哈希→注册 radix→写回 L3）+ **尾块路**（请求结束时未满尾块写一次，纯容错不进 radix）；**ref 池权威维护**（多引擎共享前缀 block 的分布式一致性，引擎不持计数），请求结束且无续推引用才减（F4 续推 ref 转移），含在途传输引用（源端冻结）。
  - **跨实例/PD 传输**：engine-to-engine 控制链**切断**，池的本地 agent 发起传输，引擎降到 `publish`/`pull`+fence、不知地址、不组装 block table；数据线仍直连 RDMA（wire 效率不变）。默认**直传**（A→B L0，PD 时序重叠主场景）+ **Drain 推 L3**（节点下线前把还被远端引用的 block 落 L3）。详见 [`architecture/kv-cache-pool.md`](architecture/kv-cache-pool.md) "跨实例 KV 传输"。
  - **重叠语义**：拒绝**引擎驱动** intra-step 重叠（SGLang `get_key_buffer` 每层 `wait_event`，绑死引擎、破坏 graph）；保留**池驱动**异步重叠——消费侧 step 间重叠 + 生产侧 prefill 层级重叠（`page_first_direct` 子块传输/"分块流水线"，支撑 PD 分离 TTFT）。引擎无感、graph 安全。
  - **持久语义**：L3 = F4 恢复点（抗 worker 失败，副本 RAM）；L4 = SSOT 永久权威（抗池级失败，L4 缺失才视为 block 不存在）。风险窗口：worker 与其 L3 副本同时失败且未落 L4 则丢尾巴（= 丢失最后一次写回 L3 之后的少量 token）。
- **技术选型已定**（P2 落地）：存储 Rust / 控制 Go / 计算 Python+Triton；元数据 etcd；SSOT 用 S3/MinIO；跨语言 gRPC+Protobuf（大块 KV 走 RDMA 旁路）。3rdparty 四个 submodule（sglang/lmcache/mooncake/vllm）作实现参考。
- **KV 类型 t-type / r-type + 投机解码机制（本轮新增）**：
  - **KV 类型**：按 HBM 存储形态分 t-type(逐 token 完整 KV,paged block,full attention/MLA)与 r-type(紧凑表示——窗口最近 W token / Mamba 定长 state,sliding window/Mamba/卷积)。**两类复用条件一致**:都需命中全部前缀才能复用;区别**仅在 HBM(L0)存储形态**,目的是降低 r-type 的 HBM 占用。HBM 两类并存、分 arena 管理(r-type 另设固定状态 arena 入图);L1–L4 统一按 block(128 token)组织(两类复用条件一致、不区分类型),r-type 落下层在 block 边界 checkpoint 紧凑状态(trailing pages / state 快照)——相对 SGLang multi-pool 物理分池,我们把类型差异收敛到 L0 存储形态 + block 内布局,而非物理分池。详见 [`architecture/storage-layer.md`](architecture/storage-layer.md) "KV 类型"节、[`architecture/kv-cache-pool.md`](architecture/kv-cache-pool.md) "t-type / r-type"。
  - **block 粒度 128 token**：缓存命中/复用/传输/写回最小单位(初版默认,待 P7 校准)。
  - **投机解码(仿 SGLang)**：drafter 与 decode(target)默认共置、同 step 串行。**pre/post 共用同一 drafter 模型,拆 `post_forward` / `pre_forward` 两阶段(同类的两个方法,非独立组件)**统一自回归类与 diffusion 类编排:**`post_forward`**(target 之后,吃 target 输出做强耦合部分)承载 MTP/EAGLE/EAGLE3 的 draft head 前向(参数与主模型一致)+ DFLASH/DSPARK 的 draft cache 准备;**`pre_forward`**(下轮 target 之前)承载 MTP/EAGLE/EAGLE3 的自回归多 token 生成 + DFLASH/DSPARK 的 diffusion 并行产 block。**prefill 阶段仍产 draft**(drafter forward 照跑),差异在产出是否使用:vLLM PD 分离下 P 侧 draft 弃用、forward 仅为保 drafter KV 同步(`llm_base_proposer.py:567`),SGLang 暂未细究——**记为遗留问题,初步判断不影响整体设计**(prefill 产出是否用属节点侧策略)。**decode 多层 MTP** 分 chain-style(每层用自己上步输出 hidden,需 FULL 暂存)/ non-chain(每层用 target hidden,只需 LAST)两范式,参考 SGLang `multi_layer_eagle_worker_v2.py::chain_mtp_hidden_states`;单层 MTP 是 non-chain 退化。draft 中间态一律 L0 r-type 暂存、不进池(自回归=hidden states,diffusion=窗口/block 状态)。MTP 重算产出 `1+num_mtp_layers` token,残差 prefill 短时左 pad。**主攻方案**:MTP/EAGLE/EAGLE3/DFLASH/DSPARK(后两者 diffusion 类,半年内进生产),不主攻 medusa/mlp_speculator/ngram/独立 draft 模型。**支持梳理**:MTP/EAGLE/EAGLE3/DFLASH 两边都有;**DSPARK 仅 SGLang**;不参考 vLLM `spec_target_max_model_len`(独立 draft 模型时代遗留)。详见 [`architecture/compute-layer.md`](architecture/compute-layer.md) "投机解码"节。
  - **缓存命中感知调度**：命中(Pool/本地)是模式选择与 batch 组成的一等输入;新增**跨请求前缀共调度**(同前缀请求组同 batch/节点,前缀 block 复用+本地命中叠加,参考 SGLang `match_prefix` cache-aware scheduling / vLLM `get_computed_blocks`);守方案 Z 单向耦合(只读视图、不指挥放置),draft 候选不进 radix。详见 [`architecture/scheduling.md`](architecture/scheduling.md) "缓存命中感知调度"节。
  - **长度边界规避(max_model_length vs runner_max_model_length)**：推理临近最大长度时的边界 bug(drafter 跳过致 EP/DP 集合通信盲等、block 申请不够致请求永不可调度)在 vLLM/sglang 均靠累积特殊逻辑兜(vLLM `reserve_full_isl`+spec pad break+PD+spec lookahead=0;sglang `init_req_max_new_tokens` admission clamp+`speculative_skip_dp_mlp_sync`+`_build_trivial_verify_input`)。lake 用双长度变量规避:对外 `max_model_length`(gateway/scheduler 守,length cap+SLO 契约)+ 计算层内部 `runner_max_model_length = max_model_length + headroom`(arena/block table/graph 预分配);headroom 吸收 draft transient,runner 不写近 max debug 逻辑。代价:额外 HBM arena。详见 [`architecture/compute-layer.md`](architecture/compute-layer.md) "长度边界规避"节。

### P1 待讨论 / 开放点

- decode 写回频率 N：多轮 agent（重前缀增强时效，N 小）vs 单轮（重带宽/容错，N 大），待 P7。
- 满块写回频率（满一个就写 vs 攒几个满块一起写），待 P7。
- 反向回传的 radix 增长时效：写回到 radix 可见的滞后上限。
- 分块流水线深度（`page_first_direct` 子块传输 k 与 prefill 层数对齐），待 P7。
- 模式选择决策树的具体阈值（本地命中判定、传输成本 vs 分离收益）待 P7；**决策树结构本身待 `data-flow.md` 落定**（结构定型、阈值留空标 P7）。
- **block 粒度 128 token** 与传输/写放大/碎片率的权衡,待 P7 校准。
- **r-type 状态 checkpoint**:Mamba/卷积 recurrent state 落 L1+ 的 checkpoint 间距/形式、sliding window trailing pages 阈值,待实现/P7 校准。
- **r-type SWA 尾段重算优化(idea,暂不实现,已记预留)**:SWA KV 不落 L1+、prefix 命中时重算匹配序列最后 `n*(w-1)+1` 个 token(position `[L+n-n*w-1, L)`)仅刷 SWA 窗口、不写非 SWA 模块(slot_mapping=-1),省存储换重算。暂不实现,但已留两处接口预留:① agent 的 slot 分配按模块差异化(只给 SWA 分 write slot,模块意识留池侧、引擎契约不破,经 Q2 张力权衡选此);② 残差路径区分"增量 prefill(未匹配尾)"与"刷新重算(已匹配尾,仅 SWA 写)"。r-type SWA 是否落 L1+(持久 vs 重算)二选一,待 P7。详见 [`architecture/compute-layer.md`](architecture/compute-layer.md) "r-type SWA 前缀复用的尾段重算优化"。
- **MTP 左 pad 策略**:是否总 pad 到固定宽度、pad token 是否复用命中 KV、pad 窗口上限,待实现/P7 校准。
- **drafter 共置 vs 独立 Draft 池**:默认共置(sglang 式);独立池的收益阈值(投机命中率 vs draft 候选传输延迟)待 P7。
- **r-type 入图**:sliding window / Mamba 固定状态 arena 与 t-type block arena 的 capture/replay 协同,待 P2/P3。
- **headroom 大小**:`runner_max_model_length − max_model_length`(覆盖 draft 深度+lookahead+block 对齐 margin+安全余量),待 P7 校准。

### P1 下一步（收尾，按此顺序）

1. ~~`architecture/data-flow.md`~~ ✅（done 2026-07-15）：请求生命周期详图 + 模式选择决策树 mermaid + 三模式执行段 + F4 分支；清掉 [`scheduling.md`](architecture/scheduling.md) ⚠️ 固定 P→D 残留（注解改为指向 data-flow）。
2. **`architecture/consistency.md`**（下一步）：一致性与故障模型。形式化本轮 B2 的持久语义（L3 F4 恢复点 / L4 SSOT）、ref 池权威、写回频率 N 的风险窗口、写一次读多次、控制面强一致/数据面最终一致、崩溃恢复点。
3. **`architecture/topology.md`**：部署拓扑（单/跨机房、网络 fabric 假设、RDMA 可用性退化）。承接本轮多处"留 topology.md"（GPUDirect RDMA 依赖 PCIe root、TCP 退化带宽-延迟模型）。

> P1 三篇补齐后满足完成判据（任一特性的"数据从哪来/写到哪/谁来调度/失败怎么办"可在此找到答案），再转 P2（proto 起草）。

---

## P2 — 模块划分与技术选型（当前首要）

**目标**：把架构落到模块，定语言、框架、接口、目录结构。

### 技术选型（已定）

| 层 | 模块 | 语言 / 框架 | 理由 |
|----|------|-------------|------|
| 存储层 | KV Pool / Weight Cache / Tiered Storage | **Rust** | 内存安全、零成本抽象、RDMA/IO 性能、长期常驻进程稳定性 |
| 控制面 | Router / Scheduler / 元数据 | **Go** | 并发原语成熟、生态利于写控制面服务、gRPC 生态 |
| 计算层 | Prefill / Decode / Draft 前向 | **Python + Triton** | Triton 写自定义 kernel、与 PyTorch/生态兼容、迭代快 |
| 对象存储 | SSOT | S3 / MinIO | 现成，不自研 |
| 控制面存储 | 元数据 | etcd | 强一致 + watch，路由表/KV 位置表 |
| 跨语言通信 | 统一 RPC | **gRPC + Protobuf** | Rust/Go/Python 都有一等支持；数据平面大块 KV 走 RDMA/共享内存旁路 gRPC |

### 模块与目录划分

```
lake/
├── docs/                       # 设计文档（语言无关）
├── rust/                       # 存储层
│   ├── kv-pool/                # KV cache 分布式池（内容寻址、分片、驱逐）
│   ├── weight-cache/           # 权重分层缓存
│   ├── tiered-store/           # L0-L4 分层缓存引擎
│   └── transfer/               # KV 传输（RDMA + TCP 退化）
├── go/                         # 控制面
│   ├── router/                 # 请求路由（无状态）
│   ├── scheduler/              # 池间/节点级调度 + 弹性
│   ├── controlplane/           # etcd 元数据、节点拓扑、KV 位置表
│   └── gateway/                # 对外 API（OpenAI 兼容）
├── python/                     # 计算层
│   ├── prefill/                # Prefill worker（Triton kernels）
│   ├── decode/                 # Decode worker（continuous batching）
│   ├── draft/                  # 投机解码 draft worker
│   ├── kernels/                # Triton kernel 集（attention/prefill/decode）
│   └── runtime/                # 与 KV Pool/Weight Cache 的 client（gRPC + RDMA）
├── proto/                      # 共享 protobuf IDL
└── deploy/                     # 部署（compose/k8s/镜像）
```

### 接口边界（P2 定稿）
- [ ] `proto/lake.proto`：Router↔Worker、Worker↔KVPool、Router↔ControlPlane 的 RPC 定义
- [ ] KV block 传输：gRPC 控制平面 + RDMA/共享内存数据平面，二进制布局规格
- [ ] KVBlockID / 元数据 schema 定稿（与 [`architecture/kv-cache-pool.md`](architecture/kv-cache-pool.md) 对齐）

### 转 P2 切入建议

P1 关键篇（execution-modes + overview）已齐，够支撑 proto 起草。建议从 **`proto/lake.proto` 的 RPC 边界草稿**切入，把这几轮定的存储池接口固化：

- **Router ↔ 存储池**：一次查询 RPC，输入 `(model_id, prompt 前缀)`，输出 `可复用 block 列表 + 各自位置（含本地命中判定）`。对应 radix + 位置视图一跳返回。
- **调度器 ↔ 存储池**：读位置视图（组 batch 用）；补拉放置请求（缺失 KV 放到指定节点 HBM）。
- **Worker ↔ 存储池**：prefill 产出写回（含反向回传的延伸 KV）；decode 读 KV；增量写回（容错 + 前缀生长）。
- **元数据 schema**：KVBlockID = `(model_id, layer_idx, block_hash)`；block 的 `locations` 为多层位置集合（L0/L1 缓存副本 + L2/L3/L4 三选一），L4 缺失才视为不存在。

**完成判据**：三个语言仓各自能编译出空壳服务；proto 可双向生成；目录结构落地。

---

## P3 — 最小可运行骨架

**目标**：跨 Rust/Go/Python 跑通一条请求，模型用 mock（返回固定 token），验证三语言联通与 KV 流转链路。

- [ ] Go gateway 接收请求 → router 路由
- [ ] Python prefill worker 产出 mock KV → 经 gRPC 写入 Rust KV Pool
- [ ] Python decode worker 从 KV Pool 读 KV → 输出 token
- [ ] 端到端冒烟脚本（替代当前 `src/` 单进程版）

**完成判据**：`deploy/` 一条命令起全栈，curl 打通。

---

## P4 — KV Pool 原型（Rust）

> 实现参考:`3rdparty/` 四个 submodule(SGLang HiCache / Mooncake / LMCache / vLLM),逐层对应与借鉴顺序见 [`research/3rdparty-reference.md`](research/3rdparty-reference.md) "实现参考顺序建议"。

- [ ] 内容寻址 block 存储 + 引用计数 + LRU 驱逐
- [ ] radix tree 前缀索引（前缀复用查询）
- [ ] 分层缓存引擎（RAM/NVMe，对象存储回填）
- [ ] gRPC 接口 + RDMA 数据平面（先 TCP 后 RDMA）
- [ ] 一致性哈希分片 + KV Node 扩缩时的 block 重分布
- [ ] **多模型生命周期**：模型注册/下线级联删、revision 失效（F11）
- [ ] **按模型配额与空间分配**（软/硬配额 + 借用 + 背压信号）
- [ ] **GC**：冷块/孤儿块回收 + 崩溃 reconcile
- [ ] **碎片整理**：逻辑共置 + 物理压实，后台节流可暂停

**完成判据**：前缀复用命中率、驱逐正确性有单测；吞吐 micro-benchmark；多模型隔离/配额/GC/碎片整理各有验证用例。

---

## P5 — 存算分离验证

- [ ] Python prefill/decode worker 接入真实（小）模型 + Triton kernel
- [ ] Prefill→Decode 的 KV 迁移流水线（计算与传输重叠）
- [ ] 故障注入：杀 Decode 节点 → 从 KV Pool 续推

**完成判据**：量化 KV 迁移带宽 vs 计算时间的比值，验证 P/Decode 物理分离可行区间。

---

## P6 — 弹性与调度（Go）

- [ ] 无状态 router + 控制面共享视图（etcd）
- [ ] 池间调度 + 反压
- [ ] 基于指标的弹性扩缩容（队列长度/TTFT/ITL/命中率）
- [ ] 冷启动压缩（权重预加载、layer-async serve、KV prefetch）

**完成判据**：扩容决策到 Ready 接受请求 < 10s（目标值，待 P7 校准）。

---

## P7 — 性能建模与验证

- [ ] 成本模型：KV 传输带宽 vs prefill/decode 计算时间
- [ ] 分层缓存的命中率/成本曲线
- [ ] 弹性冷启动时延分解
- [ ] 回填到 `docs/` 与 SLO，修正非目标与设计假设

**完成判据**：每个 P0 假设有量化结论（成立/不成立/在何条件下成立）。

---

## 当前优先级

**现在做**：P0（特性设计）→ P1（架构设计）→ P2（模块划分 + 技术选型）。

这三件是后续一切编码的前提。P2 中技术选型已定（存储 Rust / 控制 Go / 计算 Python+Triton），重点是把接口边界和目录结构定稿。

## 状态约定

- 每个阶段用 `[ ]` 标未完成、`[x]` 标完成、`[~]` 标进行中。
- 阶段完成时在对应标题后加 `(done YYYY-MM-DD)`。
