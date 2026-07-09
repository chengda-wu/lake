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
  - [x] F1 KV cache 池化与前缀复用（内容寻址、radix tree）
  - [x] F2 Prefill / Decode 物理隔离
  - [x] F3 分层缓存（HBM→RAM→NVMe→远端内存池→对象存储）
  - [x] F4 故障恢复（基于 KV Pool 续推）
  - [x] F5 无状态路由
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
- [ ] 更新 [`overview.md`](architecture/overview.md)：补全组件间接口契约（IDL/协议草稿）
- [ ] `architecture/data-flow.md` 请求生命周期详图（含故障分支）
- [ ] `architecture/consistency.md` 一致性与故障模型（KV 写一次读多次、控制面强一致/数据面最终一致、崩溃恢复点）
- [ ] `architecture/topology.md` 部署拓扑（单机房/跨机房、网络 fabric 假设、RDMA 可用性退化）

**完成判据**：任一特性的"数据从哪来、写到哪、谁来调度、失败怎么办"都可在此找到答案。

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

- [ ] 内容寻址 block 存储 + 引用计数 + LRU 驱逐
- [ ] radix tree 前缀索引（前缀复用查询）
- [ ] 分层缓存引擎（RAM/NVMe，对象存储回填）
- [ ] gRPC 接口 + RDMA 数据平面（先 TCP 后 RDMA）
- [ ] 一致性哈希分片 + KV Node 扩缩时的 block 重分布

**完成判据**：前缀复用命中率、驱逐正确性有单测；吞吐 micro-benchmark。

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
