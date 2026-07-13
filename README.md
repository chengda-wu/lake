# Lake — 彻底的存算分离推理系统

> 一个探索性仓库，目标是设计并验证一套**彻底**的存算分离（storage-compute disaggregated）大模型推理系统。

## 为什么是"彻底"

传统推理引擎把显存、KV cache、权重、调度器耦合在一个进程（甚至一张卡）里。这带来了：

- **扩缩容慢**：算力节点有状态，无法像无状态服务一样水平伸缩。
- **KV cache 利用率低**：cache 固化在产生它的节点上，前缀共享、跨请求复用、跨节点迁移都很难。
- **Prefill / Decode 互相干扰**：长 prefill 阻塞 decode，吞吐与延迟难以同时优化。
- **存储与算力绑定**：权重加载、checkpoint 恢复都依赖本地磁盘，冷启动代价高。

"彻底"意味着把**所有有状态的东西都从算力路径上剥离出去**：权重放到对象存储 + 内存池，KV cache 作为独立的分布式资源池，Prefill 与 Decode 拆成两个可独立伸缩的算力池，调度器只做无状态路由。算力节点理论上可以随时销毁、随时拉起。

更进一步，本系统**以 KV 为中心**设计，连 GPU HBM / 主机内存 / 本地 NVMe 都不归计算节点私有，而是存储池统一管理的物理载体（L0–L4）。**agent 多轮对话**是首要驱动场景：其"长共享前缀 + 逐步增长"结构使 KV 池化、前缀复用与反向回传增强的价值最大化（见 [`docs/architecture/execution-modes.md`](docs/architecture/execution-modes.md)）。

## 仓库定位

这是**探索与研究**仓库，不是生产系统。当前阶段以设计文档 + 原型代码为主，验证以下假设：

1. KV cache 池化后，前缀复用带来的收益能否覆盖跨节点传输成本？
2. Prefill/Decode 物理隔离后，P50/P99 延迟与吞吐的帕累托前沿是否真的外推？
3. 算力节点做到无状态后，秒级扩缩容是否可行，冷启动代价如何压缩？
4. 以对象存储为单一事实来源（single source of truth）时，权重/KV 的分层缓存策略如何设计？

## 路线图

先看 [`docs/00-plan.md`](docs/00-plan.md) —— 依次要做的事情与技术选型（存储 Rust / 控制 Go / 计算 Python+Triton）。

**当前阶段**：P0 特性设计 → P1 架构设计 → P2 模块划分与技术选型。

## 目录结构

```
lake/
├── README.md
├── docs/
│   ├── 00-plan.md              # 路线图与执行计划（主线入口）
│   ├── features/               # P0 特性设计 — 做什么
│   │   ├── goals.md            #   目标与非目标
│   │   ├── features.md         #   特性清单（Must/Should/Could）
│   │   ├── slo.md              #   SLO 与衡量指标
│   │   └── nonfunctional.md    #   非功能需求
│   ├── architecture/           # P1 架构设计 — 怎么搭
│   │   ├── overview.md         #   总体架构
│   │   ├── execution-modes.md  #   执行模式与 KV 流转时序（以 KV 为中心）
│   │   ├── storage-layer.md    #   存储层（权重 / KV / 对象存储分层）
│   │   ├── compute-layer.md    #   计算层（Prefill / Decode 池）
│   │   ├── kv-cache-pool.md    #   KV cache 分布式池
│   │   └── scheduling.md       #   路由与调度
│   └── research/               # 相关工作
│       ├── references.md
│       ├── 3rdparty-reference.md  # 3rdparty 源码与本设计的逐层对应(汇总)
│       ├── sglang/                #   SGLang HiCache 深度分析
│       ├── lmcache/               #   LMCache 深度分析
│       └── mooncake/              #   Mooncake 深度分析
├── 3rdparty/                   # 参考源码（git submodule，只读）
│   ├── sglang/                 #   SGLang（HiCache 分层 KV）
│   ├── lmcache/                #   LMCache（跨实例 KV 复用）
│   └── mooncake/               #   Mooncake（KVCache-centric 分离架构）
└── src/                        # 早期单进程 Python 原型（验证假设用，将被 rust/go/python 子项目取代）
```

> 首次 clone 后执行 `git submodule update --init --recursive` 拉取 `3rdparty/`。三个参考项目与本系统设计的逐层对应见 [`docs/research/3rdparty-reference.md`](docs/research/3rdparty-reference.md)。

## 设计文档速览

- **路线图**：见 [`docs/00-plan.md`](docs/00-plan.md)
- **特性 / SLO / 非功能**：见 [`docs/features/`](docs/features/)
- **总体架构**：见 [`docs/architecture/overview.md`](docs/architecture/overview.md)
- **执行模式与 KV 流转**：见 [`docs/architecture/execution-modes.md`](docs/architecture/execution-modes.md)
- **存储层**：见 [`docs/architecture/storage-layer.md`](docs/architecture/storage-layer.md)
- **计算层**：见 [`docs/architecture/compute-layer.md`](docs/architecture/compute-layer.md)
- **KV cache 池**：见 [`docs/architecture/kv-cache-pool.md`](docs/architecture/kv-cache-pool.md)
- **调度**：见 [`docs/architecture/scheduling.md`](docs/architecture/scheduling.md)
- **相关工作**：见 [`docs/research/references.md`](docs/research/references.md)

## 状态

🚧 早期设计阶段。

## License

MIT
