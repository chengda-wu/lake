# Lake — 彻底的存算分离推理系统

> 一个探索性仓库，目标是设计并验证一套**彻底**的存算分离（storage-compute disaggregated）大模型推理系统。

## 为什么是"彻底"

传统推理引擎把显存、KV cache、权重、调度器耦合在一个进程（甚至一张卡）里。这带来了：

- **扩缩容慢**：算力节点有状态，无法像无状态服务一样水平伸缩。
- **KV cache 利用率低**：cache 固化在产生它的节点上，前缀共享、跨请求复用、跨节点迁移都很难。
- **Prefill / Decode 互相干扰**：长 prefill 阻塞 decode，吞吐与延迟难以同时优化。
- **存储与算力绑定**：权重加载、checkpoint 恢复都依赖本地磁盘，冷启动代价高。

"彻底"意味着把**所有有状态的东西都从算力路径上剥离出去**：权重放到对象存储 + 内存池，KV cache 作为独立的分布式资源池，Prefill 与 Decode 拆成两个可独立伸缩的算力池，调度器只做无状态路由。算力节点理论上可以随时销毁、随时拉起。

## 仓库定位

这是**探索与研究**仓库，不是生产系统。当前阶段以设计文档 + 原型代码为主，验证以下假设：

1. KV cache 池化后，前缀复用带来的收益能否覆盖跨节点传输成本？
2. Prefill/Decode 物理隔离后，P50/P99 延迟与吞吐的帕累托前沿是否真的外推？
3. 算力节点做到无状态后，秒级扩缩容是否可行，冷启动代价如何压缩？
4. 以对象存储为单一事实来源（single source of truth）时，权重/KV 的分层缓存策略如何设计？

## 目录结构

```
lake/
├── README.md
├── docs/                       # 设计文档
│   ├── 01-goals.md             # 目标与非目标
│   ├── 02-architecture.md      # 总体架构
│   ├── 03-storage-layer.md     # 存储层（权重 / KV / 对象存储分层）
│   ├── 04-compute-layer.md     # 计算层（Prefill pool / Decode pool）
│   ├── 05-kv-cache-pool.md     # KV cache 分布式池
│   ├── 06-scheduling.md        # 路由与调度
│   └── 07-references.md        # 相关工作与文献
└── src/                        # 原型代码（Python，用于验证假设）
```

## 设计文档速览

- **总体架构**：见 [`docs/02-architecture.md`](docs/02-architecture.md)
- **存储层**：见 [`docs/03-storage-layer.md`](docs/03-storage-layer.md)
- **计算层**：见 [`docs/04-compute-layer.md`](docs/04-compute-layer.md)
- **KV cache 池**：见 [`docs/05-kv-cache-pool.md`](docs/05-kv-cache-pool.md)
- **调度**：见 [`docs/06-scheduling.md`](docs/06-scheduling.md)
- **相关工作**：见 [`docs/07-references.md`](docs/07-references.md)

## 状态

🚧 早期设计阶段。

## License

MIT
