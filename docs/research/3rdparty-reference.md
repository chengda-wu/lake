# 3rdparty 源码参考

本仓库在 `3rdparty/` 以 git submodule 引入三个项目源码,作为设计与实现的直接参考。本文是**汇总对比**;各项目的深度分析见分目录:

- [`sglang/`](sglang/) — SGLang HiCache:[总览](sglang/overview.md) · [分层机制](sglang/hicache.md) · [存储后端](sglang/storage-backends.md)
- [`lmcache/`](lmcache/) — LMCache:[总览](lmcache/overview.md) · [跨实例复用与后端](lmcache/sharing-and-backends.md)
- [`mooncake/`](mooncake/) — Mooncake:[总览](mooncake/overview.md) · [传输引擎](mooncake/transfer-engine.md) · [KV 存储与池化](mooncake/kv-store.md)

本文把它们的关键组件与本系统(`docs/architecture/`)逐层对应,并标注**借鉴点**与**关键差异**(我们的设计更彻底)。

## submodule 清单

| 路径 | 来源 | 检出 |
|------|------|------|
| `3rdparty/sglang` | [sgl-project/sglang](https://github.com/sgl-project/sglang) | main HEAD |
| `3rdparty/lmcache` | [LMCache/LMCache](https://github.com/LMCache/LMCache) | nightly |
| `3rdparty/mooncake` | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) | main HEAD |

> 三者本就生态相连:SGLang 的 HiCache 把 Mooncake 作为 L3 存储后端之一(`python/sglang/srt/mem_cache/storage/mooncake_store/`),LMCache 也是 HiCache 的可选 L3 后端。我们站在三者之上做更彻底的存算分离。

---

## 1. SGLang HiCache → 我们的 L0-L4 分层 + 放置 + 冷热

源码入口:`3rdparty/sglang/docs/advanced_features/hicache_design.md`、`python/sglang/srt/mem_cache/`。

### 借鉴点

| HiCache 设计 | 我们对应 | 说明 |
|--------------|----------|------|
| **HiRadixTree**:radix 节点记录 KV 存在哪层(GPU/CPU/L3/多层) | radix tree 归存储池 + block 的 `locations` 多层位置集合 | 见 [`../architecture/kv-cache-pool.md`](../architecture/kv-cache-pool.md)、[`../architecture/storage-layer.md`](../architecture/storage-layer.md)。HiCache 的"节点记位置"正是我们 `locations` 元数据的原型 |
| **prefetch 三策略**:best_effort / wait_complete / timeout | 迁移触发的"被动兜底"读 miss 回填 + 主动预放置 | 见 storage-layer "迁移触发"。timeout 的 `base + per_ki_token` 公式可直接借鉴为我们的 prefetch 预算模型 |
| **write-back 三策略**:write_through / write_through_selective / write_back | decode 增量写回频率 N 的策略 | 见 execution-modes "decode 写回频率"。selective(按访问频次只回写热数据)对应我们"前缀生长"的写回取舍 |
| **page-first / page_first_direct 布局** | block 粒度 + 分块流水线 | 见 kv-cache-pool "分块流水线"。page_first_direct 让同层同 page 连续,可零拷贝传 L3——我们 Rust transfer 层可照搬 |
| **计算-传输重叠**:算 layer N 时传 layer N+1 | 分块流水线与 prefill 层数对齐 | 直接对应,execution-modes 开放问题已记 |
| **MLA write-back 去重**:多 TP rank 只一个 rank 回写 | (未来 TP 支持) | 留作 compute-layer 细节参考 |
| **统一 `HiCacheStorage(ABC)` 接口** + 多后端(file/mooncake/hf3fs/nixl/aibrix) | 存储池后端抽象 | 我们存储池统一管理 L0-L4,后端可抽象;Mooncake/NIXL 等可作为 L3 物理实现 |

### 关键差异(我们更彻底)

- **HiCache 的 L1/L2 私有于推理实例,L3 才共享**;我们 **L0-L4 全归存储池统一管理,L1/L2 也是池的物理载体而非 worker 私有**。计算节点不拥有任何内存,"本地命中"是存储池放置决策的结果,不是实例私有缓存。这是我们与 HiCache 的根本分野——HiCache 仍是"实例私有分层 + 共享 L3",我们是"全层共享、放置归一"。
- HiCache 不持续同步 L3 元数据,访问时实时查后端;我们 radix + 位置视图由控制面(etcd)强一致维护,Router 一跳拿前缀复用 + 本地命中(守 5ms 预算)。
- HiCache 无"反向回传增强未来前缀"的显式机制(它的 write-back 是为跨实例共享,非为多轮前缀生长);我们把它作为 agent 多轮的核心(见 execution-modes 时序二反向)。

---

## 2. Mooncake → 我们的 KV Pool 数据面 + Transfer Bus

源码入口:`3rdparty/mooncake/mooncake-transfer-engine/`、`mooncake-store/`、`mooncake-p2p-store/`、`docs/`。

### 借鉴点

| Mooncake 组件 | 我们对应 | 说明 |
|---------------|----------|------|
| **mooncake-transfer-engine**:RDMA + 多 NIC 零拷贝传输 | Transfer Bus(RDMA 数据面,TCP 退化) | 见 overview "数据面:KV 跨节点传输"。直接参考其传输 API 与零拷贝设计 |
| **mooncake-store**:KVCache 全局池、按 segment 寻址 | KV Pool(L3 远端内存池) | 见 kv-cache-pool "物理布局"。Mooncake 的 KVCache store 是我们 L3 的工业级原型 |
| **mooncake-p2p-store**:P2P 存储拓扑 | KV Node 分片 + 一致性哈希 | 见 kv-cache-pool "空间分配与扩缩容"。参考其节点组织与扩缩 |
| **KVCache-centric disaggregation**(prefill/decode 分离 + KV 池) | 整体架构立地 | Mooncake 是我们"以 KV 为中心"的直接灵感来源(见 overview)。但 Mooncake 仍以实例为中心做 P/D 分离,我们进一步把 HBM 也剥离 |
| **PD disaggregation via TransferEngine** | 时序二正向(P→D 跨节点传输) | 见 execution-modes。Mooncake 的 P/D KV 搬运即我们时序二正向 |

### 关键差异

- Mooncake 的 KVCache 池服务于"实例间共享/迁移",实例仍拥有本地 HBM;我们连 HBM 放置都归存储池(方案 Z)。
- Mooncake 无 radix 前缀树的内容寻址复用(按 segment ID 存取);我们用内容寻址 `(model_id, layer, block_hash)` + radix 实现前缀复用,SGLang RadixAttention 的思路补上这一块。
- Mooncake 无"统一管理 L0-L4 + 冷热生命周期 + 多模型配额/GC/碎片整理"——这些是我们的存储池增量(F11)。

---

## 3. LMCache → 跨请求/跨实例 KV 复用 + 多存储后端

源码入口:`3rdparty/lmcache/lmcache/`、`csrc/storage_backends/`、`rust/`、`examples/`。

### 借鉴点

| LMCache 设计 | 我们对应 | 说明 |
|--------------|----------|------|
| 跨请求/跨实例 KV 复用,降 TTFT | 前缀复用 + D-direct | 见 features F1。LMCache 的"长 system prompt / RAG / 多轮"复用场景与我们 agent 多轮定位一致 |
| 多存储后端:CPU memory / local disk / Redis | L1-L4 分层后端 | 见 storage-layer 分层表。LMCache 的后端抽象可作 L2/L3 实现参考 |
| `csrc/storage_backends`(C++ 后端) | Rust 存储层后端 | 我们用 Rust 重写存储层,但后端策略(分片、压缩、传输)可参考 LMCache 的 C++ 实现思路 |
| `rust/` 目录(LMCache 已有 Rust 组件) | 存储层 Rust 技术栈 | 印证 Rust 适合写存储层;可参考其 Rust/C++ 桥接与 FFI 模式 |
| 与 vLLM 集成的 KV manager 拦截 | 计算层 worker ↔ 存储池 client | 见 compute-layer。LMCache 作为 vLLM 的 drop-in 优化,其"拦截 KV 读写"的模式可参考我们 Python worker 的 runtime client 设计 |

### 关键差异

- LMCache 是 vLLM 的**附加层**,不改变 vLLM 实例私有 HBM 的归属;我们是**重做存算分离架构**,HBM 归存储池。
- LMCache 无全局 radix 内容寻址(靠 prefix hash 匹配);我们 radix tree + 内容寻址 + 位置视图一跳返回。
- LMCache 无执行模式选择(PD分离/混部/D-direct);这些是我们的调度层增量。

---

## 设计取舍:站在三者之上

| 我们的设计层 | 主要参考 | 我们多做的(更彻底) |
|--------------|----------|---------------------|
| L0-L4 分层 | SGLang HiCache | L1/L2 也归存储池(非实例私有);统一冷热/生命周期 |
| KV Pool 数据面 | Mooncake transfer-engine + store | 内容寻址 + radix + 多模型配额/GC/碎片整理 |
| 前缀复用 | SGLang RadixAttention + LMCache | radix 归存储池 + 位置视图一跳 + 反向回传生长 |
| 执行模式 | DistServe/Splitwise + HiCache PD | 三模式逐请求选路 + D-direct(本地命中直跳) |
| 放置/调度边界 | (我们的方案 Z,原创) | 存储池主动放置 + 调度器单向消费 |

## 实现参考顺序建议

P4(KV Pool 原型,Rust)时按此顺序参考源码:
1. **Mooncake transfer-engine**:先抄 RDMA 零拷贝传输骨架 → 我们的 Transfer Bus。
2. **Mooncake store + LMCache storage_backends**:KV store 分片/后端 → 我们 L3 + L2/NVMe。
3. **SGLang HiCache HiRadixTree + page_first_direct**:radix 节点记位置 + 布局 → 我们 `locations` 元数据 + 分块流水线。
4. **SGLang HiCache prefetch/write-back 策略**:迁移触发与写回频率 → 我们冷热迁移 + decode 写回 N。
5. **LMCache rust/ + 跨实例复用**:Rust 存储层工程模式 + 复用场景验证。

> 参考源码时注意:三个 submodule 各带自己的 `.claude/` 规则(如 sglang 的 modify-component-must-read、mooncake 的 skills),那些是**修改它们自身代码**的约束,与我们参考其设计无关,忽略。

## submodule 使用约定

- `3rdparty/` 只读参考,**不修改** submodule 内代码。如需改造,fork 后换 URL。
- clone 本仓库后需 `git submodule update --init --recursive` 拉取。
- 升级 submodule:在对应目录 `git checkout <ref>` 后回根目录 `git add` 提交指针更新;在本文"检出"列同步记录。
- submodule 体积较大(SGLang/Mooncake),CI 如需提速可用 `--depth 1` 浅克隆。
