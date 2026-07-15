# 03 — 存储层

存储层是彻底存算分离的根基,托管两类数据:**权重**与 **KV cache**,并提供分层缓存。

存储池是长期存续、模型无关的独立基础设施:同一池同时承载多个 `(model_id, revision)` 的权重与 KV,模型上下线与池生命周期解耦。池提供按模型的空间分配/扩缩容、GC、碎片整理(详见 [`kv-cache-pool.md`](kv-cache-pool.md))。

## 数据模型

### 权重
- 不可变,按 `(model_id, revision)` 寻址。
- 粒度:按 layer / tensor 分片,支持并行加载与按需加载。
- 格式:原始 fp16/bf16,可选量化副本(int8/int4/fp8)作为独立 artifact,不在线反量化。

### KV cache
- 可变、有生命周期,按 `(model_id, layer_idx, block_hash)` **内容寻址**(见 [`kv-cache-pool.md`](kv-cache-pool.md)):相同前缀 → 相同 block hash → 命中同一 KV,前缀复用天然成立。
- 粒度:**per-block + 前缀树(radix)**索引(复用友好),per-layer/per-sequence 备选。**block 粒度 = 128 token**(缓存命中/复用/传输/写回的最小单位,初版默认,待 P7 校准;与 SGLang `--page-size`、vLLM `block_size` 同量级,常见 16/128/256)。
- 持久化:热数据在 RAM/NVMe,冷数据落对象存储。

### KV 类型:t-type / r-type

两类的**复用条件一致**:都需**命中全部前缀**才能复用——从序列起点的连续前缀 KV/state 必须在场,attention/state 才能从该点续算。区别**仅在 HBM(L0)的存储形态**:r-type 用紧凑表示替代逐 token 的完整 KV,以降低 HBM 占用。

| 类型 | HBM(L0)存储形态 | 占用 | 典型算子 |
|------|-----------------|------|----------|
| **t-type** | 逐 token 完整 KV,paged block table | 随序列长度线性增长 | full attention、MLA |
| **r-type** | 紧凑表示:滑动窗口最近 W token / 定长 recurrent state | 亚线性或常量 | sliding window attention、Mamba/state-space、卷积类 |

**各层组织**:

- **HBM(L0)**:两类并存,引擎按类型分 arena / 管理器(t-type block arena + r-type 状态 arena)。**区分的唯一目的是减少 r-type 的 HBM 占用**——r-type 不存逐 token KV 而存紧凑状态(参考 vLLM per-group `SingleTypeKVCacheManager` + `KVCacheSpec`(`FullAttentionSpec`/`MambaSpec`);SGLang multi-pool `PoolName`(Mamba/SWA/DSA/Draft))。Q1 的"固定 arena"需为 r-type 另设固定状态 arena(见 [`compute-layer.md`](compute-layer.md) "KV 类型"节)。
- **DRAM/SSD(L1–L4)**:缓存命中/存取最小单位是 block(128 token),下层**统一按 block、page-first 组织**。两类复用条件本就一致(全前缀命中),故下层**不区分类型**;r-type 落下层时在 block 边界 checkpoint 其紧凑状态:
  - sliding window:存 **trailing pages**(最近 W token,参考 SGLang `PoolHitPolicy.TRAILING_PAGES`)。
  - Mamba/state-space/卷积:每 128 token 存一份 recurrent state 快照。
  - 复用时 radix 沿前缀匹配到最长边界,取该边界处的完整 KV(t-type)或 state 快照(r-type),从该点续算。

> **关键差异(相对参考实现)**:SGLang v2 multi-pool 把 Mamba/SWA/DSA/Draft 各开独立 pool + 各自 `PoolHitPolicy`(`ALL_PAGES`/`TRAILING_PAGES`),按类型**物理分池**存(`hicache_storage.py::HiCacheStorage.batch_exists_v2` / `PoolTransfer`)。我们更彻底:类型区分**只存在于 HBM 存储形态**(降占用),L1–L4 统一按 block 存储池承载(池本就不解释张量布局,按不透明字节块存取),block 内装的是逐 token KV 还是紧凑 state 快照由布局元数据声明,而非物理分池。

r-type 状态 checkpoint 的形式/间距与 L1+ 持久化性价比(Mamba state 是否值得落 L1+)留开放,见 [`kv-cache-pool.md`](kv-cache-pool.md) "t-type / r-type"开放点。

## 分层缓存

L0–L4 五层全部由存储池统一管理(放置/驱逐/副本/冷热/生命周期)。计算节点不拥有内存——HBM/RAM/NVMe 是存储池的物理载体,计算服务向存储池申请放置而非自行管理本地缓存。所有 KV 位置(含"哪段 KV 在哪个节点 HBM")均为存储池权威元数据。统一管理的全局原则见 [`overview.md`](overview.md),本节讲分层细节。

| 层级 | 介质 | 容量 | 延迟 | 驱逐策略 | 管理主体 |
|------|------|------|------|----------|----------|
| L0 | GPU HBM | 极小 | ~ns | 存储池放置/驱逐 | 存储池(元数据强一致,物理载体在计算节点) |
| L1 | 主机 RAM | 小 | ~μs | LRU | 存储池 |
| L2 | 本地 NVMe | 中 | ~10μs | LRU + TTL | 存储池 |
| L3 | 远端内存池 (RDMA) | 大 | ~10-100μs | 全局 LRU | 存储池(强一致元数据) |
| L4 | 对象存储 (SSOT) | 无限 | ~ms | 永久(带版本) | 唯一权威 |

> **物理约束**:L0–L2 的统一管理是元数据层面的——物理访问仍受 GPU 本地性约束,attention 读 KV 必须在本机 HBM,无法高效跨节点直读。因此同一 batch 各 sequence 的 KV 必须已在同一 GPU HBM(本地命中),否则先由存储池补拉。放置与 batch 的边界见下"方案 Z"。

**读取路径**:L0 → L1 → L2 → L3 → L4,逐层回填。
**写入路径**:Prefill 产出 → L0(产出即属存储池)→ 异步写 L1/L3 → 按热度决定是否落 L4。

## 冷热与生命周期管理

存储池对每个 KV block 全权决定冷热判定、层间放置、提升/下沉、驱逐与终结,贯穿 L0→L4。

### 层间副本 vs 移动

- **L0/L1 做副本**:上层是下层的缓存副本,丢了下层还在,回填快;费空间但 L0/L1 小,可接受。
- **L2/L3/L4 间按移动**:同层冗余无意义,省空间。
- **L4 永久权威**:唯一不可丢的副本,L4 缺失才视为 block 不存在。

数据模型:一个 block 在元数据中有**多层位置**(L0/L1 各一份缓存副本,L2/L3/L4 三选一),而非单一位置。`locations` 是"层→物理位置"集合;L0/L1 缺失只是缓存未命中,L4 缺失才视为不存在。

### 冷热判据

| 维度 | 性质 | 作用 |
|------|------|------|
| 引用数(in-flight ref) | 硬门槛 | >0 → 冻结迁移/驱逐,保护在途请求 |
| 访问频次 | 连续信号 | 热度主信号,驱动 promotion |
| 最近访问(recency) | 连续信号 | 防"曾经热、现在凉"占上层,驱动 demotion |
| 前缀亲和(公共前缀加权) | 修饰项 | 驱逐 tie-breaker,公共前缀不易驱逐 |

聚合(非简单加权和):
- 引用数 >0 → 冻结,不进冷热排序(正确性约束)。
- 否则**热度分** = f(频次, recency),驱动 promotion/demotion。
- 驱逐/下沉排序时,公共前缀 block 给予加权保护。

频次用 **LFU-Aging(衰减)** 而非滑动窗口:block 数量级大,滑动窗口元数据吃不消;衰减状态小、够用。参数待 P7 校准。

### 迁移触发:主动为主 + 被动兜底

- **主动**:后台扫描,按热度分 promotion(热块上提、L0 预放置)/ demotion(冷块下沉),保持上层常驻新鲜热数据、减少首次 miss。
- **被动**:读 miss 回填、写满驱逐,即时响应兜底。

主动是主路径——否则冷块淤积上层、热块首次必 miss,本地命中率起不来。

### 带宽预算

迁移 / GC / 碎片整理共享一个后台任务带宽池,内部按优先级调度。总预算 < 10%,不额外放宽(见 [`../features/slo.md`](../features/slo.md))。

### 放置与 batch 的职责边界(方案 Z)

存储池的 HBM 放置与计算层 batch 组成**单向耦合**:

- **存储池主动放置**:按热度/前缀亲和把高频 KV 预放置到节点 HBM,发布"哪些 KV 在哪个 HBM"的位置视图。不感知任何 batch 意图,只做通用热度放置。
- **调度器读视图组 batch**:读位置视图,优先把已在同一 HBM 的 sequence 组进同一 batch(本地命中 → D-direct),缺失的再让存储池补拉。不反向指挥放置。

信息流单向(存储池发布视图 → 调度器消费)。代价:通用预判未必对齐每个具体 batch——冷门前缀被组进 batch 时无本地命中,需临时补拉。

## KV Pool 架构(详见 05)

L3(远端内存池)是 KV Pool 的物理载体,由一组 KV Node 组成:每个贡献 RAM + NVMe,通过 RDMA 暴露 block 读写,元数据由存储池控制面维护。L0–L2 物理载体在计算节点但元数据同样归存储池;五层共用一套元数据与 API。

## 一致性模型

- 权重:不可变,靠缓存失效(revision 变更)。
- KV cache:**写一次读多次**。Prefill 写入后不可变(针对该 token range),后续 token 的 KV 是新 block,只需单写者屏障,无需多写者并发控制。
- 故障恢复:KV block 写入 L3/L4 后才认为 Prefill 完成;崩溃从最近的 KV checkpoint 续推。

## 开放问题

- KV block 压缩:是否有跨 block 的低秩/量化压缩降低传输量?
- LFU-Aging 参数与前缀亲和加成取值,待 P7。
- 后台任务带宽池的内部优先级调度策略待定。
- 多租户隔离(远期预留,当前不实现):归外部控制面/部署切分;lake 侧未来若做,靠 `KVBlockID` 加 scope 维度(见 [`kv-cache-pool.md`](kv-cache-pool.md) "Block 寻址"预留、[`../features/features.md`](../features/features.md) F8)。
- **r-type 状态 checkpoint**:Mamba/卷积 recurrent state 落 L1+ 的 checkpoint 间距/形式、sliding window trailing pages 阈值,待实现/P7 校准。
- **block 粒度 128**:与传输/复用/写放大的权衡,待 P7 校准。
