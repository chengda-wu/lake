# vLLM — 计算层抽象与存算分离接入点

> 源码:`vllm/v1/`、`vllm/distributed/kv_transfer/`、`vllm/distributed/kv_events.py`、`vllm/v1/kv_offload/`、`vllm/model_executor/`。本文聚焦**计算层可复用抽象**、**KV connector 接口**、**KV Events** 与**原生多层 KV offload 子系统**——本系统 worker 接入存储池的直接参考。

## PagedAttention 内存模型

KV cache 按固定 `block_size`(token 数)分页,一个请求的 KV 由一组**不连续**的 block 组成,由 **block table**(每 sequence 一个,`block_id` 数组)映射逻辑页→物理 block。消除碎片,使 HBM 利用率近满。

- 调度器侧:`KVCacheManager` 管 block 分配/释放;`BlockPool` 持 free-list 与 `hash→block` 前缀缓存映射。
- worker 侧:`GPUModelRunner` 维护每 sequence 的 block table,attention kernel 经 block table 读散落的 KV block。

**对本系统**:block 粒度与 block table 模型可直接复用——存储池的 KV block 与 vLLM block 对齐,worker 仍用 block table 做 paged attention,只是 block 的物理位置由存储池元数据决定(而非进程私有 free-list)。

## KV cache 管理(V1 分层)

V1 把 KV 管理拆成三层:

```
KVCacheManager(facade)
  └─ SingleTypeKVCacheManager × N(每 KV-cache group 一个,如 full-attn / MLA / Mamba)
       └─ KVCacheCoordinator(ABC:alloc/free/evict 策略)
            └─ BlockPool(实际 free-list + hash→block)
```

- `KVCacheManager.get_computed_blocks(req)`:`req` 的 token 按 block hash 查 `BlockPool.cached_block_hash_to_block`,返回已缓存的 prefix block 列表 + computed token 数。这是 APC 入口。
- `allocate_slots`:为新 token 分配 block。
- `evict_blocks` / `FreeKVCacheBlockQueue`:LRU 驱逐。

**关键**:V1 **无 radix tree**——前缀匹配靠逐 block hash 查哈希表,断链即停。多模态/LoRA 的 extra key 经 `generate_block_hash_extra_keys` 入哈希。

## 前缀哈希与跨实例复用

```
block_hash = hash(group_id, tokens_in_block, extra_keys)
```

- `hash_block_tokens`(L577):核心哈希。
- `maybe_convert_block_hash`(L79):把内部 block hash 转 `ExternalBlockHash`——**这是跨实例/connector 交换的哈希形式**,使 disaggregated worker 间不必重算 token 哈希即可复用前缀。
- `make_block_hash_with_group_id`:不同 KV-cache group(如 full vs MLA)哈希隔离。

**对本系统**:`ExternalBlockHash` 是 worker 侧向存储池查询前缀的自然键——存储池按 `(model_id, layer, block_hash)` 内容寻址,vLLM 的 block hash 与之可直接对接。

## Attention 后端抽象

`AttentionBackend`(ABC)注册表式选择;`AttentionMetadata` 携带 block tables/seq lens;`AttentionImpl` 消费 paged KV 做前向。主路径 `FlashAttentionBackend`(C++/CUDA),另有 `triton_attn`(Triton)、`flashinfer`、MLA 系列。

- `AttentionMetadataBuilder`:从 `SchedulerOutput` 构造 metadata——**这是调度器输出→attention 输入的衔接点**,本系统控制面下发 batch 时可参考此 builder 形态。

## Scheduler

`Scheduler`(L68)从 waiting/running 队列组 prefill/decode batch,调 `KVCacheManager.get_computed_blocks`/`allocate_slots`,产 `SchedulerOutput` 供 worker 消费。

**对本系统**:vLLM 调度器是**单实例、进程内**视角。本系统把调度上移到控制面(Go,集群视角),worker 侧只保留**节点级 batch 组成**(continuous batching + block table)。即:vLLM 的 `Scheduler` 拆成"集群调度(控制面)+ 节点级调度(worker)"两层。

## KV Connector 接口(存算分离接入点 ★)

`KVConnectorBase_V1`(L171,ABC)是 vLLM 把 KV cache 与外部存储/传输解耦的**插件接口**。LMCache/Mooncake/NIXL/FlexKV 均已实现为 connector。这正是本系统 worker 接入存储池的参考接口形态。

### 角色与生命周期

connector 分两侧(同文件):

| 角色 | 职责 |
|------|------|
| `KVConnectorRole.SCHEDULER`(scheduler 侧) | 请求级:查外部 KV 命中、决定是否等待外部 KV 加载、产 connector metadata |
| `KVConnectorRole.WORKER`(worker 侧) | 执行级:把外部 KV 加载进本机 paged buffer、把产出 KV 存回外部 |

两侧经 `KVConnectorMetadata`/`KVConnectorHandshakeMetadata` 协调。

### worker 侧绑定

- `vllm/v1/worker/gpu/kv_connector.py::KVConnector`(L29)/`ActiveKVConnector`(L47):把 `KVConnectorBase_V1` 绑进 model runner。
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`:mixin 注入 save/load KV 钩子到 `GPUModelRunner`——**前向逐层 save KV、加载时逐层 load KV** 的 layer-wise 异步流水线在此。

### 已有 connector(可直接参考实现)

| connector | 文件 | 参考 |
|-----------|------|------|
| `LMCacheConnectorV1` | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py` (L72) | 跨实例复用 + 多后端 |
| `MooncakeStoreConnector` | `…/mooncake/store/connector.py` (L87) | RDMA 零拷贝 KV 池 |
| `NixlBaseConnector` | `…/nixl/connector.py` (L79) | NIXL 传输 |
| `FlexKVConnectorV1` | `…/flexkv_connector.py` (L35) | rank-0 leader + eventfd layerwise |
| `MultiConnector` | `…/multi_connector.py` (L128) | 组合多 connector |
| `OffloadingConnector` / `SimpleCPUOffloadConnector` | `…/offloading_connector.py`、`…/simple_cpu_offload_connector.py` | CPU offload |

### KV 布局协商

`KVCacheConfig`(L920)/`KVCacheSpec`(L100):connector 与调度器经 spec 协商 KV 布局(`FullAttentionSpec`/`MLAAttentionSpec`/`MambaSpec`)——**connector 需知道每层 KV 的形状才能与外部存储对齐**。本系统存储池按不透明字节块存(不解释布局),但 worker↔池的传输仍需 layout spec,可参考此协商。

### SupportsHMA(hybrid memory allocator)

`SupportsHMA`(L85,ABC 标记):connector 声明支持 **hybrid memory allocator (HMA)**——即 connector 可与 HMA 协同使用。HMA 是 vLLM 针对**多 KV cache group 混合架构**(如 Mamba+attention)的显存管理器,把多个 group 的 KV 放进同一池统一分配。`SupportsHMA` 要求 connector 实现 `request_finished_all_groups`:在一个请求的**所有** KV cache group 都完成后再统一异步 free block(而非逐 group 释放),使 connector 的 save/send 与 HMA 的多 group 释放时序对齐。

**对本系统**:`SupportsHMA` **不对应**方案 Z("外部管 HBM")——vLLM 的 HBM 始终由引擎自身分配,connector 只借(`register_kv_caches`)做传输,无论是否 HMA。方案 Z 的"存储池放置 KV 到 HBM、worker 消费"在 vLLM **无对应能力标记**,是我们相对 vLLM 的增量,需自行设计(见 [`../../architecture/compute-layer.md`](../../architecture/compute-layer.md) "待写:HBM 池化下的入图与 KV 管理")。本系统存储池 client ≈ 一个常驻的 `KVConnectorBase_V1`;区别:vLLM connector 是可选插件、per-instance,本系统是必经路径、集群级权威。

## KV Events(引擎→外可见性通道 ★)

`vllm/distributed/kv_events.py` 是 vLLM 把**引擎内 KV 生命周期**以事件形式发布给外部消费者的机制(经 zmq)。这是 vLLM 向"控制面外建集群 KV 索引"演进的关键设施——Dynamo/llm-d 等控制面消费这些事件构建集群级 KV 位置图,无需 mirror 引擎内部状态。

事件类型(msgspec struct,`omit_defaults` 向后兼容):

| 事件 | 字段 | 语义 |
|------|------|------|
| `BlockStored` | `block_hashes: list[ExternalBlockHash]`、`parent_block_hash`、`token_ids`、`block_size`、`medium`、`group_idx`、`kv_cache_spec_kind`、`extra_keys` | 块被写入某介质;`medium`(`"GPU"`/…)标记块所在介质——**vLLM 向位置感知迈出的一步** |
| `BlockRemoved` | `block_hashes`、`medium`、`group_idx` | 块被驱逐/释放 |
| `AllBlocksCleared` | — | 清空 |

- `KVEventBatch` 带 `ts` + `data_parallel_rank`;`KVEventAggregator` 跨 worker(TP)聚合,只返回所有 worker 都emit的事件(去重)。
- 配置:`vllm/config/kv_events.py`::`KVEventsConfig`;示例:`examples/features/kv_events/kv_events_subscriber.py`。
- 发射点:`block_pool.py` 的 `_build_block_stored_event`/`emit_cached_block_events`/`cache_partial_block`(store/removed 均在 block_pool 侧发射)。

**对本系统**:KV Events 的 schema(`BlockStored` 携 `ExternalBlockHash` + `medium` + `group_idx`)是我们控制面构建集群 KV 位置视图的**现成事件源形态**。差异:vLLM 事件只描述**单实例内**的块生命周期(`medium` 是实例内介质),无跨实例坐标;我们需在此基础上补集群级位置视图(或等 #48501 的 `session_id`/`continuation_id` 落地)。我们的存储池元数据更新可参考此事件模型,但权威在池而非引擎事件。

## KV offload 子系统(原生多层 ★)

`vllm/v1/kv_offload/` 是 vLLM **引擎内一等的多层 KV offload 子系统**(独立于 connector,2026 起密集施工)。这是 vLLM 向我们"L1/L2/L3 分层"方向最直接的收敛——但仍是 **per-instance**(跑在该引擎的 Scheduler 进程内,tier 是引擎私有,非集群池)。

### 核心抽象(`base.py`)

- `OffloadKey` = `block_hash + group_idx`(编码为 bytes 避 GC,`make_offload_key`/`get_offload_block_hash`/`get_offload_group_idx`)——**内容寻址**,与我们 `(model_id,layer,block_hash)` 同形,但无前缀树(平铺键)。
- `OffloadingManager`(ABC):`lookup() → LookupResult`(`MISS`/`HIT`/`HIT_PENDING`/`RETRY`)——三态 + pending,对应异步传输未就绪。本系统存储池查询可参考此三态(我们的"Pool 命中待传"≈ `HIT_PENDING`)。
- `ReqContext`(req_id + kv_transfer_params):请求上下文,供 session-aware 驱逐(RFC #45405 的落点)。
- `OffloadPolicy`:`BLOCK_LEVEL`(仅新算块,prefix-hit 块已 offloaded 则跳过)/ `REQUEST_LEVEL`(含前缀命中块,某些 tier 需完整 KV 上下文)。
- `LoadStoreSpec`:每层 KV 的 load/store 形状描述。

### CPU 主层(`cpu/`)

直接访问 GPU,是 GPU↔offload 网关:`SharedOffloadRegion`、驱逐策略 `lru`/`arc`(`policies/{lru,arc}.py`)、`CPUOffloadingWorker`、`swap_blocks_triton`(block 粒度 swap kernel)。

### 二级层(`tiering/`)

`SecondaryTierManager`(ABC):**二级层不能直接访问 GPU**,所有传输经 CPU 主层级联:
- **store**:GPU → CPU(主层)→ secondary(cascade)
- **load**:secondary → CPU(主层)→ GPU(promotion)

异步作业模型:`submit_load`/`submit_store`/`get_finished_jobs`,`JobMetadata`(带 `is_promotion`/`req_context`/`keys`/`block_ids`)、`JobResult`、`JobId`。已实现二级层:`fs/`(文件系统/NVMe)、`obj/`(对象存储)、`example/`;`async_lookup.py` 异步查询;`TieringOffloadingSpec` 组合 CPU 主层 + 可配置 secondary tiers。

**全部跑在 Scheduler 进程,方法须轻量非阻塞**。

### 与 connector 的关系

`OffloadingConnector`/`SimpleCPUOffloadConnector` 是把 `kv_offload` 子系统**暴露为 `KVConnectorBase_V1` 插件**的适配层(`v1/offloading_connector.py`、`v1/simple_cpu_offload_connector.py`)——即 `kv_offload` 是底层能力,connector 是其插件外壳。Mooncake connector 也已扩展为完整目录(`mooncake/`:store/connector + mooncake_connector + rdma_utils + stats),新增 `hf3fs`、`moriio` connector。

**对本系统**:
- **可借鉴**:`OffloadingManager`/`LookupResult`/`OffloadPolicy` 三态查找 + 策略枚举(直接映射我们的"Pool 命中/待传/miss");`OffloadKey` 内容寻址编码;secondary tier 的 cascade/promotion 异步作业模型(`JobMetadata.is_promotion`)——与我们"L2→L1 promotion / L1→L2 demotion"同构。
- **关键差异**:
  - vLLM `kv_offload` **per-instance**(引擎 Scheduler 进程内,tier 私有);我们归**存储池集群权威**,跨节点统一编址 L0–L3。
  - vLLM tier 间是**单实例内级联**(GPU↔CPU↔NVMe/Obj 同机);我们是**跨节点池**(DRAM/NVMe block 放本机还是远端 KV Node 由池放置决定)。
  - vLLM **无 radix**(`OffloadKey` 平铺);我们 radix + 位置视图 + D-direct。
  - vLLM HBM 仍引擎自分配(offload 只借做传输);**方案 Z"池管 HBM 放置"无对应**。
  - vLLM 的"统一管全部分层"是**单实例内**的;我们的 F3"存储池统一管理 L0–L3"是**集群级**的——这是量级差异。

## Worker / Model Runner

| 类 | 文件 | 职责 |
|----|------|------|
| `Worker` | `vllm/v1/worker/gpu_worker.py` (L124) | GPU 进程:`load_model`(L384)、`execute_model`(L932)、分布式环境初始化 |
| `GPUModelRunner` | `vllm/v1/worker/gpu/model_runner.py` (L120) | 加载模型(L280)+ 前向(L1128)、构造 attention metadata、维护 block table、驱动 spec decode |
| `WorkerBase` | `vllm/v1/worker/worker_base.py` (L39) | worker 进程基类 |

**对本系统**:worker 生命周期(`load_model`→`execute_model`→drain)与 block table 维护可直接借鉴。本系统 worker **无状态化**:模型从存储池流式加载、KV 从存储池读写、block table 物理位置由存储池元数据定。

## 权重加载与 offload(权重存算分离原型)

- `DefaultModelLoader`(L43):生产加载器(safetensors/HF),`weight_utils.download_weights_from_hf`/`_prefetch_checkpoint`(L728)做流式预取。
- `vllm/model_executor/offloader/`:`uva.py`(UVA 统一虚址,GPU 直接读 host 内存)、`prefetch.py`——**权重从远端/host 流式喂 GPU 的原型**,正是本系统"权重归存储池、计算层流式加载"的参考。

## Executor / 并行

`Executor`(L37,ABC)拥有 worker 并派发 `execute_model`:单进程 / 多进程 / Ray。`ParallelConfig`(TP/PP/DP)。**对本系统**:多卡编排参考;但本系统 worker 无状态、可独立伸缩,executor 边界会与控制面调度重叠,需重新切分。

## Speculative Decoding

- proposer(draft 侧):`vllm/v1/spec_decode/`(`EagleProposer`/`MedusaProposer`/`DraftModelProposer`/`NgramProposerGPU`)。
- verify(target 侧):`vllm/v1/worker/gpu/spec_decode/speculator.py::BaseSpeculator`/`DraftModelSpeculator` + `rejection_sampler.py::RejectionSampler`。

**对本系统**:proposer↔speculator 划分对应 Draft 池↔Decode 池;draft 候选跨节点传输延迟见 compute-layer 开放问题。

## 代码索引

> 沿代码回溯用。符号名锚定,行号会漂移——找不到时 `grep -n "符号名" 3rdparty/vllm/<文件路径>`。

### KV cache 管理

| 机制 | 文件:符号 |
|------|-----------|
| KV cache 门面 | `vllm/v1/core/kv_cache_manager.py`::`KVCacheManager` (L110) |
| 前缀块查询(scheduler 入口) | `KVCacheManager.get_computed_blocks` (L202) |
| slot 分配 / 释放 / 驱逐 | `KVCacheManager.allocate_slots` / `free` / `evict_blocks` |
| 每请求 block 句柄 | `kv_cache_manager.py`::`KVCacheBlocks` (L26) |
| 每类型组协调器 | `vllm/v1/core/kv_cache_coordinator.py`::`KVCacheCoordinator` (L61) / `KVCacheCoordinatorNoPrefixCache` (L377) |
| 单类型管理器 | `vllm/v1/core/single_type_kv_cache_manager.py`::`SingleTypeKVCacheManager` (L33) |
| block 分配器 + hash→block | `vllm/v1/core/block_pool.py`::`BlockPool` (L144;`cached_block_hash_to_block` L185、`get_cached_block` L199) |
| block 对象 + LRU 队列 | `vllm/v1/core/kv_cache_utils.py`::`KVCacheBlock` (L118) / `FreeKVCacheBlockQueue` (L179) |
| block 哈希 | `kv_cache_utils.py`::`hash_block_tokens` (L577) / `make_block_hash_with_group_id` (L57) / `get_block_hash` (L69) |
| 跨实例外部哈希 | `kv_cache_utils.py`::`maybe_convert_block_hash` (L79,→`ExternalBlockHash`) |
| 多模态/LoRA extra key | `kv_cache_utils.py`::`generate_block_hash_extra_keys` (L539) |
| worker 侧 block table | `vllm/v1/worker/gpu/block_table.py` |

### Attention

| 机制 | 文件:符号 |
|------|-----------|
| 后端抽象基类 | `vllm/v1/attention/backend.py`::`AttentionBackend` (L55) |
| metadata | `backend.py`::`AttentionMetadata` (L386) / `CommonAttentionMetadata` (L394) / `AttentionMetadataBuilder` (L573) |
| 实现(消费 paged KV) | `backend.py`::`AttentionImpl` (L820) / `AttentionImplBase` (L742) / `MLAAttentionImpl` (L903) |
| FlashAttention 后端 | `vllm/v1/attention/backends/flash_attn.py`::`FlashAttentionBackend` (L67) / `FlashAttentionImpl` (L648) |
| Triton attention | `vllm/v1/attention/backends/triton_attn.py` |
| MLA 后端集 | `vllm/v1/attention/backends/mla/`(flashattn_mla/flashinfer_mla/triton_mla/cutlass_mla/flashmla) |
| 后端注册/选择 | `vllm/v1/attention/backends/registry.py` / `fa_utils.py` |

### Scheduler

| 机制 | 文件:符号 |
|------|-----------|
| 调度器 | `vllm/v1/core/sched/scheduler.py`::`Scheduler` (L68) |
| 调度器接口 | `vllm/v1/core/sched/interface.py`::`SchedulerInterface` |
| 请求队列 | `vllm/v1/core/sched/request_queue.py` |
| 调度输出 | `vllm/v1/core/sched/output.py`::`SchedulerOutput` |

### KV Connector(★ 存算分离接入点)

| 机制 | 文件:符号 |
|------|-----------|
| connector 接口基类 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py`::`KVConnectorBase_V1` (L171) |
| 角色(scheduler/worker 侧) | `base.py`::`KVConnectorRole` (L124) |
| connector 元数据 | `base.py`::`KVConnectorMetadata` (L141) / `KVConnectorWorkerMetadata` (L150) / `KVConnectorHandshakeMetadata` (L132) |
| hybrid memory allocator 能力 | `base.py`::`SupportsHMA` (L85) / `request_finished_all_groups` (L92) |
| legacy 别名 | `vllm/distributed/kv_transfer/kv_connector/base.py`(`KVConnectorBase = KVConnectorBase_V1`) |
| worker 侧包装 | `vllm/v1/worker/gpu/kv_connector.py`::`KVConnector` (L29) / `ActiveKVConnector` (L47) |
| 注入 model runner 的 mixin | `vllm/v1/worker/kv_connector_model_runner_mixin.py` |
| LMCache connector | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`::`LMCacheConnectorV1` (L72) |
| Mooncake connector | `…/mooncake/store/connector.py`::`MooncakeStoreConnector` (L87) |
| NIXL connector | `…/nixl/connector.py`::`NixlBaseConnector` (L79) |
| FlexKV connector | `…/flexkv_connector.py`::`FlexKVConnectorV1` (L35) |
| 组合 connector | `…/multi_connector.py`::`MultiConnector` (L128) |
| offload connector | `…/offloading_connector.py`::`OffloadingConnector` (L46) / `…/simple_cpu_offload_connector.py`::`SimpleCPUOffloadConnector` (L45) |
| Mooncake connector(已扩目录) | `…/mooncake/`::`store/connector.py::MooncakeStoreConnector` (L87) / `mooncake_connector.py` / `rdma_utils.py` / `stats.py` |
| 新增 connector | `…/hf3fs/` / `…/moriio/` / `…/offloading/`(子目录带 metrics) |
| KV 布局协商 | `vllm/v1/kv_cache_interface.py`::`KVCacheConfig` (L920) / `KVCacheSpec` (L100) / `FullAttentionSpec` (L206) / `MLAAttentionSpec` (L363) / `MambaSpec` (L669) |

### KV Events(★ 引擎→外可见性)

| 机制 | 文件:符号 |
|------|-----------|
| 事件基类 / batch | `vllm/distributed/kv_events.py`::`KVCacheEvent` / `EventBatch` / `KVEventBatch`(`data_parallel_rank`) |
| 块写入事件 | `kv_events.py`::`BlockStored`(`block_hashes`/`parent_block_hash`/`medium`/`group_idx`/`kv_cache_spec_kind`/`extra_keys`) |
| 块释放事件 | `kv_events.py`::`BlockRemoved` / `AllBlocksCleared` |
| 跨 worker 聚合 | `kv_events.py`::`KVEventAggregator` |
| 配置 | `vllm/config/kv_events.py`::`KVEventsConfig` |
| 事件发射点 | `vllm/v1/core/block_pool.py`::`_build_block_stored_event` / `emit_cached_block_events` / `cache_partial_block` |
| 订阅示例 | `examples/features/kv_events/kv_events_subscriber.py` |

### KV offload 子系统(★ 原生多层)

| 机制 | 文件:符号 |
|------|-----------|
| 内容寻址键 / 请求上下文 / 策略 | `vllm/v1/kv_offload/base.py`::`OffloadKey` / `make_offload_key` / `ReqContext` / `OffloadPolicy`(BLOCK_LEVEL/REQUEST_LEVEL) |
| 管理器 ABC + 查询三态 | `vllm/v1/kv_offload/base.py`::`OffloadingManager`(`lookup`→`LookupResult`) / `LookupResult`(MISS/HIT/HIT_PENDING/RETRY) / `LoadStoreSpec` / `CanonicalKVCaches` |
| 工厂 | `vllm/v1/kv_offload/factory.py` / `vllm/v1/kv_offload/file_mapper.py` |
| CPU 主层 | `vllm/v1/kv_offload/cpu/`::`SharedOffloadRegion` / `CPUOffloadingWorker` / `spec.py::CPUOffloadingSpec` / `policies/{lru,arc}.py` / `swap_blocks_triton.py` / `shared_offload_region.py` |
| 二级层 ABC + 异步作业 | `vllm/v1/kv_offload/tiering/base.py`::`SecondaryTierManager` / `JobMetadata`(`is_promotion`/`req_context`) / `JobResult` / `JobId` |
| 二级层管理器 + spec | `vllm/v1/kv_offload/tiering/manager.py`::`TieringOffloadingManager` / `CPUPrimaryTierOffloadingManager` / `spec.py::TieringOffloadingSpec` / `factory.py::SecondaryTierFactory` |
| 二级层实现 | `vllm/v1/kv_offload/tiering/fs/`(NVMe/文件系统)、`obj/`(对象存储)、`example/` |
| 异步查询 | `vllm/v1/kv_offload/tiering/async_lookup.py` |
| simple offload(独立路径) | `vllm/v1/simple_kv_offload/`::`manager.py` / `copy_backend.py` / `worker.py` / `cuda_mem_ops.py` |

### Worker / Runner

| 机制 | 文件:符号 |
|------|-----------|
| GPU worker 进程 | `vllm/v1/worker/gpu_worker.py`::`Worker` (L124;`load_model` L384、`execute_model` L932) |
| GPU model runner | `vllm/v1/worker/gpu/model_runner.py`::`GPUModelRunner` (L120;`load_model` L280、`execute_model` L1128) |
| worker 基类 | `vllm/v1/worker/worker_base.py`::`WorkerBase` (L39) / `WorkerWrapperBase` (L187) |
| 分布式环境初始化 | `gpu_worker.py`::`init_worker_distributed_environment` (L1296) |

### 权重加载 / offload

| 机制 | 文件:符号 |
|------|-----------|
| 加载器基类 | `vllm/model_executor/model_loader/base_loader.py`::`BaseModelLoader` (L25) |
| 默认加载器 | `vllm/model_executor/model_loader/default_loader.py`::`DefaultModelLoader` (L43) |
| 权重 I/O | `vllm/model_executor/model_loader/weight_utils.py`::`download_weights_from_hf` (L431) / `_prefetch_checkpoint` (L728) |
| 权重 offload(UVA/预取) | `vllm/model_executor/offloader/`::`uva.py` / `prefetch.py` |

### Executor / 并行

| 机制 | 文件:符号 |
|------|-----------|
| 执行器抽象 | `vllm/v1/executor/abstract.py`::`Executor` (L37) |
| 单进程 | `vllm/v1/executor/uniproc_executor.py`::`UniProcExecutor` (L45) |
| 多进程 | `vllm/v1/executor/multiproc_executor.py`::`MultiprocExecutor` (L103) |
| Ray | `vllm/v1/executor/ray_executor.py`::`RayDistributedExecutor` (L64) / `ray_executor_v2.py::RayExecutorV2` (L218) |
| 并行配置 | `vllm/config/parallel.py`::`ParallelConfig` (L117) / `EPLBConfig` (L57) |

### Speculative Decoding

| 机制 | 文件:符号 |
|------|-----------|
| proposer 基类 | `vllm/v1/spec_decode/llm_base_proposer.py`::`SpecDecodeBaseProposer` (L63) |
| proposers | `vllm/v1/spec_decode/`::`eagle.EagleProposer` / `medusa.MedusaProposer` / `draft_model.DraftModelProposer` / `ngram_proposer_gpu.NgramProposerGPU` (L216) |
| 请求级 proposal metadata | `vllm/v1/spec_decode/metadata.py`::`SpecDecodeMetadata` (L10) |
| verify(target 侧)speculator | `vllm/v1/worker/gpu/spec_decode/speculator.py`::`BaseSpeculator` (L31) / `DraftModelSpeculator` (L74) |
| rejection sampling | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`::`RejectionSampler` (L43) |
| target 侧 draft 处理 | `vllm/v1/worker/gpu/spec_decode/utils.py`::`DraftTokensHandler` |

### 核与语言

| 机制 | 位置 |
|------|------|
| C++/CUDA 核(性能路径) | `csrc/`(attention/quantization/moe;经 `vllm/_custom_ops.py` 绑定) |
| Triton kernel | `vllm/kernels/triton/`、`vllm/lora/ops/triton_ops/`、各模型 `ops/` |
| Python 自定义 op 绑定 | `vllm/_custom_ops.py` |

## 本系统的借鉴点

1. **PagedAttention block + block table**:存储池 KV block 与 vLLM block 对齐,worker 仍用 block table 做 paged attention,物理位置由存储池元数据定。
2. **`KVConnectorBase_V1` 接口形态**:本系统存储池 client ≈ 常驻 connector;scheduler/worker 双侧 + metadata 协调 + layer-wise save/load 流水线 mixin 直接参考。
3. **`ExternalBlockHash`**:worker 向存储池查前缀的自然键,与存储池 `(model_id,layer,block_hash)` 内容寻址对接。
4. **`SupportsHMA` 能力标记**:声明 connector 支持 hybrid memory allocator(多 KV cache group 混合架构),要求 `request_finished_all_groups` 与多 group 释放时序对齐。**注意:它不对应方案 Z**——vLLM 的 HBM 始终引擎自分配,connector 只借做传输;方案 Z 的"池管 HBM 放置"是本系统增量,vLLM 无对应标记。
5. **KV Events 事件 schema**(★ 新):`BlockStored`/`BlockRemoved` 携 `ExternalBlockHash` + `medium` + `group_idx`,是控制面构建集群 KV 位置视图的现成事件源形态;`KVEventAggregator` 跨 worker 聚合去重可参考。
6. **`OffloadingManager` / `LookupResult` / `OffloadPolicy`**(★ 新):三态查找(MISS/HIT/HIT_PENDING/RETRY)+ 策略枚举(BLOCK_LEVEL/REQUEST_LEVEL)直接映射我们"Pool 命中/待传/miss";`OffloadKey`(hash+group_idx)内容寻址编码可参考。
7. **secondary tier cascade/promotion 异步作业模型**(★ 新):`SecondaryTierManager` 的 GPU→CPU→secondary 级联 store、secondary→CPU→GPU promotion load、`JobMetadata.is_promotion` 与我们"L2→L1 promotion / L1→L2 demotion"同构。
8. **权重 offloader(UVA/预取)**:权重流式加载原型,参考"权重归存储池、计算层流式喂 GPU"。
9. **`AttentionMetadataBuilder`**:调度器输出→attention 输入的衔接点形态。
10. **spec decode proposer↔speculator**:Draft 池↔Decode 池划分参考。

## 关键差异(我们更彻底)

> 注:vLLM 自 2026 起主动演进(见 [overview.md](overview.md) "KV 大规模管理演进"):原生 `vllm/v1/kv_offload/` 多层 + KV Events 已落地。差距已从"vLLM 完全没有"收窄为"vLLM 有 per-instance 多层 + 事件,但无集群权威"。下列差异按此理解。

- vLLM KV/调度/元数据仍 **per-instance、单实例**(`kv_offload` 跑在引擎 Scheduler 进程内,tier 私有);我们归存储池/控制面**集群权威**。
- vLLM 多层 offload 是**单实例内级联**(GPU↔CPU↔NVMe/Obj 同机);我们是**跨节点池**(block 放本机还是远端 KV Node 由池放置决定)。
- vLLM **无 radix**(APC hash 顺序匹配 + `OffloadKey` 平铺键)、**无集群位置视图/本地命中**(KV Events `medium` 仅单实例介质标记);我们 radix + 位置视图 + D-direct。
- vLLM 跨 session/实例协调(`session_id`/`continuation_id`、P2P KV Events)仍是 **RFC**(#48501/#48203,未落地);我们设计即为集群级。
- vLLM connector 是**可选 per-instance 插件**;我们是**必经集群级路径**。
- vLLM HBM **引擎自分配**(offload/connector 只借传输);我们**池管 HBM 放置**(方案 Z,vLLM 无对应)。
- vLLM attention 主路径 **C++/CUDA**;我们选 **Python + Triton**(自定义核门槛不同)。
- vLLM worker **有状态**(加载模型+HBM KV);我们 **无状态**(状态全剥离,秒级伸缩)。

详见 [3rdparty-reference.md](../3rdparty-reference.md) 的汇总对比。
