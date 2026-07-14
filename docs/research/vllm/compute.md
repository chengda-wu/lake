# vLLM — 计算层抽象与存算分离接入点

> 源码:`vllm/v1/`、`vllm/distributed/kv_transfer/`、`vllm/model_executor/`。本文聚焦**计算层可复用抽象**与**KV connector 接口**——本系统 worker 接入存储池的直接参考。

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

### SupportsHMA(host-managed-address)

`SupportsHMA`(L85,ABC 标记):connector 可声明"host-managed-address"能力——即外部存储直接管理 HBM 地址(host 拥有显存指针),worker 不自己 alloc。**这接近本系统方案 Z**(存储池主动放置 KV 到 HBM,worker 读位置视图)的雏形——vLLM 已为"外部管 HBM"预留了能力标记。

**对本系统**:本系统存储池 client ≈ 一个常驻的 `KVConnectorBase_V1`,且 `SupportsHMA` 路径对应方案 Z 的"存储池放置、worker 消费"。区别:vLLM connector 是可选插件、per-instance;本系统是必经路径、集群级权威。

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
| host-managed-address 能力 | `base.py`::`SupportsHMA` (L85) |
| legacy 别名 | `vllm/distributed/kv_transfer/kv_connector/base.py`(`KVConnectorBase = KVConnectorBase_V1`) |
| worker 侧包装 | `vllm/v1/worker/gpu/kv_connector.py`::`KVConnector` (L29) / `ActiveKVConnector` (L47) |
| 注入 model runner 的 mixin | `vllm/v1/worker/kv_connector_model_runner_mixin.py` |
| LMCache connector | `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`::`LMCacheConnectorV1` (L72) |
| Mooncake connector | `…/mooncake/store/connector.py`::`MooncakeStoreConnector` (L87) |
| NIXL connector | `…/nixl/connector.py`::`NixlBaseConnector` (L79) |
| FlexKV connector | `…/flexkv_connector.py`::`FlexKVConnectorV1` (L35) |
| 组合 connector | `…/multi_connector.py`::`MultiConnector` (L128) |
| offload connector | `…/offloading_connector.py`::`OffloadingConnector` (L46) / `…/simple_cpu_offload_connector.py`::`SimpleCPUOffloadConnector` (L45) |
| KV 布局协商 | `vllm/v1/kv_cache_interface.py`::`KVCacheConfig` (L920) / `KVCacheSpec` (L100) / `FullAttentionSpec` (L206) / `MLAAttentionSpec` (L363) / `MambaSpec` (L669) |

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
4. **`SupportsHMA` 能力标记**:对应方案 Z"存储池放置 KV 到 HBM、worker 消费"的雏形。
5. **权重 offloader(UVA/预取)**:权重流式加载原型,参考"权重归存储池、计算层流式喂 GPU"。
6. **`AttentionMetadataBuilder`**:调度器输出→attention 输入的衔接点形态。
7. **spec decode proposer↔speculator**:Draft 池↔Decode 池划分参考。

## 关键差异(我们更彻底)

- vLLM **KV/调度/元数据进程私有、单实例**;我们归存储池/控制面集群权威。
- vLLM **无 radix**(hash 顺序匹配)、**无位置视图/本地命中**;我们 radix + 位置视图 + D-direct。
- vLLM connector 是**可选 per-instance 插件**;我们是**必经集群级路径**。
- vLLM attention 主路径 **C++/CUDA**;我们选 **Python + Triton**(自定义核门槛不同)。
- vLLM worker **有状态**(加载模型+HBM KV);我们 **无状态**(状态全剥离,秒级伸缩)。

详见 [3rdparty-reference.md](../3rdparty-reference.md) 的汇总对比。
