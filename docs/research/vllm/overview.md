# vLLM — 总览

> 源码:`3rdparty/vllm`(submodule,HEAD ab132ee98)。SOSP'23 PagedAttention 的实现,工业级 LLM 推理引擎。本系统**计算层(Python + Triton)**的直接参考。

## 一句话定位

vLLM 是**高性能 LLM 推理引擎**:以 PagedAttention 块状管理 GPU KV cache、continuous batching 提升吞吐、内容寻址的 prefix caching(APC)做前缀复用,并提供 `KVConnectorBase_V1` 插件接口把 KV cache 与外部存储/传输解耦——后者正是存算分离系统的接入点。

## 与本系统的关系

本系统技术选型计算层 = **Python + Triton**,vLLM 是该选型最成熟的参照。但本系统不做"又一个 vLLM",而是**拆解其架构、提取可复用的计算面抽象,把有状态物(KV/权重/调度)剥离给存储池与控制面**:

| vLLM 概念 | 本系统对应 | 关系 |
|-----------|-----------|------|
| `KVCacheManager` + `BlockPool`(进程内 paged HBM) | 存储池统一管理 L0–L4,HBM 是池的物理载体 | **vLLM 的 KV 管理是进程私有;我们归存储池权威** |
| APC(内容寻址 hash + LRU) | radix tree + 位置视图(存储池强一致) | **vLLM 无 radix、无位置视图、单实例;我们跨节点强一致** |
| `Scheduler`(prefill/decode 组 batch) | 控制面 Router/调度器(Go) | **vLLM 调度器在引擎进程内、单实例视角;我们控制面独立、集群视角** |
| `KVConnectorBase_V1`(外部 KV 插件) | 计算层 worker ↔ 存储池 client | **直接参考:vLLM 的 connector 抽象正是 worker 接入外部存储池的接口形态** |
| `Worker` / `GPUModelRunner`(加载+前向) | 计算层 worker(Prefill/Decode) | **直接参考:worker 生命周期、execute_model、block table 维护** |
| `Executor`(TP/PP 分布) | (未来)计算层多卡编排 | 参考 |
| 权重加载 `DefaultModelLoader` + offloader/UVA | 存储池权重放置 + 计算层流式加载 | **参考:vLLM 的流式加载与 UVA offload 是权重存算分离的原型** |
| Spec decode(Eagle/Medusa/…) | Draft 池 | 参考其 proposer/speculator 划分 |

**核心结论**:vLLM 的**计算面抽象**(paged attention、worker、model runner、connector 接口、spec decode)可直接借鉴;但其**状态管理**(KV/调度/元数据进程私有、单实例)正是我们要剥离的对象。vLLM 自身已通过 `KVConnectorBase_V1` 把"外部 KV"留作插件口——Mooncake/LMCache/NIXL/FlexKV 都是它的 connector——本系统等于把这个口"扶正":KV 不再是插件可选优化,而是存储池一等公民。

详见 [compute.md](compute.md)。

## 设计哲学

- **PagedAttention**:把 KV cache 按固定大小 block 分页(类比 OS 虚拟内存分页),消除碎片,使一个请求的 KV 不必连续,极大提升 HBM 利用率与 batch 规模。这是 vLLM 的立身之本。
- **Continuous batching(Orca)**:不按 batch 边界等齐,每个 iteration 动态加入/移除请求,decode 吞吐最大化。
- **Prefix caching(APC)**:内容寻址地缓存公共前缀(system prompt/few-shot)的 KV block,跨请求复用,免重复 prefill。
- **V1 架构**:当前唯一引擎。旧 `vllm/core/`、`vllm/attention/` 已不存在,全部并入 `vllm/v1/`。无 legacy scheduler。

## 架构(V1)

```
AsyncLLM(引擎入口)
  → Scheduler(vllm/v1/core/sched/)        # 组 prefill/decode batch,单实例
      → KVCacheManager(vllm/v1/core/)      # 进程内 paged HBM + APC
          → BlockPool                      # free-list + hash→block 前缀匹配
  → Executor(vllm/v1/executor/)            # 派发到 worker(单进程/多进程/Ray)
      → Worker(vllm/v1/worker/gpu_worker)  # GPU 进程
          → GPUModelRunner                  # 加载模型 + 前向,维护 block table
              → AttentionBackend(FlashAttn) # paged KV attention
          → KVConnector(可选)              # 外部 KV 插件(LMCache/Mooncake/NIXL…)
```

核心组件:

| 组件 | 文件 | 职责 |
|------|------|------|
| `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` | KV cache 门面:前缀块查询、slot 分配、释放、驱逐 |
| `BlockPool` | `vllm/v1/core/block_pool.py` | block 分配器 + `hash→block` 前缀缓存映射 |
| `Scheduler` | `vllm/v1/core/sched/scheduler.py` | waiting/running 队列组 prefill/decode batch |
| `AttentionBackend` | `vllm/v1/attention/backend.py` | attention 抽象基类 + metadata(消费 paged KV) |
| `KVConnectorBase_V1` | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 外部 KV 插件接口 |
| `Worker` / `GPUModelRunner` | `vllm/v1/worker/gpu_worker.py`、`vllm/v1/worker/gpu/model_runner.py` | GPU 进程:加载模型 + 执行前向 |
| `Executor` | `vllm/v1/executor/abstract.py` | worker 编排与 execute_model 派发 |
| `DefaultModelLoader` | `vllm/model_executor/model_loader/default_loader.py` | 权重加载(safetensors/HF) |

## 技术栈

- **语言**:Python 主体(~1906 `.py`)。**性能关键路径是 C++/CUDA**:`csrc/`(~231 文件,attention/quantization/moe 核),经 `vllm/_custom_ops.py` 绑定。Triton kernel 散布且较少(`vllm/kernels/triton/` 仅 2 文件,另 LoRA/各模型 `ops/`)——**vLLM 的 attention 不是 Triton,是 C++/CUDA(FlashAttention)为主,Triton 是补充**。这正是本系统选"Python + **Triton**"需注意的差异:本系统倾向 Triton 自定义核,vLLM 倾向 C++/CUDA。
- **关键依赖**:torch、xformers/flash-attn、Ray(分布式)、CUDA、Triton。
- **构建**:`pip install -e .`(需 CUDA toolkit);CMake 编译 `csrc/`。

## 代码索引

> 沿代码回溯用。符号名稳定锚定,行号会漂移——找不到时 `grep -n "符号名" 3rdparty/vllm/<文件路径>`。

| 概念 | 文件:符号 |
|------|-----------|
| KV cache 门面 | `vllm/v1/core/kv_cache_manager.py`::`KVCacheManager` (L110;`get_computed_blocks` L202、`allocate_slots`、`free`、`evict_blocks`) |
| 每请求 block 句柄 | `vllm/v1/core/kv_cache_manager.py`::`KVCacheBlocks` (L26) |
| 每类型组协调器 | `vllm/v1/core/kv_cache_coordinator.py`::`KVCacheCoordinator` (L61) / `KVCacheCoordinatorNoPrefixCache` (L377) |
| 单类型管理器 | `vllm/v1/core/single_type_kv_cache_manager.py`::`SingleTypeKVCacheManager` (L33) |
| block 分配器 + hash→block | `vllm/v1/core/block_pool.py`::`BlockPool` (L144;`cached_block_hash_to_block` L185、`get_cached_block` L199) |
| block 对象 + LRU 队列 | `vllm/v1/core/kv_cache_utils.py`::`KVCacheBlock` (L118) / `FreeKVCacheBlockQueue` (L179) |
| 前缀哈希原语 | `vllm/v1/core/kv_cache_utils.py`::`hash_block_tokens` (L577) / `make_block_hash_with_group_id` (L57) / `get_block_hash` (L69) |
| 跨实例外部哈希 | `vllm/v1/core/kv_cache_utils.py`::`maybe_convert_block_hash` (L79,→`ExternalBlockHash`) |
| attention 抽象基类 | `vllm/v1/attention/backend.py`::`AttentionBackend` (L55) |
| attention metadata | `vllm/v1/attention/backend.py`::`AttentionMetadata` (L386) / `CommonAttentionMetadata` (L394) / `AttentionMetadataBuilder` (L573) |
| attention 实现(消费 paged KV) | `vllm/v1/attention/backend.py`::`AttentionImpl` (L820) / `MLAAttentionImpl` (L903) |
| FlashAttention 后端 | `vllm/v1/attention/backends/flash_attn.py`::`FlashAttentionBackend` (L67) / `FlashAttentionImpl` (L648) |
| Triton attention 路径 | `vllm/v1/attention/backends/triton_attn.py` |
| MLA 后端集 | `vllm/v1/attention/backends/mla/`(flashattn_mla/flashinfer_mla/triton_mla/cutlass_mla/flashmla) |
| 调度器 | `vllm/v1/core/sched/scheduler.py`::`Scheduler` (L68) |
| 调度器接口 | `vllm/v1/core/sched/interface.py`::`SchedulerInterface` |
| 调度输出 | `vllm/v1/core/sched/output.py`::`SchedulerOutput` |
| KV connector 接口 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py`::`KVConnectorBase_V1` (L171) |
| connector 角色/元数据 | `base.py`::`KVConnectorRole` (L124) / `KVConnectorMetadata` (L141) / `SupportsHMA` (L85,host-managed-address) |
| worker 侧 connector 包装 | `vllm/v1/worker/gpu/kv_connector.py`::`KVConnector` (L29) / `ActiveKVConnector` (L47) |
| connector 注入 model runner | `vllm/v1/worker/kv_connector_model_runner_mixin.py` |
| 已有 connector 实现 | `vllm/distributed/kv_transfer/kv_connector/v1/`::`lmcache_connector.LMCacheConnectorV1` (L72) / `mooncake/store/connector.MooncakeStoreConnector` (L87) / `nixl/connector.NixlBaseConnector` (L79) / `flexkv_connector.FlexKVConnectorV1` (L35) / `multi_connector.MultiConnector` (L128) |
| KV 布局协商 | `vllm/v1/kv_cache_interface.py`::`KVCacheConfig` (L920) / `KVCacheSpec` (L100) / `FullAttentionSpec` (L206) / `MLAAttentionSpec` (L363) |
| worker 进程 | `vllm/v1/worker/gpu_worker.py`::`Worker` (L124;`load_model` L384、`execute_model` L932) |
| GPU model runner | `vllm/v1/worker/gpu/model_runner.py`::`GPUModelRunner` (L120;`load_model` L280、`execute_model` L1128) |
| worker 基类 | `vllm/v1/worker/worker_base.py`::`WorkerBase` (L39) / `WorkerWrapperBase` (L187) |
| 权重加载基类 | `vllm/model_executor/model_loader/base_loader.py`::`BaseModelLoader` (L25) |
| 默认权重加载器 | `vllm/model_executor/model_loader/default_loader.py`::`DefaultModelLoader` (L43) |
| 权重 I/O | `vllm/model_executor/model_loader/weight_utils.py`::`download_weights_from_hf` (L431) / `_prefetch_checkpoint` (L728) |
| 权重 offload(UVA/预取) | `vllm/model_executor/offloader/`(`uva.py`、`prefetch.py`) |
| 执行器抽象 | `vllm/v1/executor/abstract.py`::`Executor` (L37) |
| 多进程/Ray 执行器 | `vllm/v1/executor/multiproc_executor.py`::`MultiprocExecutor` (L103) / `ray_executor.py::RayDistributedExecutor` (L64) |
| 并行配置 | `vllm/config/parallel.py`::`ParallelConfig` (L117;TP/PP/DP) / `EPLBConfig` (L57) |
| spec decode proposer 基类 | `vllm/v1/spec_decode/llm_base_proposer.py`::`SpecDecodeBaseProposer` (L63) |
| spec decode proposers | `vllm/v1/spec_decode/`::`eagle.EagleProposer` / `medusa.MedusaProposer` / `draft_model.DraftModelProposer` / `ngram_proposer_gpu.NgramProposerGPU` |
| spec decode verify(target 侧) | `vllm/v1/worker/gpu/spec_decode/speculator.py`::`BaseSpeculator` (L31) / `DraftModelSpeculator` (L74) |
| rejection sampling | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`::`RejectionSampler` (L43) |
| C++/CUDA 核 | `csrc/`(attention/quantization/moe;经 `vllm/_custom_ops.py` 绑定) |
| Triton kernel | `vllm/kernels/triton/`(`qkv_padded_fp8_quant.py`)、`vllm/lora/ops/triton_ops/`、各模型 `ops/` |
| block table(worker 侧) | `vllm/v1/worker/gpu/block_table.py` |

## 优势

1. **PagedAttention 工业级成熟** — block 分页消除碎片,HBM 利用率与 batch 规模领先,生产验证。
2. **APC 内容寻址前缀复用** — `hash→block` 跨请求复用公共前缀,免重复 prefill,逻辑清晰。
3. **`KVConnectorBase_V1` 标准化外部 KV 接口** — Mooncake/LMCache/NIXL/FlexKV 均已实现为 connector,证明"外部 KV 池接入"是可行且被验证的抽象。本系统存储池 client 可直接照此接口形态。
4. **V1 架构清晰** — KV 管理/调度/worker/attention 分层明确,worker 与 model runner 解耦,易于抽取计算面。
5. **continuous batching + spec decode 完备** — Orca 式动态 batching;Eagle/Medusa/n-gram 等多 proposer + rejection sampling 全链路。
6. **生态最广** — 模型支持最全、社区最大、集成最多(Mooncake/LMCache/NIXL 都是它的 connector)。

## 劣势

1. **KV/调度/元数据进程私有、单实例视角** — `KVCacheManager`/`BlockPool`/`Scheduler` 全在引擎进程内,无集群级 KV 视图。跨实例复用靠 connector 外挂,非原生。这是与本系统"存储池统一权威"的根本分野。
2. **无 radix tree** — APC 用 hash 顺序匹配(`get_computed_blocks` 逐块查 `cached_block_hash_to_block`),断链即停,无 radix 的灵活前缀匹配;前缀树在引擎外(SGLang)或不存在。
3. **无 KV 位置视图 / 本地命中概念** — block 只在"本机 HBM 或不在",无"前缀 KV 已被放置在某执行节点 HBM 可 D-direct"的跨节点放置元数据。本地性靠 connector 各自实现。
4. **attention 主路径是 C++/CUDA 非 Triton** — 自定义核门槛高、与 Python 计算层选型(Triton)不一致;深度定制 attention 需改 `csrc/`。
5. **无存算分离/弹性原生** — worker 有状态(加载的模型 + HBM KV),崩溃丢 KV(靠 connector 外部备份);扩缩容非秒级。本系统要在此基础上剥离状态。
6. **单实例调度器是瓶颈** — `Scheduler` 单进程,大规模集群需上层分片,非原生分布式调度。

## 与本系统的关键对比

| 维度 | vLLM | 本系统 |
|------|------|--------|
| KV 归属 | 进程私有(单实例 HBM) | 存储池统一权威 L0–L4 |
| 前缀复用 | APC hash 顺序匹配,单实例 | radix + 位置视图,跨节点强一致 |
| 本地命中/D-direct | 无概念 | 前缀 KV 放置在某节点 HBM → D-direct |
| 调度器 | 引擎进程内,单实例 | 控制面独立(Go),集群视角 |
| 外部 KV | `KVConnectorBase_V1` 插件(可选) | 存储池一等公民(必经) |
| attention 核 | C++/CUDA(FlashAttention) | Python + Triton |
| 弹性 | worker 有状态,扩缩慢 | 节点无状态,秒级 |
| Spec decode | 完备(Eagle/Medusa/…) | 参考其 proposer/speculator 划分 |

**本质**:vLLM 是"单实例高性能推理引擎 + 外部 KV 插件口"。本系统把它的**计算面**(paged attention、worker、model runner、connector 接口、spec decode)抽出来,把**状态面**(KV/调度/元数据)剥离给存储池与控制面,并让 connector 接口从"可选优化"升为"存储池接入的必经路径"。vLLM 的 connector 生态(Mooncake/LMCache/NIXL 已实现)印证了这条接入路径可行。详见 [3rdparty-reference.md](../3rdparty-reference.md)。
