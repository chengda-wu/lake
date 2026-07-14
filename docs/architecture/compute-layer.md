# 04 — 计算层

计算层由若干**算力池（compute pool）**组成，每个池是一组同质、可互换的算力节点。节点无状态，可随时销毁/拉起。池是按**角色**划分的逻辑分组(角色由调度器动态分配,非物理固定,见 [`overview.md`](overview.md) / [`execution-modes.md`](execution-modes.md));下文按角色画像描述资源与扩缩特征,物理节点在角色间转换(带权重迁移成本)见"开放问题"。

## 池划分

### Prefill Pool
- 任务：处理长 prompt，产出 KV cache。
- 特征：计算密集（高 FLOPS 利用率），对 HBM 容量敏感（长序列 KV 大）。
- 调度目标：最大化吞吐（batch 大、并行度高），容忍较高 TTFT。
- 产物：KV block → 写入 KV Pool；跨节点模式经存储池传输引擎零拷贝推送（混部/D-direct 本地完成,无需传输）。跨实例 KV 传输的机制（内存注册、pull/publish 异步接口、直传 vs 经池中转、布局转换、RDMA 退化）见 [`kv-cache-pool.md`](kv-cache-pool.md) "跨实例 KV 传输"节——**worker 只调异步传输接口 + fence,字节流不经 worker 进程**。

### Decode Pool
- 任务：逐 token 自回归生成。
- 特征：访存密集（每 token 读全部权重 + 增长 KV），batch 内可共享权重读取。
- 调度目标：最小化 ITL（P99），batch 连续批处理（continuous batching）。
- 输入：从 KV Pool 拉取前缀 KV。

### Draft Pool（投机解码，可选）
- 任务：用小模型快速生成候选 token。
- 特征：算力需求小，可与 Decode Pool 共置或独立。
- 产物：候选 token 序列 → 由 Decode Pool 的 target 模型并行验证。

## 节点生命周期

```
Idle → Boot (镜像拉起) → Warm (向存储池申请放置) → Ready → Serving → Drain → Terminate
```

- **Warm**：向存储池申请把权重放置到本机 L1/L0、热点前缀 KV 放置到 HBM，缩短 Ready 时延。
- **Drain**：停止接收新请求，完成 in-flight；本机 HBM 放置归还存储池（由其保留/下沉/驱逐）。
- **Terminate**：可安全销毁。节点无私有状态——HBM/RAM 中的 KV 本就是存储池的放置副本，销毁仅损失未落 L3+ 的最近增量窗口（F4 续推）。

## 冷启动压缩

冷启动是弹性能力的核心瓶颈，分层处理：
1. **镜像层**：精简容器镜像，运行时与权重解耦（权重不在镜像里）。
2. **权重加载**：由存储池把权重放置到本机 L1/L0（非对象存储直读）；按 layer 流式加载，边加载边可接受请求。
3. **CUDA 初始化**：预初始化的进程池 / 常驻 worker。
4. **KV 预取**：扩容决策做出即由存储池把热点前缀 KV 放置到新节点 HBM（支撑 D-direct）。

目标：从扩容决策到 Ready 接受请求 < 10s（待验证）。

## 资源画像

| 池 | GPU 画像 | 内存画像 | 扩缩触发 |
|----|----------|----------|----------|
| Prefill | 高 FLOPS | 大 HBM（长序列） | 队列长度 / TTFT SLO |
| Decode | 高带宽 | 中 HBM（增量 KV） | QPS / ITL SLO |
| Draft | 低端卡即可 | 小 | 投采命中率 |

## HBM 池化下的入图与 KV 管理

HBM 也归存储池后(见 [`overview.md`](overview.md) / [`kv-cache-pool.md`](kv-cache-pool.md)),计算层面临两个核心问题:**(Q1) 引擎如何入图;(Q2) 引擎如何做 KV 管理**。本节落定两轮讨论的结论。

### Q1 入图

**固定 arena(不上 VA)**:每节点固定基址 KV arena,大小分配给模型后不扩缩容、不跨模型回收物理页。graph 捕获的基址终身不动——入图地基。曾考虑用 CUDA VMM(`cuMemMap`)给池物理超订自由,但当前约束下无该场景,撤回,固定 arena 足够。

**入图三约束**(地址在 capture/replay 间不变):
1. 静态输入 buffer(token/位置等)按 max_bs 预分配。
2. 固定 KV 基址(arena 满足)。
3. 固定地址 block table tensor,每步内容在 graph 外组装后拷进固定地址。

- **decode 走 graph**:固定 batch、每步极轻,graph 主战场。
- **prefill 走 eager**:变长重计算,分段图(`BreakableCudaGraphBackend`/`TcPiecewise`)复杂且收益打折。

**block table 池组装**:由**本地 agent**(in-process,持本地视图镜像)组装并写进引擎固定地址 tensor,非全局池每步 RPC 推表(守 5ms decode 间隔)。组装只需本地 L0 状态(agent 自己放的 slot,本地权威);全局 radix 是镜像,滞后只影响命中率(miss→pull→控制面确认),不影响正确性。

> **in-process agent**:存储层 client 编成 Rust `.so`,经 PyO3 嵌进 worker 进程,共享 worker 的 CUDA context。本地视图镜像(由控制面 etcd watch / gRPC stream 刷新)、L0 free-list、block table 组装、ready/done fence、L0 内存 RDMA MR 注册全在 worker 进程内。"池是分布式系统(控制面+跨节点传输引擎,独立进程)"与"per-worker client in-process"不矛盾,后者是前者在引擎侧的本地触手。

**ready/done 双 fence 一步契约**:

```
池侧(step 前): 定 read set/write set → 保证 read set 在 L0(缺则补拉)
                 → 给 write set 分配空闲 slot(满则驱逐冷块)→ 冻结被引用 slot(ref>0)
                 → 组装完整 block table → 发 ready(fence)
引擎侧(step 中): 拷 block table 进固定地址 tensor → replay graph → 新 token KV 写进 write slots
池侧(step 后):   引擎发 done(compute fence)→ 池解冻 → 完整 block 写回 L3 + 注册 radix
                 → 驱逐冷块 → 回收已结束请求 slot
```

引擎的全部分层职责:**消费 ready → 算 → 发 done**。零 load_stream、零 `wait_event`、零 evict/write-back 逻辑。

**无 intra-step 重叠**:不照搬 SGLang HiCache "算 layer N 传 layer N+1" 的逐层重叠——那是引擎自己背分层控制器(load_stream + per-layer `wait_event`)才需要的。我们把补拉/传输整个推出引擎,引擎只调**异步传输接口 + fence**,重叠是异步的自然结果(传 step N+1 的 block 时引擎在算 step N),无需专门设计。SGLang 把"补拉与 graph 冲突"留作 TODO(`scheduler.py:2999`),我们因解耦而无此问题。跨实例传输见 [`kv-cache-pool.md`](kv-cache-pool.md) "跨实例 KV 传输"节。

**正确性地基:in-flight 跨层冻结**。graph 保证"地址不变",但不保证"地址内容不被池动掉"。池若在 replay 途中迁移/驱逐/压实一个 in-flight block,图读到半旧半新。故 step 期间被引用 block(ref>0)的物理映射冻结,step 之间池完全自由。ref 细则见 [`kv-cache-pool.md`](kv-cache-pool.md) "引用计数与驱逐"。

### Q2 KV 管理

**(1) block 对引擎纯寻址单位**。引擎的 KV 操作只剩三原语:读 ready block / 写 token 进 slot / publish 产出。block table 的索引填充都归池(本地 agent),引擎只 replay 读。引擎连"block 满没满"都不感知,只感知"写第 i 个 token 进某 slot"——满块判断、哈希、radix 注册全归池。block 是引擎的寻址单位,不是管理单位。

**(2) 写回:满块路 + 尾块路**(详见 [`kv-cache-pool.md`](kv-cache-pool.md) "写回与生命周期"):
- 满块路:block 填满 → 池算哈希 → 注册 radix → 写回 L3(持久点)。请求进行中就可能触发。
- 尾块路:请求结束时未满的尾块,在请求结束点写回一次(写全部已填 token,重放整块覆盖),纯容错不进 radix。
- 满块写回频率 N(满一个就写 vs 攒几个)留 P7。尾块只在请求结束写一次,无增量式。

**(3) ref 池权威维护**(详见 [`kv-cache-pool.md`](kv-cache-pool.md) "引用计数与驱逐"):多引擎共享前缀 block 时,引擎各自持本地计数会致分布式不一致,故 ref 是池单点计数。引擎只通过 read set/write set 间接表达引用,不持计数。ref>0 即冻结,来源有二:请求引用(请求结束且无续推引用才减;F4 续推时 ref 转移到新请求)+ 在途传输引用(传输发起 +1、完成 -1,源端冻结)。

**(4) 权重对称性**:权重也是池管、也被 graph 捕获、也要 in-flight 冻结,机制与 KV 一致(只读、跨请求共享、不写回)。本节不展开,见"冷启动压缩"。

### 参考实现与关键差异

> 按 CLAUDE.md 强制查阅规则。

- **vLLM**:`_allocate_kv_cache`(固定 `torch.zeros` 基址)+ `bind_kv_cache`(烧 data_ptr 进 kernel)+ `BlockTables`(固定地址 input_block_tables + 每步 gather)+ `CudaGraphManager.capture/run_fullgraph` + `resolve_cigraph_mode_and_sizes`(graph 降级条件)+ `KVConnectorBase_V1`(存算分离接入点)+ `ExternalBlockHash`(只对完整 block 算哈希)。见 [`../research/vllm/compute.md`](../research/vllm/compute.md)。
- **SGLang**:`memory_pool.py::MHATokenToKVPool`(L1 固定 arena + post-capture VA 原地 back,我们不取物理超订语义)+ `radix_cache.py::TreeNode`(节点记三层位置)+ `hiradix_cache.py::match_prefix`(L1前缀/L2后缀切分)+ `pool_host/mha.py::get_page_buffer_meta`(page-first 零拷贝裸指针)+ `cache_controller.py::LayerDoneCounter`/`LayerLoadingEvent`(三缓冲,我们不照搬)+ `transfer.cu::transfer_kv_per_layer_pf_lf`(layer-first↔page-first 转换核,池侧照用)。见 [`../research/sglang/{overview,hicache}.md`](../research/sglang/)。
- **关键差异**:vLLM/SGLang 引擎既拥有 KV 又发起传输(engine-to-engine connector 握手,知道地址);我们 engine-to-engine 控制链切断,池 agent 发起,引擎降到 publish/pull+fence、不知地址、不组装 block table。wire 效率不变(直连 RDMA),变的是控制权归属——"彻底存算分离"在传输/入图面的落点。

### TP

proto 预留 per-rank 字段,实现单卡先行。MLA 多 rank 回写去重(SGLang 设计 doc 提及)留作后续参考。

## 开放问题

- Prefill/Decode 比例随流量变化，是否支持节点在池间动态转换？（带权重迁移成本）
- continuous batching 与 KV 跨节点迁移如何协同（迁移中的 sequence 如何处理）？
- 投机解码的 draft 与 target 在物理分离时，候选传输延迟是否抵消收益？
