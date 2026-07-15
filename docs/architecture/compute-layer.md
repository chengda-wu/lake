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

### Draft Pool(投机解码,可选)
- 任务:用 drafter(MTP/EAGLE 一类)快速生成候选 token,target 模型并行验证。
- 特征:drafter 算力需求小;**默认与 Decode(target)共置**(仿 SGLang:drafter 在 target 之后、同节点同 step 执行,见下"投机解码"节),不强制独立 Draft 池。
- 产物:候选 token 序列 → 由同节点 target 并行验证;独立 Draft 池为可选(物理分离时 draft 候选传输延迟可能抵消收益,见"开放问题")。

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

**权重 in-flight pin**：权重归池管、也被 graph 捕获（Q2 权重对称性），机制与 KV 一致——只读、跨请求共享、in-flight 冻结。具体：节点进入 Serving 后,其 L0/L1 上的权重副本在节点生命周期内被池**钉住**(pin,ref>0 等效),池不得在节点 serving 期间迁移/驱逐/压实该权重副本(graph 捕获的是固定基址,迁移会破图)。Warm 期流式加载期间已加载的 layer 同样 pin(边加载边可接受的请求其权重不可动)。节点 Drain/Terminate 才解 pin、权重副本可被池回收/下沉。区别于 KV:权重不可变、无需写回(靠 revision 缓存失效),pin 仅是"serve 期间别动这块只读副本"。

目标：从扩容决策到 Ready 接受请求 < 10s（待验证）。

## 资源画像

| 池 | GPU 画像 | 内存画像 | 扩缩触发 |
|----|----------|----------|----------|
| Prefill | 高 FLOPS | 大 HBM（长序列） | 队列长度 / TTFT SLO |
| Decode | 高带宽 | 中 HBM（增量 KV） | QPS / ITL SLO |
| Draft | 低端卡即可 | 小 | 投采命中率 |

## KV 类型:t-type / r-type

两类 KV **复用条件一致**(都需命中全部前缀才能复用),区别**仅在 HBM 存储形态**——r-type 用紧凑表示降低 HBM 占用(详见 [`storage-layer.md`](storage-layer.md) "KV 类型"节,block 组织与复用见 [`kv-cache-pool.md`](kv-cache-pool.md) "t-type / r-type"):

| 类型 | HBM(L0)存储形态 | 占用 | 典型算子 |
|------|-----------------|------|----------|
| **t-type** | 逐 token 完整 KV,paged block table(128-token block) | 随序列线性增长 | full attention、MLA |
| **r-type** | 紧凑表示(窗口最近 W token / 定长 recurrent state) | 亚线性或常量 | sliding window、Mamba、卷积类 |

- HBM 两类并存,引擎按类型分 arena / 管理器(t-type block arena + r-type 状态 arena):参考 vLLM `SingleTypeKVCacheManager × N` + `KVCacheSpec`(`FullAttentionSpec`/`MLAAttentionSpec`/`MambaSpec`);SGLang multi-pool `PoolName`(Mamba/SWA/DSA/Draft)+ `PoolHitPolicy`。**区分的唯一目的是减少 r-type 的 HBM 占用**。
- **入图影响(Q1)**:t-type 走固定 KV arena + 固定地址 block table(已定);r-type 另设**固定状态 arena**(窗口/状态槽基址入图),block table 语义不适用——按 request 槽寻址。两类 arena 独立 capture,graph 分别 replay。
- **下层不区分**:L1–L4 按 128-token block 统一组织,两类复用条件本就一致(全前缀命中),r-type 落下层在 block 边界 checkpoint 紧凑状态(trailing pages / state 快照)。

### r-type SWA 前缀复用的尾段重算优化(idea,暂不实现)

**动机**:进一步降存储——SWA 的 KV 只在窗口内有意义,不必把前缀的 SWA KV 落 L1+ 持久(本就计划 trailing only);prefix 命中复用时再重算一小段把 SWA 窗口填回来,以一小段重算换存储节省。

**设定**:每层含一个 SWA 模块,窗口大小 w,合计 n 层。prefix 匹配**不考虑 SWA**(只按非 SWA 模块内容寻址匹配),最终匹配长度 L。

**重算量**:需重算匹配序列最后 `n*w - n + 1 = n*(w-1)+1` 个 token,即 position 在 `[L+n-n*w-1, L)` 的段。推导:顶层 SWA 在 position L-1 需本层前 w 个 token 的 SWA KV;逐层回溯每向下多需 w-1 个 token(SWA 窗口在层间圆锥展开),n 层合计 `n*(w-1)+1`。

**关键优化**:该段重算时**仅刷新 SWA 模块的 KV cache,不刷新其他模块的 cache**(通过 `slot_mapping=-1` 之类机制跳过非 SWA 模块的 KV 写)。非 SWA 模块的这段 KV 已在匹配前缀中(命中复用),只补 SWA 窗口。

**存储取舍**:SWA KV 不落 L1+ → 省存储(r-type SWA 不持久);代价是每次 prefix 命中多一段尾段重算(`n*(w-1)+1` token,仅 SWA 模块计算)。n、w 较大时该段可能 > 一个 block(128),属计算开销,非架构约束。

**暂不实现的预留评估**:若现在不实现,需在以下处留接口/语义预留,否则未来改动成本大:

1. **引擎 write-set 需可按模块表达(per-module write mask)** ★主要预留点:Q2 现把写定为"写第 i 个 token 进某 slot"(block 粒度、全模块);未来要支持"本段重算只写 SWA 模块、跳过非 SWA"。**预留**:KV 写接口的 write-set 按 `(module_type, token_range)` 表达,而非整 block 全模块。
   - **与 Q2 极简契约的张力**:Q2 定了"block 对引擎纯寻址单位、引擎连满块都不感知、零分层逻辑"。让引擎按模块跳写,等于把模块类型意识渗进引擎,破坏该极简契约。
   - **备选(下沉到池侧 agent,引擎仍无感)**:不把 per-module write mask 暴露给引擎,而是由池的本地 agent 吸收——两种做法:
     - **a. 写后丢弃**:引擎照常全模块写 step;agent 在 `done` 后按模块类型丢弃非 SWA 模块的写(只保留 SWA slot 内容,非 SWA slot 标记为未写/归还)。引擎无感,代价是一次多余的非 SWA 写 + agent 侧按模块裁剪。
     - **b. 预分配只给 SWA slot**:刷新重算段只对 SWA 模块分配 write slot、非 SWA 模块不分配(引擎写到不存在的 slot 即跳过,或 slot_mapping 在 agent 组装时就置 -1)。引擎仍只"写第 i 个 token 进某 slot",感知不到模块类型——模块意识全在 agent 的 slot 分配策略里。
   - **倾向**:选 b——模块意识留在池 agent(本就管 slot 分配/状态 arena),引擎契约不破。预留点 1 因此弱化为"agent 的 slot 分配需可按模块类型差异化(只给 SWA 分 write slot)",而非"引擎 write-set 按模块表达"。引擎接口不变,Q2 契约保住。
2. **残差 prefill 的重算范围需可延伸进已匹配前缀**:时序一/D-direct 的残差 prefill 现定义为"对未匹配尾部做增量 prefill";本优化要求重算一段**已匹配前缀的尾部**(刷 SWA,非未匹配段)。**预留**:执行路径区分"增量 prefill(未匹配尾)"与"刷新重算(已匹配尾,仅 SWA 模块写)",后者不重写非 SWA KV。见 [`execution-modes.md`](execution-modes.md) 时序一预留。
3. **模型注册 schema 预留模块类型布局 + SWA 窗口大小**:哪些层/模块是 SWA、各窗口 w——r-type arena 管理本就需要;确保注册项含 `module_type_layout` + `swa_window`,重算量公式据此算。已由 t-type/r-type 设计隐含,显式登记即可。
4. **r-type SWA 是否落 L1+ 留开放**:本优化假设**不落**(改重算)。若将来决定落 L1+(trailing pages 持久),则本优化无意义、直接命中。两条路线二选一,待 P7 存储成本 vs 重算成本权衡。见 [`kv-cache-pool.md`](kv-cache-pool.md) 开放点。

**结论**:**暂不实现可行**,需落实预留 1、2(接口语义层),3、4 已被现有设计覆盖/留开放。预留 1 经张力权衡后弱化为"agent 的 slot 分配按模块差异化"(选备选 b,引擎契约不破),预留 2 为"残差路径区分增量 prefill 与刷新重算"。预留成本小、不破坏 Q2 极简契约,不实现不影响当前 t-type/r-type 与命中复用的正确性。

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

**无引擎驱动的 intra-step 重叠;池驱动异步重叠保留**:不照搬 SGLang HiCache "引擎在 `get_key_buffer` 每层 `wait_event`、算 layer N 传 layer N+1" 的**引擎驱动**逐层重叠——那套绑死引擎、破坏 graph(SGLang 把补拉与 graph 冲突留作 TODO `scheduler.py:2999`,我们因解耦而无此问题)。我们拒绝的是**引擎驱动**的 intra-step 重叠。**池驱动异步重叠保留**:引擎只调**异步传输接口 + fence**,传输由池 agent 在独立 stream 做,引擎无感、graph 安全——消费侧 step 间重叠(传 step N+1 时算 step N)+ 生产侧层级重叠(A prefill 逐层 publish page 切片,时序二正向"与 A 计算重叠",支撑 PD 分离 TTFT)。生产侧层级重叠靠 `page_first_direct` 子块传输(分块流水线),详见 [`kv-cache-pool.md`](kv-cache-pool.md) "分块流水线"。

**正确性地基:in-flight 跨层冻结**。graph 保证"地址不变",但不保证"地址内容不被池动掉"。池若在 replay 途中迁移/驱逐/压实一个 in-flight block,图读到半旧半新。故 step 期间被引用 block(ref>0)的物理映射冻结,step 之间池完全自由。ref 细则见 [`kv-cache-pool.md`](kv-cache-pool.md) "引用计数与驱逐"。

### Q2 KV 管理

**(1) block 对引擎纯寻址单位**。引擎的 KV 操作只剩三原语:读 ready block / 写 token 进 slot / publish 产出。block table 的索引填充都归池(本地 agent),引擎只 replay 读。引擎连"block 满没满"都不感知,只感知"写第 i 个 token 进某 slot"——满块判断、哈希、radix 注册全归池。block 是引擎的寻址单位,不是管理单位。

**(2) 写回:满块路 + 尾块路**(详见 [`kv-cache-pool.md`](kv-cache-pool.md) "写回与生命周期"):
- 满块路:block 填满 → 池算哈希 → 注册 radix → 写回 L3(F4 恢复点,抗 worker 失败)。请求进行中就可能触发。
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

## 投机解码

本节落定投机解码的执行模型(仿 SGLang)、MTP/EAGLE 机制、drafter cache 与 seed hidden states 的管理,以及 PD 分离下 MTP 的左 pad 问题。

### 支持范围(lake 主攻)

面向实际生产,lake 主要考虑以下方案,不追求 vLLM 那样的全谱系覆盖:

| 方案 | 类别 | 说明 |
|------|------|------|
| **MTP** | 自回归 draft head | 主流模型自带 MTP 层(next-N 预测),当前主力 |
| **EAGLE / EAGLE3** | 特征级自回归 draft | 复用 target hidden states 的轻量 drafter,EAGLE3 用多层 aux hidden |
| **DFLASH** | diffusion 类 | 半年内可能进生产;draft 侧保留 target-token 滑动窗口(近 max 左 pad 对齐) |
| **DSPARK** | diffusion 类(self-drafting) | 半年内可能进生产;block-wise 并行 draft(gamma 块)+ ragged verify,DeepSeek-V4 self-draft |

**不主攻**:medusa、mlp_speculator、ngram/suffix-decoding、独立 draft 模型(standalone)——非当前生产诉求。选型可后续按需扩展(执行模型的 drafter-after-target 契约与 drafter cache 管理对上述方案通用)。

> **不参考 vLLM `spec_target_max_model_len`**:那是独立 draft 模型时代的历史遗留(draft 模型继承 target max_len)。主流模型自带 MTP 层、无独立 draft 模型,该设计不适用;lake 的 `runner_max_model_length` 是计算层 arena headroom(见"长度边界规避"),与之无关。

### vLLM / SGLang 支持梳理

> 按 CLAUDE.md 强制查阅规则。源码回溯:vLLM `vllm/v1/spec_decode/` + `vllm/config/speculative.py::SpeculativeMethod`;SGLang `python/sglang/srt/speculative/` + `spec_info.py::SpeculativeAlgorithm`。

| 方案 | vLLM | SGLang | 备注 |
|------|------|--------|------|
| **MTP** | ✓ 作 eagle 家族(`config/speculative.py::MTPModelTypes` 列 18+ 变体:deepseek_mtp/qwen3_next_mtp/glm4_moe_mtp/…) | ✓ `NEXTN`(别名→EAGLE)+ `FROZEN_KV_MTP`(`frozen_kv_mtp_worker_v2.py`) | 两边都把 MTP 归到 eagle 基础设施;vLLM 按模型枚举 MTP 变体,SGLang 用 NEXTN 别名 |
| **EAGLE** | ✓ `spec_decode/eagle.py::EagleProposer` | ✓ `eagle_worker_v2.py` + `multi_layer_eagle_worker_v2.py` | SGLang 另有 multi-layer eagle |
| **EAGLE3** | ✓ `eagle3` + `eagle_aux_hidden_state_layer_ids` | ✓ `spec_info.py::is_eagle3` + sliding-window drafter(Llama EAGLE-3) | 都支持多层 aux hidden |
| **DFLASH** | ✓ `spec_decode/dflash.py::DFlashProposer` | ✓ `dflash_worker_v2.py` + `dflash_info_v2.py`(draft 侧 target-token 滑窗,左 pad 对齐) | 两边都有 |
| **DSPARK** | ✗ 无 | ✓ `dspark_components/`(gamma 块 draft + ragged verify + SPS/STS 校准,DeepSeek-V4 self-draft) | **仅 SGLang**——DSPARK 只能参考 SGLang |
| medusa | ✓ `medusa.py` | ✗ | lake 不主攻 |
| mlp_speculator | ✓ | ✗ | lake 不主攻 |
| ngram / suffix | ✓ `ngram_proposer{,_gpu}.py` / `suffix_decoding.py` | ✓ `ngram_worker.py` / `cpp_ngram/` | lake 不主攻 |
| standalone draft model | ✓ `draft_model.py::DraftModelProposer` | ✓ `standalone_worker_v2.py` | lake 不主攻(独立 draft 模型) |

**对 lake 的参考取向**:
- **MTP / EAGLE / EAGLE3 / DFLASH**:两边都有,交叉参考。执行编排(drafter-after-target、hidden states 复用)以 SGLang 为主(我们仿 SGLang 共置串行),proposer↔speculator 划分参考 vLLM。
- **DSPARK**:**只有 SGLang 有**,是 diffusion 类 self-drafting 的唯一参考——`dspark_components/`(config/planner/block_accept_estimator/kernels)+ `dspark_worker_v2.py` + ragged verify。gamma 块并行 draft(默认 gamma=7)+ verify window = gamma+1,与我们"drafter 一次产多 token"的宽度语义一致,但 diffusion 的并行 draft 与 MTP 的自回归 draft 在 hidden states 依赖/verify 上不同,接入时需区分。
- **通用性**:lake 用 **drafter 的 `post_forward` / `pre_forward` 二阶段**(pre/post 共用同一 drafter 模型,同类的两个方法)统一自回归类(MTP/EAGLE/EAGLE3)与 diffusion 类(DFLASH/DSPARK)的编排(见下"执行编排")——`post_forward` 吃 target 输出做强耦合部分(自回归的 draft head 前向 / diffusion 的 cache 准备),`pre_forward` 产 draft token(自回归多 token / diffusion 并行 block)。**drafter 自己的 KV 与 target KV 同款——进存储池统一管理、跨请求前缀复用、随请求迁移**(SGLang `PoolName.DRAFT` 即此);**seed hidden states / diffusion 窗口状态**是否跨请求缓存待定,**先按 SGLang 重算式(draft-extend)推演**(见下"drafter cache 与 seed hidden states")。不改存算分离边界。

### 执行编排:drafter 的 post_forward / pre_forward 二阶段

drafter 与 target(主模型)**同节点、同 step 串行**(仿 SGLang,非并行独立池)。**pre 与 post 共用同一个 drafter 模型**——它们不是两个独立组件,而是**同一 drafter 类的两个前向阶段**(方法),按"相对 target 的时间位置 + 是否依赖 target 本步输出"划分:

| 阶段 | 位置 | 职责 | 承载 |
|------|------|------|------|
| **`post_forward`** | target **之后**(同 step) | 吃 target 本步输出(hidden states/KV),做**与主模型行为强耦合**的部分 | MTP/EAGLE/EAGLE3 的 draft head 前向(参数与主模型一致的那部分);DFLASH/DSPARK 的 **draft cache 准备**(从 target 输出构建 draft 侧所需的窗口/状态) |
| **`pre_forward`** | target **之前**(下一轮 step) | 产出实际 draft token,供本轮 target 并行验证 | MTP/EAGLE/EAGLE3 的**自回归多 token 生成**;DFLASH/DSPARK 的 **draft 生成**(diffusion 并行产 block) |

**框架落点**:一个 `Drafter` 类持有 draft 模型(MTP head / EAGLE / DFLASH / DSPARK 各自实现),暴露 `post_forward()` 与 `pre_forward()` 两个方法;二者共享同一份模型权重与 draft 侧状态(hidden states / 窗口 / block 状态),只是执行时机不同(post 紧接 target、pre 在下轮 target 前)。不拆成两个类/进程。

**划分依据**:`post_forward` 是"消费 target 输出、与主模型强绑定"的一段(必须紧接 target、读其 hidden states/KV);`pre_forward` 是"产 draft token、喂下一轮 target"的一段(时间上前置于下轮 target)。两阶段之间隔着 target 的一次前向。

```
step N:
  [target]              前向 → 产出/验证 token + KV + hidden states
  [drafter.post_forward] 吃 target 本步输出:
                           - MTP/EAGLE/EAGLE3:draft head 前向(与主模型一致的计算)
                           - DFLASH/DSPARK:准备 draft cache(窗口/状态)
step N+1:
  [drafter.pre_forward]  产 draft token(放 target 之前):
                           - MTP/EAGLE/EAGLE3:自回归生成 k 个 draft token
                           - DFLASH/DSPARK:diffusion 并行生成 draft block
  [target]              一次前向并行验证 draft → 接受若干 + 延伸 1 token → 回到 post_forward
```

- **为何分两阶段而非"drafter 一坨在 target 后"**:MTP 的自回归多 token 生成本身要迭代多步(每步吃上一个 draft token),把"依赖 target 输出的第一步(`post_forward`)"与"纯 draft 自迭代(`pre_forward`)"分开,便于:① 编排上 post 紧贴 target 复用其 hidden states/KV,pre 在下轮 target 前批量做;② diffusion 类(DFLASH/DSPARK)天然是"post 准备 cache + pre 并行生成",二阶段对它们同样成立,统一编排框架。两阶段同模型,故为方法而非独立组件。
- **对应参考**:vLLM `EagleProposer`/`MedusaProposer`(proposer 产 draft)+ `BaseSpeculator`(target 并行验证)+ `RejectionSampler` 的 proposer↔speculator 划分,大体对应我们的 `pre_forward`↔target-verify;但 vLLM 未显式区分 post/pre,把 draft head 前向与自回归生成揉在 proposer 内。我们分两阶段是为统一 MTP(自回归)与 diffusion(并行)两类的编排(见下"diffusion 类")。区别在物理编排——vLLM proposer 可独立进程,我们默认共置(下文)。

### prefill / decode 阶段差异(遗留问题,不影响整体设计)

**prefill 阶段仍产 draft**(drafter forward 照跑,`post_forward` + `pre_forward` 均执行)——差异只在**产出是否被使用**:

- **vLLM**:PD 分离部署下,prefill(P 节点)产出的 draft **不使用**(丢弃);drafter 的 prefill forward 主要用于**保持 drafter KV cache 与 target 同步**(参考 `vllm/v1/spec_decode/llm_base_proposer.py:567` 注释 "The prefill forward pass above already ran to keep the drafter KV cache in sync";`num_speculative_tokens==0` 时返回空 draft 但 forward 已跑)。
- **SGLang**:此细节暂未细究(不熟),待接入时按 `3rdparty/sglang/python/sglang/srt/speculative/` 源码回溯确认。

**判定**:是否在 prefill 期用 draft、PD 分离下 P 侧 draft 的取舍,**记为遗留问题**;初步判断**不影响整体设计**——两阶段编排(`post_forward`/`pre_forward`)对 prefill/decode 统一,是否使用 prefill 产出属节点侧策略,可后置。decode 阶段两阶段都跑且产出必用,是投机主循环(见上时序)。

### 自回归类(MTP/EAGLE/EAGLE3):一层 head 自回归产多 draft

drafter 用**一层 draft head**(MTP/EAGLE head,接续/共享 target 的 embedding 与最后一层 hidden;EAGLE3 用多层 aux hidden),以 target 末 token 的 hidden states 为种子,**自回归**生成多个 draft token(比独立小模型省算力、与 target 同分布)。`num_mtp_layers` = 一次产出的 draft token 数(也即 hidden states 需保留的最近 token 数,见下)。

映射到二阶段:
- **`post_forward`**:draft head 吃 target 末 token hidden states 做第一步前向(与主模型参数一致的计算)。
- **`pre_forward`**:在此基础上自回归迭代出剩余 draft token,放到下轮 target 之前。

### 多层 MTP(chain / non-chain,decode 阶段更复杂)

decode 阶段若用**多层 MTP**(多个 MTP layer 串联产更多/更准 draft),`pre_forward` 的自回归不再是"一层 head 迭代",而是逐层推进,hidden states 的来源有两种范式(参考 SGLang `multi_layer_eagle_worker_v2.py::chain_mtp_hidden_states`):

| 范式 | 每层 hidden states 来源 | hidden states 需求 |
|------|------------------------|--------------------|
| **chain-style MTP** | 每层用**自己上一步的输出 hidden states** 作下一步输入(层间链式传播) | 需 FULL(全 token hidden states),开销大 |
| **non-chain** | 每层都用 **target 模型的 hidden states** | 只需 LAST(target 末 token hidden states) |

- SGLang 据 `draft_arch` 判定(如 `Step3p5MTP` 走 chain);chain 需 `post_forward` 保留更全的 hidden states(FULL),non-chain 只需 LAST——直接影响 seed hidden 的窗口大小(见下"drafter cache 与 seed hidden states")。
- chain-style 的层间链式传播使 `pre_forward` 内部状态依赖更复杂(每步输出喂下一步),窗口不止"target 末 token hidden states",还含各中间层的输出;接入多层 MTP 时按 chain/non-chain 区分窗口策略。
- 约束参考:SGLang 多层 EAGLE 要求 `speculative_num_draft_tokens == num_layers`(层数与 draft token 数绑定),`enable_multi_layer_eagle` 与部分特性不兼容——接入时对齐这类约束。
- 当前先记范式,细节接入多层 MTP 时落(单层 MTP 是 non-chain 的退化)。

### drafter cache 与 seed hidden states

投机的暂存物分**两类**,管理方式不同(此前误记为"一律 L0-only 不进池",已纠正):

**1. drafter 自己的 KV(draft head/model 的 KV cache)——进池,与 target KV 同款**
- 按 token 组织、进存储池统一管理(放置/迁移/生命周期归池),**跨请求前缀命中即可复用**、随请求迁移。复用条件与 target KV 一致(全前缀命中)。
- 参考 SGLang `hicache_storage.py::PoolName.DRAFT`——drafter KV 作与 `PoolName.KV` 并列的一等池,HiCache 存取/预取/跨请求复用。命中后残差区间由 **draft-extend** 前向补齐 drafter KV(`eagle_worker_v2.py` 的 `draft_extend_attn_backend` / `eagle_draft_extend_cuda_graph_runner`)。

**2. seed hidden states(target 末 `num_mtp_layers` 个 token 的 hidden,喂 draft head 起自回归)——请求内滚动窗口,是否跨请求缓存待定**
- 每步滚动保留最近 `num_mtp_layers` 个 token 的 hidden(单层 / non-chain 多层:LAST;chain-style 多层:FULL + 各中间层输出,见上"多层 MTP")。作用:请求内继续下一步自回归 + 迁移后 resume。
- **是否进池跨请求复用 = 待定设计,先按 SGLang 重算式推演**:hidden **不**跨请求缓存,走请求内 `spec_info` 传递;前缀命中后需要的边界 hidden 由 draft-extend(`post_forward`)重算补出,不持久化 hidden(省存储,代价一小段重算)。
  - 备选(你原始"多存储 hidden"方向):按 token 把 hidden 也存池,命中时连边界 hidden 一起取回,省重算但存储开销大(hidden ≈ 每 token 一份 hidden_dim,而实际只需边界)。
  - **记为遗留问题**(见"开放问题")——请求迁移是未来特性,此处只保证框架推演不因选择而需大重构。

### diffusion 类(DFLASH/DSPARK):并行产 draft block

DFLASH/DSPARK 是 diffusion 风格投机,与自回归类的核心差异:**不逐 token 自回归,而是一次并行产出一个 draft block(多 token 同时生成)**。半年内可能进生产,当前先把编排位置定清(细节接入时按方案落)。

映射到二阶段(与自回归类共用同一 drafter 类的两个方法):
- **`post_forward`**:吃 target 本步输出,**准备 draft cache**——DFLASH 维护 draft 侧的 target-token 滑动窗口(近 max 时 paged 后端可能左留一页对齐);DSPARK 准备 self-draft 所需的 block 状态(gamma 块规划、Markov head 状态)。这段与主模型输出强耦合,故放 post。
- **`pre_forward`**:**并行生成 draft block**——DSPARK 一次产 gamma 个 draft token(默认 gamma=7,verify window = gamma+1),DFLASH 按 block size(verify window)并行产;放下轮 target 之前,target 用 ragged verify 一次验证整块。

与自回归类的差异点(接入时需区分):
- **draft 侧状态**:自回归类是 seed hidden states(逐 token 滚动窗口);diffusion 类是 **draft 侧窗口/block 状态**(DFLASH 滑窗、DSPARK gamma 块 + Markov 状态)。**drafter KV 部分同 target KV 进池复用**;seed hidden / diffusion 窗口状态按上"drafter cache 与 seed hidden states"处理(先按 SGLang 重算式,是否跨请求缓存待定)。`post_forward` 的 cache 准备产出这部分,`pre_forward` 消费它并行产 block。
- **verify**:自回归类逐 token 树形 verify;diffusion 类 **ragged verify**(整块并行验证,DSPARK 有 SPS/STS 校准表调 verify 长度)。
- **左 pad / 边界**:DFLASH 滑窗在近 max 时左留一页对齐——归"长度边界规避"的 headroom 吸收(见下)。
- **不改存算分离边界**:drafter KV 与 target KV 同归存储池权威(放置/迁移/复用);seed hidden / diffusion 窗口状态是请求内滚动态,是否进池待定(先按 SGLang 重算式)。post/pre 二分只是计算层内部编排,存储池对 target KV 与 drafter KV 一视同仁(publish/写回/前缀生长)。

> **参考**:DSPARK **仅 SGLang 有**(`3rdparty/sglang/python/sglang/srt/speculative/dspark_components/`:`dspark_config.py`/`dspark_planner.py`/`dspark_block_accept_estimator.py`/`kernels/`+ `dspark_worker_v2.py`+ `ragged_verify.py`);DFLASH 两边都有(SGLang `dflash_worker_v2.py`/`dflash_info_v2.py`,vLLM `spec_decode/dflash.py::DFlashProposer`)。见"vLLM / SGLang 支持梳理"表。

### drafter 共置 vs 独立 Draft 池

- **默认共置**(sglang 式):drafter 与 decode(target)同节点,零 draft 候选传输延迟,主路径。
- **独立 Draft 池(可选)**:仅当 drafter 算力开销显著拖累 decode 才考虑物理分离;但 draft 候选需跨节点传给 target,延迟可能抵消投机收益(见"开放问题")。

### PD 分离下 MTP 的重算与左 pad

MTP 类方案的验证 step:target 对 draft 的 k 个 token 一次前向,**重算后产出 `1 + num_mtp_layers` 个 token** 的 KV(k 个 draft 的验证 + 1 个延伸 token)。

**左 pad 问题**:残差 prefill 场景(前缀命中后做残差 prefill,见 [`execution-modes.md`](execution-modes.md) 时序一/D-direct),若**残差 prefill 的 token 数 < MTP 单次产出宽度**(`1 + num_mtp_layers`),MTP head 的多 token 输出与序列位置无法对齐 → 需**左 pad**:在残差段左侧补入若干已命中的前缀 token(从缓存重引入),把 prefill 段垫到 ≥ MTP 产出宽度,使 MTP 各 head 的位置对齐。

- 本质:MTP 一次产多 token 要求输入段长度 ≥ 产出宽度;短 prefill(尤其 D-direct 残差 prefill)不足时左 pad 补齐。
- 代价:左 pad 增加少量 prefill 算力(重算已命中的补齐 token),但保证 MTP 正确性;pad 窗口 = MTP 产出宽度 − 残差长度,上限 `num_mtp_layers`。
- PD 分离下该问题同样存在(残差在 P 侧或 D-direct 节点),左 pad 归计算层处理,pad 的 token 来自已命中前缀(本地命中则零传输,Pool 命中则需先拉取对应 block)。
- 具体左 pad 策略(是否总是 pad 到固定宽度、pad token 是否复用 KV)待实现/P7 校准。

### 参考实现与关键差异

> 按 CLAUDE.md 强制查阅规则。

- **vLLM**:`vllm/v1/spec_decode/`::`EagleProposer`/`MedusaProposer`/`DraftModelProposer`(proposer 侧产 draft)+ `spec_decode/speculator.py::BaseSpeculator`/`DraftModelSpeculator`(target 侧并行验证)+ `rejection_sampler.py::RejectionSampler`(拒绝采样)。proposer↔speculator 划分对应 drafter↔target。见 [`../research/vllm/compute.md`](../research/vllm/compute.md) "Speculative Decoding"。
- **SGLang**:speculative 执行于 `python/sglang/srt/speculative/`(drafter 在 target 之后、同 step 串行的执行模型即仿此);`spec_info.py::SpeculativeAlgorithm` 枚举 EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM/DFLASH/DSPARK/FROZEN_KV_MTP;multi-pool `PoolName` 含 **Draft** pool(`hicache_storage.py::PoolName.DRAFT` / `batch_exists_v2` / `PoolTransfer`),即 **drafter 自己的 KV** 作为与 `PoolName.KV` 并列的一等 pool 管理(跨请求存取/预取)——我们同样把 drafter KV 进池统一管理、跨请求复用;seed hidden 是否跨请求缓存待定(先按 SGLang 重算式:hidden 不进 radix、命中后 draft-extend 重算),差异见下。SGLang spec decode 未在 `docs/research/sglang/` 单列文档,执行顺序按 `3rdparty/sglang/python/sglang/srt/speculative/` 源码回溯(EAGLE `eagle_worker_v2.py`、MTP `frozen_kv_mtp_worker_v2.py`、DFLASH `dflash_worker_v2.py`、DSPARK `dspark_components/`);**多层 MTP chain/non-chain** 见 `multi_layer_eagle_worker_v2.py::chain_mtp_hidden_states`(据 `draft_arch` 判定,chain 需 FULL hidden states、non-chain 需 LAST)+ server_args `enable_multi_layer_eagle`。
- **方案支持梳理**:见上"vLLM / SGLang 支持梳理"表——DSPARK **仅 SGLang** 有,DFLASH 两边都有,MTP/EAGLE/EAGLE3 两边都有。
- **关键差异**:
  - vLLM/SGLang 的 drafter 可独立进程/proposer;我们默认 drafter 与 decode(target)**共置**(sglang 式),独立 Draft 池为可选。
  - **drafter KV**:SGLang 作 multi-pool 一类(`PoolName.DRAFT`,可落 L3);我们同样进存储池统一管理、跨请求复用、随迁——池同时承载 target KV(t-type/r-type)与 drafter KV。
  - **seed hidden states**:是否跨请求缓存 = 待定;先按 SGLang 重算式(hidden 走请求内 `spec_info`、命中后 draft-extend 重算,不进 radix)。备选"存 hidden 跨请求复用"记为遗留问题。
  - 投机不改变存算分离边界:drafter KV 与 target KV 同归存储池权威;seed hidden / diffusion 窗口是请求内滚动态(是否进池待定)。

## 长度边界规避:max_model_length vs runner_max_model_length

推理**达到/临近最大长度**时有一类难缠的边界 bug,历史实测在 vLLM 旧版屡现:① drafter 跳过(近 max 时无空间产 draft)导致 EP/DP 下其他 rank 集合通信盲等/hang;② block 申请不够,请求被准入却永不可调度,阻塞队列致健康检查失败。两个参考实现都靠**累积的特殊逻辑**兜,复杂且随特性组合(spec + PD + EP + DP + Mamba + PP async)持续增生。lake 用**双长度变量 + headroom** 规避,避免在 runner 里写复杂的近 max debug 逻辑。

### 参考实现的边界处理(事后打补丁式)

> 按 CLAUDE.md 强制查阅规则。

- **vLLM**(`3rdparty/vllm/vllm/v1/core/sched/scheduler.py`):
  - L803-815:spec decode 把 decode pad 到 `1+num_spec_tokens` 保 cudagraph 一致;若 `num_computed_tokens + num_new_tokens > max_model_len` 则 **break("Prefer to not schedule than schedule un-padded")**——放弃调度而非处理部分 draft(近 max 的 drafter 跳过)。
  - L864-872:PD 分离 + spec 的 "extra block allocated → local/remote block mismatch" edge case → `limit_lookahead_tokens`(lookahead=0)。
  - L437-451:async 占位符近 `max_tokens` 时跳过额外 step;L1899-1924:defer free by drafter look-ahead(防 drafter +1 读已释放 block)。
  - block 预留:`vllm/v1/core/kv_cache_manager.py::allocate_slots(full_sequence_must_fit)` + `vllm/config/scheduler.py:140 scheduler_reserve_full_isl=True`(默认,为整段 ISL 预留 block,防"申请不够")。
  - `vllm/config/model.py::spec_target_max_model_len`:draft 模型继承 target 的 max_len——**历史遗留(独立 draft 模型时代),lake 不参考**。主流模型自带 MTP 层(非独立 draft 模型),无"draft 模型另有 max_len"的场景;lake 的双长度是 **runner arena headroom**(计算层内部容量),与"draft 模型继承 target max_len"是两回事。
- **SGLang**(`3rdparty/sglang/python/sglang/srt/managers/scheduler.py`):
  - L1915-1934 `init_req_max_new_tokens`:**准入时** clamp `max_new_tokens` 使 `ceil_page(input_len) + max_new_tokens + page_size < max_total_num_tokens`;注释直言"否则请求被接受但永不可调度,阻塞队列致健康检查失败"——正是 block 申请不够 bug,靠 admission clamp 兜。
  - L2720-2731 `speculative_skip_dp_mlp_sync`:spec + dp-attn 时 `maybe_prepare_mlp_sync_batch` 防 prefill/decode 混批致 collective desync(drafter skip + DP 盲等);提供 flag 跳过该 sync。
  - `python/sglang/srt/speculative/eagle_worker_v2.py:1281-1288 _build_trivial_verify_input`:`speculative_num_steps==0`(drafter 跳过)时仍构造 1-node verify input 走 TARGET_VERIFY graph,保 collective/graph 一致——为 drafter skip 的特殊路径。
- **结论**:两框架的边界处理都是"近 max 打补丁"特殊逻辑,随特性组合增长,正是 lake 要规避的复杂 debug。

### lake 方案:双长度变量 + headroom

| 变量 | 值 | 谁守 | 用途 |
|------|----|------|------|
| **`max_model_length`** | 对外契约长度(prompt+output 上限) | gateway / scheduler | 请求级 length cap(`check_stop` → LENGTH_CAPPED)、计费/SLO 契约 |
| **`runner_max_model_length`** | `max_model_length + headroom` | 计算层 runner | arena / block table / graph capture 预分配尺寸 |

**headroom 构成**:`num_spec_tokens`(或 MTP 的 `1+num_mtp_layers`)+ lookahead + block 对齐 margin(向上对齐 128)+ 安全余量。draft 产出的 **transient 超出 `max_model_length` 的部分落在 headroom 内**,rejection sampling 后裁剪回 `max_model_length` 以内,runner 不需"近 max 跳 drafter / 处理部分 draft"的特殊逻辑。

**规避的边界**:
- **draft 超长 / drafter skip**:headroom 内有空间产 draft,近 `max_model_length` 仍能正常 draft,不触发 skip → 不触发 TP/EP/DP collective desync(vLLM 的 break、sglang 的 `_build_trivial_verify_input` / `speculative_skip_dp_mlp_sync` 不再需要)。
- **block 申请不够**:runner 按 `runner_max_model_length` 预留 block,请求在 `max_model_length` 内永远够(vLLM `reserve_full_isl` / sglang `init_req_max_new_tokens` clamp 的复杂 admission 逻辑不再需要)。
- **MTP 左 pad**:pad 后 prefill 段长度落在 `runner_max_model_length` 内(见"投机解码 > PD 分离下 MTP 的重算与左 pad")。

**衔接既有设计**:
- **Q1 固定 arena**:arena 大小 = `runner_max_model_length × max_bs`(含 headroom),不扩缩容;block table / graph capture 按 `runner_max_model_length`。
- **block 粒度 128**:headroom 向上对齐到 block 边界(避免半块)。
- **投机解码**:hidden states 滚动按 `runner_max_model_length`;draft transient 在 headroom 内。
- **调度/命中**:scheduler 的 stop 判定用 `max_model_length`;block 预留用 `runner_max_model_length`。两者解耦——对外契约与计算层内部容量分离。
- **职责边界**:gateway 管对外 `max_model_length` + 入口 shedding;runner 不感知对外契约,只按 `runner_max_model_length` 配 arena。

**代价**:headroom × max_bs token 的额外 HBM arena + graph capture 尺寸略大。**以空间换"免复杂近 max debug 逻辑"**。headroom 大小待 P7 校准(与 draft 深度、block 粒度 128 对齐)。

**约束**:请求级 `max_tokens ≤ max_model_length`;`headroom ≥ 最大 draft 深度 + margin`,且请求 `max_tokens` 上限绝不顶到 `runner_max_model_length`——headroom 只吸收 draft transient,不吸收请求自身的超长(超长由 gateway 在 `max_model_length` 拒/截)。

## 开放问题

- Prefill/Decode 比例随流量变化，是否支持节点在池间动态转换？（带权重迁移成本）
- continuous batching 与 KV 跨节点迁移如何协同（迁移中的 sequence 如何处理）？
- **投机解码 draft 与 target 物理分离时,候选传输延迟是否抵消收益?**(默认共置规避;独立 Draft 池的收益阈值待 P7)
- **MTP 左 pad 策略**:是否总是 pad 到固定宽度、pad token 是否复用命中 KV、pad 窗口上限,待实现/P7 校准。
- **r-type 入图**:sliding window / Mamba 的固定状态 arena 与 t-type block arena 的 capture/replay 协同。
- **headroom 大小**:`runner_max_model_length − max_model_length` 的取值(覆盖 draft 深度 + lookahead + block 对齐 margin + 安全余量),待 P7 校准;约束:请求 max_tokens ≤ max_model_length,headroom 只吸收 draft transient。
- **prefill 期 draft 取舍(遗留)**:PD 分离下 P 侧 draft 产出是否使用/丢弃、drafter prefill forward 是否只为保 KV 同步(vLLM 如此,SGLang 待细究)。初步判断不影响整体设计,接入投机时确认。
- **多层 MTP chain / non-chain**:接入多层 MTP 时的 seed hidden 窗口策略(chain 需 FULL + 中间层输出,non-chain 只需 LAST)与 `speculative_num_draft_tokens == num_layers` 等约束对齐。
- **seed hidden 是否跨请求缓存(遗留)**:默认按 SGLang 重算式(hidden 不进 radix、命中后 draft-extend 重算);备选按 token 存 hidden 进池换取跨请求复用(省重算、费存储)。待请求迁移/命中收益权衡时定,现不影响框架推演。
- **请求迁移的 seed 状态迁移(遗留)**:请求迁移(如 DP 间再均衡,见 [`scheduling.md`](scheduling.md))是未来特性;drafter KV 随池迁移,seed hidden 走"迁移后 `post_forward`/draft-extend 重建"即可,无需专门持久化通道。现阶段只需保证框架不因此需大重构(drafter KV 归池已满足)。
