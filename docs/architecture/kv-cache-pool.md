# 05 — KV Cache 池

KV Cache Pool 把 KV cache 从"附属于产生它的 GPU"提升为全局可寻址、可复用、可迁移的分布式资源。在彻底存算分离下,连 HBM/RAM/NVMe 都不归计算节点私有,而是存储池统一管理的物理载体(L0–L4:GPU HBM / 主机 RAM / 本地 NVMe / 远端内存池 / 对象存储,分层定义见 [`storage-layer.md`](storage-layer.md))。所有 KV 位置(含"哪段 KV 在哪个节点 HBM")均为存储池权威元数据。

## 设计目标

1. **前缀复用**:多请求共享公共前缀(system prompt、few-shot、共享上下文)的 KV,避免重复计算。
2. **跨节点迁移**:节点缩容/故障时,KV 迁移到他处续推。
3. **解耦计算与存储**:Prefill 不关心 Decode 在哪,只把 KV 写入 Pool;Decode 从 Pool 拉。HBM 放置亦由存储池管理,本地命中是放置决策的结果(支撑 D-direct)。
4. **可控成本**:热 KV 在 RAM,冷 KV 落 NVMe / 对象存储。
5. **模型无关、长期存续**:池生命周期与模型解耦,同一池同时承载多个 `(model_id, revision)` 的 KV。

## 数据结构

### Block 寻址
```
KVBlockID = (model_id, layer_idx, block_hash)
# block_hash 由该 block 内 token ids 内容哈希得到,相同内容 → 相同 KV
```
内容寻址使前缀复用天然成立:相同前缀 → 相同 block hash → 命中同一 KV。

**block 粒度 = 128 token**:缓存命中 / 复用 / 传输 / 写回的最小单位。128 为初版默认,待 P7 校准(与 SGLang `--page-size`、vLLM `block_size` 同量级)。L1–L4 统一按 128-token block、page-first 组织(两类复用条件一致、不区分类型,见 [`storage-layer.md`](storage-layer.md) "KV 类型"节)。HBM(L0)的 t-type block 同取 128,便于 L0↔L1 整块零拷贝;r-type 在 L0 不按 block 而按紧凑状态槽,落 L1+ 时再按 block 切(trailing pages 或 state checkpoint)。

**模型无关**:`model_id` 是寻址命名空间,Pool 不解释张量布局(层数、头维、dtype),按不透明字节块存取。接入新模型只需注册 `model_id`,无需新建池。

### t-type / r-type:复用条件一致,区别在 HBM 存储形态

两类的**复用条件相同**:都需**命中全部前缀**才能复用(从序列起点的连续前缀 KV/state 必须在场)。区别**仅在 HBM(L0)存储形态**——r-type 用紧凑表示(滑动窗口最近 W token / Mamba 定长 state)替代逐 token 完整 KV,降低 HBM 占用。HBM 之上的区分**不带入下层**:

- **L1–L4 统一按 block(128 token)组织**:两类复用条件本就一致(全前缀命中),下层不区分类型。r-type 落下层时在 block 边界 checkpoint 紧凑状态:
  - sliding window:存 trailing pages(最近 W token,参考 SGLang `PoolHitPolicy.TRAILING_PAGES`)。
  - Mamba/state-space/卷积:每 128 token 存一份 recurrent state 快照。
- **复用**:radix 沿前缀匹配到最长边界,取该边界处的完整 KV(t-type)或 state 快照(r-type),从该点续算。内容寻址 + radix 前缀匹配对两类同等成立——block hash 由 block 内 token ids 算得,与 block 内装 KV 还是 state 快照无关。

> 池按不透明字节块存,r-type 与 t-type 在存储层共享同一套 block/分层/传输机制;区别仅是 block 内布局(逐 token KV vs 紧凑 state 快照),由元数据声明。相对 SGLang multi-pool 物理分池,我们把类型差异收敛到 **L0 存储形态 + block 内布局**,而非物理分池。

### drafter 的 KV 与 seed 状态

投机解码的暂存物分**两类**,管理不同(此前误记"draft 一律 L0-only 不进池",已纠正):

**1. drafter 自己的 KV(draft head/model 的 KV)——与 target KV 同款进池**
- 按 token/block 组织、进存储池统一管理(放置/迁移/生命周期),**跨请求前缀命中即可复用**、随请求迁移;复用条件与 target KV 一致(全前缀命中)。t-type/r-type 同存储层机制对 drafter KV 一样适用。
- 参考 SGLang `hicache_storage.py::PoolName.DRAFT`——drafter KV 作与 `PoolName.KV` 并列的一等 pool(跨请求存取/预取)。命中后残差区间由 draft-extend 前向补齐。

**2. seed 状态(自回归的 seed hidden / diffusion 的窗口·block 状态)——请求内滚动窗口,是否跨请求缓存待定**
- 自回归类:target **最后 `num_mtp_layers` 个 token 的 hidden states**;diffusion 类:draft 侧窗口/block 状态(DFLASH 滑窗、DSPARK gamma 块 + Markov 状态),均由 drafter `post_forward` 从 target 输出准备。
- **是否进池跨请求复用 = 待定,先按 SGLang 重算式推演**:不进 radix、走请求内 `spec_info`,命中/迁移后由 draft-extend(`post_forward`)重建 seed。备选:按 token 存 hidden 进池换跨请求复用(省重算、费存储)。**记为遗留问题**(见 [`compute-layer.md`](compute-layer.md) "开放问题")。
- 详见 [`compute-layer.md`](compute-layer.md) "投机解码"节(drafter `post_forward`/`pre_forward` 二阶段、"drafter cache 与 seed hidden states")。

### 前缀树索引

radix tree 归存储池,按 `model_id` 分命名空间。节点 = block hash,路径 = token 序列;给定 prompt 沿树匹配最长公共前缀,确定可复用 KV block 范围。Router 一次查询同时拿到可复用 block 列表与各自位置(前缀复用 + 本地命中判定)。

```
root
├─ [sys_prompt_hash] → KV blocks [0..k]
   ├─ [fewshot_hash] → KV blocks [k..m]
   │   └─ [user_query_A_hash] → ...
   └─ [user_query_B_hash] → ...
```

## 物理布局

L3 由 N 个 KV Node 组成,每个贡献 RAM + NVMe,block 按 hash 分片:
```
node_id = hash(KVBlockID) % N
```
- 写:产出节点的 agent 通过传输引擎 RDMA write 推到目标 KV Node。
- 读:消费节点的 agent 通过传输引擎 RDMA read 拉取(跨实例传输机制见上节)。

## 传输协议

- **控制平面**:block 元数据(位置、引用计数、热度)→ etcd,强一致。
- **数据平面**:block 字节 → RDMA,最终一致,best-effort。

**无引擎驱动的 intra-step 重叠;池驱动异步重叠保留**:本系统不照搬 SGLang HiCache "引擎在 `get_key_buffer` 每层 `wait_event`、算 layer N 传 layer N+1" 的**引擎驱动**逐层重叠——那套绑死引擎、破坏 CUDA graph(SGLang 把补拉与 graph 冲突留作 TODO)。我们拒绝的是这种**引擎驱动**的 intra-step 重叠。**池驱动的异步重叠保留**:引擎只调**异步传输接口 + fence**,传输由池的 agent 在独立 stream 上做;两类重叠都是异步自然结果,引擎无感、graph 安全:
- **消费侧 step 间重叠**:传 step N+1 的 block 时引擎在算 step N(B decode)。
- **生产侧层级重叠**:A prefill 逐层产出 → A 的 agent 逐层 publish page 切片 → 传输引擎搬到 B(时序二正向"与 A 计算重叠",支撑 PD 分离 TTFT)。

生产侧层级重叠要解决一个张力:page-first 要求整块(所有层)连续才能零拷贝传,但 prefill 逐层产出,整块没满没法传 → 重叠断了。解法是 `page_first_direct` 子块传输(同 page 内同层连续)→ 层算完即传该层的 page 切片,既保 page-first 整块零拷贝(L3 用),又能层级重叠(L0→L0 用)。此即下文"分块流水线"。

## 跨实例 KV 传输

核心:**跨实例的 KV 字节流不经过任何一个 worker 进程**,走存储池的分布式传输引擎(RDMA),零拷贝直送。Q1 定的 in-process agent 只管本地 L0;跨实例是池数据面的活,完全另一层。

### 内存注册与寻址

每个 worker 的 L0 arena(或其中被引用的页)启动时向传输引擎**注册**成可寻址区域 `(segment_id, offset, len)`,拿到全局句柄。之后跨实例传输即"源地址 → 目地址"的 RDMA 写,不经过 Python、基本不占 CPU。block 的 **page-first 连续布局**(见 [`storage-layer.md`](storage-layer.md))是零拷贝前提:一个 block = 一段连续内存,传引擎拿到的是单条 `(ptr, len)`,无 gather。

### 一次跨实例传输(以 PD 分离:A prefill 产出,B decode 消费)

1. **池定源**:B 的 agent 查位置视图——目标 block 在哪。两种源,由路由/时序决定:
   - **直传(A→B)**:A 还在线且 L0 仍持热副本 → 源 = A 的 L0 注册段。低延迟,要求 A、B 时序重叠。
   - **经池中转**:A 已 `publish` 到池 → 源 = L3 segment。A 可先死、B 稍后拉,时序解耦。
2. **B 异步 pull**:B 的 agent 调 `pull(block_ids)`,传输引擎在**独立的传输 stream** 上把字节从源 RDMA 写进 B 的 L0 空闲 slot(slot 由池分配、in-flight 冻结),返回 handle。
3. **B wait ready → 算**:B 在本步 replay 前 `wait(handle)`。传 step N+1 的 block 时 B 在算 step N——异步 + fence,重叠自然。
4. **A publish**:A 产出 KV 进 L0 slot 后调 `publish(block_ids)`,池注册进 radix + 决定是否写回 L3。

引擎的全部分层职责仍是 Q1 的 **消费 ready → 算 → 发 done**;pull/publish 只是这一契约在跨实例场景的接口形态。

### PD 分离下的传输流程(engine-to-engine 控制链切断)

关键后果:Q2.1 定了"block 对引擎纯寻址、block table 池组装、引擎零地址"——于是 **engine-to-engine 控制链被彻底切断**。vLLM/SGLang 的 PD 分离是两个引擎的 connector 直接握手、用 device 网络 engine-to-engine 传(引擎既拥有 KV 又发起传输);本系统引擎不知道地址、不组装 block table、不拥有 KV,**两个引擎从不知道对方存在**,池是唯一中介。但**数据线仍是直连 RDMA**(A 的 HBM → B 的 HBM),wire 效率不变——变的是控制权归属:发起者从引擎换成池的本地 agent。

完整流程(以 A prefill 产出、B decode 消费前缀):

1. **A 产出**:KV 落进 L0 slot(slot 由 A 的 agent 分配)。A step done 时调 `publish(block_ids)`——只上报"产出了这些 block",不含地址。
2. **A 的 agent 记录**:block X → (A 的 segment, offset),写进位置视图;按写回策略决定是否同时落 L3(见"写回与生命周期")。
3. **Router 定 B**:B 的 agent 查位置视图,拿到前缀 block 的源地址(A 的 L0 段 或 L3 段)。
4. **B 的 agent 发起传输**:每个需要的 block——选源(A 在线且时序重叠→直传;否则 L3)+ 在 B 分配空闲 slot + 冻结 + 传输引擎做 RDMA → 返回 handle。
5. **B 的 agent 组装 block table**(拉来的 slot + 已在 B 本地的 slot)→ `ready`(fence) → B replay。
6. B step done → publish 新 decode KV → 回到 4。

**默认直传 + Drain 推 L3**:PD 时序重叠(A 边 prefill B 边 decode)是主场景,默认直传(A 的 L0 → B 的 L0),省一跳、最低延迟;代价是 A 的源 slot 被在途传输 ref 钉住、占 A 容量直到拉完。当 A 进入 Drain/缩容,agent 先把"还被远端引用的 block"推一份到 L3(F4 恢复点,抗 worker 销毁即可)再下线——之后 B 从 L3 拉,A 可先死、B 照常,时序解耦。Drain 语义含"把 in-flight block 落 L3"。

**在途传输 ref(源端冻结)**:RDMA 异步,源端在传完前不能被覆写/驱逐,否则 B 读到半新半旧的损坏 block(静默故障)。故发起传输 → 源 block 的 ref +1(在途引用);RDMA 完成 → ref -1。这与请求引用(下节)是**同一个 ref**,只是多一种"在途传输"的引用来源;ref>0 即冻结,统一机制。

### 布局转换

L0 是 layer-first(引擎逐层写),跨实例传输要转 page-first(整块连续好零拷贝)→ 照搬 SGLang `sgl-kernel/csrc/kvcacheio/transfer.cu::transfer_kv_per_layer_pf_lf` 那个非时间索引核(`ld.global.nc`/`st.global.cg`),在传输 stream 上一次 launch 做完,无 host staging。

### 分块流水线(page_first_direct 子块传输)

生产侧层级重叠(时序二正向)依赖 `page_first_direct` 布局:同 page 内**同层的 token 连续**,于是可按 **per-layer-page 子块**传输——A 算完某层即传该层的 page 切片,不必等整块所有层填满。这同时满足:
- **page-first 整块零拷贝**(L0→L3 / L3 间):整块所有层仍连续,传引擎拿单条 `(ptr, len)`。
- **层级重叠**(L0→L0 直传):按层切传,A 算 layer i+1 时 layer i 已在搬。

流水线深度与 prefill 层数对齐(A prefill 第 i 层时,B 已就绪到第 i-k 层),k 由传输带宽与单层计算时间比定,留 P7 校准。这是 PD 分离 TTFT 的关键——A 长 prefill 边算边把层搬到 B,B 提早就绪,而非等 A 整块完成才开始传。

### 三个边界点(非分叉,交代清楚)

- **L0 直传依赖 GPUDirect RDMA**:NIC 与 GPU 同 PCIe root 才直读 HBM;否则经 pinned host(L1)中转一次拷贝。部署拓扑(RDMA 可用性退化)留 [`topology.md`](topology.md),接口不变(传输引擎内部吸收)。
- **直传 vs 经池中转是路由决策**:A、B 时序重叠 → 直传省一跳;A 先结束 B 后到 → 经池中转。归 Router/调度器按时序选,非传输层职责。
- **in-process agent = 传输引擎的本地端点**:agent 在 worker 进程内注册本地内存、发起/接收传输;传输引擎本身是池的分布式数据面。L0 内存注册用 in-process(Q1 定的方案 a)最顺——Rust `.so` 直接拿 worker 的 CUDA 内存句柄去注册 RDMA MR,省一道 IPC。

### 双网络路径(compute network / storage network)

节点有两类物理隔离的网络([DualPath](../research/dualpath.md) 的架构前提):
- **compute network(东西向)**:GPU 间 collective 通信、L0→L0 RDMA 数据面。带宽大、呈间歇突发(集合操作亚毫秒级)。
- **storage network(南北向)**:访问 L2/L3/L4(NVMe/远端内存池/对象存储)。带宽相对小、持续。

KV 跨节点传输按"源在哪、目在哪"自然落到两类网络:
- **L0→L0 直传**(§3.2 PD 正向、§3.4 D→P 子情况 A)走 **compute network**:两台 GPU 机 HBM 间 RDMA,大带宽。
- **L3→L0 加载**(补拉、§3.4 D→P 子情况 B 的 D 侧加载、经池中转的拉取)走 **storage network**:从远端内存池/对象存储读。
- **L0→L3 写回**(满块/Drain 推 L3)走 **storage network**。

两类带宽是**池的资源**,非实例私有(本系统更彻底之处):池按 NIC 负载/带宽视图决定——

- **D→P 选路**(见 [`data-flow.md`](data-flow.md) §3.4):下一轮 prefill 所需延伸 KV 的来源,池在三条路里选:
  - 子情况 A:KV 已在 D 的 L0 → D L0 ──compute network──→ P L0(**零存储读取**,连 storage network 都不占)。
  - 子情况 B:需从 L3 加载 → 池可选 **D 侧从 L3 经 storage network 加载 + 经 compute network 回传 P**(借 D 闲置 storage 带宽 + 高带宽 compute network 回传,绕开 P 侧 storage 带宽瓶颈)。
  - 传统路:P 侧自拉 L3(§3.2 的经池中转)。
  - 三者由池按 NIC 带宽视图决策,这正是 DualPath "storage-to-decode + CNIC 回传"在本架构的原生支持——DualPath 是引擎实例视角"借用"对端闲置带宽,我们池统一管理直接分配,不存在"借"。详见 [`../research/dualpath.md`](../research/dualpath.md)。
- **与 collective 通信隔离**:compute network 既跑 GPU collective 又跑 L0→L0 KV 传输,DualPath 强调两者物理同网但 collective 是间歇突发、KV 传输在空隙插入。池调度 KV 传输时避开 collective 突发窗(避让策略留 P7),不干扰 latency-critical 的模型通信。

### 参考实现与关键差异

> 按 CLAUDE.md 强制查阅规则。

- **参考实现**:
  - **Mooncake transfer-engine**(`3rdparty/mooncake/mooncake-transfer-engine/`):RDMA 零拷贝 + 多 NIC 聚合 + segment 寻址(对象按 `(segment_id, offset, len)` 寻址,不解释内容)——这是本系统 `rust/transfer/` 的直接原型;pull/publish 的异步 handle API、GPUDirect RDMA 直读 HBM 照搬。逐层对应见 [`../research/mooncake/transfer-engine.md`](../research/mooncake/transfer-engine.md)。
  - **Mooncake mooncake-store**:KVCache 全局池、按 segment 寻址——是 L3 原型,"经池中转"那条路即它。见 [`../research/mooncake/kv-store.md`](../research/mooncake/kv-store.md)。
  - **SGLang `pool_host/mha.py::get_page_buffer_meta`**:page-first 布局让每页一段连续内存、`data_ptr()` 直出零拷贝;`transfer.cu::transfer_kv_per_layer_pf_lf` 是 layer-first↔page-first 转换核。见 [`../research/sglang/{hicache,storage-backends}.md`](../research/sglang/)。
  - **DualPath**(论文 arXiv:2602.21548v2,非 submodule):双网络隔离 + storage-to-decode-then-CNIC-to-prefill 路径——直接对应本节"双网络路径"与 [`data-flow.md`](data-flow.md) §3.4 D→P。分析见 [`../research/dualpath.md`](../research/dualpath.md)。
- **关键差异(我们更彻底)**:
  - Mooncake 传的是**实例私有 store 之间**(实例拥有本地 HBM,传输是实例间共享/迁移);我们传的是**池权威的 L0 之间**——A、B 的 L0 都是池的物理载体,源/目 slot 由池分配、in-flight 冻结,实例不"拥有"任何 KV。传输对引擎仍是异步 pull/publish,背后所有权语义彻底归池。
  - Mooncake 无内容寻址/radix(按 segment ID 存取);我们 pull 前先查 radix + 位置视图拿到 block 物理源地址(A 的 L0 段或 L3 段),一跳定位,省掉 Mooncake 按 ID 查后端。
  - **D-direct 是零传输特例**:若池已把 block 预放置在 B 的 L0(本地命中),位置视图直接返回"B 本地",B 零 pull 直跳——Mooncake 没有此路径(它总要传)。

## 引用计数与驱逐

- 每个 block 维护引用计数,由**池权威维护**(非引擎侧)——多引擎共享同一前缀 block 时,引擎各自持本地计数会导致分布式不一致,故 ref 是池的单点计数。引擎只通过"本步 read set / write set"间接表达引用,不持计数。
- ref 的来源有两种,统一为"ref>0 即冻结":
  - **请求引用**:block 进入某请求 read set 时 +1。减点只能在**请求结束且无续推引用**时——attention 每步读全部 KV(含所有前缀 block),前缀 block 全程 in-flight,不能中途早减。F4 续推时原请求结束但 KV 要续到新节点,ref 不归零而是**转移到新请求**(或持"续推引用"),避免被冷热淘汰。
  - **在途传输引用**:跨实例传输发起时,源 block +1(源端冻结,防 RDMA 半传时被覆写致损坏);RDMA 完成 -1。见"PD 分离下的传输流程"。
- 引用为 0 的 block 进冷热排序,按**热度分**(f(频次, recency),LFU-Aging)与容量阈值驱逐/下沉。
- 公共前缀 block 给予前缀亲和加权保护,驱逐时不易被选。
- 层间副本/移动、promotion/demotion、主动迁移见 [`storage-layer.md`](storage-layer.md) "冷热与生命周期管理";本节驱逐是其中一环。
- 被驱逐但 L4 仍有副本的 block,可按需回填。

## 写回与生命周期

一次请求的 KV 从产生到消亡:

```
引擎产出(在 L0 slot)
  → [满了] 池算哈希 → 注册 radix → 写回 L3(F4 恢复点,抗 worker 失败)
  → 请求结束
  → [尾块,未满] 池写回(纯容错,不进 radix)
  → KV 续存(供复用/续推) 或 按 TTL/冷热淘汰
```

两条写回路分开(满块结构性,尾块容错性):

- **满块路**:block 填满 → 池算哈希 → 注册 radix → 写回 L3。这是自然边界,radix 注册本就要等满块(vLLM `ExternalBlockHash` 也只对完整 block 算哈希),写回顺带做。decode 跨 block 边界即产生满块,请求进行中就可能触发。满块写回的频率(满一个就写 vs 攒几个一起写)即"写回频率 N",留 P7 校准。
- **尾块路**:请求结束时仍未填满的 block(尾块)→ 请求结束点写回一次,写"当前尾 block 的全部已填 token",重放时整块覆盖。纯容错,不进 radix(哈希未定,或带 partial 标记)。因尾块只在请求结束写一次,无增量式。

引擎不感知 block 满不满(Q2.1:block 对引擎纯寻址单位)——满块判断、哈希、radix 注册、写回全归池。容错点 = "KV 落 L3"的时刻;满块越频繁写回(N 小)→ 崩溃丢的越少、写放大越大,反之亦然。decode 增量写回同时服务容错 + 前缀生长(见 [`execution-modes.md`](execution-modes.md) 时序二反向)。

## 多模型生命周期

池长期存续,模型在其上动态注册/注销,二者解耦:

- **注册**:登记 `model_id`(+ revision、层数、block 规格、配额),分配初始空间。无需新建池或迁移已有数据。
- **下线**:级联删除该 `model_id` 的所有 block(KV + 元数据 + radix 子树),配额归还空闲池。进行中请求由 F4 处理完再清理。
- **revision 更新**:新 revision 视为新命名空间(内容寻址下 KV 通常不可跨 revision 复用);旧 revision 按失效策略(引用归零 + TTL)逐步淘汰。
- **池存续**:模型全下线后池仍运行。池重启不丢 L4 中的持久副本。

## 空间分配与扩缩容

总容量 = 各 KV Node 的 RAM+NVMe 之和,按**配额**在模型间分配:

- 每模型设软配额(常态上限)与硬配额(绝对上限)。
  - 软配额内自由写入;超软配额按该模型 LRU 淘汰冷块腾位。
  - 闲时借用池全局空闲空间(best-effort,遇压力可回收)。
  - 触硬配额 → 返回写入背压信号向上传播(请求级 shedding 仍归 gateway,见 [`../features/slo.md`](../features/slo.md))。
- 配额权重按模型负载/命中率动态调整(调度器决策,控制面下发)。
- **扩容**:加入 KV Node,按一致性哈希重分布,仅迁移落在新节点区间的 block。
- **缩容**:Drain 目标 Node(block 迁出或下沉 L4)再下线。
- 迁移为后台低优先级任务,可暂停让路高峰。

## GC

回收无效/不可达 block:

- **冷块回收**:引用 0 + 冷(LRU 末尾)→ 淘汰。
- **孤儿块**:Prefill 崩溃残留的部分写入 block → 写入屏障标记未完成,TTL 后回收。
- **模型下线/旧 revision**:级联删除。
- **元数据一致性**:以控制面元数据为权威,block 字节删除前确认元数据已无引用;崩溃恢复扫描 reconcile 孤儿块。
- **节流**:后台运行,受带宽/IO 预算限制,不阻塞数据面。

## 碎片整理

长期写入/删除/迁移导致两类碎片:

- **逻辑碎片**:同一序列 block 散落多 KV Node → Decode 读扇出大、传输慢。整理:把热点序列 block 迁到少数节点共置,降读扇出(热度由 radix + 访问频次判定)。
- **物理碎片**:NVMe/RAM 空闲页零散 → 写入放大、分配失败。整理:后台压实合并空闲页。
- 节流:消耗带宽与 CPU,须节流并与低峰重叠,可暂停可恢复;目标开销 < 总带宽 X%(P7 校准)。

## 故障恢复

**持久语义分层**(避免"L3 持久"与"L4 SSOT"措辞冲突):
- **L3 = F4 恢复点**:远端内存池(RDMA RAM),靠多副本/HA 抗**单 worker 失败**。block 写入 L3 即视为可 F4 续推——worker 崩溃后池把该 sequence 的 KV 放置到新节点 HBM 续推。
- **L4 = SSOT 永久权威**:对象存储,抗**池级失败**。L4 缺失才视为 block 不存在(见 [`storage-layer.md`](storage-layer.md))。

风险窗口:block 在 L0 产出 → 每 N 步写回 L3。若 worker 与其 L3 副本**同时失败**且该 block 未落 L4,则这段尾巴丢失——即"丢失最后一次写回 L3 之后的少量 token"(满块路 + 尾块路,见"写回与生命周期")。落 L4 的频率由冷热生命周期决定(冷下沉 L4),非每步;热工作集常驻 L3 副本。

- Decode 节点崩溃:存储池检测 → 把该 sequence 路由到新节点 → 由存储池把已有 KV 放置到新节点 HBM → 续推(ref 从原请求转移到新请求,见"引用计数与驱逐")。原节点 HBM 副本随销毁失效(本就是易失副本,非私有状态)。
- **Drain/缩容(主动下线)**:节点进入 Drain 时,agent 先把"还被远端引用的 block"(在途传输 ref>0 或被其他节点 read set 引用)推一份到 L3,再下线——避免销毁后远端拉取落空。这是"默认直传 + Drain 推 L3"在故障/弹性侧的落点。
- 增量写回频率(每 N 步):N 小 → 恢复快、写放大大;N 大 → 恢复慢、写放大小。另有前缀生长诉求,见 [`execution-modes.md`](execution-modes.md)。

## 开放问题

- 内容寻址哈希碰撞与安全(是否加盐区分租户)。
- RDMA 不可用时退化 TCP,带宽-延迟模型如何变化。
- 多模型配额公平性:高负载模型挤占他人时的仲裁与抢占回收代价。
- GC/碎片整理与数据面竞争的隔离(带宽预留 vs 优先级抢占)。
- 碎片整理触发判定:扇出阈值、碎片率,还是周期性?
- **block 粒度 128**:与传输带宽/写放大/碎片率的权衡,待 P7 校准。
- **r-type 状态 checkpoint**:Mamba/卷积 recurrent state 落 L1+ 的 checkpoint 间距/形式、sliding window trailing pages 阈值,待实现/P7 校准。
- **r-type SWA 是否落 L1+(二选一)**:SWA KV 落 L1+ trailing pages 直接命中 vs 不持久、prefix 命中时重算尾段 `n*(w-1)+1` 个 token 刷 SWA 窗口(省存储换重算,见 [`compute-layer.md`](compute-layer.md) "r-type SWA 前缀复用的尾段重算优化")。两条路线二选一,待 P7 存储成本 vs 重算成本权衡;若选重算路线,需 agent 的 slot 分配按模块差异化(只给 SWA 分 write slot,引擎契约不破)+ 残差路径区分"增量 prefill"与"刷新重算"。
