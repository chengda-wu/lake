# 06 — 路由与调度

> 调度分四层(请求级路由 / 池间 / 节点级 / 弹性)。**模式选择与请求生命周期权威描述见 [`data-flow.md`](data-flow.md)**(决策树 + 三模式执行段 + F4 分支);**执行模式与 KV 流转时序见 [`execution-modes.md`](execution-modes.md)**(本地完成 / 跨节点传输两条时序)。本文讲调度层各层次的机制:模式选择依存储池本地命中、prompt 规模、传输成本逐请求决策(见 [`../features/features.md`](../features/features.md) "执行模式"节)。

调度是存算分离系统能否兑现弹性与低延迟承诺的控制核心。本系统调度器**无状态**（所有决策依据来自控制面的共享视图），可水平扩展。

## 调度层次

```
1. 请求级路由 (Gateway 准入 / Router 选路)
2. 池间调度 (Prefill ↔ Decode ↔ Draft)
3. 节点级调度 (continuous batching、内存分配)
4. 弹性调度 (扩缩容)
```

## 1. 请求级路由

输入：请求 `(model_id, prompt_tokens, max_tokens, SLO)`。

步骤：
1. **前缀解析**：查存储池的 radix tree 与放置元数据，得到可复用 KV block 列表、对应物理位置，以及是否已**本地命中**（前缀 KV 是否已在某执行节点 HBM，支撑 D-direct）。
2. **Prefill 节点选择**：
   - 亲和性：优先选 KV block 所在的 KV Node 附近的 Prefill 节点（减少传输）。
   - 负载：考虑队列长度、HBM 放置余量（HBM 由存储池统一管理，Router 读存储池的 L0 容量视图）。
   - 目标：最小化 (增量 prefill 计算 + KV 传输) 的加权和。
3. **Decode 节点预分配**：在 Prefill 完成前就选定目标 Decode 节点，由存储池把 KV 放置到其 HBM（本地命中优先）。
4. **SLO 感知与优先级调度（Router 职责）**：Router 在选路时纳入 SLO 预算（如 D-direct 模式选择开销须 < 5ms，否则吃掉本地命中省传输的收益）与请求优先级——**优先级队列**决定已准入请求的执行顺序与抢占（被抢占者 KV 在存储池保留，见第 3 节）。**不设降级链**：若选不到合适节点或执行失败（故障/超时），不写 mode-to-mode 的预设 fallback、不"拒绝"已准入请求，而是触发 F4 故障恢复 → Router 依最新集群状态重跑 `f(请求, 集群状态) → (模式, 节点)` 重选模选点。
   - **边界**：过载层面的**入口准入 / 限并发 / 按优先级丢弃**归 gateway/外部控制面（决定请求"进/不进"，过载拒绝不计推理系统失败率）；Router 只对**已准入**请求做 SLO 路由与优先级**排队顺序**（决定"去哪/何时/怎么跑"，不丢请求）。Worker 不自 shedding，只上报剩余容量、队列长度、in-flight 等信号。详见 [`../features/slo.md`](../features/slo.md) "过载控制"节。

## 2. 池间调度

- Prefill → Decode 的 KV 传递通过存储池传输引擎(RDMA 数据面,见 [`kv-cache-pool.md`](kv-cache-pool.md) "跨实例 KV 传输"),需在 Prefill 完成时序上对齐 Decode 就绪（**仅 PD 分离模式**;混部/D-direct 为本地完成、无跨节点传输,见 [`execution-modes.md`](execution-modes.md)）。
- 投机解码：Draft 池在 Decode 侧生成候选，验证失败回退到标准 decode。
- 反压：Decode 池拥塞时，减缓 Prefill 速率（背压），避免 KV Pool 堆积。属池间内部流控（不丢请求、不降 batch），区别于 gateway 的请求级 shedding。

## 3. 节点级调度

- **Continuous batching**：Decode 节点动态拼接 batch。
- **PagedAttention** 风格的块状 KV 管理，与存储池的 block 粒度对齐。
- **放置与 batch 单向耦合（方案 Z）**：同一 batch 各 sequence 的 KV 必须同时在本机 HBM（attention 一次读全部）。存储池按热度主动预放置 KV 到 HBM 并发布位置视图;调度器读视图组 batch（本地命中优先），缺失补拉，不反向指挥放置。见 [`storage-layer.md`](storage-layer.md) / [`execution-modes.md`](execution-modes.md)。
- **抢占**：高优先级请求可抢占低优先级，被抢占者的 KV 在存储池中保留（不丢失，本机 HBM 放置释放归还存储池）。

## 缓存命中感知调度

缓存命中是调度的**一等输入**——模式选择(请求级)与 batch 组成(节点级)都由命中视图驱动。本节把散见于各层的命中感知要点收敛,并补一条此前未显式写的**跨请求前缀共调度**。

**两层命中(均为存储池权威元数据,见 [`features.md`](../features/features.md) "Pool 命中 vs 本地命中")**:
- **Pool 命中**:前缀 KV 在分布式池 → 省重算,但仍需传输(驱动 PD 分离 / 混部选路,见第 1 节)。
- **本地命中**:前缀 KV 已被存储池放置在某执行节点 HBM → 可 D-direct,零/极小传输(驱动节点选择 + 残差 prefill 工作量)。

**跨请求前缀共调度**(节点级):把**共享公共前缀**的多个请求尽量组到同一 batch / 同一节点,使前缀 block 在 batch 内复用、本地命中叠加。参考 SGLang RadixAttention 的 cache-aware scheduling(`radix_cache.py::match_prefix` 驱动 scheduler 把同前缀请求 co-schedule)、vLLM `KVCacheManager.get_computed_blocks`(逐请求查前缀块、命中数影响调度顺序)。收益:同前缀请求共节点 → 共享同一批前缀 block 的 HBM 放置,本地命中密度提升、传输减少。

**边界(重申,守方案 Z 单向耦合)**:
- 调度**只读**命中视图,不反向指挥存储池放置。存储池按热度主动预放置并发布位置视图;调度读视图组 batch(本地命中优先→D-direct,缺失补拉)。信息流单向。
- 命中视图由存储池权威维护,陈旧只影响命中率(miss→pull→控制面确认),不影响正确性(与 [`kv-cache-pool.md`](kv-cache-pool.md) "block table 池组装"的本地视图镜像语义一致)。
- **投机解码的 draft 候选 token(未验证)不进 radix**;但 **drafter 自己的 KV 与 target KV 同款进池、跨请求前缀复用**(SGLang `PoolName.DRAFT`),seed hidden 是否跨请求缓存待定(先按重算式,见 [`compute-layer.md`](compute-layer.md) "投机解码")。命中感知对 target KV 与 drafter KV 均适用(t-type 与 r-type 均含,复用条件一致——都按全前缀命中,见 [`kv-cache-pool.md`](kv-cache-pool.md) "t-type / r-type")。
- 前缀共调度与"前缀亲和性引发热点"(见下开放问题)存在张力:把所有同前缀请求固定到一个节点会过载。共调度目标是 batch 内复用,非全局集中;负载均衡由 Router 在节点选择时加权(见第 1 节 Prefill 节点选择)。

## DP 间在途再均衡(抢占重算式,未来特性 / 框架预留分析)

> **定位**:请求迁移是**未来特性**,本节现在讨论的目的是**确认框架无需为它做特别设计**(避免后续大重构),不是当前实现项。多 DP 部署下若各 DP 计算负载严重不均衡,需要把已在跑的请求换到别的 DP 执行。

**为什么不均衡有代价**:DP forward 是 lockstep + **padding-to-max**(vLLM `dp_utils.py::sync_cudagraph_and_dp_padding`:每 step 全体 pad 到 `num_tokens_across_dp.max()`),最闲的 rank 被最忙的拖着空跑。故**再均衡目标是降低 max(按 token/计算量衡量),不是拉平请求条数**。

**参考实现的现状**:SGLang `data_parallel_controller.py::LoadBalanceMethod`(ROUND_ROBIN/TOTAL_REQUESTS/TOTAL_TOKENS/FOLLOW_BOOTSTRAP_ROOM)与 vLLM DP 均**只在准入/派发时选 DP**,**都不做在途 DP↔DP 迁移**。lake 因存算分离,KV(target + drafter)归池权威,迁移代价远低于两者,才使在途再均衡可行。

**机制:抢占重算式(非活迁)**——参考 vLLM v1 `scheduler.py::_preempt_request`(free blocks + `num_computed_tokens=0` + 塞回 waiting,v1 已用重算式取代 v0 的 swap-to-CPU):
1. 带外监控:worker 异步上报负载信号(队列/in-flight/pad 后 token 数),**控制面带外决策,不进 hot step loop**(守原则 3:worker 只上报,不自决)。
2. 决定迁移 → **停止推理、退出 running**(不在 running 即无 in-flight 冻结冲突,天然原子)。
3. **记录已推进长度**,请求回 waiting → 重派目标 DP。
4. 目标 DP 重新准入:前缀 KV(target + drafter,均在池)**前缀命中**→ 命中式 re-prefill(seed hidden 按重算式由 draft-extend 重建);"重算"基本退化为一次命中重放。
5. "驱逐到 CPU 暂等"在 lake 里就是池把 L0→L1 降级,不需专门 swap 通道。

**控制态交接**(重算式下绝大多数随 request 重派或可由 token 历史重建,**硬核只两项**):
- **必须显式随迁**:① 采样 RNG generator state(seeded 时不迁则续写偏离);② 结构化解码 FSM 游标(guided/regex/ebnf/json,或在目标端重放已生成 token 复原)。
- **随 request 天然携带**:已生成 token 序列(= re-prefill 输入)、采样参数、max/min_tokens、优先级/SLO deadline。
- **可由 token 历史重算**:rep/freq/presence penalty、logprobs 游标、stop-string 部分匹配。
- **丢弃 + 目标端重建**:in-flight draft token(vLLM 抢占即清 `spec_token_ids`)→ 目标端 drafter `post_forward` 重新 seed。

**挑选与边界**:
- 迁移**计划单机内**(DP 间,同权重/同并行配置),传输走 NVLink/PCIe;"传输"本质是池的 **L0↔L0 同层迁移**(复用现有跨节点/同层迁移 + 后台带宽池 <10%),非 drafter/worker 私有通道。
- **优先迁短的**(重算式下短请求重放便宜);避开前缀亲和请求(迁走丢本地命中);加滞回/驻留阈值防抖动。
- **非故障(不走 F4)、非过载(不归 gateway shedding)**:这是节点级/弹性调度里的主动再均衡触发,与"模式选择纯函数"分开(模式选择是准入/重路由算一次,再均衡是对已在跑请求的周期策略)。

## 4. 弹性调度

触发指标：
- Prefill 池：队列长度 > 阈值 / TTFT 接近 SLO 上限 → 扩容。
- Decode 池：ITL P99 接近 SLO / QPS 上升 → 扩容。
- KV Pool：命中率下降 / 容量水位高 → 扩容 KV Node 或下沉冷数据到 L4。

缩容策略：
- 选最闲节点 Drain，in-flight 完成后销毁。
- KV 已在 Pool，无状态丢失。

## 调度器一致性

- **控制面强一致**（etcd）：KV block 位置表、节点拓扑、路由决策日志。
- **数据面最终一致**：调度器读到的负载视图有延迟，调度决策应**幂等可重试**，容忍基于陈旧信息的次优决策。
- **乐观并发**：多个 Router 实例并发决策，冲突时由控制面仲裁（如同一 Prefill 节点被超额分配 → 触发再路由）。

## 开放问题

- 前缀亲和性与负载均衡的冲突：把所有相同前缀请求固定到一个节点会引发热点。
- 预测性扩容：基于流量历史提前预热节点，模型如何选？
- 公平性：多租户下，KV 复用带来的成本节约如何在租户间分配？
- **DP 在途再均衡的多次迁移(遗留)**:同一请求反复迁移的防抖/防饿死(参考 vLLM `num_preemptions` 计数)、迁移次数上限、滞回阈值取值。
- **不均衡的根因判定(遗留)**:imbalance 源于 attention(请求级/序列长度)还是 MoE(专家路由倾斜,需 EPLB 而非迁几条请求),`server_args.eplb_rebalance_*`。判定后才决定迁移是否是对的手段。
- **请求迁移归存算分离开放性**:请求迁移(在途换 DP/换节点)整体归存算分离的开放问题——KV 归池权威使迁移代价低,但控制态交接(RNG/FSM)、seed 状态重建、抖动控制待请求迁移特性立项时细化;现阶段结论是框架无需为此特别预留(drafter KV 已归池)。
