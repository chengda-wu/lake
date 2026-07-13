# 06 — 路由与调度

> ⚠️ 本文档早于 P0 的"混合执行模式"设计。当前路由描述偏向固定 P→D；执行模式与 KV 流转时序见 [`execution-modes.md`](execution-modes.md)，模式选择依存储池本地命中、prompt 规模、传输成本决策（见 [`../features/features.md`](../features/features.md) "执行模式"节）。

调度是存算分离系统能否兑现弹性与低延迟承诺的控制核心。本系统调度器**无状态**（所有决策依据来自控制面的共享视图），可水平扩展。

## 调度层次

```
1. 请求级路由 (Gateway/Router)
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
4. **资源不足处理（不设降级链）**：资源不足或执行失败（故障/超时）时**不**做 mode-to-mode 的预设 fallback。调度器把"增量 prefill 计算 + KV 传输"加权和过大、或某节点不可用作为信号，触发 F4 故障恢复 → Router 依最新集群状态重跑 `f(请求, 集群状态) → (模式, 节点)` 重选模选点。过载层面的限并发/拒请求/按优先级丢弃**归 gateway**，推理系统只管执行并上报剩余容量、队列长度、in-flight 等信号（见 [`../features/slo.md`](../features/slo.md) "过载控制"节）。

## 2. 池间调度

- Prefill → Decode 的 KV 传递通过 Transfer Bus，需在 Prefill 完成时序上对齐 Decode 就绪。
- 投机解码：Draft 池在 Decode 侧生成候选，验证失败回退到标准 decode。
- 反压：Decode 池拥塞时，减缓 Prefill 速率（背压），避免 KV Pool 堆积。

## 3. 节点级调度

- **Continuous batching**：Decode 节点动态拼接 batch。
- **PagedAttention** 风格的块状 KV 管理，与存储池的 block 粒度对齐。
- **放置与 batch 单向耦合（方案 Z）**：同一 batch 各 sequence 的 KV 必须同时在本机 HBM（attention 一次读全部）。存储池按热度主动预放置 KV 到 HBM 并发布位置视图;调度器读视图组 batch（本地命中优先），缺失补拉，不反向指挥放置。见 [`storage-layer.md`](storage-layer.md) / [`execution-modes.md`](execution-modes.md)。
- **抢占**：高优先级请求可抢占低优先级，被抢占者的 KV 在存储池中保留（不丢失，本机 HBM 放置释放归还存储池）。

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
