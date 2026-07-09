# 04 — 计算层

计算层由若干**算力池（compute pool）**组成，每个池是一组同质、可互换的算力节点。节点无状态，可随时销毁/拉起。

## 池划分

### Prefill Pool
- 任务：处理长 prompt，产出 KV cache。
- 特征：计算密集（高 FLOPS 利用率），对 HBM 容量敏感（长序列 KV 大）。
- 调度目标：最大化吞吐（batch 大、并行度高），容忍较高 TTFT。
- 产物：KV block → 写入 KV Pool + Transfer Bus。

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
Idle → Boot (镜像拉起) → Warm (向存储池申请权重/KV 放置) → Ready → Serving → Drain → Terminate
```

- **Warm 阶段**：向存储池申请把所需权重放置到本机 L1/L0、预测热点前缀 KV 放置到 HBM，缩短 Ready 时延。计算节点不自行管理这些内存，而是向存储池发起放置请求。
- **Drain 阶段**：停止接收新请求，完成 in-flight 请求；本机 HBM 中的 KV 放置归还存储池（存储池决定保留/下沉/驱逐，崩溃前已异步落 L3+ 的部分由存储池权威持有）。
- **Terminate**：可安全销毁。彻底存算分离下节点无私有状态——HBM/RAM 中的 KV 本就是存储池的放置副本，节点销毁仅损失未及落 L3+ 的最近增量窗口（F4 续推）。

## 冷启动压缩

冷启动是弹性能力的核心瓶颈，分层处理：
1. **镜像层**：精简容器镜像，模型运行时与权重解耦（权重不在镜像里）。
2. **权重加载**：由存储池把权重放置到本机 L1/L0（而非对象存储直读）；按 layer 流式加载，边加载边可接受请求（layer-async serve）。
3. **CUDA 初始化**：预初始化的进程池 / 常驻 worker（类似进程预热）。
4. **KV 预取**：扩容决策一旦做出，立即由存储池把预测的热门前缀 KV 放置到新节点 HBM（支撑 D-direct 本地命中）。

目标：从扩容决策到 Ready 接受请求 < 10s（待验证）。

## 资源画像

| 池 | GPU 画像 | 内存画像 | 扩缩触发 |
|----|----------|----------|----------|
| Prefill | 高 FLOPS | 大 HBM（长序列） | 队列长度 / TTFT SLO |
| Decode | 高带宽 | 中 HBM（增量 KV） | QPS / ITL SLO |
| Draft | 低端卡即可 | 小 | 投采命中率 |

## 开放问题

- Prefill/Decode 比例随流量变化，是否支持节点在池间动态转换？（带权重迁移成本）
- continuous batching 与 KV 跨节点迁移如何协同（迁移中的 sequence 如何处理）？
- 投机解码的 draft 与 target 在物理分离时，候选传输延迟是否抵消收益？
