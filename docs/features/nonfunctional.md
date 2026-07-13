# 非功能需求

横切所有功能的质量属性。与 [`features.md`](features.md)（做什么）、[`slo.md`](slo.md)（多快多稳）正交：这里讲"在安全/成本/可观测等维度上，整个系统必须满足什么"。

---

## 可观测性

- **Metrics**：Prometheus 暴露，至少覆盖
  - 延迟：TTFT / ITL 分位（P50/P95/P99）
  - 吞吐：tokens/s、requests/s
  - KV：Pool 命中率、本地命中率（D-direct 信号）、Pool 容量水位、各层缓存命中分布
  - 池：各算力池节点数、队列长度、in-flight 请求数
  - 弹性：扩缩容事件计数、冷启动时延分布
  - **过载信号**：队列长度、in-flight、估算剩余容量实时上报 gateway，供其做准入/限流决策（推理系统自身不做过载拒绝）
- **Tracing**：每请求一条 trace，span 覆盖 gateway → router → prefill → KV transfer → decode，含跨语言边界（gRPC/RDMA）。
- **Logging**：结构化日志（JSON），含 request_id 贯穿三语言栈。
- **仪表盘**：Grafana 预置"SLO 总览""KV 健康""弹性事件"三块。

## 过载控制（职责边界）

- **归 gateway / 外部控制面**：限并发、拒请求、按优先级丢弃、准入控制。
- **推理系统只管执行**：Worker 不得为保 SLO 自行降 batch size 或丢请求；遇过载只上报指标，由 gateway 决定是否准入。
- 推理系统对已准入的请求负责执行（含 F4 续推）；过载导致的拒绝不计入推理系统失败率。

## 安全

- **多租户隔离**：KV block 按租户隔离，公共前缀共享需显式标记，禁止跨租户读取私有 KV（对应 F8）。
- **鉴权**：Gateway API 支持 token 鉴权；内部 RPC（worker↔pool↔router）mTLS 或内网信任边界。
- **权重保护**：模型权重 artifact 在对象存储加密 at-rest；传输 TLS；防未授权拉取。
- **审计**：模型版本上线/下线、扩缩容决策、租户配置变更可审计。

## 成本

- **单位成本**：定义"每百万 token 成本"指标，含算力 + 存储 + 网络三部分。
- **闲置控制**：闲时算力占比目标 < 20%（靠弹性缩容）。
- **KV 存储成本**：冷热分级，冷数据落对象存储，热数据驻 RAM/NVMe；监控 KV Pool 总存储开销。
- **成本归因**：按租户归因算力与 KV 存储消耗。

## 部署形态

- **最小形态**：单机 docker compose（gateway + router + 1 prefill + 1 decode + 1 kv-pool + minio + etcd），用于本地开发与冒烟。
- **标准形态**：k8s，各池独立 Deployment/HPA，RDMA 可选。
- **RDMA 假设**：生产推荐 RDMA；退化到 TCP 时性能预算放宽（P1 拓扑文档明确）。
- **镜像**：三语言各自镜像，权重不在镜像内（运行时从 Weight Cache 拉）。

## 可维护性

- **构建/测试**：rust（cargo）、go（go test）、python（pytest）各自 CI；proto 改动触发全语言重新生成。
- **接口版本化**：proto 带 version，跨语言兼容性测试。
- **配置**：统一配置（环境变量 + 配置文件），无硬编码地址/阈值。
- **可演进**：模块边界清晰（见 P2 目录划分），单模块可独立替换（如 KV Pool 换存储后端不动 router）。

## 可测试性

- **故障注入**：可人为杀节点、断 KV Pool、延迟 KV 传输，验证 F4 续推（重新路由重选模选点，不设降级链）。
- **mock 模型**：计算层支持 mock 前向（固定输出），用于不带 GPU 的 CI 跑通链路。
- **契约测试**：proto 定义即契约，三语言 client/server 双向校验。

---

## 与功能/SLO 的关系

非功能需求不单独排期，而是**附着在每条功能特性上验收**：
- F1 前缀复用 → 必须满足"可观测性（命中率 metrics）"+"安全（租户隔离）"。
- F7 弹性 → 必须满足"可观测性（冷启动时延）"+"成本（闲置控制）"。
- 每条 SLO → 必须有对应 metrics 支撑度量。

P1 架构设计需为以上每类提供落点（如 metrics 端点、mTLS 配置、成本归因数据流）。
