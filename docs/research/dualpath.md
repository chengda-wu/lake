# DualPath — 双路径 KV 加载(agentic 推理)

> 来源:Wu et al., "DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference", arXiv:2602.21548v2 (DeepSeek-AI / PKU / THU)。本文为**核实后的分析**(基于原文 abstract/intro/overview 提取),不是源码 submodule——DualPath 未以 submodule 引入。

## 解决的问题

agentic 多轮推理的 KV-Cache 命中率 ≥95%,瓶颈是 **KV 从存储加载的 I/O**,而非计算。在 PD 分离架构下,这个瓶颈表现为**存储 NIC 带宽不均衡**:prefill 引擎的 storage NIC(SNIC)被"从存储加载 KV"打满,decode 引擎的 SNIC 闲置。agentic 短-append 长上下文使 prefill 侧存储 I/O 需求极高,SNIC 成为整体吞吐瓶颈。

## 双路径机制(核心创新)

每节点两类物理网络,互相**隔离**(DualPath 强调这是架构前提):
- **Compute NIC(CNIC,东西向)**:GPU 间 collective 通信,model inference 的 east-west 流量,带宽大(如 8×400Gbps),呈间歇性突发(集合操作亚毫秒级 burst)。
- **Storage NIC(SNIC,南北向)**:访问数据集/checkpoint/on-disk KV cache,每节点 400Gbps。

DualPath 的双路径是**两条 KV 加载路径**(注意:不是"HBM vs DRAM",也不是"device 网络 vs DRAM 网络"——是 **用哪条物理 NIC / 从哪个引擎加载**):

| 路径 | 机制 |
|------|------|
| **storage-to-prefill**(传统) | KV 从存储经 **prefill 引擎的 SNIC** 加载进 prefill 引擎 |
| **storage-to-decode**(DualPath 新增) | KV 从存储经 **decode 引擎的闲置 SNIC** 加载进 decode 引擎,再经 **CNIC(compute network)用 RDMA 传给 prefill 引擎** |

核心洞察:decode 引擎 SNIC 闲置 → 拿来从存储加载 KV → 走 **CNIC(compute network, RDMA)** 回传 prefill。这(1)绕开 prefill 侧 SNIC 瓶颈;(2)CNIC 带宽远大于 SNIC;(3)CNIC 与 SNIC 隔离,不干扰 latency-critical 的模型 collective 通信(compute network 流量间歇,DualPath 在空隙插入 KV 传输)。

> 方向:**storage-to-decode 路径的最终目的地是 prefill 引擎**——KV 加载进 decode,再经 compute network 回传 prefill,服务下一轮 prefill。这本质是 **D→P**(decode 侧 KV 喂回 prefill),DualPath 的贡献是把它做成利用闲置 SNIC + compute network 的高带宽路径。

配合一个 **global scheduler** 在 prefill/decode 引擎间动态均衡负载。

## 借鉴点(对应我们的设计)

| DualPath 设计 | 我们对应 | 说明 |
|---------------|----------|------|
| **双网络隔离**:CNIC(计算)/ SNIC(存储)物理分离,KV 传输走 CNIC 不干扰模型通信 | 我们的 compute network(L0→L0 RDMA 数据面)/ storage network(L1/L2/L3 访问)分离 | 我们的池统一管理,NIC 带宽是池的资源;DualPath 的"借 decode SNIC"在我们这里=池可选从哪个节点的存储带宽加载 + 经 compute network 中转。见 [`../architecture/kv-cache-pool.md`](../architecture/kv-cache-pool.md) "双网络路径" |
| **storage-to-decode 路径**:借 decode 闲置 SNIC 加载 + CNIC 回传 prefill | **D→P** 作为独立正向流(服务下一轮 prefill) | 我们原生支持:data-flow §3.3 agent 多轮 loop 的 D→P。decode 侧加载/持有的 KV 经 compute network 喂回 prefill,下一轮 prefill 命中省重算 |
| **agentic 多轮是核心驱动** | 我们的首要场景(agent 多轮) | 立地一致:多轮短-append 长上下文,高命中率,KV I/O 主导。见 [`../architecture/execution-modes.md`](../architecture/execution-modes.md) |
| **global scheduler 动态均衡** | Router/调度器 + 池的负载视图 | 我们的调度读池负载视图选路,NIC 带宽分配归池(更彻底:带宽是池资源,非实例私有) |

## 关键差异(我们更彻底)

- **NIC 带宽归属**:DualPath 仍是引擎实例视角(prefill/decode 各自的 SNIC/CNIC),通过调度"借用"对端闲置带宽;我们 **NIC 带宽是池的资源**,池统一调度分配,不存在"借"——池直接决定从哪个节点的存储带宽加载、经哪条 network 中转。
- **KV 所有权**:DualPath 的 KV 仍由引擎持有(加载进引擎 HBM);我们 KV 归池权威,引擎不拥有,L0 是池的物理载体。D→P 在我们是池 agent 发起,引擎降到 publish/pull+fence 不知地址(见 [`../architecture/kv-cache-pool.md`](../architecture/kv-cache-pool.md) "PD 分离下的传输流程")。
- **D→P 的更优子情况(我们独有)**:DualPath 的 storage-to-decode 路径总要从存储读 KV(decode SNIC 加载)。我们多一条——若下一轮 prefill 所需的 KV **已在 D 的 L0**(D 自己上轮 decode 产出、未下沉),则连存储读取都省,**D L0 → P 经 compute network 直传**。这是 D→P 的零存储读取特例,DualPath 不强调(它的动机是均衡 SNIC,零读取不在其框架内)。
- **内容寻址/radix**:DualPath 关注"KV 怎么搬"(双路径带宽),不涉及前缀内容寻址复用;我们 radix + 内容寻址 + 位置视图一跳定位源,D→P 命中的前提是 radix 已生长(上轮反向回传)。
- **分层**:DualPath 聚焦 storage↔HBM 的加载路径,无 L0–L3 统一分层/冷热/多模型配额;我们全层归池统一管理。

## 我们"原生支持"的含义

DualPath 的双路径(用哪条 NIC / 从哪个引擎加载 KV)在我们架构里**已具备物理基础,不新增流类型**:
- **storage-to-prefill** = P 侧从 L1/L2/L3 经 storage network 加载(补拉)。
- **storage-to-decode + CNIC 回传** = D 侧从 L1/L2 经 storage network 加载 + 经 compute network 传 P(D→P)。
- **D L0 → P 直传** = D→P 的零存储读取特例(所需 KV 已在 D 的 HBM)。

池按 NIC 负载/带宽视图选路,因池统一管理而比 DualPath 的"借用"更彻底。选路细节见 [`../architecture/kv-cache-pool.md`](../architecture/kv-cache-pool.md) "双网络路径"。
