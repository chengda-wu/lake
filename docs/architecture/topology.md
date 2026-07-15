# 08 — 部署拓扑

本文把散见于 [`kv-cache-pool.md`](kv-cache-pool.md) "双网络路径"/"L0 直传依赖 GPUDirect RDMA"、[`features/nonfunctional.md`](../features/nonfunctional.md) "部署形态/RDMA 假设"、[`data-flow.md`](data-flow.md) §3.4 选路的部署假设，收成一篇。目标：明确 lake 运行的物理网络假设、RDMA 可用性退化时的性能模型变化、跨机房的故障域与放置边界。

架构契约不变：拓扑只影响**传输引擎内部如何选 NIC/选路径**与**SLO 预算是否放宽**，不改控制面/数据面分离、不改 ref/radix/位置视图、不改 engine-to-engine 控制链切断。退化全部在传输引擎内吸收，上层接口不变。

## 1. 双网络物理分层

节点有两类**物理隔离**的网络（[DualPath](../research/dualpath.md) 的架构前提，[`kv-cache-pool.md`](kv-cache-pool.md) "双网络路径"）：

| 网络 | 方向 | 承载 | 带宽特征 |
|------|------|------|----------|
| **compute network (CNIC, 东西向)** | GPU↔GPU | GPU collective 通信 + L0→L0 RDMA 数据面（PD 正向、D→P 子情况 A） | 大（如 8×400Gbps）、间歇突发（集合操作亚毫秒级） |
| **storage network (SNIC, 南北向)** | 节点↔存储 | 访问 L1/L2/L3（DRAM 池 / NVMe 池 / 对象存储） | 相对小、持续 |

**为何物理隔离**：KV 传输借 CNIC 大带宽回传（D→P 子情况 B），若与 SNIC 共线会挤占 latency-critical 的模型 collective 通信。DualPath 的核心就是两类带宽隔离 + 在 CNIC 空隙插入 KV 传输。两类带宽是**池的资源**（池按 NIC 带宽视图选路，非实例"借用"——比 DualPath 更彻底，见 [`../research/dualpath.md`](../research/dualpath.md) "关键差异"）。

**部署要求**：生产形态每节点配两类独立 NIC（物理或逻辑隔离）。最小/冒烟形态（单机 docker compose）可只有一类，KV 传输与模型通信共享带宽——仅用于冒烟，不满足 SLO。

## 2. RDMA 可用性分级与退化

传输引擎按硬件能力分级（参考 Mooncake `transfer_engine_impl.cpp`：`MC_FORCE_TCP` 或无 HCA → `installTransport("tcp")`，退化在引擎内部、上层接口不变）：

| 级别 | 条件 | 路径 | 性能模型 |
|------|------|------|----------|
| **A. GPUDirect RDMA（最优）** | NIC 与 GPU 同 PCIe root + 支持 peermem/DMA-BUF | HBM ↔ NIC 直读直写，零拷贝、零 CPU | 带宽满 NIC、延迟最低，SLO 预算达标 |
| **B. RDMA 经 pinned host** | 有 IB/RoCE NIC 但与 GPU 跨 PCIe root | HBM → pinned host RAM (L1) → NIC，多一次拷贝 | 带宽仍大、延迟+1 拷贝、占 CPU，SLO 预算略放宽 |
| **C. TCP 退化** | 无 RDMA（`MC_FORCE_TCP` 或无 HCA） | socket 流式，CPU 拷贝（非零拷贝） | 带宽大幅降、延迟升、占 CPU，SLO 预算显著放宽 |

**关键约束**：A 是 L0→L0 直传（PD 正向 / D→P 子情况 A）的最优前提；落到 B/C 时，L0→L0 直传收益下降，选路函数会更倾向混部或 P 侧自拉（决策树阈值随拓扑档位调整，见 [`data-flow.md`](data-flow.md) §2，阈值 P7）。退化不影响正确性，只影响模式选择与 SLO。

**GPU 内存注册门控**（参考 Mooncake `rdma_transport.cpp`）：默认走 DMA-BUF 路径（无需 nvidia-peermem），`WITH_NVIDIA_PEERMEM=1` 时直接 `ibv_reg_mr` 注册 GPU 内存。大区域（≥4GiB，如 KV arena）并行预 touch + 并行注册（`MC_ENABLE_PARALLEL_REG_MR`）——池的 in-process agent（Rust `.so` 拿 worker 的 CUDA 句柄注册 MR）照此，启动期注册一次、运行期零注册开销（见 [`kv-cache-pool.md`](kv-cache-pool.md) "in-process agent"）。

## 3. NUMA / NIC 拓扑发现与多 NIC 聚合

**NUMA-aware NIC 选择**（参考 Mooncake `topology.cpp::Topology::discover`/`selectDevice`）：

- 拓扑发现：`ibv_get_device_list` 枚举 IB 设备 + `/sys/class/infiniband/<dev>/../../numa_node` 读 NUMA 亲和。
- 每 GPU → 同 NUMA 且 PCI 距离最小的 NIC 为 `preferred_hca`，跨 NUMA 的为 `avail_hca`（fallback）。
- 每 CPU NUMA node → `preferred_hca`（同 NUMA）+ `avail_hca`（跨 NUMA）。

**多 NIC 带宽聚合**：一个请求的多个 Slice（block 切分）随机/轮询分配到不同 NIC，各自独立 lkey/rkey（参考 Mooncake `selectDevice` random/roundrobin over preferred/avail_hca）。这是 L0→L0 大块 KV 直传打满多 NIC 带宽的核心——单 NIC 打满会成瓶颈，聚合才兑现 CNIC 的 `8×400Gbps`。

**NIC 故障切换**：NIC 临时不可用自动切备选路径重传（`MC_RETRY_CNT`），连接池用 SIEVE 算法驱逐失败连接、下次重建（参考 `rdma_endpoint.cpp`）。故障切换在传输引擎内、对上层透明。

**池对 NIC 视图的使用**：池读 NIC 负载/带宽视图做选路（D→P 子情况 A/B/P 侧自拉，见 [`data-flow.md`](data-flow.md) §3.4）。该视图由各节点 agent 上报 NIC 拓扑 + 实时利用率，汇总进控制面负载视图（与节点拓扑同在 etcd）。

## 4. 单机房 vs 跨机房

### 单机房（标准生产形态）

```
┌────────── 单机房 ──────────┐
│  计算节点池(CNIC+SNIC, RDMA) │  L0 HBM / L1 本机 DRAM 物理载体在此
│  KV Node 池(RDMA DRAM+NVMe) │  L1 远端 DRAM / L2 NVMe 池物理载体在此
│  etcd + 对象存储(L3)        │  控制面 + SSOT
└────────────────────────────┘
```

- L0→L0 直传、L1/L2 池访问都在同机房 RDMA 域内（μs 级），SLO 预算达标。
- L3（对象存储）同机房或就近机房，冷下沉延迟可控。
- **主形态**，所有执行模式（PD 分离 / 混部 / D-direct）在此域内最优。

### 跨机房（容灾 / 地域分布，远期）

跨机房带来两类问题：

1. **RDMA 域不连续**：跨机房通常无 RDMA（或带宽/延迟退化），L0→L0 直传、L1/L2 池同域访问不能跨机房直连。
2. **故障域扩大**：单机房整体失败需另一机房接管（与 [`consistency.md`](consistency.md) §4 L3 SSOT 抗整机级/池级失败衔接）。

**跨机房策略（远期，不实现）**：
- **执行不跨机房**：请求路由到本机房节点完成，KV 在本机房池内流转（L0–L2 本机房闭环）。
- **L3 跨机房副本**：对象存储跨机房复制（SSOT 跨机房容灾），单机房失败后另一机房从 L3 重建。
- **前缀复用跨机房**：若两机房跑同 model_id，热前缀的 KV 可经 L3 异步同步（冷路径、最终一致），不追求跨机房 L1/L2 强一致。
- **控制面**：etcd 跨机房部署（Raft 跨机房多数派），或各机房独立 etcd + L3 间接同步（弱一致跨机房）。

**当前结论**：跨机房为远期容灾形态，P1 不细化。单机房是 SLO 兑现的基线；跨机房以"执行本机房闭环 + L3 跨机房容灾"为原则，不在跨机房链路上跑 hot path。

## 5. 故障域边界

拓扑决定故障域，与 [`consistency.md`](consistency.md) §4 持久语义分层对应：

| 故障 | 范围 | 恢复 | 见 |
|------|------|------|-----|
| 单 worker 崩溃（NPU/进程级） | 该节点 L0/L1 副本 | L2 F4 恢复点续推（本机 NVMe 通常仍在） | [`consistency.md`](consistency.md) §5 |
| 单 NIC 故障 | 该 NIC 传输 | 切备选 NIC 重传（引擎内） | §3 |
| 单 KV Node 失败 | 该 Node 的 L1/L2 副本 | 其他 KV Node 的 L1/L2 副本 + L3 回填 | [`consistency.md`](consistency.md) §4 |
| 池级失败（控制面 etcd） | 位置视图 | etcd Raft 多数派恢复 | §4、[`consistency.md`](consistency.md) §5 |
| 整机级失败（连本机 NVMe 一起没） | 该节点 L0–L2 | 退 L3（SSOT）续推 / 重算 | [`consistency.md`](consistency.md) §4 |
| 单机房失败 | 该机房全部 | 另一机房从 L3 SSOT 重建（远期） | §4 |

**RDMA 退化不计故障**：TCP 退化是性能降级非不可用，请求照常执行（SLO 放宽），不触发 F4。

## 6. 拓扑速查

| 关注点 | 单机房 | 跨机房(远期) |
|--------|--------|--------------|
| L0→L0 直传 | CNIC RDMA（μs 级） | 不跨机房（本机房闭环） |
| L1/L2 池访问 | 同域 RDMA DRAM/NVMe | 本机房池 |
| L3 SSOT | 就近机房对象存储 | 跨机房副本容灾 |
| 控制面 etcd | 单机房 Raft | 跨机房 Raft 或独立 + L3 同步 |
| SLO 预算 | 达标 | 跨机房链路不跑 hot path |

## 7. 开放问题

- GPUDirect RDMA 的 PCIe root 亲和检测与 A/B 级自动判定（运行时探测 vs 部署声明）待 P7。
- TCP 退化（C 级）下的带宽-延迟量化模型与 SLO 放宽系数待 P7（对接 [`../features/slo.md`](../features/slo.md)）。
- 多 NIC 聚合的 Slice 分配策略（random vs roundrobin vs 按负载）与 CNIC 突发窗避让待 P7（见 [`data-flow.md`](data-flow.md) §3.4 collective 突发窗）。
- 跨机房 L3 副本的同步策略（强一致 RPO=0 vs 异步 RPO>0）与跨机房 etcd 拓扑待容灾特性立项。
