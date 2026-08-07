# Mooncake — 传输引擎

> 源码:`mooncake-transfer-engine/`。这是 Mooncake 最硬核的工程价值,也是本系统 Transfer Bus 的直接参考。

## 核心抽象(`include/transport/transport.h`)

```
Transport (抽象基类)
├── SegmentID = uint64_t, SegmentHandle = SegmentID
├── BatchID = uint64_t (BatchDesc 指针的重解释,零开销)
├── TransferRequest { OpCode{READ,WRITE}; void* source; SegmentID target_id;
│                     uint64_t target_offset; size_t length; }
├── TransferStatus { TransferStatusEnum s; size_t transferred_bytes; }
│   // enum: WAITING/PENDING/COMPLETED/FAILED/TIMEOUT/...
├── Slice (硬件工作单元,含 union of rdma/tcp/nvlink/cxl/nvmeof/hccl 等传输特定字段)
├── TransferTask (聚合一个请求的所有 Slice,原子计数 success/failed/transferred_bytes)
└── BatchDesc (一个 BatchID 下的所有 TransferTask)
```

**无 `TransportSession` 或 `Buffer` 类** — "session"即 `BatchDesc`/`BatchID`,"buffer"即 `BufferEntry`/`TransferMetadata::BufferDesc`。

## 传输 API 生命周期

```
allocateBatchID(N)
  → submitTransfer(batch_id, {TransferRequest...})
  → 轮询 getTransferStatus(batch_id, task_id, status)
  → freeBatchID(batch_id)
```

`TransferEngine`(`transfer_engine.h`)是用户门面,持有 `impl_`(经典 `TransferEngineImpl`)或 `impl_tent_`(新版 TENT 引擎,构造时自动选择)。关键方法:`init()`、`installTransport(proto, args)`、`openSegment(name)`→`SegmentHandle`、`registerLocalMemory(addr, len, location, remote_accessible, update_metadata)`、`allocateBatchID()`、`submitTransfer()`、`getTransferStatus()`。

## RDMA 零拷贝实现

### 内存注册(`src/transport/rdma_transport/rdma_transport.cpp`)

- `RdmaTransport::registerLocalMemory()` → 在**每个 NIC 的 RdmaContext** 上调 `registerMemoryRegion(addr, length, access)`。
- `RdmaContext::registerMemoryRegion` → `ibv_reg_mr(pd_, addr, length, access)`。访问权限:`IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ`,可选 relaxed ordering。
- GPU 内存:默认走 DMA-BUF 路径(无需 nvidia-peermem);`WITH_NVIDIA_PEERMEM=1` 时用 `ibv_reg_mr()` 直接注册。
- 注册后构建 `BufferDesc`,收集**每个 NIC 的 lkey/rkey 向量**(一个 buffer 对多 NIC),发布到元数据服务。大区域(≥4GiB)并行预 touch + 并行注册(`MC_ENABLE_PARALLEL_REG_MR`)。

### 零拷贝传输路径(`RdmaTransport::submitTransferTask`)

1. 取本地 `SegmentDesc`,将请求按 `kBlockSize`(`MC_SLICE_SIZE`)切分为 `Slice`。
2. `selectDevice()` 按源地址定位 buffer + 用 `desc->topology.selectDevice(location, retry_count)` 选 NIC。
3. 填充 `slice->rdma.source_lkey`(本地 buffer 的 per-NIC lkey)、`slice->rdma.dest_addr`(目标 offset)、`slice->rdma.dest_rkey`(从 peer 发布的 BufferDesc 查得,per-NIC)。
4. Slice 按 RdmaContext 分组派发。

### RDMA write/read(`RdmaEndPoint::submitPostSend`,`rdma_endpoint.cpp`)

- 构建 `ibv_sge`(`sge.addr = source_addr; sge.lkey = source_lkey`)和 `ibv_send_wr`(`opcode = IBV_WR_RDMA_READ/WRITE; wr.rdma.remote_addr = dest_addr; wr.rdma.rkey = dest_rkey`)。
- `ibv_post_send(qp_list_[qp_index], ...)` — NIC 直接 DMA 读写已注册内存,**无 CPU 拷贝**。Slice 分布在多个 QP 上。
- 每个 RdmaContext 运行 `WorkerPool`,worker 线程轮询 CQ(`ibv_poll_cq`),完成时调 `slice->markSuccess()/markFailed()`。

## 多 NIC 带宽聚合

### 拓扑发现(`src/topology.cpp`)

- `Topology::discover()` 经 `ibv_get_device_list` 枚举 IB 设备,从 `/sys/class/infiniband/<dev>/../../numa_node` 读 NUMA 亲和性。
- `discoverCpuTopology()`:每 CPU NUMA node → `preferred_hca`(同 NUMA NIC)+ `avail_hca`(跨 NUMA)。Entry name = `"cpu:N"`。
- `discoverCudaTopology()`:每 GPU → 同 NUMA 且 PCI 距离最小的 NIC 为 `preferred_hca`。Entry name = `"cuda:N"`。
- 构建 `priority_matrix`,广播到集群。

### 设备选择 `Topology::selectDevice(storage_type, retry_count)`

- `retry_count==0`:从 `preferred_hca` 随机(或 `MC_PATH_ROUNDROBIN=1` 轮询)选一个 NIC;preferred 不可用则 fallback 到 `avail_hca`。
- **随机分布即多 NIC 带宽聚合的核心**:一个请求的多个 Slice 随机分配到不同 NIC,各自有独立 lkey/rkey。
- retry 时按 `(retry_count-1) % total` 轮询 preferred+avail。
- `MC_ENABLE_DEST_DEVICE_AFFINITY` 启用同名牌卡亲和(rail-optimized 拓扑下减 QP 数)。

### 故障处理

NIC 临时不可用时自动切换备选路径重传(`MC_RETRY_CNT`)。Endpoint 池用 **SIEVE 算法**(`MC_ENDPOINT_STORE_TYPE=SIEVE`,默认)管理连接驱逐,失败连接从两端池移除,下次传输重建。

## TCP 退化

`transfer_engine_impl.cpp` init:`MC_FORCE_TCP` 或未检测到 HCA → `installTransport("tcp", nullptr)`。`TcpTransport` 用 ASIO socket 流式传输,**CPU 拷贝(非零拷贝)**,单 Slice 不分块。v2 协议支持 WRITE 确认帧(接收方确认 payload 已写入目标内存),避免静默丢数据(`MC_TCP_PROTO=1` 回退 v1 无确认)。

## 传输后端清单(`include/transport/`)

16+ 后端:`rdma_transport`、`tcp_transport`、`nvlink_transport`(MNNVL 跨节点)、`intranode_nvlink_transport`(节点内)、`efa_transport`(AWS)、`nvmeof_transport`(GDS/cufile)、`hip_transport`(AMD GPU IPC/XGMI)、`cxl_transport`、`cxi_transport`(HPE Slingshot)、`barex_transport`、`kunpeng_transport`(UB/URMA)、`maca_transport`、`sunrise_link_transport`、`ascend_transport/`(ascend_direct + hccl + heterogeneous_rdma + ubshmem)、`device/`(IBGDA GPU 发起 RDMA + P2P)。

## PD 分离 KV 搬迁

prefill 节点算完 KV 后,经 Transfer Engine 将 KVCache 块从 prefill worker 的 HBM/DRAM RDMA 写到 decode worker:
```
submitTransfer(BatchID, {TransferRequest{WRITE, source=local_kv_addr,
                       target_id=decode_segment, target_offset=remote_kv_addr, length}})
```
集成侧由 vLLM `MooncakeConnector`(`kv_role=kv_producer/consumer`)或 SGLang PD disaggregation 后端驱动。

## Python 零拷贝 API

`put_from(key, buffer_ptr, size)` / `get_into(key, buffer_ptr, size)` / `batch_put_from` / `batch_get_into`,需先 `register_buffer(ptr, size)`。

## 元数据插件

`MetadataStoragePlugin`(`transfer_metadata_plugin.h`):etcd / redis / http 后端。维护 `SegmentDesc`/`BufferDesc`/`HandShakeDesc`。

## lake 落地(P4.4)

| Mooncake | lake |
|----------|------|
| `Transport` + `allocateBatchID`/`submitTransfer`/`getTransferStatus` | `rust/transfer`::`Transport` + `TcpTransport` |
| `MC_FORCE_TCP` → `TcpTransport`(CPU 拷贝) | `TcpTransport`(进程内段拷贝)+ `TcpDataService` Put/Get |
| `TransferEngine` 门面 | `TransferService` gRPC(`TransferServer`)挂 storage-agent |
| `RdmaTransport` | **P5**(同 trait,FFI) |

**关键差异**:抄 API/生命周期与 TCP 退化语义,**不**接入 Mooncake CMake/FFI;位置视图/radix 仍在 controlplane,不在 TE。

### 为何 P4 不接 FFI、P5 才接(阶段论)

FFI 成本恒定(CMake 进 Rust workspace、工具链双重钉死、CI 面、FFI 崩溃调试面),收益只在两个条件成立时存在:**有 RDMA 硬件,且传输性能是被验证的变量**。

- **P4 两个条件都不成立**:验证的是池语义(radix/位置视图/ref 冻结/durable-first),对字节怎么搬不敏感;原型机无 HCA,真接 TE 也走 `MC_FORCE_TCP`/无 HCA 的 TCP 退化——花全部成本买到已有的 TCP 兜底。
- **P5 两个条件都成立**:存算分离验证的核心变量就是传输时间(D-direct 零传输收益、5ms 模式选择预算核算、DualPath 带宽分配),TCP 进程内拷贝量不出来 → FFI 进场。
- **P5 为什么不继续自写**:性能路径要的是硬件原语工程(verbs/QP/CQ 轮询/DMA-BUF 注册 GPU 内存/拓扑多 NIC),自写 = 再造一个 TE;此时 FFI 比自研便宜,还白拿上游维护。
- **顺序的意义**:先自写 `Transport` trait 把"湖需要传输层长什么样"钉死,P5 让 TE 适配我们的形状;若 P4 就 FFI,池语义会被 TE 的 C++ 接口形状(注册/错误/batch 生命周期)拖着走。

### 三件套:接口 + 原型实现 + 未来第二实现

- `Transport` trait = **永久资产/防腐层**,业务代码(storage-agent/kv-pool)只面对它。
- `TcpTransport` = **不扔的 fallback**:P5 后仍作无 RDMA 环境的兜底(对应 TE 自己的 TCP 退化),不是一次性脚手架。
- `RdmaTransport`(P5)= 同 trait 的**第二个实现**,FFI 适配 TE;动作是"加一个实现 + 按环境选择",不是"换掉原型"。
- 备选保持开放:NIXL 可作 trait 另一实现;`TransferService` gRPC 形状也允许 TE 以 sidecar 进程外接入——FFI 与进程外两条路都留着。

## 代码索引

> 沿代码回溯用。符号名锚定,行号会漂移——找不到时 `grep -n "符号名" <文件>`。

| 机制 | 文件:符号 |
|------|-----------|
| 传输核心类型(SegmentID/BatchID/TransferRequest/Slice/TransferTask/BatchDesc) | `mooncake-transfer-engine/include/transport/transport.h` |
| 用户门面 | `mooncake-transfer-engine/include/transfer_engine.h`::`TransferEngine`(`init`/`installTransport`/`openSegment`/`registerLocalMemory`/`allocateBatchID`/`submitTransfer`/`getTransferStatus`) |
| 经典实现 | `mooncake-transfer-engine/src/transfer_engine_impl.cpp`::`TransferEngineImpl`(TCP 退化判断:`MC_FORCE_TCP`/无 HCA) |
| TENT 新引擎 | `mooncake-transfer-engine/src/tent/`(`impl_tent_`) |
| 元数据(Segment/Buffer/HandShake) | `mooncake-transfer-engine/include/transfer_metadata.h`::`TransferMetadata` |
| 元数据插件抽象 | `mooncake-transfer-engine/include/transfer_metadata_plugin.h`::`MetadataStoragePlugin`(etcd/redis/http) |
| RDMA 传输 | `mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp`::`RdmaTransport`(`registerLocalMemory`/`registerMemoryRegion`/`submitTransferTask`) |
| RDMA 上下文(per-NIC) | `rdma_transport.cpp`::`RdmaContext`(`ibv_reg_mr`) |
| RDMA 端点(ibv_post_send) | `mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp`::`RdmaEndPoint`::`submitPostSend`(`ibv_sge`/`ibv_send_wr`/`ibv_post_send`) |
| CQ 轮询 worker | `rdma_transport.cpp`::`WorkerPool`(`ibv_poll_cq` → `slice->markSuccess/markFailed`) |
| 多传输路由 | `mooncake-transfer-engine/include/multi_transport.h`::`MultiTransport` |
| NUMA 拓扑发现 | `mooncake-transfer-engine/src/topology.cpp`::`Topology::discover` / `discoverCpuTopology` / `discoverCudaTopology`(`ibv_get_device_list` + `/sys/.../numa_node`) |
| 设备选择(多 NIC 聚合核心) | `mooncake-transfer-engine/src/topology.cpp`::`Topology::selectDevice` (L706;random/roundrobin over preferred/avail_hca) |
| TCP 退化传输 | `mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp`::`TcpTransport`(ASIO;v2 确认帧 `MC_TCP_PROTO`) |
| GPU 内存注册路径门控 | `rdma_transport.cpp`(`MC_ENABLE_PARALLEL_REG_MR` / `WITH_NVIDIA_PEERMEM`) |
| Endpoint 池驱逐(SIEVE) | `rdma_endpoint.cpp`(`MC_ENDPOINT_STORE_TYPE=SIEVE`) |
| 传输后端清单 | `mooncake-transfer-engine/include/transport/`(rdma/tcp/nvlink/efa/nvmeof/cxl/cxi/hip/ascend/barex/kunpeng/... + `device/`) |
| Python 零拷贝 API | `mooncake-wheel/mooncake/`(`put_from`/`get_into`/`batch_put_from`/`batch_get_into`/`register_buffer`) |
| vLLM PD 集成 | `mooncake-wheel/mooncake/mooncake_connector_v1.py`::`MooncakeConnector`(`kv_role=kv_producer/consumer`) |
