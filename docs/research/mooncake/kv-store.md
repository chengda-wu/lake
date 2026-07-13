# Mooncake — KVCache 存储与池化

> 源码:`mooncake-store/`。Mooncake 的 Store 是"对象级分布式 KV cache",非 KVCache 语义感知的池——前缀复用/内容寻址由引擎侧或外部 Conductor 负责。

## 寻址模型

- **对象键**:opaque 字符串(`using ObjectKey = std::string;`,`types.h`)。**无内容寻址、无 radix tree、无 hash 前缀匹配**。Master 内是 `std::unordered_map<string, ObjectMetadata>` 线性哈希表。
- **Segment ID**:每个内存/NoF segment 有 `UUID id`。`MountedSegment` 映射 segment_id → allocator。Replica 记录所在 segment name。
- **对象切片**:Put 接收 `std::vector<Slice>&`(Slice = `{void* ptr; size_t size}`,caller 定义)。每 slice 分配为一个或多个 `Replica`(每 replica 落不同 segment,best-effort)。**Master 无固定 block 切分**。
- **租户隔离**:`MakeTenantScopedKey(tenant_id\0user_key)` — 租户名 + NUL + key。shard index = `hash(tenant_id) ⊕ hash(user_key)`。`tenant_id` 在 `setup()` 传入,默认 `"default"`。
- **ReplicateConfig**(`replica.h`):`replica_num`、`nof_replica_num`、`with_soft_pin`/`with_hard_pin`、`preferred_segments`、`prefer_alloc_in_same_node`、`data_type`、`host_id`、`group_ids`。

## Master 服务(元数据权威)

`MasterService`(`master_service.h`,~89KB):
- **分片内存元数据**:`std::array<MetadataShard, 1024> metadata_shards_`。每 shard 有 `SharedMutex` + `unordered_map<string, TenantState>` → `unordered_map<string, ObjectMetadata>`。**元数据全在 leader 内存,etcd 非主 KV 存储**。
- **ObjectMetadata**:持 `client_id`、`size`、`data_type`、`group_id`、`tenant_id`、`lease_timeout`(SpinLock)、optional `soft_pin_timeout`、`hard_pinned`、quota 字段、private `vector<Replica> replicas_`。提供 `GrantLease`/`IsLeaseExpired`/`IsSoftPinned`。
- **控制流/数据流分离**:Master 只管元数据(空间分配 + 副本位置),**不经手数据**。数据在 Client↔Client 间直传(经 Transfer Engine)。
- **API**:`PutStart`/`PutEnd`(两阶段写,防脏读)→ `GetReplicaList` → Client 直接 RDMA 读 → `Remove`。`MountSegment`/`UnmountSegment` 动态加减节点。

## 分配策略

`AllocationStrategy`(`allocation_strategy.h`)及 5 个子类:

| 策略 | 行为 |
|------|------|
| `random`(默认) | 纯随机 + preferred-first |
| `free_ratio_first` | 采样 6×N 候选,按空闲比降序 |
| `ssd_free_ratio_first` | SSD 感知 |
| `local_first` | 同 host 优先 |
| `cxl` | CXL 专用 |

**best-effort**:每 slice 副本保证落不同 segment;空间不足时尽量多分配(至少 1 个)。

## Allocator

`BufferAllocatorBase` / `OffsetBufferAllocator`(bin-based offset,低碎片)/ `CachelibBufferAllocator`。`OffsetBufferAllocator` 暴露 `getLargestFreeRegion()` 供策略跳过碎片化 segment,但**无 compaction 线程**。

## DRAM→SSD 分层

`FileStorage` offload + promotion,CountMinSketch 频率准入。DRAM 热数据,SSD 冷数据,按频率提升/下沉。

## HA(高可用)

- **选主**:`LeaderCoordinator` 抽象,三后端:`EtcdLeaderCoordinator`(etcd lease + keepalive)、`RedisLeaderCoordinator`、`K8sLeaderCoordinator`。`MasterServiceSupervisor` 编排生命周期。
- **OpLog 复制**:`OpLogManager` 追加 `OpLogEntry{seq, ts, OpType{PUT_END/PUT_REVOKE/REMOVE/LEASE_RENEW}, key, payload, checksum, prefix_hash}`。`EtcdOpLogStore` 持久化到 `/oplog/{cluster_id}/{seq}`。`HotStandbyService::ReplicationLoop()` 轮询 OpLog → `ApplyOpLogEntry`。
- **快照**:`MasterSnapshotManager` 周期性 fork-based COW 快照(in-memory 元数据 + segment + allocator 状态),序列化到 local/S3。Restore 从最新快照重建。
- **客户端**:`client_service.h` 持 `leader_coordinator_` + `SwitchLeader` + `LeaderMonitorThreadMain`,leader 切换时自动重连。

## 客户端架构

| 类 | 职责 |
|----|------|
| `Client`(`client_service.h`) | 核心客户端接口 |
| `RealClient`(`real_client.h`) | 资源持有者 |
| `DummyClient`(`dummy_client.h`) | 无资源,转发给 RealClient |

## mooncake-p2p-store 与 store 的区别

- **P2P Store**(`mooncake-p2p-store/`,Go-only):**无中心 master**,纯客户端架构。全局元数据由 etcd 维护。`Register`(BitTorrent 式 seeding,本地内存注册到元数据,不分发数据)→ `GetReplica`(从已注册/已下载节点并发拉取分片,拉取后自己也成数据源)。用于 checkpoint 分发。经 cgo 调 Transfer Engine C API。
- **Mooncake Store**:有中心 master 管元数据 + 副本分配 + 淘汰 + HA。面向推理时的 KVCache 池化。

## 故障恢复与一致性

- **对象写原子性**:`PutStart`→`PutEnd` 两阶段。`Get` 只读 `COMPLETE` 状态副本,不读部分写入。
- **Lease**:`GetReplicaList`/`ExistKey` 成功时授予 per-object lease(默认 TTL 5s)。Lease 期内对象免被 Remove/Evict。Lease 过期则 Get 失败(防数据竞争)。
- **Zombie 清理**:`PutStart` 后 `put_start_discard_timeout`(30s)内无 `PutEnd` → 允许新 `PutStart` 抢占;`put_start_release_timeout`(10min)后释放空间。
- **Client 故障**:`ClientMonitorFunc`(1s 间隔)检测 `client_live_ttl_sec`(10s)过期 → `ClearInvalidHandles` 移除已 unmount segment 的副本 + 死 client 的 local_disk 副本。
- **一致性**:对象写后 immutable(强一致),但无跨对象事务。Object group 是 best-effort 生命周期提示。`Get` 保证读完整数据,但"not necessarily the latest"。

## 多模型支持

- **Store 层无 model ID 概念**。对象是 opaque bytes,按任意 string key 寻址。`ObjectDataType` 枚举(KVCACHE/TENSOR/WEIGHT/SAMPLE/...)仅是标签,用于 metrics,不影响寻址或策略。
- **多模型通过 key 命名约定 + tenant 隔离**:应用层在 key 中编码 model 信息(如 vLLM 用 block hash 作 key)。Store 本身不区分模型。
- **Conductor 层有 model 概念**:prefix index 按 `ModelContext{tenant_id, model_name, lora_name, block_size, additional_salt, instance_id}` 分 scope — 但 Conductor 是外部路由组件,不是 Store 功能。
- **Tenant Quota**:可选多租户内存配额(`enable_multi_tenants`),YAML 策略文件定义 per-tenant quota。这是资源隔离,非模型隔离。

## Conductor(cache-aware router)

独立组件(`docs/source/design/conductor/`):订阅 KV 事件(`BlockStored`/`BlockRemoved`),维护全局 prefix cache table,`POST /query` 返回 per-instance longest_matched + medium hit + DP-rank hit。**这是 Mooncake 中唯一的前缀索引/radix 机制,但在 Store 之外**。

## 本系统的借鉴点

Mooncake Store 可直接作为 L3(远端内存池)的物理实现参考,但需在其上增加:
- 内容寻址 + radix 前缀复用(Store 无,Conductor 在外部)
- per-model 配额(Store 仅 tenant-level)
- 主动碎片整理(Store 无 compaction 线程)
- KVCache 语义感知(layer/head/token 寻址,Store 是 opaque bytes)

传输层(Transfer Engine)则可直接复用,见 [transfer-engine.md](transfer-engine.md)。

## 代码索引

> 沿代码回溯用。符号名锚定,行号会漂移——找不到时 `grep -n "符号名" <文件>`。

| 机制 | 文件:符号 |
|------|-----------|
| 对象键(opaque string) | `mooncake-store/include/types.h`::`ObjectKey` |
| Slice/Segment/UUID/ErrorCode | `mooncake-store/include/types.h` |
| 副本配置 | `mooncake-store/include/replica.h`::`ReplicateConfig` / `Replica` |
| 租户隔离键 | `mooncake-store/src/master_service.cpp`::`MakeTenantScopedKey`(tenant\0key;shard=hash(tenant)⊕hash(user_key)) |
| Master 元数据权威 | `mooncake-store/include/master_service.h`::`MasterService` (L88) + `mooncake-store/src/master_service.cpp`(~8000 行) |
| 1024 分片元数据 | `master_service.h`::`MetadataShard`(`std::array<MetadataShard,1024> metadata_shards_`;SharedMutex+unordered_map) |
| 对象元数据 + lease | `mooncake-store/src/master_service.cpp`::`ObjectMetadata`(`GrantLease`/`IsLeaseExpired`/`IsSoftPinned`/`replicas_`) |
| 两阶段写 API | `master_service.h`::`PutStart` / `PutEnd` / `GetReplicaList` / `Remove` |
| 动态加减节点 | `master_service.h`::`MountSegment` / `UnmountSegment` |
| 分配策略抽象 | `mooncake-store/include/allocation_strategy.h`::`AllocationStrategy` |
| 5 种策略子类 | `allocation_strategy.h`::`RandomStrategy`/`FreeRatioFirstStrategy`/`SsdFreeRatioFirstStrategy`/`LocalFirstStrategy`/`CxlStrategy` |
| Allocator(bin-based) | `mooncake-store/include/allocator.h`::`OffsetBufferAllocator` / `BufferAllocatorBase`(`getLargestFreeRegion`)/ `CachelibBufferAllocator` |
| DRAM→SSD 分层 | `mooncake-store/src/file_storage.cpp`::`FileStorage`(offload+promotion+CountMinSketch) |
| 客户端接口 | `mooncake-store/include/client_service.h`::`Client` (L63) |
| 资源持有者 | `mooncake-store/include/real_client.h`::`RealClient` (L73) |
| 无资源转发 | `mooncake-store/include/dummy_client.h`::`DummyClient` (L17) |
| 选主抽象 | `mooncake-store/include/ha/leadership/leader_coordinator.h`::`LeaderCoordinator` |
| 三种选主后端 | `ha/leadership/`::`EtcdLeaderCoordinator` / `RedisLeaderCoordinator` / `K8sLeaderCoordinator` |
| 生命周期编排 | `mooncake-store/`(MasterServiceSupervisor) |
| OpLog 复制 | `mooncake-store/`::`OpLogManager` / `OpLogEntry`(PUT_END/PUT_REVOKE/REMOVE/LEASE_RENEW)+ `EtcdOpLogStore`(`/oplog/{cluster}/{seq}`) |
| Standby 应用日志 | `mooncake-store/`::`HotStandbyService::ReplicationLoop` → `ApplyOpLogEntry` |
| 快照(fork-COW) | `mooncake-store/`::`MasterSnapshotManager` |
| 客户端 leader 切换 | `client_service.h`::`SwitchLeader` / `LeaderMonitorThreadMain` |
| lease + zombie 清理 | `master_service.cpp`(`put_start_discard_timeout`=30s / `put_start_release_timeout`=10min / `client_live_ttl_sec`=10s / `ClearInvalidHandles`) |
| 多租户配额 | `mooncake-store/`(YAML per-tenant quota,`enable_multi_tenants`) |
| Conductor(前缀索引,外部) | `mooncake/docs/source/design/conductor/`(`ModelContext` scope / `POST /query`) |
| P2P store(无 master,Go) | `mooncake-p2p-store/src/`(`Register`/`GetReplica`;cgo → Transfer Engine C API) |
