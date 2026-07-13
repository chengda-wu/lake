# LMCache — 跨实例复用与存储后端

> 源码:`lmcache/v1/`、`csrc/storage_backends/`、`rust/`、`lmcache/v1/cache_controller/`。

## 缓存键与匹配

- **键**:`CacheEngineKey` = `(model_name, world_size, worker_id, chunk_hash, dtype, request_configs)`(non-MP);MP 的 `ObjectKey` = `(chunk_hash, model_name, kv_rank, object_group_id)`。
- **切分与哈希**:`ChunkedTokenDatabase` 按固定 `chunk_size`(默认 256)切 token,**链式 prefix hash**(每 chunk hash = `hash(prev_hash, chunk_tokens)`)。另有 `SegmentTokenDatabase`(按特殊分隔符切段)。
- **匹配**:无 radix tree(在 vLLM 侧);LMCache 靠 `batched_contains` 从第 0 个 chunk 起顺序探测,断链即停。MP 的 `chunk_hash` 是内容哈希,故 L2 层内容寻址。
- **跨进程一致**:`token_database.py` 强调需 `PYTHONHASHSEED=0`,否则哈希跨进程不一致。

## 跨实例共享(两条路径)

1. **共享远程存储(L2 adapter)**:多实例共用 Redis/Mooncake/S3/Aerospike/NIXL-store,通过 `chunk_hash` 内容寻址天然去重。实例只读写同一 key。
2. **P2P 直传(CPU 内存共享)**:`P2PBackend` 经 `CreateTransferChannel`:
   - `NixlChannel`:底层 NIXL(RDMA/NVLink/TCP)。
   - `PySocketChannel`:ZMQ 控制面 + 原生 socket 数据面。
   - 需 `cache_controller` 维护"谁有哪些 chunk"元数据。
   - PD disagg 走 `NixlStorageBackend`/`pd_backend(_async)`。

## 存储后端

### Python 后端抽象(`lmcache/v1/storage_backend/abstract_backend.py`)

| 接口 | 核心方法 |
|------|----------|
| `StorageBackendInterface` | `contains` / `batched_submit_put_task` / `get_blocking` / `get_non_blocking` / `pin` / `unpin` / `remove` / `touch_cache` |
| `AllocatorBackendInterface` | `allocate` / `batched_allocate` / `get_memory_allocator` |
| `StoragePluginInterface` | 可插拔后端,从 config 加载 |

工厂:`CreateStorageBackends` 按 config 组装 `LocalCPUBackend` / `LocalDiskBackend` / `P2PBackend` / `NixlStorageBackend` / `RemoteBackend`。

### Python 后端实例

| 后端 | 文件 | 说明 |
|------|------|------|
| LocalCPUBackend | `local_cpu_backend.py` | L1 热缓存,pinned CPU 内存,LRU/自定义 cache_policy |
| LocalDiskBackend | `local_disk_backend.py` | 本地 SSD |
| RemoteBackend | `remote_backend.py` | 远程存储,封装 `ConnectorClientBase`(asyncio) |
| P2PBackend | `p2p_backend.py` | 跨实例 P2P(NIXL/socket) |
| NixlStorageBackend | `nixl_storage_backend.py` | NIXL 抽象后端(PD disagg 传输) |
| GDSBackend | `gds_backend.py` | GPUDirect Storage |

### C++ 原生后端(`csrc/storage_backends/`)

- 抽象:`IStorageConnector`(`submit_batch_get/set/exists/delete` + `drain_completions` + `close`,eventfd 驱动)、`ConnectorBase<T>`(模板基类,SQ/CQ + worker thread pool + tiling 分片)。
- 已实现:`redis/`(RESP2 over TCP,参考实现)、`aerospike/`(`BUILD_AEROSPIKE=1`)、`mooncake/`、`fs/`(本地文件)。
- **C++ 原因**:GIL-free 真并发、批量+tiling 减少提交/完成次数、eventfd 非轮询完成。一次实现同时服务 non-MP 与 MP 两模式。

### MP 模式 L2 adapter(`lmcache/v1/distributed/l2_adapters/`)

`L2AdapterInterface`(`base.py`)抽象,`StorageManager` 编排多级 L2 + serde 包装(`SerdeL2AdapterWrapper`)。已注册:`aerospike / dax / fs / fs_native / hfbucket / mock / mooncake_store / native_connector / native_plugin / nixl_store / nixl_store_dynamic / p2p / plugin / raw_block / resp / s3`。

## 分层与配额

- Non-MP:`StorageManager` 串联 LocalCPU(L1)→ LocalDisk/Remote(L2)。
- MP:`distributed/storage_manager.py::StorageManager` + `L1Manager` + 多 `L2Adapter` + `StoreController`/`PrefetchController`/`L1EvictionController`/`L2EvictionController`。tier 词汇 `distributed/tiers.py::Tier{L1,L2,ALL}`。`QuotaManager` 管配额,`store_policy` 决定写哪些 adapter。

## cache_controller(中央协调器)

`lmcache/v1/cache_controller/` + `examples/cache_controller/`:

- **职责**:**元数据协调,不是数据路径**。数据仍在 P2P 或共享存储上流动。
- **元数据**:`RegistryTree`(`utils.py`) — `(instance_id, worker_id) → location → set[chunk_hash]`。
- **命令**:lookup/pin/move/clear/compress/health/full_sync。
- **组件**:`KVController`(`controllers/kv_controller.py`)处理 lookup;`RegistrationController` 管实例注册+心跳;`LMCacheWorker`(`worker.py`)是实例侧代理;`LMCacheControllerManager` 是服务端。带 WebUI。
- **通信**:ZMQ。

## 压缩/serde

- **Non-MP**(`lmcache/v1/storage_backend/naive_serde/`):`NaiveSerializer`(原始)、`CacheGenSerializer/Deserializer`(KV 流式压缩,CUDA kernel + 算术编码)、`KIVISerializer`(2-bit 量化)。
- **MP**(`lmcache/v1/distributed/serde/`):`fp8`、`asym_k16_v8`、`turboquant/`(store/decode kernel)、`multi`(组合)。经 `SerdeL2AdapterWrapper` 透明包裹 L2 adapter。
- 统一接口:`Serializer.serialize(src,dst)→bytes`、`Deserializer`。

## 一致性

**无全局强一致**。元数据靠 controller 的 ZMQ 消息 + 心跳 + 序列号追踪(`WorkerNode.seq_tracker`)+ `RWLockWithTimeout`/`FastLockWithTimeout`。Full sync 是 best-effort(完成阈值默认 0.8,超时 300s)。数据层靠内容寻址天然去重;失效靠 TTL + 显式 clear/evict。

## Rust 组件

`rust/raw_block/`(PyO3 绑定)— 低层裸设备 I/O:
- `RawBlockDevice`:`pwrite_from_buffer`/`pread_into`。
- 后端:`posix`(同步 pread/pwrite)、`io_uring`(Rust 原生 syscall)、`io_uring_cmd`(NVMe passthrough 直通)。
- 被 `RustRawBlockBackend`(non-MP)与 `RawBlockL2Adapter`→`RawBlockCore`(MP)共用。
- **是存储后端(裸块设备 SSD),非传输层,生产可用**。

印证 Rust 适合写存储层 I/O;可参考其 Rust/C++ 桥接(PyO3)与 FFI 模式。

## 代码索引

> 沿代码回溯用。符号名锚定,行号会漂移——找不到时 `grep -n "符号名" <文件>`。

| 机制 | 文件:符号 |
|------|-----------|
| 缓存键(non-MP) | `lmcache/utils.py`::`CacheEngineKey` |
| token 切分 + 链式哈希 | `lmcache/v1/token_database.py`::`ChunkedTokenDatabase`(默认 chunk_size=256) |
| 段切分 | `lmcache/v1/token_database.py`::`SegmentTokenDatabase` |
| MP 分布式键(内容寻址) | `lmcache/v1/distributed/api.py`::`ObjectKey` |
| 顺序探测匹配 | `abstract_backend.py`::`StorageBackendInterface.batched_contains` |
| Python 后端抽象 | `lmcache/v1/storage_backend/abstract_backend.py`::`StorageBackendInterface` / `AllocatorBackendInterface` / `StoragePluginInterface` |
| 后端工厂 | `lmcache/v1/storage_backend/__init__.py`::`CreateStorageBackends` |
| L1 CPU 热缓存 | `lmcache/v1/storage_backend/local_cpu_backend.py`::`LocalCPUBackend` |
| L2 本地盘 | `lmcache/v1/storage_backend/local_disk_backend.py`::`LocalDiskBackend` |
| 远程后端(asyncio) | `lmcache/v1/storage_backend/remote_backend.py`::`RemoteBackend` / `ConnectorClientBase` |
| P2P 跨实例 | `lmcache/v1/storage_backend/p2p_backend.py`::`P2PBackend`::`CreateTransferChannel` |
| 传输通道 | `lmcache/v1/transfer_channel/nixl_channel.py`::`NixlChannel` / `py_socket_channel.py`::`PySocketChannel` |
| NIXL/PD 传输后端 | `lmcache/v1/storage_backend/nixl_storage_backend.py`::`NixlStorageBackend` + `pd_backend.py` |
| GDS 后端 | `lmcache/v1/storage_backend/gds_backend.py`::`GDSBackend` |
| C++ 后端抽象 | `csrc/storage_backends/connector_interface.h`::`IStorageConnector` |
| C++ 后端基类(SQ/CQ+tiling) | `csrc/storage_backends/connector_base.h`::`ConnectorBase<T>` |
| C++ 已实现后端 | `csrc/storage_backends/{redis,aerospike,mooncake,fs}/` |
| MP L2 adapter 抽象 | `lmcache/v1/distributed/l2_adapters/base.py`::`L2AdapterInterface` |
| MP 存储管理器 | `lmcache/v1/distributed/storage_manager.py`::`StorageManager` + `L1Manager` + `StoreController`/`PrefetchController`/`L1EvictionController`/`L2EvictionController` |
| tier 枚举 | `lmcache/v1/distributed/tiers.py`::`Tier{L1,L2,ALL}` |
| 配额管理 | `lmcache/v1/distributed/`(QuotaManager / `store_policy`) |
| serde 包装(adapter 之上) | `lmcache/v1/distributed/`(SerdeL2AdapterWrapper) |
| Non-MP serde | `lmcache/v1/storage_backend/naive_serde/`::`NaiveSerializer` / `CacheGenSerializer` / `KIVISerializer` |
| MP serde | `lmcache/v1/distributed/serde/`::`fp8` / `asym_k16_v8` / `turboquant` / `multi` |
| controller 元数据树 | `lmcache/v1/cache_controller/utils.py`::`RegistryTree`((instance,worker)→location→set[chunk_hash]) |
| controller lookup | `lmcache/v1/cache_controller/controllers/kv_controller.py`::`KVController` |
| 实例注册/心跳 | `lmcache/v1/cache_controller/`(RegistrationController) |
| 实例侧代理 | `lmcache/v1/cache_controller/worker.py`::`LMCacheWorker` |
| 服务端 | `lmcache/v1/cache_controller/controller_manager.py`::`LMCacheControllerManager` |
| Rust 裸设备 I/O | `rust/raw_block/src/lib.rs`::`RawBlockDevice`(`pwrite_from_buffer`/`pread_into`;posix/io_uring/io_uring_cmd) |
| Rust I/O 后端(non-MP/MP 共用) | `RustRawBlockBackend`(non-MP) + `RawBlockL2Adapter`→`RawBlockCore`(MP) |
