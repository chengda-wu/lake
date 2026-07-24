# UCM — 架构与插件面

> 源码：`3rdparty/ucm` @ `37af15e`。总览 [overview.md](overview.md)。  
> 对照：vLLM connector [`../vllm/compute.md`](../vllm/compute.md)；Mooncake store [`../mooncake/kv-store.md`](../mooncake/kv-store.md)；LMCache [`../lmcache/sharing-and-backends.md`](../lmcache/sharing-and-backends.md)。

## 1. Store 层（`UcmKVStoreBaseV1`）

职责：与外部 KV 持久化/池通信；稀疏算法与 PC 共用。

典型原语（Python ABC）：

| 方法 | 含义 |
|------|------|
| `lookup` / `lookup_on_prefix` | 块是否存在；前缀最长命中下标 |
| `prefetch` | 异步预取到高速缓存 |
| `load` / `dump`（及 batch/task） | 与设备内存之间搬 KV |
| `create` / `wait` / `commit` 等 | 空间与异步任务（版本间略有差异） |

C++ 侧 `StoreV1` / `CCStore` 承载热路径；Python 工厂 `UcmConnectorFactoryV1` 按名懒加载后端。

已见后端目录：`nfsstore`、`mooncakestore`、`posix`、`ds3fs`、`pcstore`、`pipeline`、`compress`、`cache`、`fake`、`empty`。

**对 lake**：字节层多后端可对照；**内容寻址 + radix + 位置视图**仍归控制面/B 复用路径，不在 UCM store 内。

## 2. Connector 层（vLLM）

`ucm/integration/vllm/ucm_connector.py` 实现 `KVConnectorBase_V1`（部分类带 `SupportsHMA`）：

| 类 | 角色（概略） |
|----|----------------|
| `UCMConnector` | 主入口，组合 store + HMA 等 |
| `UCMDirectConnector` | 直接路径基类 |
| `UCMLayerWiseConnector` | layer-wise 传输 |
| `UCMPDConnector` | PD 相关路径 |
| `UCMLiteConnector` / `UCMMockConnector` | 轻量 / 测试 |

另有 `hla_connector.py`、`hma_connector.py`、`blend_connector.py`。版本差异靠 `integration/vllm/patch/v0xxx/` 补丁树维护。

**对 lake**：scheduler/worker 双侧 metadata、layer-wise、PD connector 分裂方式可作 **P5 接入形态**对照；权威仍应是池 + 控制面，而非补丁后的引擎内状态。

## 3. 稀疏框架（`UcmSparseBase`）

Scheduler 侧：槽位估计、分配后状态、请求结束元数据。  
Worker 侧：`execute_*` / `attention_*` hooks，负责检索与 load/dump 稀疏块。

`SparseKVManager`（文档叙事）允许算法自定义块分配；与 store 通过 id/offset 解耦。子树含 `esa`、`gsa`、`kvstar`、`blend`、`rerope` 等。

**对 lake**：接口分 scheduler/worker 的做法与 vLLM connector 对称；**近期不必实现稀疏**，只记「算法插件 ≠ 存储权威」。

## 4. PD 分离（经统一池）

文档 `3rdparty/ucm/docs/source/user-guide/pd-disaggregation/` 对比三种 P↔D 传输：

1. HBM 直传  
2. 经 DRAM 间接  
3. **经统一存储池（复用 Prefix Cache）** ← UCM 主推  

论据：P/D 解耦、异常简单、实例无状态、异构（新旧卡/精度）更易。示例代理：`ucm/pd/toy_proxy_server.py`。

**对 lake**：与「KV 归池、计算可弃」同向。差异：

- UCM 仍嵌在引擎 + connector；池命中 ≠ 本地 HBM 命中 → **无 D-direct 一等公民**。  
- lake Router 还要在 PD / 混部 / D-direct 间按位置视图选路；失败走 F4 重跑纯函数，不设 mode 阶梯。

## 5. 集成面

| 路径 | 内容 |
|------|------|
| `ucm/integration/vllm/` | 主战场：connector + 版本 patch |
| `ucm/integration/sglang/` | SGLang 对接 |
| `ucm/integration/mindie/` | MindIE |
| Ascend | 文档 quickstart_vllm_ascend；patch 树含 `vllm_ascend` |

## 代码索引（补充）

| 概念 | 文件:符号 |
|------|-----------|
| Store 工厂注册 | `factory_v1.py`::`UcmConnectorFactoryV1.register_connector` |
| 前缀 lookup | `ucmstore_v1.py`::`lookup_on_prefix` |
| 稀疏角色枚举 | `sparse/base.py`::`UcmSparseRole` |
| PD toy 代理 | `pd/toy_proxy_server.py` |
