# 07 — 相关工作与文献

本仓库的设计大量借鉴以下工作。理解它们是评估本系统假设的基础。

## 存算分离 / Disaggregated Serving

- **Mooncake** (Moonshot AI): KVCache-centric disaggregated architecture，把 KV cache 作为独立分离资源池。本仓库 KV Pool 的直接灵感来源。**源码已作为 submodule 引入** `3rdparty/mooncake`,逐层对应见 [`3rdparty-reference.md`](3rdparty-reference.md)。
- **DistServe** (OSDI'24): Disaggregating prefill and decoding，物理隔离 Prefill/Decode 以分别优化吞吐与延迟。
- **Splitwise** (ISCA'24): Efficient generative LLM inference with phase-based disaggregation，按 phase 分离并建模资源。

## KV Cache 复用与传输

- **vLLM / PagedAttention** (SOSP'23): 块状 KV 内存管理，本系统 block 粒度的原型。
- **CacheGen** / **CacheBlend**: KV cache 的压缩与复用。
- **AttentionStore** (Meta): 把 KV cache 当作可复用的缓存层。
- **SGLang**: RadixAttention 前缀复用，本系统 radix tree 索引的来源。其 **HiCache**(L1 GPU / L2 host / L3 distributed 分层)是本系统 L0-L4 分层的主要参考;**源码已引入** `3rdparty/sglang`,对应与差异见 [`3rdparty-reference.md`](3rdparty-reference.md)。
- **LMCache**: 跨请求/跨实例 KV 复用,多存储后端(CPU/disk/Redis)。**源码已引入** `3rdparty/lmcache`,对应见 [`3rdparty-reference.md`](3rdparty-reference.md)。

## 弹性与冷启动

- **ServerlessLLM** (OSDI'24): serverless 场景下 LLM 的快速加载与冷启动优化。
- **dLoRA / PetS**: serverless 推理的弹性调度。

## 存储分层

- **Lakehouse / Paimon / Iceberg**: 存算分离的数据层范式（本仓库命名 "lake" 的由来），但其面向分析负载；本仓库把类似分层理念迁移到推理 KV。

## 通用参考

- **DeepSpeed-Inference**, **TensorRT-LLM**, **Orca** (continuous batching): 推理引擎基线，本系统在其上做存算分离的解耦。

> 注：以上为方向性参考。SGLang/Mooncake/LMCache 三者源码已引入 `3rdparty/`(submodule),与本项目设计的逐层对应、借鉴点与关键差异见 [`3rdparty-reference.md`](3rdparty-reference.md)。
