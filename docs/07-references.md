# 07 — 相关工作与文献

本仓库的设计大量借鉴以下工作。理解它们是评估本系统假设的基础。

## 存算分离 / Disaggregated Serving

- **Mooncake** (Moonshot AI): KVCache-centric disaggregated architecture，把 KV cache 作为独立分离资源池。本仓库 KV Pool 的直接灵感来源。
- **DistServe** (OSDI'24): Disaggregating prefill and decoding，物理隔离 Prefill/Decode 以分别优化吞吐与延迟。
- **Splitwise** (ISCA'24): Efficient generative LLM inference with phase-based disaggregation，按 phase 分离并建模资源。

## KV Cache 复用与传输

- **vLLM / PagedAttention** (SOSP'23): 块状 KV 内存管理，本系统 block 粒度的原型。
- **CacheGen** / **CacheBlend**: KV cache 的压缩与复用。
- **AttentionStore** (Meta): 把 KV cache 当作可复用的缓存层。
- **SGLang**: RadixAttention 前缀复用，本系统 radix tree 索引的来源。

## 弹性与冷启动

- **ServerlessLLM** (OSDI'24): serverless 场景下 LLM 的快速加载与冷启动优化。
- **dLoRA / PetS**: serverless 推理的弹性调度。

## 存储分层

- **Lakehouse / Paimon / Iceberg**: 存算分离的数据层范式（本仓库命名 "lake" 的由来），但其面向分析负载；本仓库把类似分层理念迁移到推理 KV。

## 通用参考

- **DeepSpeed-Inference**, **TensorRT-LLM**, **Orca** (continuous batching): 推理引擎基线，本系统在其上做存算分离的解耦。

> 注：以上为方向性参考，具体论文链接与版本号在原型验证阶段补全。
