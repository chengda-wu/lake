"""Lake — 彻底的存算分离推理系统（探索性原型）。

本包仅用于验证 docs/ 中的架构假设，非生产实现。
子模块：
  - lake.kv_pool: KV cache 分布式池抽象
  - lake.storage: 分层缓存存储
  - lake.compute: Prefill / Decode 算力池
  - lake.scheduler: 路由与调度
"""

__version__ = "0.0.1"
