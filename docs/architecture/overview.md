# 02 — 总体架构

> ⚠️ 本文档早于 P0 的"混合执行模式"设计（PD 分离 / 混部 / D-direct），当前仍以刚性 P→D 为主描述。P1 将按 [`../features/features.md`](../features/features.md) 的"执行模式"节重写。阅读时请将下文 "Prefill→Decode 固定流转" 理解为三种模式之一。

## 分层视图

```
                       ┌──────────────────────────┐
                       │      Gateway / Router    │   无状态：鉴权、限流、请求路由
                       └────────────┬─────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
      ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
      │  Prefill Pool   │ │  Decode Pool    │ │  (Speculative)  │
      │  计算密集、可扩  │ │  访存密集、有KV  │ │   Draft pool    │
      └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
               │                   │                   │
               ▼                   ▼                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │                  KV Cache Transfer Bus                  │
      │          (RDMA / TCP over fabric，分块传输)             │
      └────────────────────────┬────────────────────────────────┘
                               │
   ┌───────────────────────────┼───────────────────────────┐
   ▼                           ▼                           ▼
┌────────────┐         ┌──────────────┐          ┌──────────────────┐
│ KV Cache   │         │  Weight      │          │  Object Store    │
│ Pool       │         │  Cache (RAM  │          │  (S3 / MinIO)    │
│ (RAM+NVMe) │         │  + NVMe)     │          │  SSOT            │
└────────────┘         └──────────────┘          └──────────────────┘
```

## 关键设计决策

### 1. 算力节点无状态化
Prefill / Decode 节点不持久化状态。权重从 Weight Cache 加载，KV 从 KV Pool 拉取或写回。节点可被随时杀死。调度状态（队列、路由表）由独立的控制面维护（etcd / Redis）。

### 2. Prefill / Decode 物理隔离
借鉴 DistServe / Splitwise。Prefill 池用高算力卡（高 FLOPS）、Decode 池可用访存带宽友好的配置。两者通过 KV Transfer Bus 传递 KV cache，而非 token 隐状态耦合。

### 3. KV Transfer Bus 作为一等组件
Prefill 完成后，KV cache 被推送到 KV Pool，并按路由策略 prefetch 到目标 Decode 节点。传输与计算重叠（pipeline）。这是存算分离能否成立的物理瓶颈，单独建模。

### 4. 分层缓存（存储池统一管理）
```
GPU HBM(L0) ─ 主机 RAM(L1) ─ 本地 NVMe(L2) ─ 远端内存池(L3) ─ 对象存储(L4, SSOT)
```
越上层越快、越小、越贵；越下层越慢、越大、越便宜。**五层全部由存储池统一管理**（放置/驱逐/副本/元数据）——计算节点不拥有任何内存，HBM/RAM/NVMe 是存储池的物理载体而非 worker 私有状态。不存在计算层私有本地缓存；所有 KV 位置（含"哪段 KV 在哪个节点 HBM"）均为存储池权威元数据，使"本地命中"成为存储池放置决策的结果而非自治缓存命中。详见 [`storage-layer.md`](storage-layer.md)。

### 5. 控制面 / 数据面分离
- **控制面**：路由表、负载视图、扩缩容决策、KV 位置元数据。强一致（etcd）。
- **数据面**：实际的 KV 传输、权重加载、模型前向。最终一致、容忍延迟。

## 请求生命周期

```
Request → Gateway
  → Router 查 KV Pool 元数据：前缀是否已有 KV？
    ├─ 命中：只 prefill 增量部分，复用已有 KV
    └─ 未命中：完整 prefill
  → 分配 Prefill 节点（优先放在 KV 命中节点 / 靠近已有前缀的节点）
  → Prefill 产出 KV → 写回 KV Pool → Transfer Bus prefetch 到 Decode 节点
  → Decode 节点逐 token 生成 → 每隔 N 步把增量 KV 增量写回 Pool
  → 完成 → 释放本机 HBM 中的 KV 放置（存储池元数据更新；Pool/L3+ 中副本按 LRU/TTL 保留）
```

## 待解决的开放问题

- KV 传输带宽与 prefill/decode 计算时间的比值在什么范围内容忍 Prefill/Decode 物理分离才划算？
- KV Pool 的分片粒度（per-layer? per-block? per-sequence?）如何影响传输与复用？
- 投机解码（speculative decoding）的 draft model 放在哪一层？
- 故障恢复时，未完成的请求如何基于 KV Pool 续推？
