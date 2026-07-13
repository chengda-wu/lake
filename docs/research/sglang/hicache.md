# SGLang HiCache — 分层机制详解

> 源码:`python/sglang/srt/mem_cache/hiradix_cache.py`、`python/sglang/srt/managers/cache_controller.py`、`python/sglang/srt/mem_cache/radix_cache.py`。

## HiRadixTree 元数据

`TreeNode` 字段(`radix_cache.py`):

| 字段 | 含义 |
|------|------|
| `value` | L1 device indices tensor;`None` 表示已 evicted |
| `host_value` | L2 host indices tensor;非 None 即已 backup |
| `hash_value` | **每页一个 SHA256 链式哈希**,作为 L3 key(见下) |
| `lock_ref` | 防 GPU 驱逐引用计数 |
| `host_ref_counter` | 防 host 驱逐引用计数 |
| `hit_count` | write_through_selective 阈值判断 |
| `write_through_pending_id` | 回写任务标识 |

**关键**:节点精确记录 L1/L2 的存储地址;**不存 L3 位置**,只存 L3 的 key。L3 位置实时查后端。

### 链式哈希(前缀感知,非纯内容寻址)

`get_hash_str` → `get_native_hash` → `hash_binding.cpp::hash_page`:
```
page N 的 key = SHA256(page_{N-1}_digest ‖ page_N_tokens)   // Merkle-like 链
```
同一前缀路径 → 同一 key(故同前缀跨实例可命中);但不同前缀位置的相同 token 串 key 不同。

## local match

`match_prefix` → `_match_prefix_helper`:
- 从 root 沿子节点匹配 token 序列,`page_size>1` 时按页粒度匹配。
- 匹配在节点中间终止时自动 split(`_split_node`),保证未来匹配边界精确。
- 返回连续前缀:前段在 L1(`value`),后段在 L2(`host_value`,evicted 但 backuped)。
- 另计算 `host_hit_length`(evicted 节点 host_value 长度和)与 `last_host_node`(最深 backuped 祖先,作 L3 查询起点)。
- 纯树遍历,无数据拷贝,极快。

## prefetch from L3

`prefetch_from_storage` + controller prefetch 线程:
1. 构造 page-aligned `prefetch_key`;若长度 < `prefetch_threshold`(默认 256 token)则跳过。
2. 预分配 host 页,入队 `PrefetchOperation`。
3. `_storage_hit_query`:逐 batch 调 `storage_backend.batch_exists` 查命中页数 → `all_reduce(MIN)` 跨 TP 同步 → 不足阈值则 revoke(归还 host 内存),否则截断到命中部分。
4. `prefetch_io_aux_func`:逐 batch(`STORAGE_BATCH_SIZE=128` 页)用零拷贝 `_page_get_zero_copy` 或 generic `_generic_page_get` 从 L3 读入 host;`operation.increment` 累计完成 token,可被 `mark_terminate` 中断。
5. `check_prefetch_progress` 轮询 → `can_terminate_prefetch`(按策略判断)→ `terminate_prefetch` → `all_reduce(MIN)` 同步实际完成 token → `_insert_helper_host` 把命中段插入树(L2 only 节点)。

### prefetch 三策略(`can_terminate_prefetch`)

| 策略 | 终止条件 | 适用 |
|------|----------|------|
| `best_effort` | 立即返回,GPU 可执行 prefill 即停 | 极延迟敏感 |
| `wait_complete` | 全部完成 | 高命中率 |
| `timeout` | 完成或超时 | 生产推荐,平衡 |

timeout 公式:
```
timeout = min(cfg.max, cfg.base + cfg.per_ki_token * num_tokens / 1024)
```
默认 `base=2.0s, per_ki_token=0.1s, max=30.0s`,可经 extra_config 覆盖。

## write-back

**L1→L2**(`write_backup` → `cache_controller.write`):
- `mem_pool_host.alloc` + 入 `write_queue` → `start_writing` 在 `write_stream` 上 `backup_from_device_all_layer`(D2H 全层)→ `ack_write_queue` 记 CUDA event。

**L2→L3**(`_finish_write_through_ack` → `write_backup_storage` → `cache_controller.write_storage`):
- DMA 完成后入 `backup_queue` → `backup_thread_func` 逐 batch 调 `page_set_func` 写 L3;`backup_skip` 为真(MLA 且 rank≠0)则跳过。

### write-back 三策略

| 策略 | 阈值 | 行为 |
|------|------|------|
| `write_through` | threshold=1 | 每次 insert 命中即回写 L2→L3,带宽足时收益最强 |
| `write_through_selective` | threshold=2 | `hit_count` 超阈值才回写,只备份热数据 |
| `write_back` | — | 驱逐时才回写;`_inc_hit_count` 直接 return,驱逐走 `_evict_write_back` |

## 内存布局

定义于 `pool_host/mha.py` 的 tensor 形状:

| 布局 | dims | 页内组织 | 用途 |
|------|------|----------|------|
| `layer_first` | `(2, layer_num, size, head, dim)` | 同层 token 连续 | 兼容 GPU 计算核(默认);页数据分散,**无法整页零拷贝** |
| `page_first` | `(2, size, layer_num, head, dim)` | 同 token 全层连续 | 整页连续 → **零拷贝 L3 I/O**;仅配 `kernel` io |
| `page_first_direct` | `(2, page_num, layer_num, page_size, head, dim)` | 页为前导维 | 为 `direct` io 优化,按页批量 memcpy |
| `page_head` | `(2, page_num, head, page_size, layer, dim)` | head-major | 异构 TP split-head 零拷贝 |

兼容性(`server_args.py::_resolve_layout_io_compatibility`):`page_first_direct`+`kernel`→`direct`;`page_first`+`direct`→`page_first_direct`;ROCm 上 `page_first`+`kernel`→`layer_first`。

## 计算与传输重叠

Producer(load stream,`start_loading`):逐层 `load_to_device_per_layer` 后 `producer_event.complete(i)` 记逐层 CUDA event;`LayerDoneCounter` 三缓冲,producer 可领先 consumer。

Consumer(compute stream):`get_key_buffer`/`get_value_buffer` 每层调 `layer_transfer_counter.wait_until(layer_id)` → `LayerLoadingEvent.wait` → 计算流等该层传输 event 后才读 KV。

效果:layer N+1 在 load_stream 传输时,compute stream 算 layer N;仅当传输未赶上才阻塞。

## GPU 辅助 I/O 核

- `kernel`:sgl-kernel CUDA 核 `transfer_kv_per_layer`/`transfer_kv_all_layer`/`*_pf_lf`/`*_mla*`(`sgl-kernel/csrc/kvcacheio/transfer.cu`);JIT 路径 `jit_kernel/hicache.py`。最高 3x baseline。
- `direct`:`cudaMemcpyBatchAsync`(CUDA 12.8+,dlsym 门控)。

## MLA write-back 去重

- **控制器层**:`backup_skip = is_mla_model and tp_rank != 0` — MLA/compressed-MLA 只 rank 0 回写。
- **后端层**(Mooncake):MLA 时所有 TP rank 共享同一 key `{page_hash}_k`(K/V 融合);MHA 每 rank 独立 `{rank}_k`/`{rank}_v`。`batch_set_v1` 先 `_batch_exist`,已存在即跳过。

## 零拷贝传输

`HostKVCache.get_page_buffer_meta(indices)` 返回 `(ptr_list, element_size_list)` — 裸 host 指针 + 字节数,非 tensor。
- `page_first`/`page_first_direct`:每页一个指针(全层连续),整页一次传。
- `layer_first`:每(页,层)两指针,需多次小传输。
- 后端(Mooncake `batch_put_from`/`batch_get_into`、NIXL `register_mr`)直接读写这些指针。需 host buffer 页对齐以支持 O_DIRECT。

## Multi-Rank 同步

`_all_reduce_attn_groups` 对 `attn_cp_group`、`attn_tp_group` 做 all_reduce:
- **命中数同步**:`all_reduce(MIN)` 保证所有 rank 取相同 L3 命中数,避免 threshold 判断不一致。
- **完成数同步**:`all_reduce(MIN)` 取最小完成 token。
- **terminate 同步**:`all_reduce(MAX)`,任一 rank 已终止则全体终止。
- **write/load ack 同步**:`all_reduce(MIN)`,防 NCCL op 序列分叉导致 TP 死锁。

## PP(Pipeline Parallel)同步

`_pp_sync`:PP0 all_reduce 后用 `isend/recv`(tag=`P2PTag.HIRADIX_PP_SYNC`)沿 PP 链向后传;无全局 barrier,异步发送累积,每轮 `_drain_async_work` 收割。

## 与 PD-Disaggregation 集成

- **Prefill 侧**:直接启用 HiCache,跨 prefill 实例共享 KV。
- **Decode 侧异步卸载**:`--disaggregation-decode-enable-offload-kvcache`,实现于 `python/sglang/srt/disaggregation/decode_hicache_mixin.py`:
  - `DecodeHiCachePreallocMixin`:admission 后查 L3 命中长度,发起 L3→L2 prefetch。
  - `DecodeHiCacheTransferMixin`:等 prefetch 完成后做 L2→L1 load_back,分阶段推进 DMA。
  - decode 产出 KV 通过 write-back 写入 L3,供 prefill 节点多轮对话复用。

## 元数据管理(为何 L3 实时查后端)

设计哲学(`hicache_design.md`):HiRadixTree 不存或不持续同步 L3 元数据,访问时实时查后端。

理由:跨实例 L3 元数据强一致成本高(每实例 radix tree 视角不同、驱逐/写入并发),实时查后端避免分布式一致性问题,代价是每次访问一次 `batch_exists` RPC(后端可缓存)。

**kv_events 旁路**:`BlockStored`/`BlockRemoved`(含 `StorageMedium`)经 `ZmqEventPublisher` 发布,供外部 router/indexer 构建跨实例前缀索引——但这是旁路,不回填树自身的 L3 元数据。
