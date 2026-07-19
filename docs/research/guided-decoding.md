# Guided / Structured Decoding — SGLang × vLLM（async / overlap 下的同步）

> 源码:`3rdparty/sglang`、`3rdparty/vllm`。本文对照两边对 **grammar-guided / structured output** 的支持，重点回答：在 overlap / async scheduling 下能否避免 host↔device 同步、让 GPU 执行路径**完全无空闲**；以及 xgrammar / llguidance / outlines 等库是否把 FSM 放到了 GPU。
>
> 不涉及 HiCache / KV connector（见各 overview）；计算层总览见 [sglang/model-runner.md](sglang/model-runner.md)、[vllm/compute.md](vllm/compute.md)。thinking 开关/长度预算见 [sglang/thinking-control.md](sglang/thinking-control.md)（与 grammar 正交，可叠加）。

## 一句话结论

**两边都不能在所有场景下做到 device 完全无空闲。** 库侧只有「bitmask 打到 logits」的 GPU kernel；FSM / `accept_token` / `fill_*_bitmask` 仍在 **CPU**。overlap / async 能把这段 CPU 藏进 forward 阴影里（mask 填得比 forward 短时接近零开销），但 **async + structured（尤其再加 spec）仍会被迫等上一轮真实 token**，在 sample 前出现 GPU 气泡。

## 库层：GPU 支持到哪一步

| 能力 | xgrammar | llguidance | outlines |
|------|----------|------------|----------|
| FSM / `accept_token` / `fill_*_bitmask` | **CPU** | **CPU** | **CPU** |
| bitmask 分配 | CPU `int32` packed | CPU | CPU / 散列表 |
| `apply_token_bitmask` | CUDA / Triton / CPU | `llguidance.torch` GPU apply | 多为 `masked_fill_` |

xgrammar 官方集成文档写死：bitmask 在 CPU 填，logits 在 GPU 时再 H2D + GPU kernel。**没有「GrammarMatcher 上 GPU」这条路径。**

引擎侧对应：

| 引擎 | 填 mask | H2D | Apply |
|------|---------|-----|-------|
| SGLang | `GrammarMatcher.fill_next_token_bitmask`（CPU） | `move_vocab_mask(..., non_blocking=True)` | Triton（CUDA）/ `sgl_kernel`（HIP）/ NPU op |
| vLLM | `StructuredOutputManager.grammar_bitmask`（CPU；大批量线程池并行 fill） | `to(..., non_blocking=True)`；V2 `StructuredOutputsWorker` 专用 copy stream | xgrammar CUDA/Triton，或 V2 自研 Triton kernel |

## 理想重叠模型（两边共同追求）

```text
GPU:  forward(N) ──────────────────────────► apply_bitmask + sample(N)
CPU:           accept(N-1) + fill_bitmask(N) ─┘
```

依赖：下一步 bitmask **只依赖已生成 token**（与本步 forward 无关）；故可先 launch forward，再在 CPU 上推进 FSM / 填 mask，仅在 sample 前汇合。xgrammar 论文与 TensorRT-LLM tech blog 同一套路。

破裂点：async / overlap 下本步 sample 往往还依赖**上一 in-flight 步的真实 token**；token 未回 CPU 前不能诚实填 mask → defer sample 或关掉 overlap → GPU 空转。

## SGLang：overlap schedule × grammar

后端选择：`--grammar-backend` ∈ `{xgrammar, outlines, llguidance, none}`（默认 xgrammar）。

### 非 spec：可重叠

`event_loop_overlap` 顺序：

1. `run_batch` 起 GPU forward  
2. `process_batch_result`（`copy_done.synchronize` → token 回 CPU → `accept_token`）  
3. `launch_batch_sample_if_needed` → `update_regex_vocab_mask` → apply → sample  

注释写明：**sample 依赖上一 batch 的 grammar 状态**。正常 decode 下这就是「forward 与 CPU grammar 重叠」；mask 时延 ≪ forward 时开销接近零。

### 硬缺口：overlap + spec + grammar

```1630:1641:3rdparty/sglang/python/sglang/srt/managers/scheduler.py
        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )
```

`is_disable_overlap_for_batch` 在此强制 drain `result_queue`，**关掉 overlap**。

额外同步：grammar 启用时 TP 对 sampled token ids 做 `all_reduce(MIN)`（`sampler.py::_sync_token_ids_across_tp`），防各 rank 采样非确定性导致 FSM 分叉。

**SGLang 判断**：非 spec guided ≈ 能藏进 overlap；**spec + grammar 明确不支持 overlap，做不到无空闲**。

## vLLM：async scheduling × structured output

后端：`xgrammar`（默认）/ `outlines` / `guidance`(llguidance) / `lm-format-enforcer`。

### 理想路径（`EngineCore.step`）

```text
execute_model(non_block=True)
  → get_grammar_bitmask(...)   # CPU，与 GPU forward 并行
  → sample_tokens(grammar_output)
```

与 xgrammar 重叠模型一致；H2D 已 `non_blocking`，V2 runner 用独立 copy stream 避免阻塞 memcpy（历史 PR #12563 同类问题）。

### async + structured：defer sample

`AsyncScheduler` 给 in-flight 请求加 `num_output_placeholders`；若 structured 请求尚有未兑现 placeholder，置 `pending_structured_output_tokens`。

`step_with_batch_queue`：

- **无 pending**：立刻 `get_grammar_bitmask` + `sample_tokens(non_block)`  
- **有 pending**：**defer** `sample_tokens`，等先验 batch 产出真实 token（及可选 draft 校验）后再填 bitmask  

→ 当前 batch 的 forward 可能已结束，GPU 在等 sample → **气泡**。再加 speculative decoding 时还要 draft token 回 CPU，defer 更重。

**vLLM 判断**：常见「非 pending」路径接近零开销；**async 下只要 structured 依赖 in-flight token，就不能保证 device 全程忙**。

## 场景对照

| 场景 | SGLang | vLLM |
|------|--------|------|
| 普通 guided + overlap/async | 可重叠，近零 | 可重叠，近零 |
| guided + speculative | **强制关 overlap** | defer sample / 等 draft |
| FSM 上 GPU？ | 否 | 否 |
| GPU kernel | apply bitmask | apply bitmask（+ copy stream） |
| 编译异步 | grammar_queue + future | `_use_async_grammar_compilation` + executor |

TensorRT-LLM 更进一步：把 grammar advance / mask gen 挂 **CUDA callback**，塞进 CUDA graph，减轻 spec+guided 同步——**本仓 submodule 的 SGLang / vLLM 均未做到这一档**。

## 与 lake 的关系

| 关注点 | 参考实现 | lake |
|--------|----------|------|
| 结构化约束正确性 | xgrammar / llguidance FSM + bitmask | 可直接复用同库或等价接口；FSM 游标属请求控制态（抢占重算时须随迁或重放 token 复原，见 [`../architecture/scheduling.md`](../architecture/scheduling.md)） |
| 隐藏 CPU 开销 | forward ∥ fill_bitmask，sample 前汇合 | **应照搬**此重叠契约；worker 上报信号、gateway 管过载，不在引擎内为 grammar 降 batch |
| device 绝对无空闲 | 未做到（async/spec 破洞） | 若要绝对零气泡：自研 GPU FSM 或 CUDA-callback-in-graph（TRT-LLM 方向），代价远高于接 xgrammar——**默认接受「mask ≪ forward」近零，不把绝对无空闲当硬 SLO** |
| grammar 归属 | host `Req` / scheduler 侧 manager | 与 lake「语义状态在 host、device 只镜像执行必要张量」一致（见 model-runner「请求数据结构」） |

**结论**：接现成库 + 重叠调度即可覆盖主流 structured output；不要假设「库已把 guided decoding 全部 GPU 化」。spec + guided 的无气泡路径是增量课题，不是开箱能力。

## 代码索引

> 符号名稳定锚定，行号会漂移——找不到时 `grep -n "符号名" 3rdparty/<repo>/<文件路径>`。

### 库 / 后端

| 机制 | 文件:符号 |
|------|-----------|
| SGLang xgrammar grammar 对象 | `python/sglang/srt/constrained/xgrammar_backend.py`::`XGrammarGrammar`（`accept_token` / `fill_vocab_mask` / `move_vocab_mask` / `apply_vocab_mask`） |
| SGLang llguidance | `python/sglang/srt/constrained/llguidance_backend.py`::`GuidanceBackend` |
| SGLang outlines | `python/sglang/srt/constrained/outlines_backend.py` |
| SGLang grammar 编译队列 | `python/sglang/srt/constrained/grammar_manager.py`::`GrammarManager` |
| SGLang Triton apply | `python/sglang/kernels/ops/grammar/bitmask_ops.py`::`apply_token_bitmask_inplace_triton` |
| vLLM xgrammar backend | `vllm/v1/structured_output/backend_xgrammar.py`::`XgrammarGrammar.fill_bitmask` |
| vLLM 引擎侧 manager | `vllm/v1/structured_output/__init__.py`::`StructuredOutputManager.grammar_bitmask` |
| vLLM apply（V1 runner 路径） | `vllm/v1/structured_output/utils.py`::`apply_grammar_bitmask` |
| vLLM apply（V2 + copy stream） | `vllm/v1/worker/gpu/structured_outputs.py`::`StructuredOutputsWorker.apply_grammar_bitmask` |

### 调度 / overlap / async

| 机制 | 文件:符号 |
|------|-----------|
| SGLang overlap 主循环 | `python/sglang/srt/managers/scheduler.py`::`event_loop_overlap` / `launch_batch_sample_if_needed` |
| SGLang 关 overlap（spec+grammar） | `scheduler.py`::`is_disable_overlap_for_batch`（`need_grammar_sync`） |
| SGLang 填/打 mask | `python/sglang/srt/sampling/sampling_batch_info.py`::`update_regex_vocab_mask` / `apply_logits_bias` |
| SGLang accept token | `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`::`_accept_grammar_tokens` |
| SGLang TP sync（grammar） | `python/sglang/srt/layers/sampler.py`::`_sync_token_ids_across_tp` |
| SGLang spec 路径 bitmask | `python/sglang/srt/speculative/spec_utils.py`::`generate_token_bitmask` |
| vLLM sync step 重叠 | `vllm/v1/engine/core.py`::`step`（`execute_model` → `get_grammar_bitmask` → `sample_tokens`） |
| vLLM async batch queue + defer | `vllm/v1/engine/core.py`::`step_with_batch_queue`（`pending_structured_output_tokens`） |
| vLLM async placeholder | `vllm/v1/core/sched/async_scheduler.py`::`AsyncScheduler._update_after_schedule` |
| vLLM bitmask 入口 | `vllm/v1/core/sched/scheduler.py`::`get_grammar_bitmask` |
| vLLM sample 打 mask | `vllm/v1/worker/gpu_model_runner.py`::`sample_tokens` |

### 外部文档（非本仓）

| 主题 | 出处 |
|------|------|
| xgrammar CPU fill + GPU apply + 与 forward 重叠 | [XGrammar paper](https://arxiv.org/abs/2411.15100)；[Engine integration](https://xgrammar.mlc.ai/docs/tutorials/engine_integration.html) |
| CUDA callback + CUDA graph（spec+guided） | [TensorRT-LLM tech blog](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog12_Combining_Guided_Decoding_and_Speculative_Decoding.html) |
| vLLM 非阻塞 bitmask H2D | [vllm#12563](https://github.com/vllm-project/vllm/pull/12563) |
