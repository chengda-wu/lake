# Sampling Parameters — SGLang × vLLM

> 源码:`3rdparty/sglang/python/sglang/srt/sampling/sampling_params.py`、`managers/io_struct.py::GenerateReqInput`；`3rdparty/vllm/vllm/sampling_params.py`。  
> 结构化约束细节见 [guided-decoding.md](guided-decoding.md)；thinking 预算见 [sglang/thinking-control.md](sglang/thinking-control.md)。

## 一句话结论

两边共享 OpenAI 风格核心采样（temperature / top_p / top_k / min_p / penalties / stop / n），但**命名、挂载位置、扩展能力不同**。SGLang 的 `n` 是**同 prompt 复制成 n 条独立采样**（不是 beam search）；vLLM 在开启 speculative decoding 时**硬禁** `min_p` 与 `logit_bias`——因为 spec 路径根本不装这两个 logits processor，装了也会与 rejection sampling 分布不一致。

## 两边都有（核心采样）

| 参数 | vLLM | SGLang | 备注 |
|------|------|--------|------|
| `temperature` | ✓ 默认 1.0；≈0 → greedy | ✓ 默认 1.0；≈0 → 内部改 `top_k=1` | greedy 语义不同 |
| `top_p` | ✓ (0,1] | ✓ | |
| `top_k` | ✓ `0`/`-1`=全词表 | ✓ `-1`→内部 `TOP_K_ALL` | 禁用约定不同 |
| `min_p` | ✓ | ✓ | vLLM：与 spec decode **硬不兼容**（见下） |
| `presence_penalty` | ✓ [-2,2] | ✓ | |
| `frequency_penalty` | ✓ [-2,2] | ✓ | |
| `repetition_penalty` | ✓ >0 | ✓ (0,2] | vLLM 上界更松 |
| `n` | ✓ | ✓ | 并行样本数；**不是 beam**（见下） |
| `stop` / `stop_token_ids` | ✓ | ✓ | |
| `ignore_eos` | ✓ | ✓ | |
| `skip_special_tokens` | ✓ | ✓ | |
| `spaces_between_special_tokens` | ✓ | ✓ | |
| `logit_bias` | ✓ `dict[int,float]` | ✓ `dict[str,float]` | vLLM：与 spec decode **硬不兼容** |
| seed | `seed` | `sampling_seed` | 命名不同 |
| 最大/最小生成长度 | `max_tokens` / `min_tokens` | `max_new_tokens` / `min_new_tokens` | 命名不同 |
| stop 串是否保留在输出 | `include_stop_str_in_output` | `no_stop_trim`（语义相反） | |

## 结构化约束

| | vLLM | SGLang |
|--|------|--------|
| 形态 | 嵌套 `structured_outputs` | 扁平字段 |
| JSON schema | `structured_outputs.json` | `json_schema` |
| JSON object | `json_object` | （用 schema / `$$ANY$$` 等） |
| regex | ✓ | ✓ |
| grammar | `grammar` | `ebnf` |
| choice | ✓ | ✗（引擎 SamplingParams 无） |
| structural_tag | ✓ | ✓ |

互斥：两边都要求多种 grammar 约束只能选一种。详见 [guided-decoding.md](guided-decoding.md)。

## 仅 vLLM（引擎 `SamplingParams`）

| 参数 | 作用 |
|------|------|
| `bad_words` | 禁止词（tokenize 成 token 序列） |
| `allowed_token_ids` | 白名单 token |
| `thinking_token_budget` | thinking 长度预算（`-1`=不限） |
| `repetition_detection` | N-gram 重复早停 |
| `logprobs` / `prompt_logprobs` | 采样 / prompt logprobs |
| `logprob_token_ids` | 指定 token 的 logprob |
| `flat_logprobs` | 扁平 logprob 格式 |
| `detokenize` | 是否 detokenize |
| `output_kind` | CUMULATIVE / DELTA / FINAL_ONLY |
| `extra_args` | 插件扩展 |
| `BeamSearchParams` | **独立** beam search API（`beam_width` / `length_penalty`…） |

## 仅 SGLang

**在 `SamplingParams` 上：**

| 参数 | 作用 |
|------|------|
| `stop_regex` | 正则 stop |
| `custom_params` | 给 custom logit processor 的 JSON 参数 |
| `stream_interval` | 流式间隔 |

**在请求级 `GenerateReqInput`（不进 SamplingParams）：**

| 参数 | 对应 vLLM |
|------|-----------|
| `return_logprob` / `top_logprobs_num` / `logprob_start_len` / `token_ids_logprob` | ≈ `logprobs` + `prompt_logprobs` + `logprob_token_ids` |
| `custom_logit_processor` | vLLM 靠 `logits_processors` / 插件，非同构字段 |
| `return_hidden_states` / `return_routed_experts` | vLLM 另有 routed experts 相关配置 |

内置 custom processor 示例：`ThinkingBudgetLogitProcessor`、`DisallowedTokensLogitsProcessor`（近似 vLLM 的 thinking budget / allowed·bad tokens，但是扩展路径）。

---

## 深挖 1：SGLang「主路径是 n + sampling」是什么意思？

**不是 beam search。** OpenAI / 两边引擎的 `n` 语义都是：同一 prompt 产出 **n 条彼此独立的完整 completion**。SGLang 的实现是请求展开：

1. `GenerateReqInput._handle_parallel_sampling` 读 `sampling_params["n"]` → `parallel_sample_num`  
2. `_normalize_batch_inputs` 把 batch 扩成 `batch_size * n`  
3. `_expand_inputs` 把 `text` / `input_ids` / `sampling_params` 等 **按 n 份复制**  
4. 之后每条副本走普通 continuous-batching **独立采样**（各自 temperature / top_p / RNG）

```401:451:3rdparty/sglang/python/sglang/srt/managers/io_struct.py
    def _handle_parallel_sampling(self):
        ...
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        ...
    def _normalize_batch_inputs(self):
        ...
            num = self.batch_size * self.parallel_sample_num
```

| | `n` + sampling（SGLang / OpenAI 主路径） | 经典 beam search（vLLM `BeamSearchParams`） |
|--|------------------------------------------|---------------------------------------------|
| 假设数关系 | n 条**独立**序列，互不看对方分数 | 共享前缀的 beam，按联合分数剪枝 |
| 每步动作 | 各自 sample 一个 next token | 对每条 beam 扩词表 / top-k，再全局取 top-`beam_width` |
| 打分 | 无跨序列比较 | `length_penalty` 等联合打分 |
| 多样性来源 | 随机采样（temperature 等） | 搜索宽度 + 可选温度 |
| SGLang | **这就是 `n` 的全部含义** | 引擎主路径**无**等价 `BeamSearchParams` |

所以说「主路径是 n + sampling」=：要多候选时，用 **独立并行采样**；不要指望 SGLang 默认 decode 环里有 HF/vLLM 那种 beam 状态机。vLLM 则把 beam 拆成**另一条 API**（`BeamSearchParams`），与 `SamplingParams.n` 分开。

### 实现结果会不会有差异？

**会有系统性差异——它们不是同一算法的两种实现。**

| | vLLM beam search（`BeamSearchParams`） | SGLang `n` + sampling（≈ vLLM 的 `SamplingParams.n`） |
|--|----------------------------------------|------------------------------------------------------|
| 每步 | 各 beam 扩候选，**全局**按累计 logprob 留 `beam_width` 条 | n 条序列**各自** sample，互不看对方 |
| 目标 | 近似高概率序列（MAP / 搜索） | 从分布里独立抽 n 个样本 |
| 分数 | `cum_logprob / len^length_penalty`（见 `beam_search/utils.py`） | 无跨序列打分 |
| 多样性 | 宽度内的竞争路径；`temperature=0` 时偏确定 | 靠 temperature / top_p / seed，路径可完全分叉 |
| 前缀 | 共享前缀、剪掉弱分支 | 无共享剪枝；弱路径也会跑完 |

典型对照：

1. **同一 prompt、`beam_width = n` 且 temperature=0**：beam 更偏向「模型认为概率高」的几条；`n` 采样若每条都 greedy（top-1）会因无搜索而**全部相同**（同一条贪心路径）；有温度时则是 n 条随机轨迹——候选集合通常与 beam 的 top-n **对不齐**。
2. **有温度时**：`n` 采样方差大、可重复性依赖 seed；beam 仍由联合分数主导，结果更像「几条高分变体」。
3. **等价对比对象**：不要拿「vLLM beam」对「SGLang `n`」。要比两边「多候选」是否一致，应对 **vLLM `SamplingParams.n` ↔ SGLang `n`**（都是独立并行采样）；beam 是第三条路径。

用途分工：要「几条好答案 / 搜索」→ beam；要「多样本估计 / 投票 / 多样性」→ `n`。

---

## 深挖 2：为何 vLLM 禁止 `min_p` / `logit_bias` 与 speculative decoding 同开？

硬拒绝在 `SamplingParams._validate_spec_decode`：

```877:882:3rdparty/vllm/vllm/sampling_params.py
        if self.min_p > _SAMPLING_EPS or self.logit_bias:
            raise ValueError(
                "The min_p and logit_bias sampling parameters "
                "are not yet supported with speculative decoding."
            )
```

**根因不是「理论上不能」，而是「spec 路径没接上」。**

### 普通采样路径

`build_logitsprocs` 默认装三个内置 processor：

```49:53:3rdparty/vllm/vllm/v1/sample/logits_processor/__init__.py
BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
]
```

`MinPLogitsProcessor.apply`：softmax → 按 `max_prob * min_p` 阈值 → 填 `-inf`。  
`LogitBiasLogitsProcessor.apply`：对指定 `(req, token)` 加 bias。两者都假定 logits 形状是 **`[num_reqs, vocab]`**（每请求一行）。

### Spec 路径

开启 speculative decoding 时，**同一函数故意只留 `MinTokensLogitsProcessor`**：

```200:209:3rdparty/vllm/vllm/v1/sample/logits_processor/__init__.py
    if vllm_config.speculative_config:
        ...
        logger.warning(
            "min_p and logit_bias parameters won't work with speculative decoding."
        )
        return LogitsProcessors(
            [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)]
        )
```

Rejection sampler 侧的 logits 是 **展平的** `[Σ(draft_i)+batch, vocab]`（每条 draft 位置一行 + bonus）。它对已支持的约束做了「按 draft 展开」：

- penalties / `allowed_token_ids`：`repeat_interleave(num_draft_tokens)` 后 mask  
- `bad_words`：`apply_bad_words_with_drafts`  
- `min_tokens`：`MinTokensLogitsProcessor.apply_with_spec_decode`  

**没有** `MinPLogitsProcessor.apply_with_spec_decode`，也**没有**在 rejection 路径里调用 `LogitBiasLogitsProcessor`。

### 若强行同开会怎样？

Rejection sampling（Leviathan 等）要求：用 **与最终采样一致的 target 分布** `q(·)` 去验收 draft token。若 `min_p` / `logit_bias` 只在「无 draft 的普通 sampler」上生效、却未作用到 verify 的每一行 logits，则：

1. accept/reject 用的是**未约束**（或错误约束）的 `q`；  
2. 最终输出分布 ≠ 用户请求的 min_p / bias 分布；  
3. 更糟时静默错误（看起来在跑，结果不对）。

因此引擎选择 **请求级硬失败**（`ValueError`），而不是装上半截处理器。自定义 logits processor 在 spec 下同样直接拒绝（`STR_SPEC_DEC_REJECTS_LOGITSPROCS`）。

> 旁注：rejection sampler 注释还提到 draft verify 阶段对 top_p/top_k 的支持与 bonus token 采样不同——那是另一条限制；与本次「禁 min_p/logit_bias」直接相关的是 **processor 未装 + 无 draft 展开 apply**。

### 对本系统

| 点 | 借鉴 |
|----|------|
| `n` | 多候选默认用独立并行采样即可；若要真 beam，需单独状态机与 API，勿与 `n` 混名；**结果集合与 beam 不对齐**（见上） |
| min_p / logit_bias × spec | 任何「改 target 分布」的约束，必须进入 **verify 展平行** 的同一 apply 路径；做不到就明确拒绝，禁止静默忽略 |

---

## 命名与挂载差异速查

1. `max_tokens` ↔ `max_new_tokens`；`seed` ↔ `sampling_seed`；`include_stop_str_in_output` ↔ `no_stop_trim`。  
2. greedy：vLLM `temperature≈0`；SGLang 强制 `top_k=1`。  
3. logprobs：vLLM 在 `SamplingParams`；SGLang 在 `GenerateReqInput`。  
4. 白/黑名单与 thinking 预算：vLLM 一等字段；SGLang 多用 `custom_logit_processor`。  
5. beam：vLLM 独立 `BeamSearchParams`；SGLang 主路径只有 `n` 并行独立采样。

## 代码索引

| 机制 | 文件:符号 |
|------|-----------|
| vLLM SamplingParams | `vllm/sampling_params.py`::`SamplingParams` / `StructuredOutputsParams` / `BeamSearchParams` |
| vLLM beam 打分 / 剪枝 | `vllm/entrypoints/generate/beam_search/utils.py`::`get_beam_search_score`；`offline.py` / `online.py`::`_beam_search_step` |
| vLLM spec 禁 min_p/logit_bias | `sampling_params.py`::`_validate_spec_decode` |
| vLLM spec 时裁剪 logitsprocs | `vllm/v1/sample/logits_processor/__init__.py`::`build_logitsprocs` |
| vLLM MinP / LogitBias | `vllm/v1/sample/logits_processor/builtin.py`::`MinPLogitsProcessor` / `LogitBiasLogitsProcessor` |
| vLLM rejection + 部分约束 | `vllm/v1/sample/rejection_sampler.py`::`RejectionSampler`（`apply_with_spec_decode` / penalties / bad_words） |
| SGLang SamplingParams | `python/sglang/srt/sampling/sampling_params.py`::`SamplingParams` |
| SGLang n 展开 | `python/sglang/srt/managers/io_struct.py`::`GenerateReqInput._handle_parallel_sampling` / `_expand_inputs` |
| SGLang 请求级 logprob / custom processor | `io_struct.py`::`GenerateReqInput`（`return_logprob` / `custom_logit_processor`） |
| SGLang custom processor 示例 | `python/sglang/srt/sampling/custom_logit_processor.py` |
| 官方参数表 | SGLang `docs_new/docs/basic_usage/sampling_params.mdx` |
