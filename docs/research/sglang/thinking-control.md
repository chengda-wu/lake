# SGLang — thinking 控制(reasoning 模型的思考开关/分离)

> 源码:`3rdparty/sglang`(submodule)。本文聚焦 SGLang 对 reasoning 模型(DeepSeek-R1 / Qwen3 / Kimi K2 / GPT-OSS / Apertus 等)thinking 行为的控制与输出分离,**不涉及 HiCache/分层**(见 [overview.md](overview.md) / [hicache.md](hicache.md))。
>
> 调研动机:评估参考实现是否支持"限定长度的 thinking"(在特定 position 控制 logits、或修改已得 token、或未考虑),供 lake 决定是否做 thinking 长度预算。

## 一句话定位

SGLang 的 thinking 控制**全部在 chat-template(prompt 侧)+ 事后文本解析层**,不碰 logits、不改已得 token;**thinking 长度本身未作为一等控制目标**。三层机制:`enable_thinking`/`thinking`(模板开关)→ `reasoning_effort`(模板力度 hint)→ `reasoning_parser`(事后文本切分)。

## 三层机制

### 1. 开关:`enable_thinking` / `thinking`(chat template kwarg)

`ChatCompletionRequest` 把请求的 `thinking` 字段同时 setdefault 进 `chat_template_kwargs` 的 `thinking` 与 `enable_thinking` 两个键,适配不同模型的模板约定:

```827:833:3rdparty/sglang/python/sglang/srt/entrypoints/openai/protocol.py
            # different models check different keys:
            # - "thinking" for deepseek-v3, kimi_k2
            # - "enable_thinking" for qwen3, glm45, nemotron_3, interns1
            ctk.setdefault("thinking", thinking)
            ctk.setdefault("enable_thinking", thinking)
            values["chat_template_kwargs"] = ctk
```

这两个键**只透传给 jinja chat template**,由模型仓库的 tokenizer_config 模板决定 generation prompt 形态。SGLang 自身不据此动 sampling/logits。

> **"强制跳过 thinking"是模板的 trick,不是 SGLang 逻辑**:`enable_thinking=False` 时,Qwen3 的 jinja 模板会在 assistant 前缀预置一段空 `<think>\n\n\n\n`(由 transformers tokenizer 渲染),让模型一开局就"已结束思考"。这是**生成前注入 prompt token**,既不是改 logits、也不是改"已得 token";且该 trick 在模型仓库的模板里,**不在 SGLang 代码内**——SGLang 只透传 `enable_thinking`。

### 2. 力度:`reasoning_effort`(none/low/medium/high/max)——仍是 template hint

`reasoning_effort` 是 OpenAI schema 的 SGLang 扩展(`protocol.py` L697)。`'none'` 把 `thinking`/`enable_thinking` 默认置 false(即"关思考")。在 `serving_chat.py` 里它被塞进 `extra_template_kwargs` 喂模板;只有 dsv4 走专门 encoder(`encode_messages(..., reasoning_effort=...)`,仍是 prompt 编码侧,非 logits):

```840:841:3rdparty/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py
            if request.reasoning_effort is not None:
                extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
```

`_set_reasoning` / `_get_reasoning_from_request`(`serving_chat.py` L1782-1862)全在读写 `chat_template_kwargs[toggle_param]`——**纯模板开关逻辑,无 logits processor 介入**。`reasoning_effort` 是给模型的一个"力度"hint,模型据此自行缩短/拉长 thinking(效果取决于模型训练),框架不强制 token 数。

### 3. 分离:`reasoning_parser`(事后文本切分)

`reasoning_parser.py`::`BaseReasoningFormatDetector`(L21)+ `detect_and_parse`(L64):按 `<think>`…`</think>`(或 Kimi 的 `◁think▷`、GPT-OSS 的 `<|channel|>analysis<|message|>`、Apertus 的 `<|inner_prefix|>` 等定界符)把**已生成的文本**切成 `reasoning_text` / `normal_text`,对应 OpenAI 字段 `reasoning_content` / `content`。**纯字符串解析,不接触 logits、不改 token**。

支持的模型/定界符/parser 名见 `docs_new/docs/advanced_features/separate_reasoning.mdx`(deepseek-r1 / deepseek-v3 / qwen3 / qwen3-thinking / kimi_k2 / gpt-oss / apertus2509)。`separate_reasoning` / `stream_reasoning` 控制是否分离及流式分离(`protocol.py` L734-735)。

## 限定 thinking 长度:未考虑

搜 `thinking_budget|max_think|think_budget|max_thinking|reasoning_effort` + `logits.*think|think.*logits|reasoning.*logits_processor|think.*budget`——**全无命中**。结论:

- **不在特定 position 对 logits 控制**(无"thinking 阶段抬升 `</think>` 概率"的 logits processor)。
- **不直接修改已得 token**(无"检测 think 超长就插入/替换 `</think>`"的逻辑;唯一的 token 注入是上面模板侧的 prompt 预置,属生成前)。
- thinking 长度实际由**模型自己决定**(何时吐 `</think>`),框架侧只有三个间接手段:
  1. `reasoning_effort` 给模板一个"力度" hint,模型自行缩短/拉长(软控制,效果依赖模型);
  2. `max_new_tokens` 兜底——但这是**总长度**上限,不区分 thinking/answer,超了直接截断(可能截在 think 中间,输出半截思考);
  3. 通用 `stop_token_ids` / `regex` / `ebnf` / `min_tokens`(受约束解码)——通用能力,非 thinking 专用,且只能"到点停",不能"限 thinking 段长度后强制转 answer"。

## vLLM 对比(本仓 submodule 版本)

`3rdparty/vllm` 里 `enable_thinking|enable-thinking|reasoning_parser|ReasoningParser|reasoning|<think` **全无命中**。本版本无 reasoning parser、无 thinking 参数——thinking 的开关/分离全靠用户自己套 chat template + 外部解析。

> 注:上游更新的 vLLM 有 `reasoning_parser`(位于 `vllm/entrypoints/`),但本仓 submodule 检出版本(ab132ee98)未含。即便上游版本,其 reasoning parser 同样是**事后文本切分**,不涉及 logits/token 层的长度控制——结论与 SGLang 一致。

## 与 lake 的关系

| 关注点 | SGLang | vLLM(本版) | lake(若要做) |
|--------|--------|-------------|----------------|
| thinking 开关 | `enable_thinking`/`thinking`(模板 kwarg) | 无 | 透传 chat template(同 SGLang) |
| thinking 力度 | `reasoning_effort`(模板 hint) | 无 | 可透传;若做长度预算则需更强机制 |
| thinking/answer 分离 | `reasoning_parser`(事后文本) | 无 | gateway/后处理层做(同 SGLang,纯文本) |
| **限定 thinking 长度** | **未考虑** | **未考虑** | **相对参考实现的增量,需自设计** |

**结论:你问的"限定长度的 thinking"——两个参考实现都没做。** 若 lake 要支持"thinking 长度预算"(think 满 N token 后强制转 answer),是相对 SGLang/vLLM 的增量,落在 logits/token 控制面,两条候选路线:

1. **logits processor 路线**:在 think 段挂 per-step logits processor,到预算位置抬升 `</think>`(及后续 answer 起始 token)概率,让模型自然转出。优点:不改 token 流、与采样兼容;缺点:需在引擎采样路径插入 processor,与 lake"引擎零分层逻辑"的极简契约有张力(类似 SWA 优化的 per-module write-set 张力)。
2. **token 注入路线**:检测 think 段超预算后,直接在输出流注入 `</think>` token(改已得 token),强制转 answer。优点:简单硬切;缺点:改 token 流、可能与模型后续生成不一致(模型没"决定"结束思考)。

两者都触及 lake 计算层与采样/输出流的边界,属"thinking 长度预算"特性(P0 未列)的设计输入,待需要时单独立项。

## 代码索引

> 沿代码回溯用。符号名稳定锚定,行号会漂移——找不到时 `grep -n "符号名" 3rdparty/sglang/<文件路径>`。

| 机制 | 文件:符号 |
|------|-----------|
| 请求 thinking 字段 → chat_template_kwargs | `python/sglang/srt/entrypoints/openai/protocol.py`::`ChatCompletionRequest`(`thinking`/`enable_thinking` setdefault L827-833) |
| reasoning_effort 字段(none/low/medium/high/max) | `protocol.py`::`ChatCompletionRequest.reasoning_effort`(L697) |
| separate_reasoning / stream_reasoning 开关 | `protocol.py`(L734-735) |
| reasoning_effort 喂模板 | `python/sglang/srt/entrypoints/openai/serving_chat.py`(L840-841) |
| dsv4 encoder 消费 reasoning_effort | `serving_chat.py`::`encode_messages(..., reasoning_effort=...)`(L796-800) |
| thinking 开关读写(toggle_param) | `serving_chat.py`::`_set_reasoning`(L1782-1784)/ `_get_reasoning_from_request`(L1786-1862) |
| reasoning_default 模式判定 | `serving_chat.py`::`_reasoning_default_mode`(L1711-1718) |
| 事后文本切分基类 | `python/sglang/srt/parser/reasoning_parser.py`::`BaseReasoningFormatDetector`(L21)/ `detect_and_parse`(L64)/ `StreamingParseResult`(L9) |
| 模板自动检测 | `python/sglang/srt/parser/template_detection.py` |
| 文档(支持模型/parser 表) | `docs_new/docs/advanced_features/separate_reasoning.mdx` |
