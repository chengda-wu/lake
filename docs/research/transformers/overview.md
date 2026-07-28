# Hugging Face Transformers Qwen3 参考

Transformers 是 lake 计算层的**模型定义参考**，用于校准模型类层级、HF config 字段、`nn.Module` 边界和 `forward` 形态。它不是推理服务框架参考；服务端调度、KV 管理、并行通信仍以 vLLM/SGLang/Dynamo 为主。

## 借鉴点

| Transformers 机制 | lake 对应 | 说明 |
|------------------|-----------|------|
| `Qwen3Config` 继承 `PreTrainedConfig`，声明 `model_type`、GQA、RoPE、sliding window、layer types 等 HF 字段 | `Qwen3Config` 固定 Qwen3-0.6B 字段 | 先保留 lake 需要的 dense/full-attention 子集，避免引入完整 HF 配置生态 |
| `Qwen3Model` 是 decoder backbone，`Qwen3ForCausalLM` 顶层持有 `self.model = Qwen3Model(config)` 和 `lm_head` | `Qwen3Model(nn.Module)` + `Qwen3ForCausalLM(nn.Module)` | lake 的模型类应是普通 PyTorch 模型骨架，不应把 dummy 逻辑塞进模型类名或 `load_weights(dummy=True)` |
| `Qwen3ForCausalLM.forward()` 调用 `self.model(...)` 后计算 logits，并返回 causal LM 输出 | lake 保留 `forward` / `compute_logits` 边界 | 当前 dummy decode 仍在 runner 层，后续真 Torch/Triton 后端接入时可替换为真实 hidden/logits |

## 关键差异

- Transformers 面向训练/通用生成，KV 由 `past_key_values`/`Cache` 贯穿模型 API；lake 的 KV 归存储池权威管理，模型 runner 只消费 `AttentionMetadata` / slot mapping，不让模型 API 拥有 KV 生命周期。
- Transformers 的 `Qwen3ForCausalLM` 直接持有 embedding、decoder layers、norm、lm_head；lake 当前只实现 `nn.Module` 结构和 `load_weights(weights)` 边界，真实 layer/Triton kernel 在后续 C 阶段补齐。
- Transformers 不提供 vLLM/SGLang 的 paged attention、weight loader、TP/PP、KV connector、dummy init 语义；lake 的 dummy load 继续放在通用 loader 层，对齐 vLLM 的 loader/模型职责边界。

## 代码索引

- `3rdparty/transformers/src/transformers/models/qwen3/configuration_qwen3.py:Qwen3Config`
- `3rdparty/transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3Attention`
- `3rdparty/transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3DecoderLayer`
- `3rdparty/transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3Model`
- `3rdparty/transformers/src/transformers/models/qwen3/modeling_qwen3.py:Qwen3ForCausalLM`
