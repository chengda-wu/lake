"""Qwen3 model backend skeleton."""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.model_runner import ModelRunner
from engine.models.loader import DummyModelLoader
from engine.models.qwen3_meta import (
    QWEN3_0_6B_CONFIG,
    QWEN3_0_6B_MODEL_ID,
    QWEN3_DUMMY_WEIGHT_NAMES,
)
from engine.models.qwen3 import (
    Qwen3ForCausalLM,
)
from engine.pool_iface import PoolIface
from engine.pool_types import ReadyHandle
from runtime.req import Req
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig
from runtime.scheduler_output import ForwardMode, GrammarOutput, SamplingParams, SchedulerOutput


def test_qwen3_config_matches_dense_0_6b_shape() -> None:
    cfg = QWEN3_0_6B_CONFIG
    cfg.validate()
    assert cfg.model_type == "qwen3"
    assert cfg.architecture == "Qwen3ForCausalLM"
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 28
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 8
    assert cfg.head_dim == 128
    assert cfg.vocab_size == 151936
    assert cfg.max_position_embeddings == 40960
    assert cfg.torch_dtype == "bfloat16"


def test_dummy_model_loader_loads_qwen3() -> None:
    loader = DummyModelLoader(
        Qwen3ForCausalLM,
        QWEN3_0_6B_CONFIG,
        QWEN3_DUMMY_WEIGHT_NAMES,
    )
    model = loader.load_model()
    assert model.loaded_dummy_weights is True
    assert "lm_head.weight" in model.loaded_weights


def test_qwen3_load_weights_keeps_model_api_plain() -> None:
    model = Qwen3ForCausalLM(QWEN3_0_6B_CONFIG)
    assert hasattr(model, "training")
    assert model.model.config is QWEN3_0_6B_CONFIG
    assert len(model.model.layers) == QWEN3_0_6B_CONFIG.num_hidden_layers
    assert model.model.layers[0].layer_idx == 0
    assert model.lm_head is model.model.embed_tokens
    assert model.forward([1, 2], [0, 1]) == [1, 2]
    assert model.compute_logits("hidden") == "hidden"
    loaded = model.load_weights([("model.norm.weight", object())])
    assert model.loaded_dummy_weights is False
    assert loaded == {"model.norm.weight"}


def test_scheduler_qwen3_dummy_finishes() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(model_backend="qwen3", enable_overlap=False, max_running_reqs=2)
    runner = ModelRunner(pool, model_backend="qwen3")
    runner.load_model()
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("q1", QWEN3_0_6B_MODEL_ID, list(range(8)), 3, "n0"))
    sched.run_until_idle()
    done = sched.get_req("q1")
    assert done.finished
    assert done.num_output_tokens == 3
    assert all(0 <= t < QWEN3_0_6B_CONFIG.vocab_size for t in done.output_token_ids)
    assert ag.finished == ["q1"]


def test_qwen3_dummy_decode_uses_sampling_bitmask() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = ModelRunner(pool, model_backend="qwen3")
    runner.load_model()
    req = Req(
        req_id="q-mask",
        model_id=QWEN3_0_6B_MODEL_ID,
        prompt_token_ids=[3],
        sampling_params=SamplingParams(max_new_tokens=1, structured_output="json"),
    )
    req.num_computed_tokens = 1
    forced_token = 7
    mask = [False] * QWEN3_0_6B_CONFIG.vocab_size
    mask[forced_token] = True
    output = SchedulerOutput(
        step_id=11,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={req.req_id: 1},
        total_num_scheduled_tokens=1,
        req_forward_modes={req.req_id: ForwardMode.DECODE},
        req_num_computed_at_schedule={req.req_id: 1},
        grammar_output=GrammarOutput(
            req_ids=[req.req_id],
            token_bitmask_by_req={req.req_id: mask},
        ),
        has_structured_output=True,
    )
    ready = ReadyHandle(
        step_id=11,
        block_table_by_req={req.req_id: [0]},
        slot_mapping_by_req={req.req_id: [0]},
    )

    out = runner.execute_model(output, ready, {req.req_id: req})
    assert out.next_token_ids == {req.req_id: [forced_token]}
    assert ag.done_calls == 0
