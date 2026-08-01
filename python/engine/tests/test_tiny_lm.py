"""C3：TinyLM 前向 + node_scheduler(model_backend=tiny_lm)。"""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.model_runner import ModelRunner
from engine.model_executor.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.sample.grammar import apply_token_bitmask
from engine.sample.greedy import greedy_sample
from runtime.req import Req
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig
from runtime.scheduler_output import ForwardMode, GrammarOutput, SamplingParams, SchedulerOutput


def test_tiny_lm_deterministic() -> None:
    m = TinyLM(vocab_size=64, d_model=16, n_heads=4, seed=3)
    a = m.greedy_token([1, 2, 3])
    b = m.greedy_token([1, 2, 3])
    assert a == b
    assert 0 <= a < 64


def test_greedy_sample() -> None:
    assert greedy_sample([0.1, 0.9, 0.2]) == 1


def test_apply_token_bitmask() -> None:
    masked = apply_token_bitmask([0.1, 0.9, 0.2], [True, False, True])
    assert greedy_sample(masked) == 2


def test_sample_tokens_uses_grammar_bitmask() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = ModelRunner(pool, model_backend="tiny_lm", tiny_lm=TinyLM(vocab_size=3, d_model=8, n_heads=1))
    req = Req(
        req_id="g1",
        model_id="tiny",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(max_new_tokens=1, structured_output="json"),
    )
    output = SchedulerOutput(
        step_id=1,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"g1": 1},
        total_num_scheduled_tokens=1,
        grammar_output=GrammarOutput(
            req_ids=["g1"],
            token_bitmask_by_req={"g1": [True, False, True]},
        ),
        has_structured_output=True,
    )
    sampled, _ = runner.sample_tokens(output, {"g1": req}, {"g1": [0.1, 0.9, 0.2]})
    assert sampled == {"g1": [2]}


def test_sample_tokens_can_defer_structured_output() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    runner = ModelRunner(pool, model_backend="tiny_lm", tiny_lm=TinyLM(vocab_size=3, d_model=8, n_heads=1))
    req = Req(
        req_id="g2",
        model_id="tiny",
        prompt_token_ids=[0],
        sampling_params=SamplingParams(max_new_tokens=1, structured_output="json"),
    )
    output = SchedulerOutput(
        step_id=2,
        forward_mode=ForwardMode.DECODE,
        num_scheduled_tokens={"g2": 1},
        total_num_scheduled_tokens=1,
        grammar_output=GrammarOutput(
            req_ids=["g2"],
            deferred_req_ids=["g2"],
            reason="waiting for prior token",
        ),
        has_structured_output=True,
    )
    sampled, _ = runner.sample_tokens(output, {"g2": req}, {"g2": [0.1, 0.9, 0.2]})
    assert sampled == {}


def test_scheduler_tiny_lm_finishes() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(model_backend="tiny_lm", enable_overlap=False, max_running_reqs=2)
    runner = ModelRunner(pool, model_backend="tiny_lm", tiny_lm=TinyLM(vocab_size=128, d_model=16, n_heads=4))
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("t1", "tiny", list(range(8)), 3, "n0"))
    sched.run_until_idle()
    done = sched.get_req("t1")
    assert done.finished
    assert done.num_output_tokens == 3
    assert all(0 <= t < 128 for t in done.output_token_ids)
    assert ag.finished == ["t1"]


if __name__ == "__main__":
    test_tiny_lm_deterministic()
    test_greedy_sample()
    test_apply_token_bitmask()
    test_sample_tokens_uses_grammar_bitmask()
    test_sample_tokens_can_defer_structured_output()
    test_scheduler_tiny_lm_finishes()
    print("test_tiny_lm OK")
