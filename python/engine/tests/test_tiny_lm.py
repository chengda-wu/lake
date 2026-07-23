"""C3：TinyLM 前向 + node_scheduler(model_backend=tiny_lm)。"""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.model_runner import ModelRunner
from engine.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.sample.greedy import greedy_sample
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig


def test_tiny_lm_deterministic() -> None:
    m = TinyLM(vocab_size=64, d_model=16, n_heads=4, seed=3)
    a = m.greedy_token([1, 2, 3])
    b = m.greedy_token([1, 2, 3])
    assert a == b
    assert 0 <= a < 64


def test_greedy_sample() -> None:
    assert greedy_sample([0.1, 0.9, 0.2]) == 1


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
    test_scheduler_tiny_lm_finishes()
    print("test_tiny_lm OK")
