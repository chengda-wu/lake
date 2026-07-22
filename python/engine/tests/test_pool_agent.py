"""C2：StorageAgent FFI 契约 + D5 交互序（InMemoryAgent）。"""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.pool_iface import PoolIface
from engine.pool_types import FinishRequest, PoolError, PoolErrorCode, PreparePlan
from runtime.req import Req
from runtime.scheduler_output import ForwardMode, ReqIoSet, SamplingParams, SchedulerOutput


def _plan(step: int, *, read=None, write=None, budget=0, partial=False) -> PreparePlan:
    return PreparePlan(
        step_id=step,
        forward_mode=ForwardMode.DECODE,
        read_set=read or [],
        write_set=write or [],
        num_scheduled_tokens={"r1": 1},
        pull_budget_ms=budget,
        allow_partial_hit=partial,
    )


def test_ready_done_mismatch() -> None:
    ag = InMemoryAgent()
    try:
        ag.done(1)
        raise AssertionError("expected PoolError")
    except PoolError as e:
        assert e.code == PoolErrorCode.NOT_READY


def test_prepare_done_roundtrip() -> None:
    ag = InMemoryAgent()
    plan = _plan(1, write=[ReqIoSet(req_id="r1", token_start=0, token_end=8)])
    ready = ag.prepare_step(plan)
    assert ready.step_id == 1
    ag.done(1)
    assert ag.done_calls == 1


def test_pull_budget_timeout() -> None:
    ag = InMemoryAgent()
    ag.force_pull_reqs.add("r1")
    ag.pull_cost_ms = 30
    plan = _plan(
        1,
        read=[ReqIoSet(req_id="r1", token_start=0, token_end=8)],
        budget=10,
        partial=False,
    )
    try:
        ag.prepare_step(plan)
        raise AssertionError("expected TIMEOUT")
    except PoolError as e:
        assert e.code == PoolErrorCode.TIMEOUT


def test_allow_partial_hit_drops_req() -> None:
    ag = InMemoryAgent()
    ag.force_pull_reqs.add("r1")
    ag.pull_cost_ms = 30
    ag.l0_token_end["r2"] = 4
    plan = _plan(
        1,
        read=[
            ReqIoSet(req_id="r1", token_start=0, token_end=8),
            ReqIoSet(req_id="r2", token_start=0, token_end=4),
        ],
        write=[ReqIoSet(req_id="r2", token_start=4, token_end=5)],
        budget=10,
        partial=True,
    )
    ready = ag.prepare_step(plan)
    assert all(io.req_id != "r1" for io in ready.effective_read_set)
    assert any(io.req_id == "r2" for io in ready.effective_write_set)


def test_deferred_finish_until_done() -> None:
    ag = InMemoryAgent()
    plan = _plan(1, write=[ReqIoSet(req_id="r1", token_start=0, token_end=8)])
    ag.prepare_step(plan)
    ag.on_request_finished(FinishRequest(req_id="r1", node_id="n0"))
    assert ag.finished == []
    ag.done(1)
    assert ag.finished == ["r1"]


def test_pool_iface_facade() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag, pull_budget_ms=0)
    req = Req(
        req_id="r1",
        model_id="m",
        prompt_token_ids=list(range(8)),
        sampling_params=SamplingParams(max_new_tokens=1),
    )
    out = SchedulerOutput(
        step_id=7,
        forward_mode=ForwardMode.EXTEND,
        num_scheduled_tokens={"r1": 8},
        total_num_scheduled_tokens=8,
        write_set=[ReqIoSet(req_id="r1", token_start=0, token_end=8)],
        req_forward_modes={"r1": ForwardMode.EXTEND},
    )
    ready = pool.prepare_step(out, {"r1": req})
    assert ready.step_id == 7
    pool.done(7)
    pool.on_request_finished(req)
    assert "r1" in ag.finished


if __name__ == "__main__":
    test_ready_done_mismatch()
    test_prepare_done_roundtrip()
    test_pull_budget_timeout()
    test_allow_partial_hit_drops_req()
    test_deferred_finish_until_done()
    test_pool_iface_facade()
    print("test_pool_agent OK")
