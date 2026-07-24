"""C10：Worker 生命周期 + 容量信号上报。"""

from __future__ import annotations

from engine.model_runner import ModelRunner
from runtime.lifecycle import WorkerLifecycle, WorkerState
from runtime.role import RoleConfig
from runtime.worker_engine import WorkerEngine


class FakePool:
    def prepare_step(self, output, reqs):
        from engine.pool_iface import ReadyHandle

        return ReadyHandle(step_id=output.step_id)

    def done(self, step_id: int) -> None:
        return None

    def on_request_finished(self, req) -> None:
        return None

    def commit_write_extent(self, req_id: str, token_end: int) -> None:
        return None


def test_lifecycle_forward_and_drain() -> None:
    life = WorkerLifecycle()
    life.warm()
    assert life.state == WorkerState.WARM
    life.ready()
    life.serve()
    assert life.accepts_new_requests()
    life.drain()
    assert not life.accepts_new_requests()


def test_engine_capacity_signal() -> None:
    pool = FakePool()
    runner = ModelRunner(pool)  # type: ignore[arg-type]
    eng = WorkerEngine(pool, runner, RoleConfig(model_backend="mock"), coalesce_s=0)  # type: ignore[arg-type]
    eng.start()
    try:
        sig = eng.capacity_signal()
        assert sig.state == WorkerState.SERVING
        assert sig.max_running_reqs == 8
        assert sig.remaining_slots == 8
        assert sig.role == "hybrid"
    finally:
        eng.stop()
    assert eng.lifecycle.state == WorkerState.TERMINATE


def test_stop_fails_orphaned_inflight() -> None:
    """stop 后必须唤醒仍卡在 done.wait 的 inflight（审出：哨兵抢先会孤儿化）。"""
    import threading

    from runtime.node_scheduler import build_req_from_generate
    from runtime.worker_engine import _Inbound

    pool = FakePool()
    runner = ModelRunner(pool)  # type: ignore[arg-type]
    eng = WorkerEngine(pool, runner, RoleConfig(model_backend="mock"), coalesce_s=0)  # type: ignore[arg-type]
    eng.start()
    req = build_req_from_generate("orphan", "m", list(range(4)), 2, "n0")
    item = _Inbound(req=req, hint=None, done=threading.Event(), error=[], result=[])
    eng._inflight[req.req_id] = item  # noqa: SLF001
    eng.scheduler.add_request(req)
    eng.stop(timeout=2.0)
    assert item.done.is_set()
    assert item.error and "stopped" in str(item.error[0]).lower()
