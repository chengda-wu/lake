"""C6：WorkerEngine 长期单环 + 并发入队 continuous batching。"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from engine.model_runner import ModelRunner
from engine.pool_iface import ReadyHandle, StepStats
from runtime.node_scheduler import build_req_from_generate
from runtime.role import RoleConfig, WorkerRole
from runtime.scheduler_output import ForwardMode, SchedulerOutput
from runtime.worker_engine import WorkerEngine


class FakePool:
    def __init__(self) -> None:
        self.finished: List[str] = []
        self.prepared_req_sets: List[frozenset] = []

    def prepare_step(self, output: SchedulerOutput, reqs: Dict) -> ReadyHandle:
        self.prepared_req_sets.append(frozenset(output.num_scheduled_tokens))
        stats = {}
        for rid in output.num_scheduled_tokens:
            req = reqs[rid]
            phase = output.req_forward_modes.get(rid, output.forward_mode)
            if phase == ForwardMode.EXTEND:
                nblocks = (len(req.prompt_token_ids) + 7) // 8
                stats[rid] = StepStats(reused_blocks=0, prefill_blocks=nblocks)
            else:
                stats[rid] = StepStats(reused_blocks=0, prefill_blocks=0)
        return ReadyHandle(step_id=output.step_id, stats_by_req=stats)

    def done(self, step_id: int) -> None:
        return None

    def on_request_finished(self, req) -> None:
        self.finished.append(req.req_id)

    def commit_write_extent(self, req_id: str, token_end: int) -> None:
        return None


def _make_engine(max_running: int = 8) -> Tuple[WorkerEngine, FakePool]:
    pool = FakePool()
    runner = ModelRunner(pool)  # type: ignore[arg-type]
    role = RoleConfig(enable_overlap=True, max_running_reqs=max_running, model_backend="mock")
    eng = WorkerEngine(pool, runner, role)  # type: ignore[arg-type]
    eng.start()
    return eng, pool


def test_submit_single_request() -> None:
    eng, pool = _make_engine()
    try:
        req = build_req_from_generate("r1", "m", list(range(8)), 2, "n0")
        done = eng.submit(req)
        assert done.finished
        assert done.num_output_tokens == 2
        assert pool.finished == ["r1"]
        assert not eng.scheduler.has_req("r1")  # release after return
    finally:
        eng.stop()


def test_concurrent_submit_shared_scheduler() -> None:
    """两并发 submit 进同一 Scheduler；至少一步同批多 req。"""
    eng, pool = _make_engine(max_running=4)
    barrier = threading.Barrier(2)

    def _one(rid: str, prompt: List[int]):
        barrier.wait()
        return eng.submit(build_req_from_generate(rid, "m", prompt, 2, "n0"))

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [
                ex.submit(_one, "a", list(range(8))),
                ex.submit(_one, "b", list(range(8, 16))),
            ]
            results = [f.result(timeout=10) for f in as_completed(futs)]
        assert all(r.finished for r in results)
        assert set(pool.finished) == {"a", "b"}
        # 同 step 多 req：prepare 见到的 req 集合 size>=2
        assert any(len(s) >= 2 for s in pool.prepared_req_sets), pool.prepared_req_sets
    finally:
        eng.stop()


def test_role_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LAKE_WORKER_ROLE", "prefill")
    monkeypatch.setenv("LAKE_MODEL_BACKEND", "tiny_lm")
    monkeypatch.setenv("LAKE_ENABLE_DRAFTER", "1")
    monkeypatch.setenv("LAKE_MAX_RUNNING_REQS", "3")
    monkeypatch.setenv("LAKE_ENABLE_OVERLAP", "0")
    cfg = RoleConfig.from_env()
    assert cfg.role == WorkerRole.PREFILL
    assert cfg.model_backend == "tiny_lm"
    assert cfg.enable_drafter is True
    assert cfg.max_running_reqs == 3
    assert cfg.enable_overlap is False
    # 清掉以免污染其它测试（monkeypatch 会自动还原）
    assert "LAKE_WORKER_ROLE" in os.environ
