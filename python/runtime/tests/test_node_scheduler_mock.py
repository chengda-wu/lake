"""C0/C1：node_scheduler continuous batching + overlap 冒烟（Fake pool）。"""

from __future__ import annotations

from typing import Dict, List, Tuple

from engine.model_runner import ModelRunner
from engine.pool_iface import ReadyHandle, StepStats
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig
from runtime.scheduler_output import ForwardMode, SchedulerOutput


class FakePool:
    def __init__(self) -> None:
        self.finished: List[str] = []
        self.prepared_modes: List[ForwardMode] = []

    def prepare_step(self, output: SchedulerOutput, reqs: Dict) -> ReadyHandle:
        self.prepared_modes.append(output.forward_mode)
        stats = {}
        for rid in output.num_scheduled_tokens:
            req = reqs[rid]
            phase = output.req_forward_modes.get(rid, output.forward_mode)
            if phase == ForwardMode.EXTEND:
                nblocks = (len(req.prompt_token_ids) + 7) // 8
                # 只回 StepStats；由 scheduler._apply_ready_stats 写入 Host Req
                st = StepStats(reused_blocks=0, prefill_blocks=nblocks)
            else:
                # 模拟「decode 误报 reused」：不得覆盖冷启动 reused==0
                st = StepStats(reused_blocks=99, prefill_blocks=0)
            stats[rid] = st
        return ReadyHandle(step_id=output.step_id, stats_by_req=stats)

    def done(self, step_id: int) -> None:
        return None

    def on_request_finished(self, req) -> None:
        self.finished.append(req.req_id)

    def commit_write_extent(self, req_id: str, token_end: int) -> None:
        return None


def _make_sched(overlap: bool = True, max_running: int = 8) -> Tuple[NodeScheduler, FakePool]:
    pool = FakePool()
    runner = ModelRunner(pool)  # type: ignore[arg-type]
    role = RoleConfig(enable_overlap=overlap, max_running_reqs=max_running)
    return NodeScheduler(pool, runner, role), pool  # type: ignore[arg-type]


def test_extend_then_decode_finishes() -> None:
    sched, pool = _make_sched()
    req = build_req_from_generate("r1", "mock-llm", list(range(16)), max_new_tokens=3, node_id="n0")
    sched.add_request(req)
    sched.run_until_idle()
    done = sched.get_req("r1")
    assert done.finished
    assert done.num_output_tokens == 3
    assert done.output_token_ids == [1016, 1017, 1018]
    assert pool.finished == ["r1"]
    assert ForwardMode.EXTEND in pool.prepared_modes
    assert ForwardMode.DECODE in pool.prepared_modes
    # 冷启动：EXTEND prefill>0 写入后，decode 步 reused=99 不得覆盖
    assert done.reused_blocks == 0
    assert done.prefill_blocks == 2


def test_continuous_batching_two_reqs() -> None:
    sched, pool = _make_sched(max_running=4)
    sched.add_request(build_req_from_generate("a", "m", list(range(8)), 2, "n0"))
    sched.add_request(build_req_from_generate("b", "m", list(range(8, 16)), 2, "n0"))
    sched.run_until_idle()
    assert sched.get_req("a").finished and sched.get_req("b").finished
    assert set(pool.finished) == {"a", "b"}
    # 至少有一步同批（EXTEND 双请求或 MIXED）
    assert any(m in (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.DECODE) for m in pool.prepared_modes)


def test_overlap_process_lags_execute() -> None:
    """多请求 decode 稳态：execute(N) 先入队，再 process 上批（对齐 event_loop_overlap）。

    单请求 EXTEND 不可重入，会先 drain 再 decode，看不出重叠；用双请求 decode 段验证。
    """
    sched, _ = _make_sched(overlap=True)
    sched.add_request(build_req_from_generate("a", "m", list(range(8)), 3, "n0"))
    sched.add_request(build_req_from_generate("b", "m", list(range(8, 16)), 3, "n0"))
    sched.run_until_idle()
    tl = sched.timeline
    # 存在某次：execute(s) 出现在 process(s) 之前，且中间夹了另一次 execute
    found = False
    for i, (op, step) in enumerate(tl):
        if op != "process":
            continue
        # 同一 step 的 execute 必须更早
        assert ("execute", step) in tl[:i]
        # 在 process(step) 之前，应已启动过更新的 execute（重叠）
        later_exec = [s for o, s in tl[:i] if o == "execute" and s > step]
        if later_exec:
            found = True
            break
    assert found, tl


def test_sync_loop_process_before_next_execute() -> None:
    sched, _ = _make_sched(overlap=False)
    sched.add_request(build_req_from_generate("r1", "m", list(range(8)), 2, "n0"))
    sched.run_until_idle()
    tl = sched.timeline
    # 同步：process(1) 在 execute(2) 之前
    assert tl.index(("process", 1)) < tl.index(("execute", 2)), tl


def test_future_map_holds_last_token() -> None:
    sched, _ = _make_sched(overlap=True)
    sched.add_request(build_req_from_generate("r1", "m", [1, 2, 3, 4, 5, 6, 7, 8], 2, "n0"))
    sched.run_until_idle()
    done = sched.get_req("r1")
    # 结束后 FutureMap 已 clear
    assert sched.future_map.resolve("r1") is None
    assert done.output_token_ids[-1] == 1010  # seed=8 → 1009,1010


if __name__ == "__main__":
    test_extend_then_decode_finishes()
    test_continuous_batching_two_reqs()
    test_overlap_process_lags_execute()
    test_sync_loop_process_before_next_execute()
    test_future_map_holds_last_token()
    print("test_node_scheduler_mock OK")
