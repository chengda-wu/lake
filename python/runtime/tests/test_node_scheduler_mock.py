"""C0：无 gRPC 的 node_scheduler 主循环冒烟（Fake pool）。"""

from __future__ import annotations

from typing import Dict

from engine.model_runner import ModelRunner
from engine.pool_iface import ReadyHandle, StepStats
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.scheduler_output import ForwardMode, SchedulerOutput


class FakePool:
    def __init__(self) -> None:
        self.finished = []
        self.prepared_modes = []

    def prepare_step(self, output: SchedulerOutput, reqs: Dict) -> ReadyHandle:
        self.prepared_modes.append(output.forward_mode)
        stats = {}
        for rid in output.num_scheduled_tokens:
            req = reqs[rid]
            if output.forward_mode == ForwardMode.EXTEND:
                nblocks = (len(req.prompt_token_ids) + 7) // 8
                st = StepStats(reused_blocks=0, prefill_blocks=nblocks)
                req.reused_blocks = 0
                req.prefill_blocks = nblocks
            else:
                st = StepStats(reused_blocks=req.reused_blocks, prefill_blocks=0)
            stats[rid] = st
        return ReadyHandle(step_id=output.step_id, stats_by_req=stats)

    def done(self, step_id: int) -> None:
        return None

    def on_request_finished(self, req) -> None:
        self.finished.append(req.req_id)


def test_extend_then_decode_finishes() -> None:
    pool = FakePool()
    runner = ModelRunner(pool)  # type: ignore[arg-type]
    sched = NodeScheduler(pool, runner)  # type: ignore[arg-type]
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


if __name__ == "__main__":
    test_extend_then_decode_finishes()
    print("test_node_scheduler_mock OK")
