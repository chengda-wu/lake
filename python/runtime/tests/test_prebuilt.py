"""C5：PREBUILT + 三模式选路骨架。"""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.model_runner import ModelRunner
from engine.pool_iface import PoolIface
from runtime.exec_mode import ExecMode
from runtime.mode_select import select_exec_mode, should_prebuilt
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.prefix_hint import PrefixHint
from runtime.role import RoleConfig, WorkerRole
from runtime.scheduler_output import ForwardMode


def test_mode_select_d_direct() -> None:
    h = PrefixHint(computed_tokens=16, local_hit=True)
    assert select_exec_mode(h, prompt_len=16) == ExecMode.D_DIRECT
    assert should_prebuilt(h, 16)


def test_mode_select_pd_role() -> None:
    h = PrefixHint()
    assert select_exec_mode(h, prompt_len=8, role=WorkerRole.DECODE) == ExecMode.PD_DISAGG
    assert select_exec_mode(h, prompt_len=8, role=WorkerRole.HYBRID) == ExecMode.COLOCATED


def test_prebuilt_then_decode() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(model_backend="mock", enable_overlap=False)
    runner = ModelRunner(pool, model_backend="mock")
    sched = NodeScheduler(pool, runner, role)

    prompt = list(range(8))
    ag.seed_local_prefix("d1", len(prompt))
    hint = PrefixHint(
        computed_tokens=len(prompt),
        reused_blocks=1,
        local_hit=True,
        prebuilt=True,
    )
    req = build_req_from_generate("d1", "m", prompt, 2, "n0")
    sched.add_request(req, hint=hint)
    assert req.exec_mode == ExecMode.D_DIRECT

    modes = []
    while sched._waiting or sched._running or sched._result_queue:  # noqa: SLF001
        out = sched.schedule()
        if out.total_num_scheduled_tokens == 0:
            sched._drain_results()  # noqa: SLF001
            if not sched._waiting and not sched._running:  # noqa: SLF001
                break
            continue
        modes.append(out.forward_mode)
        sched._run_batch(out)  # noqa: SLF001
        sched._pop_and_process()  # noqa: SLF001

    assert ForwardMode.PREBUILT in modes
    assert ForwardMode.EXTEND not in modes
    assert ForwardMode.DECODE in modes
    done = sched.get_req("d1")
    assert done.finished
    assert done.prebuilt_done
    assert done.exec_mode == ExecMode.D_DIRECT
    assert done.num_output_tokens == 2


if __name__ == "__main__":
    test_mode_select_d_direct()
    test_mode_select_pd_role()
    test_prebuilt_then_decode()
    print("test_prebuilt OK")
