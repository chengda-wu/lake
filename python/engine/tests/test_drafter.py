"""C4：TinyMTPDrafter + TARGET_VERIFY + chain reject。"""

from __future__ import annotations

from engine.agents.memory import InMemoryAgent
from engine.drafter.tiny_mtp import TinyMTPDrafter
from engine.model_runner import ModelRunner
from engine.models.tiny_lm import TinyLM
from engine.pool_iface import PoolIface
from engine.sample.reject import chain_reject_sample
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig
from runtime.scheduler_output import ForwardMode


def test_chain_reject_all_match() -> None:
    # target always returns draft[i] then bonus 9
    drafts = [1, 2]
    calls = {"i": 0}

    def tg(ctx):
        i = calls["i"]
        calls["i"] += 1
        if i < len(drafts):
            return drafts[i]
        return 9

    acc = chain_reject_sample([0], drafts, tg)
    assert acc == [1, 2, 9]


def test_chain_reject_divergence() -> None:
    def tg(ctx):
        return 7

    acc = chain_reject_sample([0], [1, 2], tg)
    assert acc == [7]


def test_drafter_post_pre() -> None:
    d = TinyMTPDrafter(num_draft_tokens=2, vocab_size=64, d_model=16, n_heads=4, seed=11)
    d.post_forward("r", [1, 2, 3, 4])
    drafts = d.pre_forward("r")
    assert len(drafts) == 2
    assert all(0 <= t < 64 for t in drafts)


def test_scheduler_spec_verify_path() -> None:
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    target = TinyLM(vocab_size=64, d_model=16, n_heads=4, seed=1)
    role = RoleConfig(
        model_backend="tiny_lm",
        enable_drafter=True,
        num_draft_tokens=2,
        enable_overlap=False,
        max_running_reqs=2,
    )
    runner = ModelRunner(
        pool,
        model_backend="tiny_lm",
        tiny_lm=target,
        enable_drafter=True,
        num_draft_tokens=2,
    )
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("s1", "tiny", list(range(8)), 5, "n0"))
    sched.run_until_idle()
    done = sched.get_req("s1")
    assert done.finished
    assert done.num_output_tokens == 5
    assert all(0 <= t < 64 for t in done.output_token_ids)
    assert ag.finished == ["s1"]


def test_scheduler_records_verify_mode() -> None:
    """同步跑一步步观察：extend 后下一步应为 TARGET_VERIFY。"""
    ag = InMemoryAgent()
    pool = PoolIface(ag)
    role = RoleConfig(
        model_backend="tiny_lm",
        enable_drafter=True,
        num_draft_tokens=2,
        enable_overlap=False,
    )
    runner = ModelRunner(pool, model_backend="tiny_lm", enable_drafter=True, num_draft_tokens=2)
    sched = NodeScheduler(pool, runner, role)
    sched.add_request(build_req_from_generate("s2", "tiny", list(range(8)), 4, "n0"))

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

    assert ForwardMode.EXTEND in modes
    assert ForwardMode.TARGET_VERIFY in modes


if __name__ == "__main__":
    test_chain_reject_all_match()
    test_chain_reject_divergence()
    test_drafter_post_pre()
    test_scheduler_spec_verify_path()
    test_scheduler_records_verify_mode()
    print("test_drafter OK")
