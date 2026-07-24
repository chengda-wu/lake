"""长期 WorkerEngine 单环（C6）——无 gRPC 依赖，便于单测。

参考:vLLM EngineCore 单环；lake：一份 NodeScheduler + 入队 drain。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from engine.model_runner import ModelRunner
from engine.pool_iface import PoolIface
from runtime.node_scheduler import NodeScheduler
from runtime.prefix_hint import PrefixHint
from runtime.req import Req
from runtime.role import RoleConfig

LOG = logging.getLogger("lake.worker_engine")


@dataclass
class _Inbound:
    req: Req
    hint: Optional[PrefixHint]
    done: threading.Event
    error: list  # 0-or-1 Exception
    result: list  # 0-or-1 Req


class WorkerEngine:
    """长期 EngineCore 式进程内核：入队 + 单 step 环。

    对齐 vLLM 单 EngineCore：禁止每请求新建 Scheduler / 并行 prepare。
    """

    def __init__(
        self,
        pool: PoolIface,
        runner: ModelRunner,
        role: Optional[RoleConfig] = None,
        *,
        coalesce_s: float = 0.005,
    ) -> None:
        self._role = role or RoleConfig()
        self._pool = pool
        self._runner = runner
        self._coalesce_s = max(0.0, coalesce_s)
        self._sched = NodeScheduler(
            pool, runner, self._role, on_req_finished=self._on_req_finished
        )
        self._inbound: "queue.Queue[Optional[_Inbound]]" = queue.Queue()
        self._inflight: Dict[str, _Inbound] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lake-worker-engine", daemon=True)
        self._started = False

    @property
    def scheduler(self) -> NodeScheduler:
        return self._sched

    @property
    def role(self) -> RoleConfig:
        return self._role

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._stop.set()
        self._inbound.put(None)
        self._thread.join(timeout=timeout)
        self._started = False

    def submit(self, req: Req, hint: Optional[PrefixHint] = None) -> Req:
        """阻塞直到该请求 finished（或引擎故障）。返回完成后的 Host Req。"""
        if self._stop.is_set() or not self._started:
            raise RuntimeError("WorkerEngine is not running")
        item = _Inbound(req=req, hint=hint, done=threading.Event(), error=[], result=[])
        self._inbound.put(item)
        item.done.wait()
        if item.error:
            raise item.error[0]
        if not item.result:
            raise RuntimeError(f"submit finished without result req_id={req.req_id}")
        return item.result[0]

    def _drain_inbound(self) -> None:
        while True:
            try:
                item = self._inbound.get_nowait()
            except queue.Empty:
                break
            if item is None:
                if self._stop.is_set():
                    break
                continue
            rid = item.req.req_id
            with self._lock:
                if rid in self._inflight or self._sched.has_req(rid):
                    item.error.append(ValueError(f"duplicate req_id={rid}"))
                    item.done.set()
                    continue
                self._inflight[rid] = item
            try:
                self._sched.add_request(item.req, hint=item.hint)
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._inflight.pop(rid, None)
                item.error.append(e)
                item.done.set()

    def _on_req_finished(self, req: Req) -> None:
        with self._lock:
            inn = self._inflight.pop(req.req_id, None)
        if inn is None:
            return
        finished = self._sched.release_req(req.req_id) or req
        inn.result.append(finished)
        inn.done.set()

    def _fail_inflight(self, exc: BaseException) -> None:
        with self._lock:
            items = list(self._inflight.values())
            self._inflight.clear()
        for inn in items:
            self._sched.abandon_req(inn.req.req_id)
            if not inn.done.is_set():
                inn.error.append(exc)
                inn.done.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain_inbound()
            if not self._sched.has_work():
                try:
                    item = self._inbound.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    if self._stop.is_set():
                        break
                    continue
                self._inbound.put(item)
                continue
            # 空闲→忙碌时短窗 coalesce，吞掉几乎同时到达的并发 submit（C6）
            if self._coalesce_s > 0:
                t0 = time.monotonic()
                while time.monotonic() - t0 < self._coalesce_s:
                    self._drain_inbound()
                    time.sleep(min(0.001, self._coalesce_s))
            try:
                self._sched.run_until_idle(before_schedule=self._drain_inbound)
            except Exception as e:  # noqa: BLE001
                LOG.exception("WorkerEngine step loop failed: %s", e)
                self._fail_inflight(e)
