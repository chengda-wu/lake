"""池边界门面：把 SchedulerOutput 编成 PreparePlan，转调 StorageAgent（D2）。

生产：agent = PyO3 `lake-storage-agent`；P3：`GrpcSkeletonAgent`。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from engine.agent import StorageAgent
from engine.agents.grpc_skeleton import GrpcSkeletonAgent, chain_block_hashes, mock_kv_bytes
from engine.pool_types import FinishRequest, PreparePlan, ReadyHandle, StepStats
from runtime.req import Req
from runtime.scheduler_output import SchedulerOutput

LOG = logging.getLogger("lake.pool_iface")

# 兼容旧 import
__all__ = [
    "PoolIface",
    "ReadyHandle",
    "StepStats",
    "chain_block_hashes",
    "mock_kv_bytes",
]


class PoolIface:
    def __init__(
        self,
        agent: StorageAgent,
        *,
        pull_budget_ms: int = 0,
        allow_partial_hit: bool = False,
    ) -> None:
        self._agent = agent
        self.pull_budget_ms = pull_budget_ms
        self.allow_partial_hit = allow_partial_hit
        self._last_ready: Optional[ReadyHandle] = None

    @classmethod
    def from_grpc(cls, cp, kv, **kwargs) -> "PoolIface":
        return cls(GrpcSkeletonAgent(cp, kv), **kwargs)

    def prepare_step(self, output: SchedulerOutput, reqs: Dict[str, Req]) -> ReadyHandle:
        plan = PreparePlan(
            step_id=output.step_id,
            forward_mode=output.forward_mode,
            read_set=list(output.read_set),
            write_set=list(output.write_set),
            num_scheduled_tokens=dict(output.num_scheduled_tokens),
            pull_budget_ms=self.pull_budget_ms,
            allow_partial_hit=self.allow_partial_hit,
        )
        if isinstance(self._agent, GrpcSkeletonAgent):
            self._agent.bind_host_reqs(reqs)
        handle = self._agent.prepare_step(plan)
        self._last_ready = handle
        return handle

    def done(self, step_id: int) -> None:
        self._agent.done(step_id)
        self._last_ready = None

    def on_request_finished(self, req: Req) -> None:
        self._agent.on_request_finished(
            FinishRequest(req_id=req.req_id, node_id=req.node_id, model_id=req.model_id)
        )
