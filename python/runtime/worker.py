"""WorkerService gRPC 门面：Dispatch 后的 Generate 挂到 node_scheduler + engine。

P3 仍：Router → AgentService.Dispatch(ack) → WorkerService.Generate。
生产路径：Router Dispatch → agent 组 batch → FFI 引擎；本门面可收窄为控制面 RPC。

参考:vLLM KVConnectorBase_V1(worker↔池);SGLang Scheduler+薄 ModelRunner。
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from engine.model_runner import ModelRunner
from engine.pool_iface import PoolIface, chain_block_hashes, mock_kv_bytes
from lake_pb import lake_pb2, lake_pb2_grpc
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig, WorkerRole

LOG = logging.getLogger("lake.worker")

NODE_ID = "worker-0"

# 兼容 scripts/verify-p3.sh 等旧 import
__all__ = ["WorkerServicer", "serve", "chain_block_hashes", "mock_kv_bytes", "NODE_ID"]


def _abort_rpc(context: grpc.ServicerContext, exc: grpc.RpcError) -> None:
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.INTERNAL
    details = exc.details() if hasattr(exc, "details") else str(exc)
    if code == grpc.StatusCode.UNAVAILABLE:
        context.abort(grpc.StatusCode.UNAVAILABLE, f"downstream: {details}")
    context.abort(grpc.StatusCode.INTERNAL, f"downstream: {details}")


class WorkerServicer(lake_pb2_grpc.WorkerServiceServicer):
    def __init__(self, cp: lake_pb2_grpc.ControlPlaneServiceStub, kv: lake_pb2_grpc.SkeletonKvServiceStub):
        self._pool = PoolIface(cp, kv)
        self._runner = ModelRunner(self._pool)
        self._role = RoleConfig(role=WorkerRole.HYBRID)

    def Generate(self, request: lake_pb2.GenerateRequest, context: grpc.ServicerContext) -> lake_pb2.GenerateResponse:
        node = request.requester_node_id or NODE_ID
        req = build_req_from_generate(
            request_id=request.request_id,
            model_id=request.model_id or "mock-llm",
            prompt_tokens=list(request.prompt_tokens),
            max_new_tokens=request.max_new_tokens or 4,
            node_id=node,
        )
        sched = NodeScheduler(self._pool, self._runner, self._role)
        try:
            sched.add_request(req)
            sched.run_until_idle()
        except grpc.RpcError as e:
            _abort_rpc(context, e)
        except RuntimeError as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        done = sched.get_req(req.req_id)
        # P3 只注册 L2,无 L0 本地命中 → 固定混部,不报 D_DIRECT。
        return lake_pb2.GenerateResponse(
            request_id=request.request_id,
            output_tokens=done.output_token_ids,
            reused_blocks=done.reused_blocks,
            prefill_blocks=done.prefill_blocks,
            mode="COLOCATED",
        )


def serve(bind: str, cp_addr: str, kv_addr: str) -> None:
    cp_chan = grpc.insecure_channel(cp_addr)
    kv_chan = grpc.insecure_channel(kv_addr)
    cp = lake_pb2_grpc.ControlPlaneServiceStub(cp_chan)
    kv = lake_pb2_grpc.SkeletonKvServiceStub(kv_chan)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    lake_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(cp, kv), server)
    server.add_insecure_port(bind)
    server.start()
    LOG.info("WorkerService on %s (cp=%s kv=%s) via node_scheduler+engine", bind, cp_addr, kv_addr)
    server.wait_for_termination()
