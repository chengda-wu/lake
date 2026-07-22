"""WorkerService gRPC 门面：Dispatch 后的 Generate 挂到 node_scheduler + engine。

P3 仍：Router → AgentService.Dispatch(ack) → WorkerService.Generate。
生产路径：Router Dispatch → agent 组 batch → FFI 引擎；本门面可收窄为控制面 RPC。

参考:vLLM EngineCore 单环 schedule→execute（非每请求并行占 runner）；
KVConnectorBase_V1；SGLang Scheduler+薄 ModelRunner。
骨架期：共享 PoolIface/ModelRunner 单槽 ready → Generate 必须串行；
真正 continuous batching 要共享一个长期 NodeScheduler（入队 + 单 step 环），
而不是每请求新建 scheduler 并行 prepare。
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures

import grpc

from engine.model_runner import ModelRunner
from engine.pool_iface import PoolIface, chain_block_hashes, mock_kv_bytes
from engine.pool_types import PoolError, PoolErrorCode
from lake_pb import lake_pb2, lake_pb2_grpc
from runtime.exec_mode import ExecMode
from runtime.mode_select import select_exec_mode
from runtime.node_scheduler import NodeScheduler, build_req_from_generate
from runtime.role import RoleConfig, WorkerRole

LOG = logging.getLogger("lake.worker")

NODE_ID = "worker-0"

# 兼容 scripts/verify-p3.sh 等旧 import
__all__ = ["WorkerServicer", "serve", "chain_block_hashes", "mock_kv_bytes", "NODE_ID"]

_POOL_ERROR_STATUS = {
    PoolErrorCode.TIMEOUT: grpc.StatusCode.UNAVAILABLE,  # 触发 F4 重路由
    PoolErrorCode.CAPACITY: grpc.StatusCode.RESOURCE_EXHAUSTED,
    PoolErrorCode.DOWNSTREAM: grpc.StatusCode.UNAVAILABLE,
    PoolErrorCode.PROTOCOL_ERROR: grpc.StatusCode.UNAVAILABLE,  # fence 乱 → F4
    PoolErrorCode.INVALID_ARG: grpc.StatusCode.INVALID_ARGUMENT,
}


def _abort_rpc(context: grpc.ServicerContext, exc: grpc.RpcError) -> None:
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.INTERNAL
    details = exc.details() if hasattr(exc, "details") else str(exc)
    if code == grpc.StatusCode.UNAVAILABLE:
        context.abort(grpc.StatusCode.UNAVAILABLE, f"downstream: {details}")
    context.abort(grpc.StatusCode.INTERNAL, f"downstream: {details}")


def _abort_pool_error(context: grpc.ServicerContext, exc: PoolError) -> None:
    status = _POOL_ERROR_STATUS.get(exc.code, grpc.StatusCode.INTERNAL)
    context.abort(status, str(exc))


class WorkerServicer(lake_pb2_grpc.WorkerServiceServicer):
    def __init__(self, cp: lake_pb2_grpc.ControlPlaneServiceStub, kv: lake_pb2_grpc.SkeletonKvServiceStub):
        self._role = RoleConfig(role=WorkerRole.HYBRID, model_backend="mock")
        self._pool = PoolIface.from_grpc(
            cp,
            kv,
            pull_budget_ms=self._role.pull_budget_ms,
            allow_partial_hit=self._role.allow_partial_hit,
        )
        self._runner = ModelRunner(
            self._pool,
            model_backend=self._role.model_backend,
            enable_drafter=self._role.enable_drafter,
            num_draft_tokens=self._role.num_draft_tokens,
        )
        # 共享 agent 单槽 _ready_step；对齐 vLLM 单 EngineCore 环，禁止并行 prepare
        self._generate_lock = threading.Lock()

    def Generate(self, request: lake_pb2.GenerateRequest, context: grpc.ServicerContext) -> lake_pb2.GenerateResponse:
        with self._generate_lock:
            return self._generate_locked(request, context)

    def _generate_locked(
        self, request: lake_pb2.GenerateRequest, context: grpc.ServicerContext
    ) -> lake_pb2.GenerateResponse:
        node = request.requester_node_id or NODE_ID
        req = build_req_from_generate(
            request_id=request.request_id,
            model_id=request.model_id or "mock-llm",
            prompt_tokens=list(request.prompt_tokens),
            max_new_tokens=request.max_new_tokens or 4,
            node_id=node,
        )
        # 每请求临时 scheduler：骨架可跑通；生产应改为长期 NodeScheduler + 入队
        sched = NodeScheduler(self._pool, self._runner, self._role)
        try:
            hint = self._pool.probe_prefix(req)
            # P3：无 L0 预放置 → 通常 COLOCATED；命中块数仍进 hint
            req.exec_mode = select_exec_mode(
                hint, prompt_len=len(req.prompt_token_ids), role=self._role.role
            )
            sched.add_request(req, hint=hint)
            sched.run_until_idle()
        except PoolError as e:
            _abort_pool_error(context, e)
        except grpc.RpcError as e:
            _abort_rpc(context, e)
        except RuntimeError as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))

        done = sched.get_req(req.req_id)
        mode = done.exec_mode.value if isinstance(done.exec_mode, ExecMode) else str(done.exec_mode)
        return lake_pb2.GenerateResponse(
            request_id=request.request_id,
            output_tokens=done.output_token_ids,
            reused_blocks=done.reused_blocks,
            prefill_blocks=done.prefill_blocks,
            mode=mode,
        )


def serve(bind: str, cp_addr: str, kv_addr: str) -> None:
    cp_chan = grpc.insecure_channel(cp_addr)
    kv_chan = grpc.insecure_channel(kv_addr)
    cp = lake_pb2_grpc.ControlPlaneServiceStub(cp_chan)
    kv = lake_pb2_grpc.SkeletonKvServiceStub(kv_chan)

    # Generate 已在 servicer 内加锁；max_workers=1 再收一层，避免误并行占槽
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    lake_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(cp, kv), server)
    server.add_insecure_port(bind)
    server.start()
    LOG.info("WorkerService on %s (cp=%s kv=%s) via node_scheduler+engine", bind, cp_addr, kv_addr)
    server.wait_for_termination()
