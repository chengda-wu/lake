"""P3 mock worker:LookupPrefix → SkeletonKv Get/Put → RegisterBlocks → mock decode.

生产路径:Router Dispatch → agent → FFI 引擎。P3 用 WorkerService.Generate 直连。
参考:vLLM KVConnectorBase_V1(worker↔池 Get/Put 必须校验);早期 src/ 前缀复用冒烟。
"""

from __future__ import annotations

import hashlib
import logging
from concurrent import futures
from typing import List, Sequence

import grpc

from lake_pb import lake_pb2, lake_pb2_grpc, schema_pb2

LOG = logging.getLogger("lake.worker")

BLOCK_SIZE = 8  # P3 mock;生产默认 128
NODE_ID = "worker-0"


def chain_block_hashes(token_ids: Sequence[int], block_size: int = BLOCK_SIZE) -> List[bytes]:
    """前缀链式 hash:h_i = sha256(h_{i-1} || tokens_i),起点 parent=空。"""
    hashes: List[bytes] = []
    parent = b""
    for i in range(0, len(token_ids), block_size):
        chunk = list(token_ids[i : i + block_size])
        if not chunk:
            break
        h = hashlib.sha256()
        h.update(parent)
        for t in chunk:
            h.update(int(t).to_bytes(4, "little", signed=False))
        digest = h.digest()
        hashes.append(digest)
        parent = digest
    return hashes


def mock_kv_bytes(block_hash: bytes) -> bytes:
    return b"KV:" + block_hash[:16]


def mock_decode_tokens(prompt: Sequence[int], max_new: int) -> List[int]:
    """可复现 mock:基于 prompt 末 token 递推固定序列。"""
    seed = prompt[-1] if prompt else 0
    return [((seed + i + 1) % 1000) + 1000 for i in range(max_new)]


def _abort_rpc(context: grpc.ServicerContext, exc: grpc.RpcError) -> None:
    code = exc.code() if hasattr(exc, "code") else grpc.StatusCode.INTERNAL
    details = exc.details() if hasattr(exc, "details") else str(exc)
    if code == grpc.StatusCode.UNAVAILABLE:
        context.abort(grpc.StatusCode.UNAVAILABLE, f"downstream: {details}")
    context.abort(grpc.StatusCode.INTERNAL, f"downstream: {details}")


class WorkerServicer(lake_pb2_grpc.WorkerServiceServicer):
    def __init__(self, cp: lake_pb2_grpc.ControlPlaneServiceStub, kv: lake_pb2_grpc.SkeletonKvServiceStub):
        self._cp = cp
        self._kv = kv

    def Generate(self, request: lake_pb2.GenerateRequest, context: grpc.ServicerContext) -> lake_pb2.GenerateResponse:
        prompt = list(request.prompt_tokens)
        model_id = request.model_id or "mock-llm"
        node = request.requester_node_id or NODE_ID
        max_new = request.max_new_tokens or 4

        hashes = chain_block_hashes(prompt)
        ids = [
            schema_pb2.KVBlockID(
                model_id=model_id,
                block_hash=h,
                pool_kind=schema_pb2.TARGET,
                scope="public",
            )
            for h in hashes
        ]

        try:
            lookup = self._cp.LookupPrefix(
                lake_pb2.LookupPrefixRequest(
                    model_id=model_id,
                    prefix_hashes=hashes,
                    requester_node_id=node,
                )
            )
        except grpc.RpcError as e:
            _abort_rpc(context, e)

        reused = int(lookup.hit_length)
        miss_ids = ids[reused:]

        # 校验 SkeletonKv 字节路径:Lookup 命中必须真能 Get 到 bytes。
        if reused:
            try:
                got = self._kv.GetBlocks(lake_pb2.GetBlocksRequest(ids=ids[:reused]))
            except grpc.RpcError as e:
                _abort_rpc(context, e)
            if len(got.blocks) != reused:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"GetBlocks mismatch: lookup hit={reused} got={len(got.blocks)}",
                )
            # 按请求 id 顺序对账 block_hash(同长度错块集合也要拒)。
            for i, blk in enumerate(got.blocks):
                want = bytes(ids[i].block_hash)
                if not blk.id or bytes(blk.id.block_hash) != want:
                    got_h = bytes(blk.id.block_hash) if blk.id else b""
                    context.abort(
                        grpc.StatusCode.INTERNAL,
                        f"GetBlocks hash mismatch at {i}: want={want.hex()} got={got_h.hex()}",
                    )
                if not blk.data.startswith(b"KV:"):
                    context.abort(grpc.StatusCode.INTERNAL, "GetBlocks: bad mock KV payload")
            LOG.info("GetBlocks hit=%d ok", reused)

        # mock prefill:为 miss block 写不透明 KV + 注册元数据
        if miss_ids:
            opaques = [
                lake_pb2.OpaqueBlock(id=bid, data=mock_kv_bytes(bid.block_hash)) for bid in miss_ids
            ]
            try:
                put = self._kv.PutBlocks(lake_pb2.PutBlocksRequest(node_id=node, blocks=opaques))
            except grpc.RpcError as e:
                _abort_rpc(context, e)
            if not put.ok:
                context.abort(grpc.StatusCode.INTERNAL, put.err or "PutBlocks failed")

            metas = []
            for bid in miss_ids:
                metas.append(
                    schema_pb2.BlockMeta(
                        id=bid,
                        block_kind=schema_pb2.T_TYPE,
                        locations=[
                            schema_pb2.Location(
                                tier=schema_pb2.L2,
                                node_id=node,
                                segment_id=1,
                                offset=0,
                            )
                        ],
                        l3_present=False,
                        ref_count=1,
                    )
                )
            try:
                reg = self._cp.RegisterBlocks(
                    lake_pb2.RegisterBlocksRequest(node_id=node, blocks=metas)
                )
            except grpc.RpcError as e:
                _abort_rpc(context, e)
            if not reg.ok:
                context.abort(grpc.StatusCode.INTERNAL, reg.err or "RegisterBlocks failed")

        try:
            self._cp.RequestBarrier(
                lake_pb2.RequestBarrierRequest(request_id=request.request_id, node_id=node)
            )
        except grpc.RpcError as e:
            _abort_rpc(context, e)

        out = mock_decode_tokens(prompt, max_new)
        # P3 只注册 L2,无 L0 本地命中 → 固定混部,不报 D_DIRECT。
        return lake_pb2.GenerateResponse(
            request_id=request.request_id,
            output_tokens=out,
            reused_blocks=reused,
            prefill_blocks=len(miss_ids),
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
    LOG.info("WorkerService on %s (cp=%s kv=%s)", bind, cp_addr, kv_addr)
    server.wait_for_termination()
