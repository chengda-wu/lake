"""P3：ControlPlane + SkeletonKv 实现的 StorageAgent（同步 mock）。"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Mapping, Optional, Sequence

import grpc

from engine.pool_types import (
    FinishRequest,
    PoolError,
    PoolErrorCode,
    PreparePlan,
    ReadyHandle,
    StepStats,
)
from lake_pb import lake_pb2, lake_pb2_grpc, schema_pb2
from runtime.req import Req

LOG = logging.getLogger("lake.agent.grpc")

BLOCK_SIZE = 8  # P3 mock；生产默认 128


def chain_block_hashes(token_ids: Sequence[int], block_size: int = BLOCK_SIZE) -> List[bytes]:
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


class GrpcSkeletonAgent:
    def __init__(
        self,
        cp: lake_pb2_grpc.ControlPlaneServiceStub,
        kv: lake_pb2_grpc.SkeletonKvServiceStub,
    ) -> None:
        self._cp = cp
        self._kv = kv
        self._ready_step: Optional[int] = None
        self._host_reqs: Mapping[str, Req] = {}

    def bind_host_reqs(self, reqs: Mapping[str, Req]) -> None:
        """PoolIface 在 prepare 前注入 Host Req（协议层不持权威）。"""
        self._host_reqs = reqs

    def prepare_step(self, plan: PreparePlan) -> ReadyHandle:
        if self._ready_step is not None:
            raise PoolError(PoolErrorCode.NOT_READY, f"prepare while ready={self._ready_step}")

        stats: Dict[str, StepStats] = {}
        try:
            for io in plan.write_set:
                req = self._host_reqs[io.req_id]
                st = self._ensure_prefix_kv(req)
                stats[io.req_id] = st
                req.reused_blocks = st.reused_blocks
                req.prefill_blocks = st.prefill_blocks
            for io in plan.read_set:
                if io.req_id in stats:
                    continue
                req = self._host_reqs[io.req_id]
                stats[io.req_id] = StepStats(reused_blocks=req.reused_blocks, prefill_blocks=0)
        except grpc.RpcError as e:
            raise PoolError(PoolErrorCode.DOWNSTREAM, e.details() or str(e)) from e
        except RuntimeError as e:
            raise PoolError(PoolErrorCode.DOWNSTREAM, str(e)) from e

        self._ready_step = plan.step_id
        return ReadyHandle(
            step_id=plan.step_id,
            stats_by_req=stats,
            effective_read_set=list(plan.read_set),
            effective_write_set=list(plan.write_set),
        )

    def done(self, step_id: int) -> None:
        if self._ready_step is None or self._ready_step != step_id:
            raise PoolError(PoolErrorCode.NOT_READY, f"done step={step_id} ready={self._ready_step}")
        self._ready_step = None

    def on_request_finished(self, finish: FinishRequest) -> None:
        try:
            self._cp.RequestBarrier(
                lake_pb2.RequestBarrierRequest(request_id=finish.req_id, node_id=finish.node_id)
            )
        except grpc.RpcError as e:
            raise PoolError(PoolErrorCode.DOWNSTREAM, e.details() or str(e)) from e

    def _ensure_prefix_kv(self, req: Req) -> StepStats:
        prompt = req.prompt_token_ids
        hashes = chain_block_hashes(prompt)
        ids = [
            schema_pb2.KVBlockID(
                model_id=req.model_id,
                block_hash=h,
                pool_kind=schema_pb2.TARGET,
                scope="public",
            )
            for h in hashes
        ]

        lookup = self._cp.LookupPrefix(
            lake_pb2.LookupPrefixRequest(
                model_id=req.model_id,
                prefix_hashes=hashes,
                requester_node_id=req.node_id,
            )
        )
        reused = int(lookup.hit_length)
        miss_ids = ids[reused:]

        if reused:
            got = self._kv.GetBlocks(lake_pb2.GetBlocksRequest(ids=ids[:reused]))
            if len(got.blocks) != reused:
                raise RuntimeError(f"GetBlocks mismatch: lookup hit={reused} got={len(got.blocks)}")
            for i, blk in enumerate(got.blocks):
                want = bytes(ids[i].block_hash)
                if not blk.id or bytes(blk.id.block_hash) != want:
                    got_h = bytes(blk.id.block_hash) if blk.id else b""
                    raise RuntimeError(f"GetBlocks hash mismatch at {i}: want={want.hex()} got={got_h.hex()}")
                if not blk.data.startswith(b"KV:"):
                    raise RuntimeError("GetBlocks: bad mock KV payload")
            LOG.info("GetBlocks hit=%d ok", reused)

        if miss_ids:
            opaques = [
                lake_pb2.OpaqueBlock(id=bid, data=mock_kv_bytes(bid.block_hash)) for bid in miss_ids
            ]
            put = self._kv.PutBlocks(lake_pb2.PutBlocksRequest(node_id=req.node_id, blocks=opaques))
            if not put.ok:
                raise RuntimeError(put.err or "PutBlocks failed")

            metas = [
                schema_pb2.BlockMeta(
                    id=bid,
                    block_kind=schema_pb2.T_TYPE,
                    locations=[
                        schema_pb2.Location(
                            tier=schema_pb2.L2,
                            node_id=req.node_id,
                            segment_id=1,
                            offset=0,
                        )
                    ],
                    l3_present=False,
                    ref_count=1,
                )
                for bid in miss_ids
            ]
            reg = self._cp.RegisterBlocks(lake_pb2.RegisterBlocksRequest(node_id=req.node_id, blocks=metas))
            if not reg.ok:
                raise RuntimeError(reg.err or "RegisterBlocks failed")

        return StepStats(reused_blocks=reused, prefill_blocks=len(miss_ids))
