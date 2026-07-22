"""池边界（替代 vLLM kv_connector 插件）：ready/done/pull/publish + 请求结束。

C0：用 ControlPlane + SkeletonKv 同步 mock，语义对齐 ready/done 一步契约。
生产：PyO3/FFI → storage-agent（边6）；引擎不知物理地址、不组装 block table。

参考:vLLM `KVConnectorBase_V1`（scheduler/worker 双侧钩子）；
lake 差异：必经路径 + 表组装归 agent（Q1/Q2）。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import grpc

from lake_pb import lake_pb2, lake_pb2_grpc, schema_pb2
from runtime.req import Req
from runtime.scheduler_output import ReqIoSet, SchedulerOutput

LOG = logging.getLogger("lake.pool_iface")

# P3 mock 粒度；生产默认 128（见 architecture）
BLOCK_SIZE = 8


def chain_block_hashes(token_ids: Sequence[int], block_size: int = BLOCK_SIZE) -> List[bytes]:
    """前缀链式 hash:h_i = sha256(h_{i-1} || tokens_i)。"""
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


@dataclass
class StepStats:
    reused_blocks: int = 0
    prefill_blocks: int = 0


@dataclass
class ReadyHandle:
    """agent.prepare_step 完成信号；C0 只携带统计，无 device 指针。"""

    step_id: int
    stats_by_req: Dict[str, StepStats] = field(default_factory=dict)


class PoolIface:
    """Python↔池边界。C0 同步；生产异步 fence。"""

    def __init__(
        self,
        cp: lake_pb2_grpc.ControlPlaneServiceStub,
        kv: lake_pb2_grpc.SkeletonKvServiceStub,
    ) -> None:
        self._cp = cp
        self._kv = kv
        self._ready: Optional[ReadyHandle] = None

    def prepare_step(self, output: SchedulerOutput, reqs: Dict[str, Req]) -> ReadyHandle:
        """对应 ready fence：按 read/write set 补齐 KV 元数据与 mock 字节。"""
        stats: Dict[str, StepStats] = {}
        for io in output.write_set:
            req = reqs[io.req_id]
            st = self._ensure_prefix_kv(req)
            stats[io.req_id] = st
            req.reused_blocks = st.reused_blocks
            req.prefill_blocks = st.prefill_blocks
            req.num_computed_tokens = max(req.num_computed_tokens, len(req.prompt_token_ids))
        for io in output.read_set:
            if io.req_id in stats:
                continue
            req = reqs[io.req_id]
            # decode 步：已注册前缀只需校验可读（首版跳过重复 Get）
            stats[io.req_id] = StepStats(reused_blocks=req.reused_blocks, prefill_blocks=0)

        handle = ReadyHandle(step_id=output.step_id, stats_by_req=stats)
        self._ready = handle
        return handle

    def done(self, step_id: int) -> None:
        """对应 done fence：引擎 replay 结束，池可解冻/写回/注册。C0 无-op。"""
        if self._ready is None or self._ready.step_id != step_id:
            LOG.warning("done step_id=%s without matching ready", step_id)
        self._ready = None

    def on_request_finished(self, req: Req) -> None:
        """唯一 KV 收尾入口（对齐 compute-layer：引擎不 free KV）。"""
        try:
            self._cp.RequestBarrier(
                lake_pb2.RequestBarrierRequest(request_id=req.req_id, node_id=req.node_id)
            )
        except grpc.RpcError:
            raise

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


# 供类型检查 / 测试：read_set 元素形状
_ = ReqIoSet
