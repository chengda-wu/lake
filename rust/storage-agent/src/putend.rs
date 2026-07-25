//! PutEnd 两阶段 + writeback 会话（P4.3）。
//!
//! 时序（对齐 `kv-cache-pool.md` / `consistency.md` §3，覆盖 writeback 不变量）：
//! 1. PutStart：本地记账（未进 radix）
//! 2. RegisterBlocks（PutEnd 控制面侧）+ `ReportRef(WRITEBACK,+1)`
//! 3. 本地 `MemoryL2::put_durable`（可与 2 重叠；屏障前必须完成）
//! 4. RequestBarrier：确认 durable → `ReportRef(WRITEBACK,-1)` → CP `RequestBarrier`
//!
//! Proto 注释的「先 durable 再 Register」是严序 PutEnd；本路径用 WRITEBACK
//! 覆盖「先注册后落盘」窗口（doc 主路径）。真 gRPC 客户端接线后续。
//!
//! 参考：Mooncake PutStart/PutEnd；SGLang `_evict_write_back`。

use lake_proto::lake::*;
use lake_tiered_store::MemoryL2;

/// One full block pending PutEnd / barrier on this agent.
#[derive(Clone, Debug)]
pub struct PendingBlock {
    pub id: KvBlockId,
    pub bytes: Vec<u8>,
    pub durable: bool,
}

/// Per-request PutEnd ledger (agent-local).
#[derive(Debug)]
pub struct PutEndSession {
    pub request_id: String,
    pub node_id: String,
    pub model_id: String,
    /// Ordered prefix hashes for lineage (full chain).
    pub prefix_hashes: Vec<Vec<u8>>,
    blocks: Vec<PendingBlock>,
    /// WRITEBACK still held on controlplane for these flats.
    writeback_open: bool,
}

impl PutEndSession {
    pub fn new(
        request_id: impl Into<String>,
        node_id: impl Into<String>,
        model_id: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            node_id: node_id.into(),
            model_id: model_id.into(),
            prefix_hashes: Vec::new(),
            blocks: Vec::new(),
            writeback_open: false,
        }
    }

    /// PutStart: stage a full block (not yet durable / registered).
    pub fn put_start(&mut self, block_hash: Vec<u8>, bytes: Vec<u8>) {
        self.prefix_hashes.push(block_hash.clone());
        self.blocks.push(PendingBlock {
            id: KvBlockId {
                model_id: self.model_id.clone(),
                block_hash,
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
            },
            bytes,
            durable: false,
        });
    }

    /// Build `RegisterBlocksRequest` for the controlplane PutEnd RPC.
    pub fn register_request(&self) -> RegisterBlocksRequest {
        let blocks = self
            .blocks
            .iter()
            .map(|b| BlockMeta {
                id: Some(b.id.clone()),
                block_kind: BlockKind::TType as i32,
                locations: vec![Location {
                    tier: Tier::L2 as i32,
                    node_id: self.node_id.clone(),
                    segment_id: 1,
                    offset: 0,
                }],
                l3_present: false,
                ref_count: 0,
            })
            .collect();
        RegisterBlocksRequest {
            node_id: self.node_id.clone(),
            blocks,
            prefix_hashes: self.prefix_hashes.clone(),
        }
    }

    /// After successful RegisterBlocks: WRITEBACK +1 deltas (合账进 global_refs).
    pub fn writeback_plus_deltas(&mut self) -> Vec<RefDelta> {
        self.writeback_open = true;
        self.blocks
            .iter()
            .map(|b| RefDelta {
                id: Some(b.id.clone()),
                kind: RefKind::Writeback as i32,
                delta: 1,
                node_id: self.node_id.clone(),
            })
            .collect()
    }

    /// Persist all staged blocks to local L2 stand-in.
    pub fn flush_durable(&mut self, store: &mut MemoryL2) {
        for b in &mut self.blocks {
            store.put_durable(&b.id.block_hash, &b.bytes);
            b.durable = true;
        }
    }

    pub fn all_durable(&self) -> bool {
        self.blocks.iter().all(|b| b.durable)
    }

    /// Barrier prep: require durable, then WRITEBACK -1 + RequestBarrierRequest.
    pub fn finish_barrier(&mut self) -> Result<(Vec<RefDelta>, RequestBarrierRequest), String> {
        if !self.all_durable() {
            return Err("RequestBarrier: L2 not durable for all registered blocks".into());
        }
        if !self.writeback_open {
            return Err(
                "RequestBarrier: writeback not held (call writeback_plus after register)".into(),
            );
        }
        let deltas = self
            .blocks
            .iter()
            .map(|b| RefDelta {
                id: Some(b.id.clone()),
                kind: RefKind::Writeback as i32,
                delta: -1,
                node_id: self.node_id.clone(),
            })
            .collect();
        self.writeback_open = false;
        let barrier = RequestBarrierRequest {
            request_id: self.request_id.clone(),
            node_id: self.node_id.clone(),
        };
        Ok((deltas, barrier))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lake_controlplane::Authority;

    #[test]
    fn putend_writeback_freeze_then_barrier() {
        let mut store = MemoryL2::new();
        let mut sess = PutEndSession::new("r1", "n0", "m");
        sess.put_start(b"h0".to_vec(), b"KV:h0".to_vec());

        let mut auth = Authority::default();
        let reg = sess.register_request();
        auth.register(&reg.node_id, &reg.prefix_hashes, reg.blocks)
            .unwrap();
        auth.report_refs(&sess.writeback_plus_deltas()).unwrap();
        assert_eq!(auth.evict_n("m", 1), 0);

        sess.flush_durable(&mut store);
        assert!(store.is_durable(b"h0"));

        let (minus, barrier) = sess.finish_barrier().unwrap();
        auth.report_refs(&minus).unwrap();
        auth.complete_barrier(&barrier.request_id, &barrier.node_id)
            .unwrap();
        assert!(auth.barrier_completed("r1"));
        assert_eq!(auth.evict_n("m", 1), 1);
    }

    #[test]
    fn barrier_rejects_undurable() {
        let mut sess = PutEndSession::new("r2", "n0", "m");
        sess.put_start(b"h1".to_vec(), b"x".to_vec());
        let _ = sess.writeback_plus_deltas();
        assert!(sess.finish_barrier().is_err());
    }
}
