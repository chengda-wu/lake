//! PutEnd 两阶段 + writeback 会话（P4.3）。
//!
//! 时序（`kv-cache-pool.md` / `consistency.md` §3）：
//! 1. PutStart 本地记账
//! 2. RegisterBlocks + `ReportRef(WRITEBACK,+1)`
//! 3. `LocalTierEngine::put_durable`（L2，可选 L3）
//! 4. Barrier：WRITEBACK-1 → CP `RequestBarrier`
//!
//! 经 [`ControlPlanePort`] 接线（进程内或未来 tonic）。

use lake_proto::lake::*;
use lake_tiered_store::LocalTierEngine;

use crate::cp_port::ControlPlanePort;

#[derive(Clone, Debug)]
pub struct PendingBlock {
    pub id: KvBlockId,
    pub bytes: Vec<u8>,
    pub durable: bool,
}

#[derive(Debug)]
pub struct PutEndSession {
    pub request_id: String,
    pub node_id: String,
    pub model_id: String,
    pub prefix_hashes: Vec<Vec<u8>>,
    blocks: Vec<PendingBlock>,
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

    fn writeback_deltas(&self, delta: i32) -> Vec<RefDelta> {
        self.blocks
            .iter()
            .map(|b| RefDelta {
                id: Some(b.id.clone()),
                kind: RefKind::Writeback as i32,
                delta,
                node_id: self.node_id.clone(),
            })
            .collect()
    }

    pub fn flush_durable(&mut self, store: &mut LocalTierEngine, also_l3: bool) {
        for b in &mut self.blocks {
            store.put_durable(&b.id.block_hash, &b.bytes, also_l3);
            b.durable = true;
        }
    }

    pub fn all_durable(&self) -> bool {
        self.blocks.iter().all(|b| b.durable)
    }

    /// Full PutEnd → barrier against a [`ControlPlanePort`].
    pub fn commit_through<P: ControlPlanePort>(
        &mut self,
        store: &mut LocalTierEngine,
        cp: &mut P,
        also_l3: bool,
    ) -> Result<(), String> {
        let reg = self.register_request();
        cp.register_blocks(reg)?;
        cp.report_refs(&self.writeback_deltas(1))?;
        self.writeback_open = true;

        self.flush_durable(store, also_l3);
        for b in &self.blocks {
            if also_l3 {
                cp.set_l3_present(&self.model_id, &b.id.block_hash, true)?;
            }
        }

        if !self.all_durable() {
            return Err("commit: L2 not durable".into());
        }
        cp.report_refs(&self.writeback_deltas(-1))?;
        self.writeback_open = false;
        cp.request_barrier(RequestBarrierRequest {
            request_id: self.request_id.clone(),
            node_id: self.node_id.clone(),
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp_port::AuthorityPort;
    use lake_controlplane::Authority;
    use lake_tiered_store::{LocalTier, LocalTierEngine, TierCaps};

    #[test]
    fn commit_through_writeback_then_evictable() {
        let mut store = LocalTierEngine::with_caps(TierCaps::default());
        let mut auth = Authority::default();
        let mut sess = PutEndSession::new("r1", "n0", "m");
        sess.put_start(b"h0".to_vec(), b"KV:h0".to_vec());

        {
            let mut port = AuthorityPort { auth: &mut auth };
            sess.commit_through(&mut store, &mut port, true).unwrap();
        }
        assert!(store.is_l2_durable(b"h0"));
        assert!(store.l3_present(b"h0"));
        assert!(auth.barrier_completed("r1"));
        // After WRITEBACK cleared, 0→正→0 path: need +1/-1 REQUEST to enter inactive,
        // or register left ref=0 without inactive insert. Force cycle:
        let d = RefDelta {
            id: Some(KvBlockId {
                model_id: "m".into(),
                block_hash: b"h0".to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
            }),
            kind: RefKind::Request as i32,
            delta: 1,
            node_id: "n0".into(),
        };
        auth.report_ref(&d).unwrap();
        let mut d0 = d;
        d0.delta = -1;
        auth.report_ref(&d0).unwrap();
        assert_eq!(auth.evict_n("m", 1), 1);
    }

    #[test]
    fn promote_publishes_l0() {
        let mut store = LocalTierEngine::new();
        let mut auth = Authority::default();
        let mut sess = PutEndSession::new("r2", "n0", "m");
        sess.put_start(b"p".to_vec(), b"P".to_vec());
        {
            let mut port = AuthorityPort { auth: &mut auth };
            sess.commit_through(&mut store, &mut port, false).unwrap();
            store.promote_to_l0(b"p").unwrap();
            port.publish_location("m", b"p", Tier::L0, "n0", true)
                .unwrap();
        }
        assert_eq!(store.local_tier(b"p"), Some(LocalTier::L0));
        assert!(auth.has_l0_on("m", b"p", "n0"));
        let (_, _, all_local) = auth.lookup_prefix("m", &[b"p".to_vec()], "n0");
        assert!(all_local);
    }
}
