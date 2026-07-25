//! PutEnd 两阶段 + writeback 会话（P4.3）。
//!
//! 时序（#20 / Mooncake PutEnd；对齐 `consistency.md`「注册即有后盾」）：
//! 1. PutStart 本地记账
//! 2. `LocalTierEngine::put_durable`（先落稳 L2|L3）
//! 3. RegisterBlocks（按真实 settle tier 标 L2 / `l3_present`）+ `ReportRef(WRITEBACK,+1)`
//! 4. Barrier：WRITEBACK-1 → CP `RequestBarrier`（ledger；解冻靠 WRITEBACK−1）
//!
//! 经 [`ControlPlanePort`] 接线（进程内或未来 tonic）。

use lake_proto::lake::*;
use lake_tiered_store::LocalTierEngine;

use crate::cp_port::ControlPlanePort;

#[derive(Clone, Debug)]
pub struct PendingBlock {
    pub id: KvBlockId,
    pub bytes: Vec<u8>,
    /// True iff engine reports settled (L2|L3) after flush.
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

    /// Build Register from **engine** settle state (call after [`flush_durable`]).
    pub fn register_request(&self, store: &LocalTierEngine) -> RegisterBlocksRequest {
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let h = b.id.block_hash.as_slice();
                let mut locations = Vec::new();
                if store.is_l2_durable(h) {
                    locations.push(Location {
                        tier: Tier::L2 as i32,
                        node_id: self.node_id.clone(),
                        segment_id: 1,
                        offset: 0,
                    });
                }
                BlockMeta {
                    id: Some(b.id.clone()),
                    block_kind: BlockKind::TType as i32,
                    locations,
                    l3_present: store.l3_present(h),
                    ref_count: 0,
                }
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

    /// Flush bytes; returns hashes demoted L2→L3 under cap (for CP view sync).
    pub fn flush_durable(
        &mut self,
        store: &mut LocalTierEngine,
        also_l3: bool,
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut demoted_all = Vec::new();
        for b in &mut self.blocks {
            let (_tier, demoted) = store.put_durable(&b.id.block_hash, &b.bytes, also_l3)?;
            demoted_all.extend(demoted);
            // Durable = real settle, not a sticky flag.
            b.durable = store.is_settled(&b.id.block_hash);
            if !b.durable {
                return Err(format!(
                    "flush: block {:?} not settled on L2|L3",
                    b.id.block_hash
                ));
            }
        }
        Ok(demoted_all)
    }

    pub fn all_durable(&self) -> bool {
        self.blocks.iter().all(|b| b.durable)
    }

    /// Full PutEnd → barrier against a [`ControlPlanePort`].
    ///
    /// Order: durable first → Register (COMPLETE-equivalent) → WRITEBACK freeze
    /// until barrier ledger + WRITEBACK−1.
    pub fn commit_through<P: ControlPlanePort>(
        &mut self,
        store: &mut LocalTierEngine,
        cp: &mut P,
        also_l3: bool,
    ) -> Result<(), String> {
        let demoted = self.flush_durable(store, also_l3)?;
        if !self.all_durable() {
            return Err("commit: not settled on L2|L3".into());
        }

        // Cap demotions of *prior* blocks: drop stale L2, mark L3 on the view.
        for h in &demoted {
            if !self.blocks.iter().any(|b| b.id.block_hash == *h) {
                let _ = cp.publish_location(&self.model_id, h, Tier::L2, &self.node_id, false);
                let _ = cp.set_l3_present(&self.model_id, h, true);
            }
        }

        let reg = self.register_request(store);
        // Sync presence from register metas (L2 marker only if L2 location present).
        cp.register_blocks(reg)?;
        for b in &self.blocks {
            let h = b.id.block_hash.as_slice();
            if store.l3_present(h) {
                cp.set_l3_present(&self.model_id, h, true)?;
            }
            // Self demoted under cap during flush: register already omitted L2.
            if !store.is_l2_durable(h) {
                let _ = cp.publish_location(&self.model_id, h, Tier::L2, &self.node_id, false);
            }
        }

        cp.report_refs(&self.writeback_deltas(1))?;
        self.writeback_open = true;
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

    #[test]
    fn commit_flush_before_register_l2_cap_syncs_l3() {
        let mut store = LocalTierEngine::with_caps(TierCaps {
            l0: 4,
            l1: 8,
            l2: 1,
        });
        let mut auth = Authority::default();
        // First block fills L2.
        let mut s0 = PutEndSession::new("r-a", "n0", "m");
        s0.put_start(b"x".to_vec(), b"X".to_vec());
        {
            let mut port = AuthorityPort { auth: &mut auth };
            s0.commit_through(&mut store, &mut port, false).unwrap();
        }
        assert!(store.is_l2_durable(b"x"));

        // Second PutEnd demotes x → L3; register must not claim L2 for x.
        let mut s1 = PutEndSession::new("r-b", "n0", "m");
        s1.put_start(b"y".to_vec(), b"Y".to_vec());
        {
            let mut port = AuthorityPort { auth: &mut auth };
            s1.commit_through(&mut store, &mut port, false).unwrap();
        }
        assert!(store.is_l2_durable(b"y"));
        assert!(!store.is_l2_durable(b"x"));
        assert!(store.l3_present(b"x"));

        // View for x: no L2 location, l3_present.
        let entry_l3 = auth
            .lookup_prefix("m", &[b"x".to_vec()], "n0")
            .0
            .into_iter()
            .next()
            .and_then(|r| r.meta);
        let meta_x = entry_l3.expect("x still in radix");
        assert!(meta_x.l3_present);
        assert!(!meta_x
            .locations
            .iter()
            .any(|l| l.tier == Tier::L2 as i32 && l.node_id == "n0"));
    }
}
