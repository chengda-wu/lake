//! PutEnd 两阶段（P4.3）。
//!
//! **COMPLETE 语义**（Mooncake `PutEnd` → `mark_complete`；#20）：
//! 1. PutStart 本地记账
//! 2. `put_durable` 落稳 L2|L3（先有字节）
//! 3. `RegisterBlocks`（按真实 settle 标 L2 / `l3_present`）= COMPLETE
//! 4. 本地 `pin`（≈ SGLang `lock_ref`）覆盖提交窗口；CP `ReportRef(WRITEBACK)`
//!    仍喂 `global_refs` 冻 radix 驱逐（P4.2 骨架忽略 RefKind）
//! 5. WRITEBACK−1 + `RequestBarrier` ledger
//!
//! 中途 register-before-durable 的 writeback 窗口已不走；勿与 consistency.md
//! 旧「先注册再写回」叙述混读——以 durable-first COMPLETE 为准。

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

    pub fn flush_durable(
        &mut self,
        store: &mut LocalTierEngine,
        also_l3: bool,
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut demoted_all = Vec::new();
        for b in &mut self.blocks {
            let (_tier, fx) = store.put_durable(&b.id.block_hash, &b.bytes, also_l3)?;
            demoted_all.extend(fx.l2_demoted_to_l3);
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

    pub fn commit_through<P: ControlPlanePort>(
        &mut self,
        store: &mut LocalTierEngine,
        cp: &mut P,
        also_l3: bool,
    ) -> Result<(), String> {
        // Local freeze during commit (SGLang lock_ref shape).
        for b in &self.blocks {
            store.pin(&b.id.block_hash);
        }

        let demoted = self.flush_durable(store, also_l3)?;
        if !self.all_durable() {
            for b in &self.blocks {
                let _ = store.unpin(&b.id.block_hash);
            }
            return Err("commit: not settled on L2|L3".into());
        }

        for h in &demoted {
            if !self.blocks.iter().any(|b| b.id.block_hash == *h) {
                let _ = cp.publish_location(&self.model_id, h, Tier::L2, &self.node_id, false);
                let _ = cp.set_l3_present(&self.model_id, h, true);
            }
        }

        let reg = self.register_request(store);
        cp.register_blocks(reg)?;
        for b in &self.blocks {
            let h = b.id.block_hash.as_slice();
            if store.l3_present(h) {
                cp.set_l3_present(&self.model_id, h, true)?;
            }
            if !store.is_l2_durable(h) {
                let _ = cp.publish_location(&self.model_id, h, Tier::L2, &self.node_id, false);
            }
        }

        // CP global_refs freeze (radix evict); engine already pinned.
        cp.report_refs(&self.writeback_deltas(1))?;
        self.writeback_open = true;
        cp.report_refs(&self.writeback_deltas(-1))?;
        self.writeback_open = false;
        cp.request_barrier(RequestBarrierRequest {
            request_id: self.request_id.clone(),
            node_id: self.node_id.clone(),
        })?;

        for b in &self.blocks {
            store.unpin(&b.id.block_hash)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cp_port::{apply_location_events, AuthorityPort};
    use lake_controlplane::Authority;
    use lake_tiered_store::{
        BandwidthPool, LocalTier, LocalTierEngine, PipelineAction, TierCaps, TierPipeline,
    };

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
    }

    #[test]
    fn commit_flush_before_register_l2_cap_syncs_l3() {
        let mut store = LocalTierEngine::with_caps(TierCaps {
            l0: 4,
            l1: 8,
            l2: 1,
        });
        let mut auth = Authority::default();
        let mut s0 = PutEndSession::new("r-a", "n0", "m");
        s0.put_start(b"x".to_vec(), b"X".to_vec());
        {
            let mut port = AuthorityPort { auth: &mut auth };
            s0.commit_through(&mut store, &mut port, false).unwrap();
        }
        let mut s1 = PutEndSession::new("r-b", "n0", "m");
        s1.put_start(b"y".to_vec(), b"Y".to_vec());
        {
            let mut port = AuthorityPort { auth: &mut auth };
            s1.commit_through(&mut store, &mut port, false).unwrap();
        }
        assert!(!store.is_l2_durable(b"x"));
        assert!(store.l3_present(b"x"));
        let meta_x = auth
            .lookup_prefix("m", &[b"x".to_vec()], "n0")
            .0
            .into_iter()
            .next()
            .and_then(|r| r.meta)
            .expect("x");
        assert!(meta_x.l3_present);
        assert!(!meta_x
            .locations
            .iter()
            .any(|l| l.tier == Tier::L2 as i32 && l.node_id == "n0"));
    }

    #[test]
    fn pipeline_tick_apply_drops_victim_l0_on_cp() {
        // Audit HIGH#1: promote under L0 cap must clear victim L0 on CP.
        let mut store = LocalTierEngine::with_caps(TierCaps {
            l0: 1,
            l1: 8,
            l2: 8,
        });
        let mut auth = Authority::default();
        for (id, h, bytes) in [("r-a", b"a", b"A"), ("r-b", b"b", b"B")] {
            let mut s = PutEndSession::new(id, "n0", "m");
            s.put_start(h.to_vec(), bytes.to_vec());
            let mut port = AuthorityPort { auth: &mut auth };
            s.commit_through(&mut store, &mut port, false).unwrap();
        }
        let mut pipe = TierPipeline::new(store, BandwidthPool::new(1 << 20));
        pipe.enqueue(PipelineAction::Promote {
            hash: b"a".to_vec(),
        });
        let (n, ev) = pipe.tick(1);
        assert_eq!(n, 1);
        {
            let mut port = AuthorityPort { auth: &mut auth };
            apply_location_events(&mut port, "m", "n0", &ev).unwrap();
        }
        assert!(auth.has_l0_on("m", b"a", "n0"));

        pipe.enqueue(PipelineAction::Promote {
            hash: b"b".to_vec(),
        });
        let (n2, ev2) = pipe.tick(1);
        assert_eq!(n2, 1);
        {
            let mut port = AuthorityPort { auth: &mut auth };
            apply_location_events(&mut port, "m", "n0", &ev2).unwrap();
        }
        assert!(auth.has_l0_on("m", b"b", "n0"));
        assert!(
            !auth.has_l0_on("m", b"a", "n0"),
            "victim L0 must be cleared on CP after collateral demote"
        );
        let (_, _, all_local_a) = auth.lookup_prefix("m", &[b"a".to_vec()], "n0");
        assert!(!all_local_a, "must not false D-direct on demoted a");
    }
}
