//! 位置视图权威：每 model_id 一个 `BlockRegistry` + 强句柄 + InactiveIndex。
//!
//! 参考:Dynamo `BlockRegistry` / `InactiveIndex`+`MultiLruBackend`；
//! 不用 `BlockManager`/`BlockStore`。EventsManager 不接线。

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use kvbm_logical::registry::BlockRegistrationHandle;
use kvbm_logical::{
    BlockId, BlockRegistry, FrequencyTrackingCapacity, InactiveIndex, MultiLruBackend, SequenceHash,
};

use crate::hash_chain::lineage_from_prefix;
use crate::tier::TierL2;
use lake_proto::lake::*;

const INACTIVE_CAP: usize = 4096;
const MULTI_LRU_THRESHOLDS: [u8; 3] = [3, 8, 15];

struct Entry {
    seq_hash: SequenceHash,
    meta: BlockMeta,
    block_id: BlockId,
}

struct Namespace {
    registry: BlockRegistry,
    /// Keep strong refs so Weak entries in the radix tree stay alive.
    handles: HashMap<SequenceHash, BlockRegistrationHandle>,
    by_flat: HashMap<Vec<u8>, Entry>,
    seq_to_flat: HashMap<SequenceHash, Vec<u8>>,
    inactive: Box<dyn InactiveIndex>,
    /// Aggregate global refs (REQUEST + IN_FLIGHT + WRITEBACK).
    global_refs: HashMap<SequenceHash, i64>,
    next_block_id: BlockId,
}

impl Namespace {
    fn new() -> Self {
        let tracker = FrequencyTrackingCapacity::Small.create_tracker();
        let registry = BlockRegistry::builder()
            .frequency_tracker(Arc::clone(&tracker) as _)
            .build();
        let cap = NonZeroUsize::new(INACTIVE_CAP).expect("INACTIVE_CAP > 0");
        let inactive = Box::new(
            MultiLruBackend::new_with_thresholds(cap, &MULTI_LRU_THRESHOLDS, tracker)
                .expect("MultiLru thresholds"),
        );
        Self {
            registry,
            handles: HashMap::new(),
            by_flat: HashMap::new(),
            seq_to_flat: HashMap::new(),
            inactive,
            global_refs: HashMap::new(),
            next_block_id: 1,
        }
    }

    fn alloc_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.saturating_add(1);
        id
    }
}

/// Process-local authority state.
#[derive(Default)]
pub struct Authority {
    namespaces: HashMap<String, Namespace>,
}

impl Authority {
    fn ns_mut(&mut self, model_id: &str) -> &mut Namespace {
        self.namespaces
            .entry(model_id.to_string())
            .or_insert_with(Namespace::new)
    }

    fn ns(&self, model_id: &str) -> Option<&Namespace> {
        self.namespaces.get(model_id)
    }

    /// Register durable blocks. `prefix_hashes` must be the full ordered chain;
    /// `metas` may be a miss suffix (hashes ⊆ prefix_hashes).
    pub fn register(
        &mut self,
        node_id: &str,
        prefix_hashes: &[Vec<u8>],
        metas: Vec<BlockMeta>,
    ) -> Result<(), String> {
        if metas.is_empty() {
            return Ok(());
        }
        let model_id = metas
            .iter()
            .find_map(|m| m.id.as_ref().map(|i| i.model_id.clone()))
            .ok_or_else(|| "RegisterBlocks: no KVBlockID".to_string())?;

        if prefix_hashes.is_empty() {
            return Err("RegisterBlocks: prefix_hashes required (P4.2 lineage)".into());
        }

        let index_of: HashMap<&[u8], usize> = prefix_hashes
            .iter()
            .enumerate()
            .map(|(i, h)| (h.as_slice(), i))
            .collect();

        for meta in &metas {
            let Some(id) = meta.id.as_ref() else {
                continue;
            };
            if id.model_id != model_id {
                return Err(format!(
                    "RegisterBlocks: mixed model_id {} vs {}",
                    model_id, id.model_id
                ));
            }
            if !index_of.contains_key(id.block_hash.as_slice()) {
                return Err(format!(
                    "RegisterBlocks: block hash not in prefix_hashes (len={})",
                    id.block_hash.len()
                ));
            }
        }

        let lineage = lineage_from_prefix(prefix_hashes);
        let ns = self.ns_mut(&model_id);

        for mut meta in metas {
            let Some(id) = meta.id.clone() else { continue };
            let flat = id.block_hash.clone();
            let pos = *index_of.get(flat.as_slice()).expect("checked");
            let seq = lineage[pos];

            if meta.locations.is_empty() {
                meta.locations.push(Location {
                    tier: Tier::L2 as i32,
                    node_id: node_id.to_string(),
                    segment_id: 1,
                    offset: 0,
                });
            }

            let handle = ns.registry.register_sequence_hash(seq);
            handle.mark_present::<TierL2>();
            ns.handles.insert(seq, handle);

            let block_id = if let Some(prev) = ns.by_flat.get(&flat) {
                prev.block_id
            } else {
                ns.alloc_block_id()
            };
            // Fresh register: not inactive candidate while ref unknown; start refs at 0
            // until ReportRef. Agent historically set ref_count=1 on meta — keep field
            // for Locate display but global_refs drives eviction.
            ns.seq_to_flat.insert(seq, flat.clone());
            ns.by_flat.insert(
                flat,
                Entry {
                    seq_hash: seq,
                    meta,
                    block_id,
                },
            );
        }
        Ok(())
    }

    pub fn lookup_prefix(
        &mut self,
        model_id: &str,
        prefix_hashes: &[Vec<u8>],
        requester: &str,
    ) -> (Vec<ReusableBlock>, u32, bool) {
        if prefix_hashes.is_empty() {
            return (Vec::new(), 0, false);
        }
        // Ensure namespace exists so we can lazy-index flats registered earlier.
        let _ = self.ns_mut(model_id);
        let lineage = lineage_from_prefix(prefix_hashes);
        let ns = self.ns_mut(model_id);

        let mut out = Vec::new();
        let mut hit = 0u32;
        let mut all_local = true;

        for (i, flat) in prefix_hashes.iter().enumerate() {
            let seq = lineage[i];
            let Some(entry) = ns.by_flat.get(flat) else {
                all_local = false;
                break;
            };
            // Ensure radix + presence (lazy repair if handle was lost).
            if !ns.handles.contains_key(&seq) {
                let handle = ns.registry.register_sequence_hash(seq);
                handle.mark_present::<TierL2>();
                ns.handles.insert(seq, handle);
            } else {
                let _ = ns.registry.match_sequence_hash(seq, true);
            }

            let meta = entry.meta.clone();
            let local = meta
                .locations
                .iter()
                .any(|l| l.tier == Tier::L0 as i32 && l.node_id == requester);
            if !local {
                all_local = false;
            }
            out.push(ReusableBlock {
                id: meta.id.clone(),
                meta: Some(meta),
                local_hit: local,
            });
            hit += 1;
        }
        if hit == 0 {
            all_local = false;
        }
        (out, hit, all_local && hit > 0)
    }

    pub fn locate(&self, ids: &[KvBlockId]) -> Vec<BlockMeta> {
        let mut blocks = Vec::new();
        for id in ids {
            if let Some(ns) = self.ns(&id.model_id) {
                if let Some(entry) = ns.by_flat.get(&id.block_hash) {
                    blocks.push(entry.meta.clone());
                }
            }
        }
        blocks
    }

    /// Apply one ref delta. Returns error if block unknown.
    pub fn report_ref(&mut self, delta: &RefDelta) -> Result<(), String> {
        let id = delta
            .id
            .as_ref()
            .ok_or_else(|| "RefDelta missing id".to_string())?;
        let ns = self
            .namespaces
            .get_mut(&id.model_id)
            .ok_or_else(|| format!("unknown model_id {}", id.model_id))?;
        let entry = ns
            .by_flat
            .get(&id.block_hash)
            .ok_or_else(|| "RefDelta: unknown block_hash".to_string())?;
        let seq = entry.seq_hash;
        let block_id = entry.block_id;

        let cur = ns.global_refs.entry(seq).or_insert(0);
        let before = *cur;
        *cur = cur.saturating_add(i64::from(delta.delta));
        if *cur < 0 {
            *cur = 0;
        }
        let after = *cur;

        if before > 0 && after == 0 {
            // Candidate for eviction — do not delete.
            if !ns.inactive.has(seq) {
                ns.inactive.insert(seq, block_id);
            }
        } else if before == 0 && after > 0 {
            // Frozen again — leave inactive index (take if present).
            let _ = ns.inactive.take(seq, block_id);
        }
        Ok(())
    }

    /// Test / pressure hook: evict up to `n` inactive (ref==0) blocks.
    /// Returns number of blocks removed from the location view.
    pub fn evict_n(&mut self, model_id: &str, n: usize) -> usize {
        let Some(ns) = self.namespaces.get_mut(model_id) else {
            return 0;
        };
        let victims = ns.inactive.allocate(n);
        let mut removed = 0;
        for (seq, _bid) in victims {
            if ns.global_refs.get(&seq).copied().unwrap_or(0) > 0 {
                // Should not happen; skip.
                continue;
            }
            ns.handles.remove(&seq);
            ns.global_refs.remove(&seq);
            if let Some(flat) = ns.seq_to_flat.remove(&seq) {
                ns.by_flat.remove(&flat);
                removed += 1;
            }
        }
        removed
    }

    /// Inactive index size (tests).
    pub fn inactive_len(&self, model_id: &str) -> usize {
        self.ns(model_id).map(|n| n.inactive.len()).unwrap_or(0)
    }

    pub fn global_ref(&self, model_id: &str, flat: &[u8]) -> i64 {
        let Some(ns) = self.ns(model_id) else {
            return 0;
        };
        let Some(entry) = ns.by_flat.get(flat) else {
            return 0;
        };
        ns.global_refs.get(&entry.seq_hash).copied().unwrap_or(0)
    }
}
