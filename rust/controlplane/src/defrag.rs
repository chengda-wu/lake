//! P4.8:碎片整理计划（逻辑共置 + 物理压实）。
//!
//! 参考:Mooncake 无 compaction 线程（靠 `OffsetBufferAllocator` 减碎片）；
//! SGLang group semantics ≈ 共置提示；Dynamo Pipeline 已承载带宽节流。
//! 关键差异:CP 出计划 + 更新 Location 权威；字节/段布局在 tiered-store SegmentArena。

use std::collections::HashMap;

use lake_proto::lake::*;

use crate::authority::{resolve_pool_kind, Authority};
use crate::tier::TierL2;

/// Default slot bytes when TriggerDefrag.slot_bytes == 0 (align SegmentArena).
pub const DEFAULT_DEFRAG_SLOT_BYTES: u64 = 4096;

impl Authority {
    /// Build defrag moves from the location view (no byte mutation).
    pub fn plan_defrag(
        &self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        mode: DefragMode,
        slot_bytes: u64,
    ) -> Result<Vec<DefragMove>, String> {
        let slot = if slot_bytes == 0 {
            DEFAULT_DEFRAG_SLOT_BYTES
        } else {
            slot_bytes
        };
        let pk = resolve_pool_kind(pool_kind)?;
        let Some(ns) = self.ns(model_id, revision) else {
            return Err(format!(
                "plan_defrag: unknown namespace ({model_id}, rev={revision:?})"
            ));
        };
        let Some(pool) = ns.pools.get(&pk) else {
            return Ok(Vec::new());
        };

        let mode = if mode == DefragMode::Unspecified {
            DefragMode::Both
        } else {
            mode
        };

        let mut moves = Vec::new();
        if mode == DefragMode::Compact || mode == DefragMode::Both {
            moves.extend(plan_compact(pool, slot));
        }
        if mode == DefragMode::Colocate || mode == DefragMode::Both {
            moves.extend(plan_colocate(pool, slot));
        }
        Ok(moves)
    }

    /// Update L0/L1/L2 placement coordinates (P4.8 Moved).
    pub fn relocate_in_view(
        &mut self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        flat: &[u8],
        tier: Tier,
        node_id: &str,
        segment_id: u64,
        offset: u64,
    ) -> Result<(), String> {
        let pk = resolve_pool_kind(pool_kind)?;
        let ns = self
            .ns_mut(model_id, revision)
            .ok_or_else(|| format!("unknown namespace ({model_id}, rev={revision:?})"))?;
        let pool = ns
            .pools
            .get_mut(&pk)
            .ok_or_else(|| format!("relocate: unknown pool_kind {pk}"))?;
        let entry = pool
            .by_flat
            .get_mut(flat)
            .ok_or_else(|| "relocate: unknown block".to_string())?;
        let tier_i = tier as i32;
        let mut found = false;
        for loc in &mut entry.meta.locations {
            if loc.tier == tier_i && loc.node_id == node_id {
                loc.segment_id = segment_id;
                loc.offset = offset;
                found = true;
                break;
            }
        }
        if !found {
            // Upsert: treat as present at new coords.
            entry.meta.locations.push(Location {
                tier: tier_i,
                node_id: node_id.to_string(),
                segment_id,
                offset,
            });
            if let Some(handle) = pool.handles.get(&entry.seq_hash) {
                if tier == Tier::L2 {
                    handle.mark_present::<TierL2>();
                }
            }
        }
        Ok(())
    }
}

fn plan_compact(pool: &crate::authority::PoolView, slot: u64) -> Vec<DefragMove> {
    // (node, segment) → sorted offsets
    let mut groups: HashMap<(String, u64), Vec<u64>> = HashMap::new();
    for entry in pool.by_flat.values() {
        if pool.global_refs.get(&entry.seq_hash).copied().unwrap_or(0) > 0 {
            continue;
        }
        for loc in &entry.meta.locations {
            if loc.tier != Tier::L2 as i32 {
                continue;
            }
            groups
                .entry((loc.node_id.clone(), loc.segment_id))
                .or_default()
                .push(loc.offset);
        }
    }
    let mut out = Vec::new();
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();
    for (node_id, segment_id) in keys {
        let mut offs = groups.remove(&(node_id.clone(), segment_id)).unwrap();
        offs.sort_unstable();
        offs.dedup();
        if offs.len() < 2 {
            continue;
        }
        let dense = (0..offs.len())
            .all(|i| offs[i] == (i as u64).saturating_mul(slot));
        if dense {
            continue;
        }
        out.push(DefragMove {
            id: None,
            node_id,
            from_segment: segment_id,
            from_offset: 0,
            to_segment: segment_id,
            to_offset: 0,
            compact_segment: true,
            segment_id,
        });
    }
    out
}

fn plan_colocate(pool: &crate::authority::PoolView, slot: u64) -> Vec<DefragMove> {
    // Group by full prefix_chain key (blocks that share a chain).
    // Collect L2 placements per chain; if fan-out across segments, pack onto first.
    let mut chains: HashMap<Vec<Vec<u8>>, Vec<(Vec<u8>, String, u64, u64)>> = HashMap::new();
    for entry in pool.by_flat.values() {
        if entry.prefix_chain.len() < 2 {
            continue;
        }
        if pool.global_refs.get(&entry.seq_hash).copied().unwrap_or(0) > 0 {
            continue;
        }
        let Some(l2) = entry
            .meta
            .locations
            .iter()
            .find(|l| l.tier == Tier::L2 as i32)
        else {
            continue;
        };
        let flat = entry
            .meta
            .id
            .as_ref()
            .map(|i| i.block_hash.clone())
            .unwrap_or_default();
        // Index under the full chain of the longest member — use each block's own chain
        // as key; later merge by shared root prefix of length >=2.
        chains
            .entry(entry.prefix_chain.clone())
            .or_default()
            .push((flat, l2.node_id.clone(), l2.segment_id, l2.offset));
    }

    // Also gather by chain root: for chains [h0]/[h0,h1] register under longest.
    // Rebuild: map root → ordered positions from all entries sharing that root.
    let mut by_root: HashMap<Vec<u8>, Vec<(usize, Vec<u8>, String, u64, u64, Vec<Vec<u8>>)>> =
        HashMap::new();
    for entry in pool.by_flat.values() {
        if entry.prefix_chain.is_empty() {
            continue;
        }
        if pool.global_refs.get(&entry.seq_hash).copied().unwrap_or(0) > 0 {
            continue;
        }
        let Some(l2) = entry
            .meta
            .locations
            .iter()
            .find(|l| l.tier == Tier::L2 as i32)
        else {
            continue;
        };
        let flat = entry
            .meta
            .id
            .as_ref()
            .map(|i| i.block_hash.clone())
            .unwrap_or_default();
        let root = entry.prefix_chain[0].clone();
        let pos = entry
            .prefix_chain
            .iter()
            .position(|h| h == &flat)
            .unwrap_or(entry.prefix_chain.len().saturating_sub(1));
        by_root.entry(root).or_default().push((
            pos,
            flat,
            l2.node_id.clone(),
            l2.segment_id,
            l2.offset,
            entry.prefix_chain.clone(),
        ));
    }

    let mut out = Vec::new();
    let mut roots: Vec<_> = by_root.keys().cloned().collect();
    roots.sort();
    for root in roots {
        let mut members = by_root.remove(&root).unwrap();
        if members.len() < 2 {
            continue;
        }
        members.sort_by_key(|(pos, _, _, _, _, _)| *pos);
        let segments: std::collections::HashSet<u64> =
            members.iter().map(|(_, _, _, s, _, _)| *s).collect();
        let nodes: std::collections::HashSet<&str> =
            members.iter().map(|(_, _, n, _, _, _)| n.as_str()).collect();
        let needs = segments.len() > 1
            || nodes.len() > 1
            || !is_adjacent_on_segment(&members, slot);
        if !needs {
            continue;
        }
        let dest_node = members[0].2.clone();
        let dest_seg = members[0].3;
        for (i, (_pos, flat, node, seg, off, _chain)) in members.iter().enumerate() {
            let to_off = (i as u64).saturating_mul(slot);
            if *seg == dest_seg && *off == to_off && node == &dest_node {
                continue;
            }
            let id = pool.by_flat.get(flat).and_then(|e| e.meta.id.clone());
            out.push(DefragMove {
                id,
                node_id: dest_node.clone(),
                from_segment: *seg,
                from_offset: *off,
                to_segment: dest_seg,
                to_offset: to_off,
                compact_segment: false,
                segment_id: dest_seg,
            });
        }
    }
    let _ = chains; // reserved for future chain-key grouping
    out
}

fn is_adjacent_on_segment(
    members: &[(usize, Vec<u8>, String, u64, u64, Vec<Vec<u8>>)],
    slot: u64,
) -> bool {
    if members.is_empty() {
        return true;
    }
    let seg0 = members[0].3;
    let node0 = &members[0].2;
    for (i, (_, _, node, seg, off, _)) in members.iter().enumerate() {
        if node != node0 || *seg != seg0 {
            return false;
        }
        if *off != (i as u64).saturating_mul(slot) {
            return false;
        }
    }
    true
}
