//! 位置视图权威：每 `(model_id, revision)` 一个命名空间；
//! 其内按 `pool_kind`(TARGET/DRAFT) 各持一棵 `BlockRegistry`（schema 寻址含 pool_kind）。
//!
//! 参考:Dynamo `BlockRegistry` / `InactiveIndex`；驱逐主路径 =
//! `LineageBackend::with_frequency`（只驱叶子 ≈ 前缀亲和 + TinyLFU 冷叶优先 ≈ LFU-Aging）。
//! 不用 `BlockManager`/`BlockStore`——因此必须自己守 inactive 上界：
//! `report_ref` 满容只 skip insert（对齐 Dynamo `inactive.insert`）；
//! 压力 `allocate` 只在显式 `evict_n`（对齐 `allocate_atomic`）。
//! EventsManager 不接线。
//!
//! P4.5:显式 `RegisterModel` / `DeregisterModel`；禁止懒建命名空间。
//! 下线 = 整表 drop（强句柄释放 → radix Weak 失效 ≈ 按命名空间剪枝）。
//! **关键差异**(相对 Dynamo):lake 一等 `(model_id, revision)` + `TARGET|DRAFT` 同池寻址，
//! 不能共用单 registry 的 flat→entry 表。

use std::collections::HashMap;
use std::sync::Arc;

use kvbm_logical::registry::BlockRegistrationHandle;
use kvbm_logical::{
    BlockId, BlockRegistry, FrequencyTrackingCapacity, InactiveIndex, LineageBackend, SequenceHash,
};

use crate::hash_chain::lineage_from_prefix;
use crate::tier::{TierL0, TierL1, TierL2};
use lake_proto::lake::*;

const INACTIVE_CAP: usize = 4096;
/// Same threshold shape as `MultiLruBackend` / Frequency LeafPolicy.
const FREQ_THRESHOLDS: [u8; 3] = [3, 8, 15];

/// Control-plane model namespace key = `(model_id, revision)`.
/// Block identity within = `(pool_kind, block_hash)` → separate [`PoolView`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NamespaceKey {
    pub model_id: String,
    pub revision: String,
}

impl NamespaceKey {
    pub fn new(model_id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            revision: revision.into(),
        }
    }

    pub fn from_id(id: &KvBlockId) -> Self {
        Self::new(id.model_id.clone(), id.revision.clone())
    }
}

/// Wire `POOL_UNSPECIFIED`(0) → TARGET. Reject unknown kinds.
pub fn resolve_pool_kind(raw: i32) -> Result<i32, String> {
    // POOL_UNSPECIFIED = 0 → TARGET (LookupPrefix / wire default).
    if raw == 0 {
        return Ok(PoolKind::Target as i32);
    }
    if raw == PoolKind::Target as i32 || raw == PoolKind::Draft as i32 {
        return Ok(raw);
    }
    Err(format!("unsupported pool_kind {raw}"))
}

struct Entry {
    seq_hash: SequenceHash,
    meta: BlockMeta,
    block_id: BlockId,
}

/// One Dynamo-shaped registry domain per `pool_kind`.
struct PoolView {
    registry: BlockRegistry,
    handles: HashMap<SequenceHash, BlockRegistrationHandle>,
    /// Flat content hash → entry **within this pool_kind**.
    by_flat: HashMap<Vec<u8>, Entry>,
    seq_to_flat: HashMap<SequenceHash, Vec<u8>>,
    inactive: Box<dyn InactiveIndex>,
    inactive_cap: usize,
    global_refs: HashMap<SequenceHash, i64>,
    next_block_id: BlockId,
}

impl PoolView {
    fn new(inactive_cap: usize) -> Self {
        let cap = inactive_cap.max(1);
        let tracker = FrequencyTrackingCapacity::Small.create_tracker();
        let registry = BlockRegistry::builder()
            .frequency_tracker(Arc::clone(&tracker) as _)
            .build();
        let inactive = Box::new(
            LineageBackend::with_frequency(cap, FREQ_THRESHOLDS, tracker)
                .expect("Lineage+Frequency thresholds"),
        );
        Self {
            registry,
            handles: HashMap::new(),
            by_flat: HashMap::new(),
            seq_to_flat: HashMap::new(),
            inactive,
            inactive_cap: cap,
            global_refs: HashMap::new(),
            next_block_id: 1,
        }
    }

    fn alloc_block_id(&mut self) -> BlockId {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.saturating_add(1);
        id
    }

    fn drop_inactive_victims(&mut self, n: usize) -> usize {
        let victims = self.inactive.allocate(n);
        let mut removed = 0;
        for (seq, _bid) in victims {
            if self.global_refs.get(&seq).copied().unwrap_or(0) > 0 {
                continue;
            }
            self.handles.remove(&seq);
            self.global_refs.remove(&seq);
            if let Some(flat) = self.seq_to_flat.remove(&seq) {
                self.by_flat.remove(&flat);
                removed += 1;
            }
        }
        removed
    }
}

struct Namespace {
    descriptor: ModelDescriptor,
    /// `pool_kind` → independent radix / inactive / refs.
    pools: HashMap<i32, PoolView>,
    inactive_cap: usize,
}

impl Namespace {
    fn new(descriptor: ModelDescriptor, inactive_cap: usize) -> Self {
        Self {
            descriptor,
            pools: HashMap::new(),
            inactive_cap: inactive_cap.max(1),
        }
    }

    fn pool_mut(&mut self, pool_kind: i32) -> &mut PoolView {
        let cap = self.inactive_cap;
        self.pools
            .entry(pool_kind)
            .or_insert_with(|| PoolView::new(cap))
    }

    fn pool(&self, pool_kind: i32) -> Option<&PoolView> {
        self.pools.get(&pool_kind)
    }

    fn block_count(&self) -> usize {
        self.pools.values().map(|p| p.by_flat.len()).sum()
    }
}

/// Immutable identity of a registered model (quota is mutable).
fn descriptor_identity_eq(a: &ModelDescriptor, b: &ModelDescriptor) -> bool {
    a.model_id == b.model_id
        && a.revision == b.revision
        && a.num_layers == b.num_layers
        && a.hash_algo == b.hash_algo
        && a.block_spec == b.block_spec
}

/// Process-local authority state.
pub struct Authority {
    namespaces: HashMap<NamespaceKey, Namespace>,
    inactive_cap: usize,
    /// Completed request barriers: `(request_id, node_id)`.
    completed_barriers: HashMap<String, String>,
}

impl Default for Authority {
    fn default() -> Self {
        Self::with_inactive_cap(INACTIVE_CAP)
    }
}

impl Authority {
    pub fn with_inactive_cap(inactive_cap: usize) -> Self {
        Self {
            namespaces: HashMap::new(),
            inactive_cap: inactive_cap.max(1),
            completed_barriers: HashMap::new(),
        }
    }

    /// P4.5: register `(model_id, revision)` namespace.
    ///
    /// Idempotent only when immutable fields match (`num_layers` / `block_spec` /
    /// `hash_algo` / ids). Re-register may update **quota** only; other changes
    /// require a new `revision`.
    pub fn register_model(&mut self, desc: ModelDescriptor) -> Result<(), String> {
        if desc.model_id.is_empty() {
            return Err("RegisterModel: model_id required".into());
        }
        let key = NamespaceKey::new(desc.model_id.clone(), desc.revision.clone());
        if let Some(ns) = self.namespaces.get_mut(&key) {
            if !descriptor_identity_eq(&ns.descriptor, &desc) {
                return Err(format!(
                    "RegisterModel: immutable fields changed for ({}, rev={:?}); use a new revision \
                     (num_layers/block_spec/hash_algo must match)",
                    desc.model_id, desc.revision
                ));
            }
            // Same identity → allow quota refresh (P4.6 enforcement later).
            ns.descriptor.quota = desc.quota;
            return Ok(());
        }
        let cap = self.inactive_cap;
        self.namespaces.insert(key, Namespace::new(desc, cap));
        Ok(())
    }

    /// Cascade-delete one namespace (all pool_kinds). Bytes → P4.7.
    pub fn deregister_model(&mut self, model_id: &str, revision: &str) -> Result<(), String> {
        if model_id.is_empty() {
            return Err("DeregisterModel: model_id required".into());
        }
        let key = NamespaceKey::new(model_id, revision);
        if self.namespaces.remove(&key).is_none() {
            return Err(format!(
                "DeregisterModel: unknown namespace ({model_id}, rev={revision:?})"
            ));
        }
        Ok(())
    }

    pub fn has_namespace(&self, model_id: &str, revision: &str) -> bool {
        self.namespaces
            .contains_key(&NamespaceKey::new(model_id, revision))
    }

    pub fn model_descriptor(&self, model_id: &str, revision: &str) -> Option<&ModelDescriptor> {
        self.namespaces
            .get(&NamespaceKey::new(model_id, revision))
            .map(|n| &n.descriptor)
    }

    pub fn block_count(&self, model_id: &str, revision: &str) -> usize {
        self.ns(model_id, revision)
            .map(|n| n.block_count())
            .unwrap_or(0)
    }

    fn ns_mut(&mut self, model_id: &str, revision: &str) -> Option<&mut Namespace> {
        self.namespaces
            .get_mut(&NamespaceKey::new(model_id, revision))
    }

    fn ns(&self, model_id: &str, revision: &str) -> Option<&Namespace> {
        self.namespaces.get(&NamespaceKey::new(model_id, revision))
    }

    /// Register durable blocks. One `RegisterBlocks` batch = one `pool_kind`
    /// contiguous segment of `prefix_hashes`.
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
        let revision = metas
            .iter()
            .find_map(|m| m.id.as_ref().map(|i| i.revision.clone()))
            .unwrap_or_default();
        let pool_kind = {
            let raw = metas
                .iter()
                .find_map(|m| m.id.as_ref().map(|i| i.pool_kind))
                .unwrap_or(PoolKind::Target as i32);
            resolve_pool_kind(raw)?
        };

        if prefix_hashes.is_empty() {
            return Err("RegisterBlocks: prefix_hashes required (P4.2 lineage)".into());
        }

        let index_of: HashMap<&[u8], usize> = prefix_hashes
            .iter()
            .enumerate()
            .map(|(i, h)| (h.as_slice(), i))
            .collect();

        let mut positions: Vec<usize> = Vec::with_capacity(metas.len());
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
            if id.revision != revision {
                return Err(format!(
                    "RegisterBlocks: mixed revision {:?} vs {:?}",
                    revision, id.revision
                ));
            }
            let pk = resolve_pool_kind(id.pool_kind)?;
            if pk != pool_kind {
                return Err(format!(
                    "RegisterBlocks: mixed pool_kind {pk} vs {pool_kind}"
                ));
            }
            let Some(&pos) = index_of.get(id.block_hash.as_slice()) else {
                return Err(format!(
                    "RegisterBlocks: block hash not in prefix_hashes (len={})",
                    id.block_hash.len()
                ));
            };
            positions.push(pos);
        }
        if positions.is_empty() {
            return Err("RegisterBlocks: no KVBlockID".into());
        }
        positions.sort_unstable();
        positions.dedup();
        for w in positions.windows(2) {
            if w[1] != w[0] + 1 {
                return Err(
                    "RegisterBlocks: blocks must be a contiguous segment of prefix_hashes".into(),
                );
            }
        }

        if !self.has_namespace(&model_id, &revision) {
            return Err(format!(
                "RegisterBlocks: model not registered ({model_id}, rev={revision:?}); call RegisterModel first"
            ));
        }

        let lineage = lineage_from_prefix(prefix_hashes);
        let ns = self
            .ns_mut(&model_id, &revision)
            .expect("checked has_namespace");
        let pool = ns.pool_mut(pool_kind);

        for mut meta in metas {
            let Some(id) = meta.id.clone() else { continue };
            // Normalize unspecified → TARGET on stored identity.
            if let Some(ref mut mid) = meta.id {
                mid.pool_kind = pool_kind;
            }
            let flat = id.block_hash.clone();
            let pos = *index_of.get(flat.as_slice()).expect("checked");
            let seq = lineage[pos];

            if meta.locations.is_empty() && !meta.l3_present {
                return Err(
                    "RegisterBlocks: need L2 location or l3_present (durable first)".into(),
                );
            }
            for loc in &mut meta.locations {
                if loc.node_id.is_empty() {
                    loc.node_id = node_id.to_string();
                }
            }

            let handle = pool.registry.register_sequence_hash(seq);
            for loc in &meta.locations {
                if loc.tier == Tier::L0 as i32 {
                    handle.mark_present::<TierL0>();
                } else if loc.tier == Tier::L1 as i32 {
                    handle.mark_present::<TierL1>();
                } else if loc.tier == Tier::L2 as i32 {
                    handle.mark_present::<TierL2>();
                }
            }
            pool.handles.insert(seq, handle);

            let block_id = if let Some(prev) = pool.by_flat.get(&flat) {
                prev.block_id
            } else {
                pool.alloc_block_id()
            };
            pool.seq_to_flat.insert(seq, flat.clone());
            pool.by_flat.insert(
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

    /// Prefix lookup in one `pool_kind` domain.
    ///
    /// `pool_kind == UNSPECIFIED` → TARGET (P4.5 default; draft prefix opt-in).
    pub fn lookup_prefix(
        &mut self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        prefix_hashes: &[Vec<u8>],
        requester: &str,
    ) -> (Vec<ReusableBlock>, u32, bool) {
        if prefix_hashes.is_empty() {
            return (Vec::new(), 0, false);
        }
        let Ok(pool_kind) = resolve_pool_kind(pool_kind) else {
            return (Vec::new(), 0, false);
        };
        if self.ns(model_id, revision).is_none() {
            return (Vec::new(), 0, false);
        }
        let lineage = lineage_from_prefix(prefix_hashes);
        let ns = self.ns_mut(model_id, revision).expect("checked");
        let Some(pool) = ns.pools.get_mut(&pool_kind) else {
            return (Vec::new(), 0, false);
        };

        let mut out = Vec::new();
        let mut hit = 0u32;
        let mut all_local = true;

        for (i, flat) in prefix_hashes.iter().enumerate() {
            let seq = lineage[i];
            let Some(entry) = pool.by_flat.get(flat) else {
                all_local = false;
                break;
            };
            if entry.seq_hash != seq {
                all_local = false;
                break;
            }
            let meta = entry.meta.clone();
            if !pool.handles.contains_key(&seq) {
                let handle = pool.registry.register_sequence_hash(seq);
                for loc in &meta.locations {
                    if loc.tier == Tier::L0 as i32 {
                        handle.mark_present::<TierL0>();
                    } else if loc.tier == Tier::L1 as i32 {
                        handle.mark_present::<TierL1>();
                    } else if loc.tier == Tier::L2 as i32 {
                        handle.mark_present::<TierL2>();
                    }
                }
                pool.handles.insert(seq, handle);
            } else {
                let _ = pool.registry.match_sequence_hash(seq, true);
            }

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
            let Ok(pk) = resolve_pool_kind(id.pool_kind) else {
                continue;
            };
            if let Some(ns) = self.ns(&id.model_id, &id.revision) {
                if let Some(pool) = ns.pool(pk) {
                    if let Some(entry) = pool.by_flat.get(&id.block_hash) {
                        blocks.push(entry.meta.clone());
                    }
                }
            }
        }
        blocks
    }

    pub fn check_report_ref(&self, delta: &RefDelta) -> Result<(), String> {
        let id = delta
            .id
            .as_ref()
            .ok_or_else(|| "RefDelta missing id".to_string())?;
        let pk = resolve_pool_kind(id.pool_kind)?;
        let ns = self
            .ns(&id.model_id, &id.revision)
            .ok_or_else(|| format!("unknown namespace ({}, rev={:?})", id.model_id, id.revision))?;
        let pool = ns
            .pool(pk)
            .ok_or_else(|| format!("RefDelta: unknown pool_kind {pk}"))?;
        if !pool.by_flat.contains_key(&id.block_hash) {
            return Err("RefDelta: unknown block_hash".to_string());
        }
        Ok(())
    }

    pub fn report_ref(&mut self, delta: &RefDelta) -> Result<(), String> {
        self.check_report_ref(delta)?;
        let id = delta.id.as_ref().expect("checked");
        let pk = resolve_pool_kind(id.pool_kind).expect("checked");
        let key = NamespaceKey::from_id(id);
        let ns = self.namespaces.get_mut(&key).expect("checked");
        let pool = ns.pools.get_mut(&pk).expect("checked");
        let entry = pool.by_flat.get(&id.block_hash).expect("checked");
        let seq = entry.seq_hash;
        let block_id = entry.block_id;
        let _kind = delta.kind;

        let cur = pool.global_refs.entry(seq).or_insert(0);
        let before = *cur;
        *cur = cur.saturating_add(i64::from(delta.delta));
        if *cur < 0 {
            *cur = 0;
        }
        let after = *cur;

        if before > 0 && after == 0 {
            if !pool.inactive.has(seq) && pool.inactive.len() < pool.inactive_cap {
                pool.inactive.insert(seq, block_id);
            }
        } else if before == 0 && after > 0 {
            let _ = pool.inactive.take(seq, block_id);
        }
        Ok(())
    }

    pub fn report_refs(&mut self, deltas: &[RefDelta]) -> Result<(), String> {
        for (i, d) in deltas.iter().enumerate() {
            self.check_report_ref(d)
                .map_err(|e| format!("ReportRef batch[{i}]: {e}"))?;
        }
        for d in deltas {
            self.report_ref(d)
                .expect("report_ref after successful check_report_ref");
        }
        Ok(())
    }

    pub fn evict_n(&mut self, model_id: &str, revision: &str, pool_kind: i32, n: usize) -> usize {
        let Ok(pk) = resolve_pool_kind(pool_kind) else {
            return 0;
        };
        let Some(ns) = self.ns_mut(model_id, revision) else {
            return 0;
        };
        let Some(pool) = ns.pools.get_mut(&pk) else {
            return 0;
        };
        pool.drop_inactive_victims(n)
    }

    pub fn inactive_len(&self, model_id: &str, revision: &str, pool_kind: i32) -> usize {
        let Ok(pk) = resolve_pool_kind(pool_kind) else {
            return 0;
        };
        self.ns(model_id, revision)
            .and_then(|n| n.pool(pk))
            .map(|p| p.inactive.len())
            .unwrap_or(0)
    }

    pub fn global_ref(&self, model_id: &str, revision: &str, pool_kind: i32, flat: &[u8]) -> i64 {
        let Ok(pk) = resolve_pool_kind(pool_kind) else {
            return 0;
        };
        let Some(ns) = self.ns(model_id, revision) else {
            return 0;
        };
        let Some(pool) = ns.pool(pk) else {
            return 0;
        };
        let Some(entry) = pool.by_flat.get(flat) else {
            return 0;
        };
        pool.global_refs.get(&entry.seq_hash).copied().unwrap_or(0)
    }

    pub fn complete_barrier(&mut self, request_id: &str, node_id: &str) -> Result<(), String> {
        if request_id.is_empty() {
            return Err("RequestBarrier: request_id required".into());
        }
        if node_id.is_empty() {
            return Err("RequestBarrier: node_id required".into());
        }
        self.completed_barriers
            .insert(request_id.to_string(), node_id.to_string());
        Ok(())
    }

    pub fn barrier_completed(&self, request_id: &str) -> bool {
        self.completed_barriers.contains_key(request_id)
    }

    #[allow(clippy::too_many_arguments)] // wire-shaped presence update; pack later if needed
    pub fn publish_location(
        &mut self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        flat: &[u8],
        tier: Tier,
        node_id: &str,
        present: bool,
    ) -> Result<(), String> {
        let pk = resolve_pool_kind(pool_kind)?;
        let ns = self
            .ns_mut(model_id, revision)
            .ok_or_else(|| format!("unknown namespace ({model_id}, rev={revision:?})"))?;
        let pool = ns
            .pools
            .get_mut(&pk)
            .ok_or_else(|| format!("publish_location: unknown pool_kind {pk}"))?;
        let entry = pool
            .by_flat
            .get_mut(flat)
            .ok_or_else(|| "publish_location: unknown block".to_string())?;
        let handle = pool
            .handles
            .get(&entry.seq_hash)
            .ok_or_else(|| "publish_location: missing handle".to_string())?;

        let tier_i = tier as i32;
        let had = entry
            .meta
            .locations
            .iter()
            .any(|l| l.tier == tier_i && l.node_id == node_id);

        if present && !had {
            entry.meta.locations.push(Location {
                tier: tier_i,
                node_id: node_id.to_string(),
                segment_id: 1,
                offset: 0,
            });
            match tier {
                Tier::L0 => handle.mark_present::<TierL0>(),
                Tier::L1 => handle.mark_present::<TierL1>(),
                Tier::L2 => handle.mark_present::<TierL2>(),
                _ => {}
            }
        } else if !present && had {
            entry
                .meta
                .locations
                .retain(|l| !(l.tier == tier_i && l.node_id == node_id));
            match tier {
                Tier::L0 => handle.mark_absent::<TierL0>(),
                Tier::L1 => handle.mark_absent::<TierL1>(),
                Tier::L2 => handle.mark_absent::<TierL2>(),
                _ => {}
            }
        }
        Ok(())
    }

    pub fn set_l3_present(
        &mut self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        flat: &[u8],
        present: bool,
    ) -> Result<(), String> {
        let pk = resolve_pool_kind(pool_kind)?;
        let ns = self
            .ns_mut(model_id, revision)
            .ok_or_else(|| format!("unknown namespace ({model_id}, rev={revision:?})"))?;
        let pool = ns
            .pools
            .get_mut(&pk)
            .ok_or_else(|| format!("set_l3_present: unknown pool_kind {pk}"))?;
        let entry = pool
            .by_flat
            .get_mut(flat)
            .ok_or_else(|| "set_l3_present: unknown block".to_string())?;
        entry.meta.l3_present = present;
        Ok(())
    }

    pub fn has_l0_on(
        &self,
        model_id: &str,
        revision: &str,
        pool_kind: i32,
        flat: &[u8],
        node_id: &str,
    ) -> bool {
        let Ok(pk) = resolve_pool_kind(pool_kind) else {
            return false;
        };
        self.ns(model_id, revision)
            .and_then(|n| n.pool(pk))
            .and_then(|p| p.by_flat.get(flat))
            .map(|e| {
                e.meta
                    .locations
                    .iter()
                    .any(|l| l.tier == Tier::L0 as i32 && l.node_id == node_id)
            })
            .unwrap_or(false)
    }
}
