//! 本地分层字节引擎：L0(HBM 站位) / L1(DRAM) / L2(NVMe) / L3(对象存储站位)。
//!
//! 参考：SGLang `hiradix_cache.py::_evict_write_back`（`lock_ref>0` 跳过）+
//! `write_backup` 后再丢 device；Dynamo offload 副作用需回写 presence。
//! 关键差异：无 BlockStore；位置权威在 CP；本 crate 通过 [`TierSideEffects`] 上报副作用。

use std::collections::{HashMap, VecDeque};

use crate::stats::{AccessKind, HitStats};

/// Soft capacity knobs (block counts). `0` = unlimited.
#[derive(Debug, Clone, Copy)]
pub struct TierCaps {
    pub l0: usize,
    pub l1: usize,
    pub l2: usize,
}

impl Default for TierCaps {
    fn default() -> Self {
        Self {
            l0: 64,
            l1: 256,
            l2: 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTier {
    L0,
    L1,
    L2,
    L3,
}

/// Collateral tier mutations that **must** be published to the controlplane view.
///
/// Mirrors Dynamo transfer→presence apply: engine mutates bytes, caller syncs markers.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TierSideEffects {
    pub l0_demoted: Vec<Vec<u8>>,
    pub l2_demoted_to_l3: Vec<Vec<u8>>,
}

impl TierSideEffects {
    pub fn merge(&mut self, other: TierSideEffects) {
        self.l0_demoted.extend(other.l0_demoted);
        self.l2_demoted_to_l3.extend(other.l2_demoted_to_l3);
    }

    pub fn is_empty(&self) -> bool {
        self.l0_demoted.is_empty() && self.l2_demoted_to_l3.is_empty()
    }
}

/// In-process L0–L3 byte maps + LRU + local pin (≈ SGLang `lock_ref`).
pub struct LocalTierEngine {
    caps: TierCaps,
    l0: HashMap<Vec<u8>, Vec<u8>>,
    l1: HashMap<Vec<u8>, Vec<u8>>,
    l2: HashMap<Vec<u8>, Vec<u8>>,
    l3: HashMap<Vec<u8>, Vec<u8>>,
    l0_order: VecDeque<Vec<u8>>,
    l1_order: VecDeque<Vec<u8>>,
    l2_order: VecDeque<Vec<u8>>,
    /// Local freeze (request / writeback / in-flight). Demote skips when >0.
    pins: HashMap<Vec<u8>, u32>,
    pub stats: HitStats,
}

impl Default for LocalTierEngine {
    fn default() -> Self {
        Self::with_caps(TierCaps::default())
    }
}

impl LocalTierEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_caps(caps: TierCaps) -> Self {
        Self {
            caps,
            l0: HashMap::new(),
            l1: HashMap::new(),
            l2: HashMap::new(),
            l3: HashMap::new(),
            l0_order: VecDeque::new(),
            l1_order: VecDeque::new(),
            l2_order: VecDeque::new(),
            pins: HashMap::new(),
            stats: HitStats::default(),
        }
    }

    pub fn caps(&self) -> TierCaps {
        self.caps
    }

    pub fn l0_len(&self) -> usize {
        self.l0.len()
    }
    pub fn l1_len(&self) -> usize {
        self.l1.len()
    }
    pub fn l2_len(&self) -> usize {
        self.l2.len()
    }
    pub fn l3_len(&self) -> usize {
        self.l3.len()
    }

    pub fn is_l2_durable(&self, h: &[u8]) -> bool {
        self.l2.contains_key(h)
    }

    pub fn l3_present(&self, h: &[u8]) -> bool {
        self.l3.contains_key(h)
    }

    pub fn is_settled(&self, h: &[u8]) -> bool {
        self.l2.contains_key(h) || self.l3.contains_key(h)
    }

    pub fn has_durable_backing(&self, h: &[u8]) -> bool {
        self.is_settled(h)
    }

    pub fn pin_count(&self, h: &[u8]) -> u32 {
        self.pins.get(h).copied().unwrap_or(0)
    }

    /// ≈ SGLang `inc_lock_ref` — freeze demote while request/writeback holds the block.
    pub fn pin(&mut self, h: &[u8]) {
        *self.pins.entry(h.to_vec()).or_insert(0) += 1;
    }

    pub fn unpin(&mut self, h: &[u8]) -> Result<(), String> {
        let Some(c) = self.pins.get_mut(h) else {
            return Err("unpin: not pinned".into());
        };
        *c = c.saturating_sub(1);
        if *c == 0 {
            self.pins.remove(h);
        }
        Ok(())
    }

    pub(crate) fn l0_nbytes(&self, h: &[u8]) -> u64 {
        self.l0.get(h).map(|b| b.len() as u64).unwrap_or(0)
    }

    pub(crate) fn l2_nbytes(&self, h: &[u8]) -> u64 {
        self.l2.get(h).map(|b| b.len() as u64).unwrap_or(0)
    }

    /// Multi-hop promote cost estimate (L3→L2→L1→L0 may each move `nbytes`).
    pub fn estimate_promote_cost(&self, h: &[u8]) -> u64 {
        let Some(n) = self.get(h).map(|b| b.len() as u64) else {
            return 0;
        };
        if self.l0.contains_key(h) {
            return 0;
        }
        let mut hops = 1u64; // into L0
        if !self.l1.contains_key(h) {
            hops += 1;
        }
        if !self.l2.contains_key(h) && self.l3.contains_key(h) {
            hops += 1; // L3→L2 fill
        }
        n * hops
    }

    fn touch(order: &mut VecDeque<Vec<u8>>, h: &[u8]) {
        order.retain(|x| x.as_slice() != h);
        order.push_back(h.to_vec());
    }

    /// PutEnd / 满块写回：只落 L2（F4 恢复点）。
    ///
    /// 对齐 `storage-layer.md`：写回不写 L1、不写 L3。L3 仅经 [`demote_l2_to_l3`]
    /// 或 L2 容量压力（迁移窗后稳态 XOR）。参考：SGLang 满块不写 host；
    /// Mooncake PutEnd 完成当前副本；Dynamo offload 链式 demote 而非 Put 双写。
    pub fn put_durable(
        &mut self,
        h: &[u8],
        bytes: &[u8],
    ) -> Result<(LocalTier, TierSideEffects), String> {
        self.l2.insert(h.to_vec(), bytes.to_vec());
        Self::touch(&mut self.l2_order, h);
        let l2_demoted_to_l3 = self.ensure_l2_cap();
        let effects = TierSideEffects {
            l0_demoted: Vec::new(),
            l2_demoted_to_l3,
        };
        if self.l2.contains_key(h) {
            Ok((LocalTier::L2, effects))
        } else if self.l3.contains_key(h) {
            // This hash itself was immediately demoted under L2 cap → L3 only.
            Ok((LocalTier::L3, effects))
        } else {
            Err("put_durable: block lost after L2 cap demote".into())
        }
    }

    fn ensure_l2_cap(&mut self) -> Vec<Vec<u8>> {
        let mut demoted = Vec::new();
        let cap = self.caps.l2;
        if cap == 0 {
            return demoted;
        }
        let mut skipped = 0usize;
        while self.l2.len() > cap {
            if skipped >= self.l2.len() {
                break; // all remaining pinned / stuck
            }
            let Some(victim) = self.l2_order.pop_front() else {
                break;
            };
            if !self.l2.contains_key(&victim) {
                continue;
            }
            if self.pin_count(&victim) > 0 {
                self.l2_order.push_back(victim);
                skipped += 1;
                continue;
            }
            skipped = 0;
            if let Some(bytes) = self.l2.remove(&victim) {
                self.l3.entry(victim.clone()).or_insert(bytes);
                demoted.push(victim);
            }
        }
        demoted
    }

    pub fn probe(&mut self, h: &[u8]) -> AccessKind {
        let kind = if self.l0.contains_key(h) {
            AccessKind::L0Hit
        } else if self.l1.contains_key(h) {
            AccessKind::L1Hit
        } else if self.l2.contains_key(h) {
            AccessKind::L2Hit
        } else if self.l3.contains_key(h) {
            AccessKind::L3Hit
        } else {
            AccessKind::Miss
        };
        self.stats.record(kind);
        kind
    }

    pub fn get(&self, h: &[u8]) -> Option<&[u8]> {
        self.l0
            .get(h)
            .or_else(|| self.l1.get(h))
            .or_else(|| self.l2.get(h))
            .or_else(|| self.l3.get(h))
            .map(|v| v.as_slice())
    }

    pub fn local_tier(&self, h: &[u8]) -> Option<LocalTier> {
        if self.l0.contains_key(h) {
            Some(LocalTier::L0)
        } else if self.l1.contains_key(h) {
            Some(LocalTier::L1)
        } else if self.l2.contains_key(h) {
            Some(LocalTier::L2)
        } else if self.l3.contains_key(h) {
            Some(LocalTier::L3)
        } else {
            None
        }
    }

    /// Promote toward L0. Returns `(bytes_moved_estimate, collateral effects)`.
    pub fn promote_to_l0(&mut self, h: &[u8]) -> Result<(u64, TierSideEffects), String> {
        let mut effects = TierSideEffects::default();
        if self.l0.contains_key(h) {
            Self::touch(&mut self.l0_order, h);
            return Ok((0, effects));
        }
        let bytes = self
            .get(h)
            .ok_or_else(|| "promote: block missing all tiers".to_string())?
            .to_vec();
        let n = self.estimate_promote_cost(h).max(bytes.len() as u64);
        if !self.l2.contains_key(h) && self.l3.contains_key(h) {
            self.l2.insert(h.to_vec(), bytes.clone());
            Self::touch(&mut self.l2_order, h);
            effects.l2_demoted_to_l3 = self.ensure_l2_cap();
            if !self.is_settled(h) {
                return Err("promote: lost durable backing under L2 cap".into());
            }
        }
        if !self.l1.contains_key(h) {
            self.ensure_l1_room();
            self.l1.insert(h.to_vec(), bytes.clone());
            Self::touch(&mut self.l1_order, h);
        }
        effects.merge(self.ensure_l0_room()?);
        self.l0.insert(h.to_vec(), bytes);
        Self::touch(&mut self.l0_order, h);
        Ok((n, effects))
    }

    /// Demote L0. Requires durable backing; refuses if locally pinned (`lock_ref`).
    pub fn demote_l0(&mut self, h: &[u8]) -> Result<u64, String> {
        if self.pin_count(h) > 0 {
            return Err("demote_l0: pinned (lock_ref)".into());
        }
        if !self.has_durable_backing(h) {
            return Err("demote_l0: need L2 or L3 durable backing".into());
        }
        let n = self.l0.remove(h).map(|b| b.len() as u64).unwrap_or(0);
        self.l0_order.retain(|x| x.as_slice() != h);
        Ok(n)
    }

    pub fn fill_read_miss(&mut self, h: &[u8]) -> Result<(u64, TierSideEffects), String> {
        if self.l0.contains_key(h) {
            Self::touch(&mut self.l0_order, h);
            return Ok((0, TierSideEffects::default()));
        }
        self.promote_to_l0(h)
    }

    pub fn evict_l0_pressure(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.caps.l0 == 0 || self.l0.len() < self.caps.l0 {
            return Ok(None);
        }
        let victim = self
            .l0_order
            .iter()
            .find(|h| self.pin_count(h) == 0 && self.has_durable_backing(h.as_slice()))
            .cloned()
            .ok_or_else(|| {
                "evict_l0: no durable unpinned victim (pin/writeback first)".to_string()
            })?;
        self.demote_l0(&victim)?;
        Ok(Some(victim))
    }

    fn ensure_l0_room(&mut self) -> Result<TierSideEffects, String> {
        let mut effects = TierSideEffects::default();
        let cap = self.caps.l0;
        if cap == 0 {
            return Ok(effects);
        }
        while self.l0.len() >= cap {
            match self.evict_l0_pressure()? {
                Some(v) => effects.l0_demoted.push(v),
                None => return Err("l0 full: cannot make room".into()),
            }
        }
        Ok(effects)
    }

    fn ensure_l1_room(&mut self) {
        let cap = self.caps.l1;
        if cap == 0 {
            return;
        }
        while self.l1.len() >= cap {
            let Some(v) = self.l1_order.pop_front() else {
                break;
            };
            self.l1.remove(&v);
        }
    }

    /// Steady L2/L3 XOR: move L2 → L3, drop L2.
    pub fn demote_l2_to_l3(&mut self, h: &[u8]) -> Result<u64, String> {
        if self.pin_count(h) > 0 {
            return Err("demote_l2: pinned".into());
        }
        let bytes = self
            .l2
            .remove(h)
            .ok_or_else(|| "demote_l2: not on L2".to_string())?;
        let n = bytes.len() as u64;
        self.l2_order.retain(|x| x.as_slice() != h);
        self.l3.insert(h.to_vec(), bytes);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_promote_demote() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 2,
            l1: 4,
            l2: 8,
        });
        e.put_durable(b"a", b"A").unwrap();
        assert!(!e.l3_present(b"a"), "PutEnd must not dual-write L3");
        e.promote_to_l0(b"a").unwrap();
        assert_eq!(e.local_tier(b"a"), Some(LocalTier::L0));
        e.demote_l0(b"a").unwrap();
        assert_eq!(e.local_tier(b"a"), Some(LocalTier::L1));
    }

    #[test]
    fn l0_pressure_reports_collateral() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 1,
            l1: 8,
            l2: 8,
        });
        e.put_durable(b"c0", b"0").unwrap();
        e.put_durable(b"c1", b"1").unwrap();
        e.promote_to_l0(b"c0").unwrap();
        let (_, fx) = e.promote_to_l0(b"c1").unwrap();
        assert_eq!(fx.l0_demoted, vec![b"c0".to_vec()]);
        assert_eq!(e.l0_len(), 1);
    }

    #[test]
    fn read_miss_from_l3() {
        let mut e = LocalTierEngine::new();
        e.l3.insert(b"z".to_vec(), b"ZZ".to_vec());
        e.fill_read_miss(b"z").unwrap();
        assert_eq!(e.local_tier(b"z"), Some(LocalTier::L0));
        assert!(e.is_l2_durable(b"z"));
    }

    #[test]
    fn l0_demote_allows_l3_only_backing() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 1,
            l1: 8,
            l2: 8,
        });
        e.put_durable(b"p", b"P").unwrap();
        e.promote_to_l0(b"p").unwrap();
        e.demote_l2_to_l3(b"p").unwrap();
        assert!(!e.is_l2_durable(b"p") && e.l3_present(b"p"));
        e.put_durable(b"q", b"Q").unwrap();
        let (_, fx) = e.promote_to_l0(b"q").unwrap();
        assert_eq!(fx.l0_demoted, vec![b"p".to_vec()]);
        assert_eq!(e.l0_len(), 1);
    }

    #[test]
    fn pin_blocks_demote_like_lock_ref() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 1,
            l1: 8,
            l2: 8,
        });
        e.put_durable(b"p", b"P").unwrap();
        e.promote_to_l0(b"p").unwrap();
        e.pin(b"p");
        e.put_durable(b"q", b"Q").unwrap();
        assert!(e.promote_to_l0(b"q").is_err());
        e.unpin(b"p").unwrap();
        let (_, fx) = e.promote_to_l0(b"q").unwrap();
        assert_eq!(fx.l0_demoted, vec![b"p".to_vec()]);
    }

    #[test]
    fn put_durable_l2_only_l3_via_cap_xor() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 4,
            l1: 8,
            l2: 1,
        });
        let (tx, d0) = e.put_durable(b"x", b"X").unwrap();
        assert_eq!(tx, LocalTier::L2);
        assert!(!e.l3_present(b"x"));
        assert!(d0.is_empty());
        // Cap pressure moves cold x L2→L3 (XOR), not PutEnd dual-write.
        let (ty, demoted) = e.put_durable(b"y", b"Y").unwrap();
        assert_eq!(ty, LocalTier::L2);
        assert_eq!(demoted.l2_demoted_to_l3, vec![b"x".to_vec()]);
        assert!(!e.is_l2_durable(b"x") && e.l3_present(b"x"));
        assert!(e.is_l2_durable(b"y") && !e.l3_present(b"y"));
    }
}
