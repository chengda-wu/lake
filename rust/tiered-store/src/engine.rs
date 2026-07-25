//! 本地分层字节引擎：L0(HBM 站位) / L1(DRAM) / L2(NVMe) / L3(对象存储站位)。

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

/// In-process L0–L3 byte maps + LRU eviction order for capacity pressure.
pub struct LocalTierEngine {
    caps: TierCaps,
    l0: HashMap<Vec<u8>, Vec<u8>>,
    l1: HashMap<Vec<u8>, Vec<u8>>,
    l2: HashMap<Vec<u8>, Vec<u8>>,
    l3: HashMap<Vec<u8>, Vec<u8>>,
    /// Insertion / touch order for L0 victims (front = coldest).
    l0_order: VecDeque<Vec<u8>>,
    l1_order: VecDeque<Vec<u8>>,
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

    pub(crate) fn l0_nbytes(&self, h: &[u8]) -> u64 {
        self.l0.get(h).map(|b| b.len() as u64).unwrap_or(0)
    }

    pub(crate) fn l2_nbytes(&self, h: &[u8]) -> u64 {
        self.l2.get(h).map(|b| b.len() as u64).unwrap_or(0)
    }

    fn touch(order: &mut VecDeque<Vec<u8>>, h: &[u8]) {
        order.retain(|x| x.as_slice() != h);
        order.push_back(h.to_vec());
    }

    /// PutEnd / writeback: bytes become L2 durable (and optionally L3 SSOT copy).
    pub fn put_durable(&mut self, h: &[u8], bytes: &[u8], also_l3: bool) {
        self.l2.insert(h.to_vec(), bytes.to_vec());
        if also_l3 {
            self.l3.insert(h.to_vec(), bytes.to_vec());
        }
        self.ensure_l2_cap();
    }

    fn ensure_l2_cap(&mut self) {
        let cap = self.caps.l2;
        if cap == 0 {
            return;
        }
        // L2 cold demote → L3 only (XOR-ish: keep L3, drop L2 when over cap).
        while self.l2.len() > cap {
            let Some(victim) = self.l2.keys().next().cloned() else {
                break;
            };
            if let Some(bytes) = self.l2.remove(&victim) {
                self.l3.entry(victim).or_insert(bytes);
            }
        }
    }

    /// Lookup with stats; does not mutate tiers.
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

    /// Promote toward L0 (L3→L2→L1→L0 as needed). Returns bytes moved estimate.
    pub fn promote_to_l0(&mut self, h: &[u8]) -> Result<u64, String> {
        let bytes = self
            .get(h)
            .ok_or_else(|| "promote: block missing all tiers".to_string())?
            .to_vec();
        let n = bytes.len() as u64;
        if !self.l2.contains_key(h) && self.l3.contains_key(h) {
            self.l2.insert(h.to_vec(), bytes.clone());
        }
        if !self.l1.contains_key(h) {
            self.ensure_l1_room();
            self.l1.insert(h.to_vec(), bytes.clone());
            Self::touch(&mut self.l1_order, h);
        }
        self.ensure_l0_room();
        self.l0.insert(h.to_vec(), bytes);
        Self::touch(&mut self.l0_order, h);
        Ok(n)
    }

    /// Demote L0 → keep L2 (must be durable). Drops L0 replica only.
    pub fn demote_l0(&mut self, h: &[u8]) -> Result<u64, String> {
        if !self.l2.contains_key(h) {
            return Err("demote_l0: need L2 durable first".into());
        }
        let n = self.l0.remove(h).map(|b| b.len() as u64).unwrap_or(0);
        self.l0_order.retain(|x| x.as_slice() != h);
        Ok(n)
    }

    /// Read-miss fill: pull from deepest available tier up to L0.
    pub fn fill_read_miss(&mut self, h: &[u8]) -> Result<u64, String> {
        if self.l0.contains_key(h) {
            Self::touch(&mut self.l0_order, h);
            return Ok(0);
        }
        self.promote_to_l0(h)
    }

    /// L0 full → demote coldest L0 that is L2-durable. Returns victim hash.
    pub fn evict_l0_pressure(&mut self) -> Result<Option<Vec<u8>>, String> {
        if self.caps.l0 == 0 || self.l0.len() < self.caps.l0 {
            return Ok(None);
        }
        let victim = self
            .l0_order
            .iter()
            .find(|h| self.l2.contains_key(h.as_slice()))
            .cloned()
            .ok_or_else(|| "evict_l0: no durable victim (writeback first)".to_string())?;
        self.demote_l0(&victim)?;
        Ok(Some(victim))
    }

    fn ensure_l0_room(&mut self) {
        let cap = self.caps.l0;
        if cap == 0 {
            return;
        }
        while self.l0.len() >= cap {
            if self.evict_l0_pressure().ok().flatten().is_none() {
                break;
            }
        }
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
            // Drop L1 replica; L2/L3 remain.
            self.l1.remove(&v);
        }
    }

    /// Cold path: move L2 → L3 and drop L2 (L2/L3 steady XOR-ish).
    pub fn demote_l2_to_l3(&mut self, h: &[u8]) -> Result<u64, String> {
        let bytes = self
            .l2
            .remove(h)
            .ok_or_else(|| "demote_l2: not on L2".to_string())?;
        let n = bytes.len() as u64;
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
        e.put_durable(b"a", b"A", true);
        assert!(e.is_l2_durable(b"a"));
        assert!(e.l3_present(b"a"));
        e.promote_to_l0(b"a").unwrap();
        assert_eq!(e.local_tier(b"a"), Some(LocalTier::L0));
        e.demote_l0(b"a").unwrap();
        assert_eq!(e.local_tier(b"a"), Some(LocalTier::L1)); // L1 kept from promote
    }

    #[test]
    fn l0_pressure_evicts_cold() {
        let mut e = LocalTierEngine::with_caps(TierCaps {
            l0: 1,
            l1: 8,
            l2: 8,
        });
        e.put_durable(b"c0", b"0", false);
        e.put_durable(b"c1", b"1", false);
        e.promote_to_l0(b"c0").unwrap();
        e.promote_to_l0(b"c1").unwrap(); // should demote c0
        assert_eq!(e.l0_len(), 1);
        assert!(e.l0.contains_key(b"c1".as_slice()));
        assert!(!e.l0.contains_key(b"c0".as_slice()));
    }

    #[test]
    fn read_miss_from_l3() {
        let mut e = LocalTierEngine::new();
        e.l3.insert(b"z".to_vec(), b"ZZ".to_vec());
        e.fill_read_miss(b"z").unwrap();
        assert_eq!(e.local_tier(b"z"), Some(LocalTier::L0));
        assert!(e.is_l2_durable(b"z"));
    }
}
