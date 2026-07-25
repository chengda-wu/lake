//! L0–L3 分层字节存储（P4.3 最小可用）。
//!
//! 参考：Dynamo kvbm-engine `OffloadPolicy` / `Pipeline`（demote-only）；
//! SGLang `hiradix_cache.py::_evict_write_back`（驱逐前写回）。
//! 关键差异：lake 无 `BlockStore` 固定槽；位置视图权威在 controlplane；
//! 本 crate 只持**不透明字节**，promote/demote 更新由 agent/CP 协作。

use std::collections::HashMap;

pub use lake_proto::lake::*;

/// Content-addressed durable bytes at L2 (NVMe stand-in).
///
/// P4.3 MVP: in-memory. Real NVMe / object (L3) later.
#[derive(Default)]
pub struct MemoryL2 {
    blocks: HashMap<Vec<u8>, Vec<u8>>,
}

impl MemoryL2 {
    pub fn new() -> Self {
        Self::default()
    }

    /// PutEnd 本地阶段：写满块字节，返回是否为新写入。
    pub fn put_durable(&mut self, block_hash: &[u8], bytes: &[u8]) -> bool {
        self.blocks
            .insert(block_hash.to_vec(), bytes.to_vec())
            .is_none()
    }

    pub fn get(&self, block_hash: &[u8]) -> Option<&[u8]> {
        self.blocks.get(block_hash).map(|v| v.as_slice())
    }

    pub fn is_durable(&self, block_hash: &[u8]) -> bool {
        self.blocks.contains_key(block_hash)
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Soft tier view for a block on this node (HBM cache vs L2 durable).
/// L3 is SSOT elsewhere (`l3_present` on controlplane meta).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTier {
    /// Hot replica in HBM (L0).
    L0,
    /// Durable on this node's L2 stand-in.
    L2,
}

/// Minimal promote / demote bookkeeping over [`MemoryL2`].
///
/// Does **not** own location-view authority — agent publishes via
/// `RegisterBlocks` / future `Publish`. This only moves local bytes/flags.
#[derive(Default)]
pub struct LocalTierEngine {
    l2: MemoryL2,
    /// Blocks currently cached as L0 replicas (hash → true).
    l0: HashMap<Vec<u8>, ()>,
}

impl LocalTierEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn l2(&self) -> &MemoryL2 {
        &self.l2
    }

    pub fn l2_mut(&mut self) -> &mut MemoryL2 {
        &mut self.l2
    }

    /// Demote: drop L0 replica; bytes must already be L2-durable.
    pub fn demote_l0(&mut self, block_hash: &[u8]) -> Result<(), String> {
        if !self.l2.is_durable(block_hash) {
            return Err("demote_l0: L2 not durable (need writeback first)".into());
        }
        self.l0.remove(block_hash);
        Ok(())
    }

    /// Promote: copy L2 → mark L0 present (bytes stay in L2 map; L0 is a flag).
    pub fn promote_l0(&mut self, block_hash: &[u8]) -> Result<(), String> {
        if !self.l2.is_durable(block_hash) {
            return Err("promote_l0: miss L2 (read-miss fill first)".into());
        }
        self.l0.insert(block_hash.to_vec(), ());
        Ok(())
    }

    /// Read-miss fill: ensure durable then promote.
    pub fn fill_from_l2(&mut self, block_hash: &[u8]) -> Result<(), String> {
        self.promote_l0(block_hash)
    }

    pub fn local_tier(&self, block_hash: &[u8]) -> Option<LocalTier> {
        if self.l0.contains_key(block_hash) {
            Some(LocalTier::L0)
        } else if self.l2.is_durable(block_hash) {
            Some(LocalTier::L2)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_durable_then_get() {
        let mut s = MemoryL2::new();
        assert!(s.put_durable(b"h0", b"KV:0"));
        assert!(!s.put_durable(b"h0", b"KV:0")); // overwrite, not new
        assert_eq!(s.get(b"h0"), Some(b"KV:0".as_slice()));
    }

    #[test]
    fn demote_requires_durable() {
        let mut e = LocalTierEngine::new();
        e.l0.insert(b"x".to_vec(), ());
        assert!(e.demote_l0(b"x").is_err());
        e.l2.put_durable(b"x", b"bytes");
        assert!(e.demote_l0(b"x").is_ok());
        assert_eq!(e.local_tier(b"x"), Some(LocalTier::L2));
    }

    #[test]
    fn promote_from_l2() {
        let mut e = LocalTierEngine::new();
        e.l2.put_durable(b"y", b"yy");
        e.promote_l0(b"y").unwrap();
        assert_eq!(e.local_tier(b"y"), Some(LocalTier::L0));
        e.demote_l0(b"y").unwrap();
        assert_eq!(e.local_tier(b"y"), Some(LocalTier::L2));
    }
}
