// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend storage strategies for the inactive index.
//!
//! lake P4.2: only export backends that do not leak `pub(crate)` types
//! (`InactiveBlock` / `FifoPolicy` / `TickPolicy`) into the public API.
//! Fifo/HashMap/ReusePolicy/LeafPolicy stay `pub(crate)`.

use super::*;

mod fifo;
mod hashmap_backend;
mod lineage;
mod lru_backend;
mod multi_lru_backend;
mod reuse_policy;

#[cfg(test)]
mod tests;

pub(crate) use fifo::FifoReusePolicy;
pub(crate) use hashmap_backend::HashMapBackend;
pub(crate) use lineage::LeafPolicy;
pub use lineage::LineageBackend;
pub use lru_backend::LruBackend;
pub use multi_lru_backend::MultiLruBackend;
pub(crate) use reuse_policy::ReusePolicy; // hashmap_backend: `use super::ReusePolicy`
