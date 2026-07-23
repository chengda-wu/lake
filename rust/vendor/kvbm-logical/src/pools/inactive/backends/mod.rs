// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend storage strategies for the inactive index.

use super::*;

mod fifo;
mod hashmap_backend;
mod lineage;
mod lru_backend;
mod multi_lru_backend;
mod reuse_policy;

#[cfg(test)]
mod tests;

pub use fifo::FifoReusePolicy;
pub use hashmap_backend::HashMapBackend;
pub use lineage::{LeafPolicy, LineageBackend};
pub use lru_backend::LruBackend;
pub use multi_lru_backend::MultiLruBackend;
pub use reuse_policy::ReusePolicy;
