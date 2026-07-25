//! Promote / demote 流水线（抄 Dynamo `Pipeline` 骨架，补 lake promote）。
//!
//! 上游 demote-only；lake：按动作消费 [`BandwidthPool`]。
//! **P4.3 缺口**：本 crate 不持 CP 句柄；`tick` 返回 [`LocationEvent`]，
//! 由 agent/`AuthorityPort` 调 `publish_location`（见 storage-agent 接线）。

use crate::bandwidth::BandwidthPool;
use crate::engine::{LocalTier, LocalTierEngine};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineAction {
    /// Hot block → L0.
    Promote { hash: Vec<u8> },
    /// Cold L0 replica drop (needs L2|L3).
    DemoteL0 { hash: Vec<u8> },
    /// Cold L2 → L3, drop L2.
    DemoteL2 { hash: Vec<u8> },
    /// Read miss fill to L0.
    FillMiss { hash: Vec<u8> },
}

/// View-sync hints for controlplane after a successful step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocationEvent {
    Present { hash: Vec<u8>, tier: LocalTier },
    Absent { hash: Vec<u8>, tier: LocalTier },
}

/// Minimal async-less pipeline: drain a queue under bandwidth budget.
pub struct TierPipeline {
    pub engine: LocalTierEngine,
    pub bandwidth: BandwidthPool,
    queue: std::collections::VecDeque<PipelineAction>,
}

impl TierPipeline {
    pub fn new(engine: LocalTierEngine, bandwidth: BandwidthPool) -> Self {
        Self {
            engine,
            bandwidth,
            queue: std::collections::VecDeque::new(),
        }
    }

    pub fn enqueue(&mut self, action: PipelineAction) {
        self.queue.push_back(action);
    }

    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    fn estimate_cost(&self, action: &PipelineAction) -> u64 {
        match action {
            PipelineAction::Promote { hash } | PipelineAction::FillMiss { hash } => {
                if self.engine.local_tier(hash) == Some(LocalTier::L0) {
                    0
                } else {
                    self.engine.get(hash).map(|b| b.len() as u64).unwrap_or(0)
                }
            }
            PipelineAction::DemoteL0 { hash } => self.engine.l0_nbytes(hash),
            PipelineAction::DemoteL2 { hash } => self.engine.l2_nbytes(hash),
        }
    }

    /// Run up to `max_steps` **attempts**. Returns `(successes, location_events)`.
    /// Failed actions are dropped (refund bandwidth) and do **not** count as successes.
    pub fn tick(&mut self, max_steps: usize) -> (usize, Vec<LocationEvent>) {
        let mut done = 0;
        let mut events = Vec::new();
        let mut attempts = 0;
        while attempts < max_steps {
            let Some(action) = self.queue.pop_front() else {
                break;
            };
            attempts += 1;
            let cost = self.estimate_cost(&action);
            if cost > 0 && !self.bandwidth.try_consume(cost) {
                self.queue.push_front(action);
                break;
            }
            let result = match &action {
                PipelineAction::Promote { hash } => self.engine.promote_to_l0(hash),
                PipelineAction::DemoteL0 { hash } => self.engine.demote_l0(hash),
                PipelineAction::DemoteL2 { hash } => self.engine.demote_l2_to_l3(hash),
                PipelineAction::FillMiss { hash } => self.engine.fill_read_miss(hash),
            };
            match (&action, result) {
                (PipelineAction::Promote { hash }, Ok(_)) => {
                    events.push(LocationEvent::Present {
                        hash: hash.clone(),
                        tier: LocalTier::L0,
                    });
                    done += 1;
                }
                (PipelineAction::DemoteL0 { hash }, Ok(_)) => {
                    events.push(LocationEvent::Absent {
                        hash: hash.clone(),
                        tier: LocalTier::L0,
                    });
                    done += 1;
                }
                (PipelineAction::DemoteL2 { hash }, Ok(_)) => {
                    events.push(LocationEvent::Absent {
                        hash: hash.clone(),
                        tier: LocalTier::L2,
                    });
                    events.push(LocationEvent::Present {
                        hash: hash.clone(),
                        tier: LocalTier::L3,
                    });
                    done += 1;
                }
                (PipelineAction::FillMiss { hash }, Ok(n)) => {
                    if n > 0 {
                        events.push(LocationEvent::Present {
                            hash: hash.clone(),
                            tier: LocalTier::L0,
                        });
                    }
                    done += 1;
                }
                (_, Err(_)) => {
                    if cost > 0 {
                        self.bandwidth.refund(cost);
                    }
                    // Drop failed action; do not count as success.
                }
            }
        }
        (done, events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{LocalTier, TierCaps};

    #[test]
    fn pipeline_promote_under_budget() {
        let eng = LocalTierEngine::with_caps(TierCaps::default());
        let mut p = TierPipeline::new(eng, BandwidthPool::new(1024));
        p.engine.put_durable(b"h", b"hello", false).unwrap();
        p.enqueue(PipelineAction::Promote {
            hash: b"h".to_vec(),
        });
        let (n, ev) = p.tick(1);
        assert_eq!(n, 1);
        assert_eq!(p.engine.local_tier(b"h"), Some(LocalTier::L0));
        assert!(ev.iter().any(|e| matches!(
            e,
            LocationEvent::Present {
                tier: LocalTier::L0,
                ..
            }
        )));
    }

    #[test]
    fn pipeline_stops_when_budget_exhausted() {
        let eng = LocalTierEngine::with_caps(TierCaps::default());
        let mut p = TierPipeline::new(eng, BandwidthPool::new(3));
        p.engine.put_durable(b"a", b"abcd", false).unwrap(); // 4 bytes
        p.enqueue(PipelineAction::Promote {
            hash: b"a".to_vec(),
        });
        let (n, _) = p.tick(1);
        assert_eq!(n, 0);
        assert_eq!(p.pending(), 1);
        // Bandwidth gate is before mutate — must not half-apply.
        assert_ne!(p.engine.local_tier(b"a"), Some(LocalTier::L0));
    }

    #[test]
    fn pipeline_err_does_not_count_as_done() {
        let eng = LocalTierEngine::with_caps(TierCaps::default());
        let mut p = TierPipeline::new(eng, BandwidthPool::new(1024));
        // Demote without durable backing → Err, dropped, done=0.
        p.enqueue(PipelineAction::DemoteL0 {
            hash: b"missing".to_vec(),
        });
        let (n, ev) = p.tick(1);
        assert_eq!(n, 0);
        assert!(ev.is_empty());
        assert_eq!(p.pending(), 0);
    }
}
