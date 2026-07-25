//! Agent → ControlPlane 端口（P4.3）。
//!
//! 进程内 [`AuthorityPort`] 供单测 / 同进程联调；真 tonic 客户端可另实现本 trait。

use lake_controlplane::Authority;
use lake_proto::lake::*;

/// Subset of ControlPlane RPCs used by PutEnd / tier publish.
pub trait ControlPlanePort {
    fn register_blocks(&mut self, req: RegisterBlocksRequest) -> Result<(), String>;
    fn report_refs(&mut self, deltas: &[RefDelta]) -> Result<(), String>;
    fn request_barrier(&mut self, req: RequestBarrierRequest) -> Result<(), String>;
    fn publish_location(
        &mut self,
        model_id: &str,
        flat: &[u8],
        tier: Tier,
        node_id: &str,
        present: bool,
    ) -> Result<(), String>;
    fn set_l3_present(&mut self, model_id: &str, flat: &[u8], present: bool) -> Result<(), String>;
}

/// In-process Authority (no gRPC).
pub struct AuthorityPort<'a> {
    pub auth: &'a mut Authority,
}

impl ControlPlanePort for AuthorityPort<'_> {
    fn register_blocks(&mut self, req: RegisterBlocksRequest) -> Result<(), String> {
        self.auth
            .register(&req.node_id, &req.prefix_hashes, req.blocks)
    }

    fn report_refs(&mut self, deltas: &[RefDelta]) -> Result<(), String> {
        self.auth.report_refs(deltas)
    }

    fn request_barrier(&mut self, req: RequestBarrierRequest) -> Result<(), String> {
        self.auth.complete_barrier(&req.request_id, &req.node_id)
    }

    fn publish_location(
        &mut self,
        model_id: &str,
        flat: &[u8],
        tier: Tier,
        node_id: &str,
        present: bool,
    ) -> Result<(), String> {
        self.auth
            .publish_location(model_id, flat, tier, node_id, present)
    }

    fn set_l3_present(&mut self, model_id: &str, flat: &[u8], present: bool) -> Result<(), String> {
        self.auth.set_l3_present(model_id, flat, present)
    }
}
