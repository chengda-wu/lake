//! 存储控制面:位置视图权威(进程内存)。
//!
//! P4.2:Dynamo `BlockRegistry` + `PositionalRadixTree` + `InactiveIndex` 薄驱动。
//! 参考:`registry/mod.rs::register_sequence_hash` / `match_sequence_hash`；
//! `pools/store.rs::InactiveIndex` + `MultiLruBackend`。
//! 关键差异:不用 BlockManager/BlockStore；EventsManager 不接线；
//! presence 与 Authority 同锁 → 进程内线性一致。

mod authority;
mod hash_chain;
mod tier;

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

pub use authority::Authority;
pub use lake_proto::lake::*;

use control_plane_service_server::ControlPlaneService;

#[derive(Clone, Default)]
pub struct ControlPlane {
    inner: Arc<Mutex<Authority>>,
}

#[tonic::async_trait]
impl ControlPlaneService for ControlPlane {
    type SubscribeViewStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<ViewUpdate, Status>> + Send + 'static>>;

    async fn subscribe_view(
        &self,
        _request: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeViewStream>, Status> {
        Err(Status::unimplemented(
            "SubscribeView 未实现;Router 冷路径用 LookupPrefix",
        ))
    }

    async fn lookup_prefix(
        &self,
        request: Request<LookupPrefixRequest>,
    ) -> Result<Response<LookupPrefixResponse>, Status> {
        let req = request.into_inner();
        let mut auth = self.inner.lock().unwrap();
        let (blocks, hit_length, all_local_hit) =
            auth.lookup_prefix(&req.model_id, &req.prefix_hashes, &req.requester_node_id);
        Ok(Response::new(LookupPrefixResponse {
            blocks,
            hit_length,
            all_local_hit,
        }))
    }

    async fn locate(
        &self,
        request: Request<LocateRequest>,
    ) -> Result<Response<LocateResponse>, Status> {
        let req = request.into_inner();
        let auth = self.inner.lock().unwrap();
        let blocks = auth.locate(&req.ids);
        Ok(Response::new(LocateResponse { blocks }))
    }

    async fn register_blocks(
        &self,
        request: Request<RegisterBlocksRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut auth = self.inner.lock().unwrap();
        match auth.register(&req.node_id, &req.prefix_hashes, req.blocks) {
            Ok(()) => Ok(Response::new(Ack {
                ok: true,
                err: String::new(),
            })),
            Err(e) => Ok(Response::new(Ack { ok: false, err: e })),
        }
    }

    async fn report_ref(
        &self,
        request: Request<Streaming<RefDelta>>,
    ) -> Result<Response<Ack>, Status> {
        let mut stream = request.into_inner();
        let mut deltas = Vec::new();
        while let Some(delta) = stream.message().await? {
            deltas.push(delta);
        }
        let mut auth = self.inner.lock().unwrap();
        for delta in &deltas {
            if let Err(e) = auth.report_ref(delta) {
                return Ok(Response::new(Ack { ok: false, err: e }));
            }
        }
        Ok(Response::new(Ack {
            ok: true,
            err: String::new(),
        }))
    }

    async fn request_barrier(
        &self,
        _request: Request<RequestBarrierRequest>,
    ) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {
            ok: true,
            err: String::new(),
        }))
    }

    type LeaseStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<LeaseAck, Status>> + Send + 'static>>;

    async fn lease(
        &self,
        _request: Request<Streaming<LeaseHeartbeat>>,
    ) -> Result<Response<Self::LeaseStream>, Status> {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        let stream: Self::LeaseStream = Box::pin(ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}

#[allow(dead_code)]
type _CpServer = lake_proto::lake::control_plane_service_server::ControlPlaneServiceServer<()>;
#[allow(dead_code)]
const _ANCHOR: fn() = || {
    let _ = RegisterBlocksRequest::default();
    let _ = LookupPrefixRequest::default();
    let _ = RefDelta::default();
};

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(model: &str, hash: &[u8]) -> BlockMeta {
        BlockMeta {
            id: Some(KvBlockId {
                model_id: model.into(),
                block_hash: hash.to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
            }),
            block_kind: BlockKind::TType as i32,
            locations: vec![],
            l3_present: false,
            ref_count: 1,
        }
    }

    fn prefix(hashes: &[&[u8]]) -> Vec<Vec<u8>> {
        hashes.iter().map(|h| h.to_vec()).collect()
    }

    #[test]
    fn lookup_prefix_contiguous_then_gap() {
        let mut auth = Authority::default();
        let full = prefix(&[b"h0", b"h1", b"h2"]);
        auth.register(
            "n0",
            &full,
            vec![meta("m", b"h0"), meta("m", b"h1"), meta("m", b"h2")],
        )
        .unwrap();
        let (blocks, hit, local) = auth.lookup_prefix("m", &prefix(&[b"h0", b"gap", b"h2"]), "n0");
        assert_eq!(hit, 1);
        assert_eq!(blocks.len(), 1);
        assert!(!local);
    }

    #[test]
    fn lookup_prefix_full_hit_not_local_without_l0() {
        let mut auth = Authority::default();
        let full = prefix(&[b"a", b"b"]);
        auth.register("n0", &full, vec![meta("m", b"a"), meta("m", b"b")])
            .unwrap();
        let (_, hit, local) = auth.lookup_prefix("m", &full, "n0");
        assert_eq!(hit, 2);
        assert!(!local);
    }

    #[test]
    fn cross_model_isolation() {
        let mut auth = Authority::default();
        let full = prefix(&[b"shared"]);
        auth.register("n0", &full, vec![meta("m1", b"shared")])
            .unwrap();
        auth.register("n0", &full, vec![meta("m2", b"shared")])
            .unwrap();
        let (_, hit1, _) = auth.lookup_prefix("m1", &full, "n0");
        let (_, hit2, _) = auth.lookup_prefix("m2", &full, "n0");
        assert_eq!(hit1, 1);
        assert_eq!(hit2, 1);
        // miss suffix only for m1 should not affect m2
        let (_, miss, _) = auth.lookup_prefix("m1", &prefix(&[b"other"]), "n0");
        assert_eq!(miss, 0);
    }

    #[test]
    fn register_requires_prefix_hashes() {
        let mut auth = Authority::default();
        let err = auth.register("n0", &[], vec![meta("m", b"a")]).unwrap_err();
        assert!(err.contains("prefix_hashes"));
    }

    #[test]
    fn register_miss_suffix_with_full_chain() {
        let mut auth = Authority::default();
        let full = prefix(&[b"h0", b"h1", b"h2"]);
        auth.register("n0", &full, vec![meta("m", b"h0"), meta("m", b"h1")])
            .unwrap();
        // miss suffix only
        auth.register("n0", &full, vec![meta("m", b"h2")]).unwrap();
        let (_, hit, _) = auth.lookup_prefix("m", &full, "n0");
        assert_eq!(hit, 3);
    }

    #[test]
    fn ref_freeze_and_evict() {
        let mut auth = Authority::default();
        let full = prefix(&[b"x"]);
        auth.register("n0", &full, vec![meta("m", b"x")]).unwrap();

        let d = RefDelta {
            id: Some(KvBlockId {
                model_id: "m".into(),
                block_hash: b"x".to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
            }),
            kind: RefKind::Request as i32,
            delta: 1,
            node_id: "n0".into(),
        };
        auth.report_ref(&d).unwrap();
        assert_eq!(auth.global_ref("m", b"x"), 1);
        assert_eq!(auth.inactive_len("m"), 0);

        // still held → evict should remove 0 from view if we force-insert...
        // first drop ref
        let mut d0 = d.clone();
        d0.delta = -1;
        auth.report_ref(&d0).unwrap();
        assert_eq!(auth.global_ref("m", b"x"), 0);
        assert_eq!(auth.inactive_len("m"), 1);

        // ref>0 again removes from inactive
        auth.report_ref(&d).unwrap();
        assert_eq!(auth.inactive_len("m"), 0);
        auth.report_ref(&d0).unwrap();
        assert_eq!(auth.inactive_len("m"), 1);

        let n = auth.evict_n("m", 1);
        assert_eq!(n, 1);
        let (_, hit, _) = auth.lookup_prefix("m", &full, "n0");
        assert_eq!(hit, 0);
    }

    #[test]
    fn ref_gt_zero_not_evicted() {
        let mut auth = Authority::default();
        let full = prefix(&[b"y"]);
        auth.register("n0", &full, vec![meta("m", b"y")]).unwrap();
        let d = RefDelta {
            id: Some(KvBlockId {
                model_id: "m".into(),
                block_hash: b"y".to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
            }),
            kind: RefKind::Request as i32,
            delta: 1,
            node_id: "n0".into(),
        };
        auth.report_ref(&d).unwrap();
        // not in inactive while held
        assert_eq!(auth.evict_n("m", 10), 0);
        let (_, hit, _) = auth.lookup_prefix("m", &full, "n0");
        assert_eq!(hit, 1);
    }

    #[test]
    fn multi_lru_evicts_colder_first() {
        let mut auth = Authority::default();
        let cold = prefix(&[b"cold"]);
        let hot = prefix(&[b"hot"]);
        auth.register("n0", &cold, vec![meta("m", b"cold")])
            .unwrap();
        auth.register("n0", &hot, vec![meta("m", b"hot")]).unwrap();
        // bump TinyLFU for hot via LookupPrefix → match_sequence_hash(touch)
        for _ in 0..64 {
            let _ = auth.lookup_prefix("m", &hot, "n0");
        }
        for flat in [b"cold".as_slice(), b"hot".as_slice()] {
            let d = RefDelta {
                id: Some(KvBlockId {
                    model_id: "m".into(),
                    block_hash: flat.to_vec(),
                    pool_kind: PoolKind::Target as i32,
                    scope: "public".into(),
                }),
                kind: RefKind::Request as i32,
                delta: 1,
                node_id: "n0".into(),
            };
            auth.report_ref(&d).unwrap();
            let mut d0 = d;
            d0.delta = -1;
            auth.report_ref(&d0).unwrap();
        }
        assert_eq!(auth.inactive_len("m"), 2);
        assert_eq!(auth.evict_n("m", 1), 1);
        let (_, cold_hit, _) = auth.lookup_prefix("m", &cold, "n0");
        let (_, hot_hit, _) = auth.lookup_prefix("m", &hot, "n0");
        assert_eq!(cold_hit, 0, "colder block should be MultiLru victim");
        assert_eq!(
            hot_hit, 1,
            "touched hot block should survive first allocate"
        );
    }

    #[test]
    fn lineage_backend_evicts_leaf_before_parent() {
        use crate::hash_chain::lineage_from_prefix;
        use kvbm_logical::{InactiveIndex, LineageBackend};

        let mut idx = LineageBackend::new();
        let chain = lineage_from_prefix(&[b"parent".to_vec(), b"child".to_vec()]);
        idx.insert(chain[0], 10);
        idx.insert(chain[1], 11);
        let victims = idx.allocate(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(
            victims[0].0, chain[1],
            "LineageBackend must protect prefix parent until leaf gone"
        );
    }
}
