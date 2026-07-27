//! 存储控制面:位置视图权威(进程内存)。
//!
//! P4.2:Dynamo `BlockRegistry` + `PositionalRadixTree` + `InactiveIndex` 薄驱动。
//! P4.5:`RegisterModel` / `DeregisterModel`——每 `(model_id, revision)` 一命名空间。
//! 参考:`registry/mod.rs::register_sequence_hash` / `match_sequence_hash`；
//! `InactiveIndex` + `LineageBackend::with_frequency`（叶子 + TinyLFU）。
//! 关键差异:不用 BlockManager/BlockStore；EventsManager 不接线；
//! presence 与 Authority 同锁 → 进程内线性一致；命名空间显式注册(非懒建)。

mod authority;
mod hash_chain;
mod tier;

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

pub use authority::{Authority, NamespaceKey};
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
        let (blocks, hit_length, all_local_hit) = auth.lookup_prefix(
            &req.model_id,
            &req.revision,
            &req.prefix_hashes,
            &req.requester_node_id,
        );
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
        match auth.report_refs(&deltas) {
            Ok(()) => Ok(Response::new(Ack {
                ok: true,
                err: String::new(),
            })),
            Err(e) => Ok(Response::new(Ack { ok: false, err: e })),
        }
    }

    async fn request_barrier(
        &self,
        request: Request<RequestBarrierRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut auth = self.inner.lock().unwrap();
        // P4.3: agent must flush L2 + ReportRef(WRITEBACK,-1) before this call.
        match auth.complete_barrier(&req.request_id, &req.node_id) {
            Ok(()) => Ok(Response::new(Ack {
                ok: true,
                err: String::new(),
            })),
            Err(e) => Ok(Response::new(Ack { ok: false, err: e })),
        }
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

    async fn register_model(
        &self,
        request: Request<RegisterModelRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let Some(model) = req.model else {
            return Ok(Response::new(Ack {
                ok: false,
                err: "RegisterModel: model required".into(),
            }));
        };
        let mut auth = self.inner.lock().unwrap();
        match auth.register_model(model) {
            Ok(()) => Ok(Response::new(Ack {
                ok: true,
                err: String::new(),
            })),
            Err(e) => Ok(Response::new(Ack { ok: false, err: e })),
        }
    }

    async fn deregister_model(
        &self,
        request: Request<DeregisterModelRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut auth = self.inner.lock().unwrap();
        match auth.deregister_model(&req.model_id, &req.revision) {
            Ok(()) => Ok(Response::new(Ack {
                ok: true,
                err: String::new(),
            })),
            Err(e) => Ok(Response::new(Ack { ok: false, err: e })),
        }
    }
}

#[allow(dead_code)]
type _CpServer = lake_proto::lake::control_plane_service_server::ControlPlaneServiceServer<()>;
#[allow(dead_code)]
const _ANCHOR: fn() = || {
    let _ = RegisterBlocksRequest::default();
    let _ = LookupPrefixRequest::default();
    let _ = RegisterModelRequest::default();
    let _ = DeregisterModelRequest::default();
    let _ = RefDelta::default();
};

#[cfg(test)]
mod tests {
    use super::*;

    fn ensure_model(auth: &mut Authority, model: &str) {
        ensure_model_rev(auth, model, "");
    }

    fn ensure_model_rev(auth: &mut Authority, model: &str, revision: &str) {
        auth.register_model(ModelDescriptor {
            model_id: model.into(),
            revision: revision.into(),
            num_layers: 32,
            block_spec: Some(BlockSpec {
                block_tokens: 128,
                bytes_per_block: 0,
            }),
            hash_algo: HashAlgo::HashSha256256 as i32,
            quota: Some(Quota {
                soft_bytes: 0,
                hard_bytes: 0,
                borrow_enabled: false,
            }),
        })
        .unwrap();
    }

    fn meta(model: &str, hash: &[u8]) -> BlockMeta {
        meta_rev(model, "", hash)
    }

    fn meta_rev(model: &str, revision: &str, hash: &[u8]) -> BlockMeta {
        BlockMeta {
            id: Some(KvBlockId {
                model_id: model.into(),
                block_hash: hash.to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
                revision: revision.into(),
            }),
            block_kind: BlockKind::TType as i32,
            locations: vec![Location {
                tier: Tier::L2 as i32,
                node_id: "n0".into(),
                segment_id: 1,
                offset: 0,
            }],
            l3_present: false,
            ref_count: 1,
        }
    }

    fn prefix(hashes: &[&[u8]]) -> Vec<Vec<u8>> {
        hashes.iter().map(|h| h.to_vec()).collect()
    }

    fn delta(model: &str, flat: &[u8], d: i32) -> RefDelta {
        delta_rev(model, "", flat, d)
    }

    fn delta_rev(model: &str, revision: &str, flat: &[u8], d: i32) -> RefDelta {
        RefDelta {
            id: Some(KvBlockId {
                model_id: model.into(),
                block_hash: flat.to_vec(),
                pool_kind: PoolKind::Target as i32,
                scope: "public".into(),
                revision: revision.into(),
            }),
            kind: RefKind::Request as i32,
            delta: d,
            node_id: "n0".into(),
        }
    }

    #[test]
    fn lookup_prefix_contiguous_then_gap() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"h0", b"h1", b"h2"]);
        auth.register(
            "n0",
            &full,
            vec![meta("m", b"h0"), meta("m", b"h1"), meta("m", b"h2")],
        )
        .unwrap();
        let (blocks, hit, local) =
            auth.lookup_prefix("m", "", &prefix(&[b"h0", b"gap", b"h2"]), "n0");
        assert_eq!(hit, 1);
        assert_eq!(blocks.len(), 1);
        assert!(!local);
    }

    #[test]
    fn lookup_prefix_full_hit_not_local_without_l0() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"a", b"b"]);
        auth.register("n0", &full, vec![meta("m", b"a"), meta("m", b"b")])
            .unwrap();
        let (_, hit, local) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 2);
        assert!(!local);
    }

    #[test]
    fn cross_model_isolation() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m1");
        ensure_model(&mut auth, "m2");
        let full = prefix(&[b"shared"]);
        auth.register("n0", &full, vec![meta("m1", b"shared")])
            .unwrap();
        auth.register("n0", &full, vec![meta("m2", b"shared")])
            .unwrap();
        let (_, hit1, _) = auth.lookup_prefix("m1", "", &full, "n0");
        let (_, hit2, _) = auth.lookup_prefix("m2", "", &full, "n0");
        assert_eq!(hit1, 1);
        assert_eq!(hit2, 1);
        let (_, miss, _) = auth.lookup_prefix("m1", "", &prefix(&[b"other"]), "n0");
        assert_eq!(miss, 0);
    }

    #[test]
    fn register_requires_prefix_hashes() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let err = auth.register("n0", &[], vec![meta("m", b"a")]).unwrap_err();
        assert!(err.contains("prefix_hashes"));
    }

    #[test]
    fn register_requires_registered_model() {
        let mut auth = Authority::default();
        let full = prefix(&[b"a"]);
        let err = auth
            .register("n0", &full, vec![meta("m", b"a")])
            .unwrap_err();
        assert!(err.contains("not registered"));
    }

    #[test]
    fn register_miss_suffix_with_full_chain() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"h0", b"h1", b"h2"]);
        auth.register("n0", &full, vec![meta("m", b"h0"), meta("m", b"h1")])
            .unwrap();
        auth.register("n0", &full, vec![meta("m", b"h2")]).unwrap();
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 3);
    }

    #[test]
    fn register_rejects_non_contiguous_subset() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"h0", b"h1", b"h2"]);
        let err = auth
            .register("n0", &full, vec![meta("m", b"h0"), meta("m", b"h2")])
            .unwrap_err();
        assert!(err.contains("contiguous"));
    }

    #[test]
    fn lookup_stops_on_lineage_mismatch() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let chain = prefix(&[b"A", b"B"]);
        auth.register("n0", &chain, vec![meta("m", b"A"), meta("m", b"B")])
            .unwrap();
        let (blocks, hit, _) = auth.lookup_prefix("m", "", &prefix(&[b"B"]), "n0");
        assert_eq!(hit, 0);
        assert!(blocks.is_empty());
    }

    #[test]
    fn ref_freeze_and_evict() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"x"]);
        auth.register("n0", &full, vec![meta("m", b"x")]).unwrap();

        let d = delta("m", b"x", 1);
        auth.report_ref(&d).unwrap();
        assert_eq!(auth.global_ref("m", "", b"x"), 1);
        assert_eq!(auth.inactive_len("m", ""), 0);

        let mut d0 = d.clone();
        d0.delta = -1;
        auth.report_ref(&d0).unwrap();
        assert_eq!(auth.global_ref("m", "", b"x"), 0);
        assert_eq!(auth.inactive_len("m", ""), 1);

        auth.report_ref(&d).unwrap();
        assert_eq!(auth.inactive_len("m", ""), 0);
        auth.report_ref(&d0).unwrap();
        assert_eq!(auth.inactive_len("m", ""), 1);

        let n = auth.evict_n("m", "", 1);
        assert_eq!(n, 1);
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 0);
    }

    #[test]
    fn report_ref_batch_all_or_nothing() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"p"]);
        auth.register("n0", &full, vec![meta("m", b"p")]).unwrap();
        let plus = delta("m", b"p", 1);
        auth.report_ref(&plus).unwrap();
        assert_eq!(auth.global_ref("m", "", b"p"), 1);

        let mut minus = plus.clone();
        minus.delta = -1;
        let unknown = delta("m", b"missing", -1);
        let err = auth.report_refs(&[minus, unknown]).unwrap_err();
        assert!(err.contains("unknown block_hash") || err.contains("batch"));
        assert_eq!(
            auth.global_ref("m", "", b"p"),
            1,
            "failed batch must not apply prefix deltas"
        );
    }

    #[test]
    fn ref_gt_zero_not_evicted() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"y"]);
        auth.register("n0", &full, vec![meta("m", b"y")]).unwrap();
        auth.report_ref(&delta("m", b"y", 1)).unwrap();
        assert_eq!(auth.evict_n("m", "", 10), 0);
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 1);
    }

    #[test]
    fn frequency_evicts_colder_leaf_first() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let cold = prefix(&[b"cold"]);
        let hot = prefix(&[b"hot"]);
        auth.register("n0", &cold, vec![meta("m", b"cold")])
            .unwrap();
        auth.register("n0", &hot, vec![meta("m", b"hot")]).unwrap();
        for _ in 0..64 {
            let _ = auth.lookup_prefix("m", "", &hot, "n0");
        }
        for flat in [b"cold".as_slice(), b"hot".as_slice()] {
            auth.report_ref(&delta("m", flat, 1)).unwrap();
            auth.report_ref(&delta("m", flat, -1)).unwrap();
        }
        assert_eq!(auth.inactive_len("m", ""), 2);
        assert_eq!(auth.evict_n("m", "", 1), 1);
        let (_, cold_hit, _) = auth.lookup_prefix("m", "", &cold, "n0");
        let (_, hot_hit, _) = auth.lookup_prefix("m", "", &hot, "n0");
        assert_eq!(cold_hit, 0, "colder leaf should be Frequency victim");
        assert_eq!(hot_hit, 1, "touched hot leaf should survive first allocate");
    }

    #[test]
    fn authority_evicts_leaf_before_prefix_parent() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let chain = prefix(&[b"parent", b"child"]);
        auth.register(
            "n0",
            &chain,
            vec![meta("m", b"parent"), meta("m", b"child")],
        )
        .unwrap();
        for flat in [b"parent".as_slice(), b"child".as_slice()] {
            auth.report_ref(&delta("m", flat, 1)).unwrap();
            auth.report_ref(&delta("m", flat, -1)).unwrap();
        }
        assert_eq!(auth.evict_n("m", "", 1), 1);
        let (_, parent_only, _) = auth.lookup_prefix("m", "", &prefix(&[b"parent"]), "n0");
        assert_eq!(parent_only, 1, "prefix parent must survive first evict");
        let (_, chain_hit, _) = auth.lookup_prefix("m", "", &chain, "n0");
        assert_eq!(chain_hit, 1, "leaf gone → gap after parent");
    }

    #[test]
    fn inactive_cap_skip_insert_no_zombie() {
        let cap = 4;
        let mut auth = Authority::with_inactive_cap(cap);
        ensure_model(&mut auth, "m");
        let n = cap * 2;
        let mut flats: Vec<Vec<u8>> = Vec::with_capacity(n);
        for i in 0..n {
            let flat = format!("leaf{i:02}").into_bytes();
            let full = prefix(&[flat.as_slice()]);
            auth.register("n0", &full, vec![meta("m", &flat)]).unwrap();
            auth.report_ref(&delta("m", &flat, 1)).unwrap();
            auth.report_ref(&delta("m", &flat, -1)).unwrap();
            flats.push(flat);
            assert!(
                auth.inactive_len("m", "") <= cap,
                "inactive must stay ≤ cap after insert #{i}"
            );
        }
        assert_eq!(auth.inactive_len("m", ""), cap);
        let (_, early_hit, _) = auth.lookup_prefix("m", "", &prefix(&[flats[0].as_slice()]), "n0");
        assert_eq!(early_hit, 1, "skip-insert must not drop view entries");
        let removed = auth.evict_n("m", "", cap);
        assert_eq!(removed, cap, "allocate must clear all inactive at cap");
        assert_eq!(auth.inactive_len("m", ""), 0);
    }

    #[test]
    fn report_refs_mid_batch_must_not_panic_or_drop_peer() {
        let mut auth = Authority::with_inactive_cap(1);
        ensure_model(&mut auth, "m");
        let held = prefix(&[b"held"]);
        let cand = prefix(&[b"cand"]);
        auth.register("n0", &held, vec![meta("m", b"held")])
            .unwrap();
        auth.register("n0", &cand, vec![meta("m", b"cand")])
            .unwrap();

        auth.report_ref(&delta("m", b"cand", 1)).unwrap();
        auth.report_ref(&delta("m", b"cand", -1)).unwrap();
        auth.report_ref(&delta("m", b"held", 1)).unwrap();
        assert_eq!(auth.inactive_len("m", ""), 1);
        assert_eq!(auth.global_ref("m", "", b"held"), 1);
        assert_eq!(auth.global_ref("m", "", b"cand"), 0);

        auth.report_refs(&[delta("m", b"held", -1), delta("m", b"cand", 1)])
            .unwrap();
        assert_eq!(auth.global_ref("m", "", b"held"), 0);
        assert_eq!(auth.global_ref("m", "", b"cand"), 1);
        assert_eq!(auth.inactive_len("m", ""), 0);
        let (_, cand_hit, _) = auth.lookup_prefix("m", "", &cand, "n0");
        assert_eq!(cand_hit, 1, "peer must not be pressure-evicted mid-batch");
    }

    #[test]
    fn writeback_ref_blocks_evict_until_cleared() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"wb0"]);
        auth.register("n0", &full, vec![meta("m", b"wb0")]).unwrap();
        let mut plus = delta("m", b"wb0", 1);
        plus.kind = RefKind::Writeback as i32;
        auth.report_ref(&plus).unwrap();
        assert_eq!(auth.global_ref("m", "", b"wb0"), 1);
        assert_eq!(auth.evict_n("m", "", 10), 0, "writeback must freeze");

        let mut minus = plus.clone();
        minus.delta = -1;
        auth.report_ref(&minus).unwrap();
        auth.complete_barrier("req-wb", "n0").unwrap();
        assert!(auth.barrier_completed("req-wb"));
        assert_eq!(auth.evict_n("m", "", 1), 1);
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 0);
    }

    #[test]
    fn request_barrier_requires_ids() {
        let mut auth = Authority::default();
        assert!(auth.complete_barrier("", "n0").is_err());
        assert!(auth.complete_barrier("r", "").is_err());
    }

    #[test]
    fn register_rejects_invented_l2() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"bare"]);
        let mut m = meta("m", b"bare");
        m.locations.clear();
        assert!(auth.register("n0", &full, vec![m]).is_err());
    }

    #[test]
    fn publish_l0_enables_local_hit() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"loc"]);
        auth.register("n0", &full, vec![meta("m", b"loc")]).unwrap();
        auth.publish_location("m", "", b"loc", Tier::L0, "n0", true)
            .unwrap();
        assert!(auth.has_l0_on("m", "", b"loc", "n0"));
        let (_, _, all_local) = auth.lookup_prefix("m", "", &full, "n0");
        assert!(all_local);
        auth.publish_location("m", "", b"loc", Tier::L0, "n0", false)
            .unwrap();
        let (_, _, all_local) = auth.lookup_prefix("m", "", &full, "n0");
        assert!(!all_local);
    }

    /// P4.5: two models, same prefix hash → no cross-namespace hit.
    #[test]
    fn p45_two_models_same_hash_no_crosstalk() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "llama");
        ensure_model(&mut auth, "qwen");
        let full = prefix(&[b"samehash"]);
        auth.register("n0", &full, vec![meta("llama", b"samehash")])
            .unwrap();
        // qwen has no blocks yet
        let (_, hit_q, _) = auth.lookup_prefix("qwen", "", &full, "n0");
        assert_eq!(hit_q, 0);
        auth.register("n0", &full, vec![meta("qwen", b"samehash")])
            .unwrap();
        let (_, hit_l, _) = auth.lookup_prefix("llama", "", &full, "n0");
        let (_, hit_q, _) = auth.lookup_prefix("qwen", "", &full, "n0");
        assert_eq!(hit_l, 1);
        assert_eq!(hit_q, 1);
        auth.deregister_model("llama", "").unwrap();
        let (_, hit_l, _) = auth.lookup_prefix("llama", "", &full, "n0");
        let (_, hit_q, _) = auth.lookup_prefix("qwen", "", &full, "n0");
        assert_eq!(hit_l, 0, "llama cascaded away");
        assert_eq!(hit_q, 1, "qwen untouched");
    }

    /// P4.5: DeregisterModel clears radix/locations for that namespace.
    #[test]
    fn p45_deregister_cascades_view() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"a", b"b"]);
        auth.register("n0", &full, vec![meta("m", b"a"), meta("m", b"b")])
            .unwrap();
        assert_eq!(auth.block_count("m", ""), 2);
        auth.deregister_model("m", "").unwrap();
        assert!(!auth.has_namespace("m", ""));
        assert_eq!(auth.block_count("m", ""), 0);
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 0);
        // re-register model + blocks works on a fresh namespace
        ensure_model(&mut auth, "m");
        auth.register("n0", &full, vec![meta("m", b"a")]).unwrap();
        let (_, hit, _) = auth.lookup_prefix("m", "", &prefix(&[b"a"]), "n0");
        assert_eq!(hit, 1);
    }

    /// P4.5: new revision = new namespace; old revision remains until deregistered.
    #[test]
    fn p45_revision_isolation_and_invalidate() {
        let mut auth = Authority::default();
        ensure_model_rev(&mut auth, "m", "r1");
        ensure_model_rev(&mut auth, "m", "r2");
        let full = prefix(&[b"pfx"]);
        auth.register("n0", &full, vec![meta_rev("m", "r1", b"pfx")])
            .unwrap();
        auth.register("n0", &full, vec![meta_rev("m", "r2", b"pfx")])
            .unwrap();
        let (_, h1, _) = auth.lookup_prefix("m", "r1", &full, "n0");
        let (_, h2, _) = auth.lookup_prefix("m", "r2", &full, "n0");
        assert_eq!(h1, 1);
        assert_eq!(h2, 1);
        // miss across revision
        let (_, h_cross, _) = auth.lookup_prefix("m", "r1", &full, "n0");
        assert_eq!(h_cross, 1);
        auth.deregister_model("m", "r1").unwrap();
        let (_, h1, _) = auth.lookup_prefix("m", "r1", &full, "n0");
        let (_, h2, _) = auth.lookup_prefix("m", "r2", &full, "n0");
        assert_eq!(h1, 0, "r1 invalidated");
        assert_eq!(h2, 1, "r2 still live");
        assert!(auth.has_namespace("m", "r2"));
        assert!(!auth.has_namespace("m", "r1"));
    }

    #[test]
    fn register_model_idempotent_keeps_blocks() {
        let mut auth = Authority::default();
        ensure_model(&mut auth, "m");
        let full = prefix(&[b"keep"]);
        auth.register("n0", &full, vec![meta("m", b"keep")])
            .unwrap();
        ensure_model(&mut auth, "m"); // re-register
        let (_, hit, _) = auth.lookup_prefix("m", "", &full, "n0");
        assert_eq!(hit, 1);
        assert_eq!(
            auth.model_descriptor("m", "")
                .map(|d| d.num_layers)
                .unwrap(),
            32
        );
    }
}
