//! 存储控制面:位置视图权威(进程内存)。
//!
//! P3:内存 HashMap + 前缀 hash 链匹配(LookupPrefix)。
//! 参考:Mooncake PutEnd 控制面侧;SGLang match_prefix;Dynamo RadixTree::find_match_details。

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

// tokio-stream 供 Lease 空流;SubscribeView 未实现。

pub use lake_proto::lake::*;

use control_plane_service_server::ControlPlaneService;

/// 进程内权威状态。
#[derive(Default)]
pub struct Authority {
    /// block_hash → BlockMeta(LookupPrefix 沿 prefix_hashes 链探测)。
    blocks: HashMap<Vec<u8>, BlockMeta>,
}

impl Authority {
    fn register(&mut self, node_id: &str, metas: Vec<BlockMeta>) {
        for mut meta in metas {
            let Some(id) = meta.id.clone() else { continue };
            let hash = id.block_hash.clone();
            // 保证至少有一个 L2 占位 location(P3 mock durable)。
            if meta.locations.is_empty() {
                meta.locations.push(Location {
                    tier: Tier::L2 as i32,
                    node_id: node_id.to_string(),
                    segment_id: 1,
                    offset: 0,
                });
            }
            self.blocks.insert(hash, meta);
        }
    }

    fn lookup_prefix(
        &self,
        model_id: &str,
        prefix_hashes: &[Vec<u8>],
        requester: &str,
    ) -> (Vec<ReusableBlock>, u32, bool) {
        let mut out = Vec::new();
        let mut hit = 0u32;
        let mut all_local = true;
        for h in prefix_hashes {
            match self.blocks.get(h) {
                Some(meta) if meta.id.as_ref().map(|i| i.model_id.as_str()) == Some(model_id) => {
                    let local = meta.locations.iter().any(|l| {
                        l.tier == Tier::L0 as i32 && l.node_id == requester
                    });
                    if !local {
                        all_local = false;
                    }
                    out.push(ReusableBlock {
                        id: meta.id.clone(),
                        meta: Some(meta.clone()),
                        local_hit: local,
                    });
                    hit += 1;
                }
                _ => {
                    all_local = false;
                    break;
                }
            }
        }
        if hit == 0 {
            all_local = false;
        }
        (out, hit, all_local && hit > 0)
    }
}

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
            "P3:SubscribeView 未实现;Router 冷路径用 LookupPrefix",
        ))
    }

    async fn lookup_prefix(
        &self,
        request: Request<LookupPrefixRequest>,
    ) -> Result<Response<LookupPrefixResponse>, Status> {
        let req = request.into_inner();
        let auth = self.inner.lock().unwrap();
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
        let mut blocks = Vec::new();
        for id in req.ids {
            if let Some(meta) = auth.blocks.get(&id.block_hash) {
                blocks.push(meta.clone());
            }
        }
        Ok(Response::new(LocateResponse { blocks }))
    }

    async fn register_blocks(
        &self,
        request: Request<RegisterBlocksRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut auth = self.inner.lock().unwrap();
        auth.register(&req.node_id, req.blocks);
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
        // 占位空流。
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        let stream: Self::LeaseStream = Box::pin(ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}

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

    #[test]
    fn lookup_prefix_contiguous_then_gap() {
        let mut auth = Authority::default();
        auth.register(
            "n0",
            vec![meta("m", b"h0"), meta("m", b"h1"), meta("m", b"h2")],
        );
        // 中间缺 h1 → 只命中 h0
        let (blocks, hit, local) =
            auth.lookup_prefix("m", &[b"h0".to_vec(), b"gap".to_vec(), b"h2".to_vec()], "n0");
        assert_eq!(hit, 1);
        assert_eq!(blocks.len(), 1);
        assert!(!local); // 仅 L2,无 L0
    }

    #[test]
    fn lookup_prefix_full_hit_not_local_without_l0() {
        let mut auth = Authority::default();
        auth.register("n0", vec![meta("m", b"a"), meta("m", b"b")]);
        let (_, hit, local) =
            auth.lookup_prefix("m", &[b"a".to_vec(), b"b".to_vec()], "n0");
        assert_eq!(hit, 2);
        assert!(!local);
    }
}
// 编译期锚定:引用具体生成符号,防 proto 改名/删字段后 Rust 仍编译通过(Go/Python 已锚定)。
#[allow(dead_code)]
type _CpServer = lake_proto::lake::control_plane_service_server::ControlPlaneServiceServer<()>;
#[allow(dead_code)]
const _ANCHOR: fn() = || {
    let _ = RegisterBlocksRequest::default();
    let _ = LookupPrefixRequest::default();
};
