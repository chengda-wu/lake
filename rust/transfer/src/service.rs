//! `TransferService` gRPC → `TcpTransport` + 内容寻址字节库。
//!
//! 挂在 storage-agent(边7/8,非控制面)。Pull 仿 SGLang prefetch 三策略;
//! Publish 仿 `on_publish` layer-wise + seq fence。

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use lake_proto::lake::transfer_service_server::TransferService;
use lake_proto::lake::*;

use crate::bytes::InMemoryByteStore;
use crate::error::TransferError;
use crate::location::validate_location_tier;
use crate::tcp::TcpTransport;
use crate::transport::{TaskState, TransferOp, Transport};

/// Pull 会话(handle → 结果);后续 GetPullStatus / cancel 会读这些字段。
#[allow(dead_code)]
struct PullHandle {
    pulled_length: u32,
    completed: bool,
    /// Pull 落入的 requester L0 段。
    dest_segment: u64,
}

struct PublishState {
    last_seq: u32,
    /// (block_hash, layer_idx) → 累计长度(layer-wise 增量)。
    layers: HashMap<(Vec<u8>, u32), u64>,
}

/// TransferService 实现。
pub struct TransferServer {
    transport: Arc<TcpTransport>,
    bytes: Arc<InMemoryByteStore>,
    pulls: Mutex<HashMap<u64, PullHandle>>,
    next_pull: AtomicU64,
    publish: Mutex<PublishState>,
}

impl TransferServer {
    pub fn new(transport: Arc<TcpTransport>, bytes: Arc<InMemoryByteStore>) -> Self {
        Self {
            transport,
            bytes,
            pulls: Mutex::new(HashMap::new()),
            next_pull: AtomicU64::new(1),
            publish: Mutex::new(PublishState {
                last_seq: 0,
                layers: HashMap::new(),
            }),
        }
    }

    pub fn transport(&self) -> &Arc<TcpTransport> {
        &self.transport
    }

    pub fn bytes(&self) -> &Arc<InMemoryByteStore> {
        &self.bytes
    }

    fn task_state_to_proto(s: TaskState) -> i32 {
        match s {
            TaskState::Pending => transfer_status_response::State::Pending as i32,
            TaskState::InFlight => transfer_status_response::State::InFlight as i32,
            TaskState::Done => transfer_status_response::State::Done as i32,
            TaskState::Failed => transfer_status_response::State::Failed as i32,
        }
    }
}

#[tonic::async_trait]
impl TransferService for TransferServer {
    async fn submit_transfer(
        &self,
        request: Request<TransferBatchRequest>,
    ) -> Result<Response<TransferBatchAck>, Status> {
        let req = request.into_inner();
        if req.reqs.is_empty() {
            return Err(TransferError::EmptyBatch.into());
        }
        let mut ops = Vec::with_capacity(req.reqs.len());
        for r in &req.reqs {
            let source = r
                .source
                .clone()
                .ok_or_else(|| Status::invalid_argument("TransferRequest.source required"))?;
            validate_location_tier(&source).map_err(Status::from)?;
            ops.push(TransferOp {
                source,
                target_segment_id: r.target_segment_id,
                target_offset: r.target_offset,
                length: r.length,
            });
        }
        let batch_id = self
            .transport
            .allocate_batch_id(ops.len())
            .map_err(Status::from)?;
        match self.transport.submit_transfer(batch_id, &ops) {
            Ok(()) => Ok(Response::new(TransferBatchAck { batch_id })),
            Err(e) => {
                let _ = self.transport.free_batch_id(batch_id);
                Err(e.into())
            }
        }
    }

    async fn get_transfer_status(
        &self,
        request: Request<TransferStatusRequest>,
    ) -> Result<Response<TransferStatusResponse>, Status> {
        let req = request.into_inner();
        let st = self
            .transport
            .get_transfer_status(req.batch_id, req.task_id)
            .map_err(Status::from)?;
        Ok(Response::new(TransferStatusResponse {
            state: Self::task_state_to_proto(st.state),
            bytes_done: st.bytes_done,
        }))
    }

    async fn pull(&self, request: Request<PullRequest>) -> Result<Response<PullResponse>, Status> {
        let req = request.into_inner();
        let policy = PullPolicy::try_from(req.policy).unwrap_or(PullPolicy::PullBestEffort);

        // TIMEOUT:budget_ms 截断可拉数量(粗粒度站位;P7 校准 base+per_ki_token)。
        let budget_cap = match policy {
            PullPolicy::PullTimeout if req.budget_ms > 0 => req.budget_ms as usize,
            _ => usize::MAX,
        };

        let mut payloads: Vec<(KvBlockId, Vec<u8>)> = Vec::new();
        for id in &req.ids {
            if payloads.len() >= budget_cap {
                break;
            }
            match self.bytes.get(id) {
                Some(data) => payloads.push((id.clone(), data)),
                None if policy == PullPolicy::PullWaitComplete => {
                    return Err(Status::not_found(format!(
                        "Pull WAIT_COMPLETE missing block hash_len={}",
                        id.block_hash.len()
                    )));
                }
                None => {}
            }
        }

        let total_bytes: usize = payloads.iter().map(|(_, d)| d.len()).sum();
        let dest = self
            .transport
            .open_segment(
                &format!("pull-{}", req.requester_node_id),
                total_bytes.max(1),
            )
            .map_err(Status::from)?;
        let mut offset = 0u64;
        for (_id, data) in &payloads {
            self.transport
                .write_segment(dest, offset, data)
                .map_err(Status::from)?;
            offset += data.len() as u64;
        }

        let pulled = payloads.len() as u32;
        let truncated =
            matches!(policy, PullPolicy::PullTimeout) && (pulled as usize) < req.ids.len();
        let completed = match policy {
            PullPolicy::PullWaitComplete => (pulled as usize) == req.ids.len(),
            PullPolicy::PullTimeout => !truncated,
            PullPolicy::PullBestEffort => true,
        };

        let handle = self.next_pull.fetch_add(1, Ordering::Relaxed);
        self.pulls.lock().unwrap().insert(
            handle,
            PullHandle {
                pulled_length: pulled,
                completed,
                dest_segment: dest,
            },
        );

        Ok(Response::new(PullResponse {
            handle,
            pulled_length: pulled,
            completed,
        }))
    }

    async fn publish(&self, request: Request<PublishRequest>) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        let mut st = self.publish.lock().unwrap();
        // seq fence:允许 seq==0(未设置)、seq==last(重试)、seq>last(步进);拒绝 0<seq<last。
        if req.seq > 0 && req.seq < st.last_seq {
            return Ok(Response::new(Ack {
                ok: false,
                err: format!(
                    "Publish seq={} < last_seq={} (overlap fence)",
                    req.seq, st.last_seq
                ),
            }));
        }
        if req.seq > st.last_seq {
            st.last_seq = req.seq;
        }
        for slice in &req.slices {
            let Some(id) = slice.id.as_ref() else {
                continue;
            };
            let key = (id.block_hash.clone(), slice.layer_idx);
            *st.layers.entry(key).or_insert(0) += slice.length;
        }
        Ok(Response::new(Ack {
            ok: true,
            err: String::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lake_proto::lake::transfer_service_server::TransferService;

    fn block(hash: &[u8]) -> KvBlockId {
        KvBlockId {
            model_id: "m".into(),
            block_hash: hash.to_vec(),
            pool_kind: PoolKind::Target as i32,
            scope: "public".into(),
        }
    }

    #[tokio::test]
    async fn submit_get_status_e2e() {
        let transport = Arc::new(TcpTransport::new());
        let bytes = Arc::new(InMemoryByteStore::new());
        let svc = TransferServer::new(transport.clone(), bytes);

        let src = transport.open_segment("s", 32).unwrap();
        let dst = transport.open_segment("d", 32).unwrap();
        transport.write_segment(src, 0, b"abcd").unwrap();

        let ack = svc
            .submit_transfer(Request::new(TransferBatchRequest {
                reqs: vec![TransferRequest {
                    source: Some(Location {
                        tier: Tier::L1 as i32,
                        node_id: "n".into(),
                        segment_id: src,
                        offset: 0,
                    }),
                    target_segment_id: dst,
                    target_offset: 4,
                    length: 4,
                }],
            }))
            .await
            .unwrap()
            .into_inner();

        let st = svc
            .get_transfer_status(Request::new(TransferStatusRequest {
                batch_id: ack.batch_id,
                task_id: 0,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(st.state, transfer_status_response::State::Done as i32);
        assert_eq!(st.bytes_done, 4);
        assert_eq!(&transport.read_segment(dst, 4, 4).unwrap(), b"abcd");
    }

    #[tokio::test]
    async fn submit_rejects_tier_l3() {
        let transport = Arc::new(TcpTransport::new());
        let svc = TransferServer::new(transport.clone(), Arc::new(InMemoryByteStore::new()));
        let src = transport.open_segment("s", 8).unwrap();
        let dst = transport.open_segment("d", 8).unwrap();
        let err = svc
            .submit_transfer(Request::new(TransferBatchRequest {
                reqs: vec![TransferRequest {
                    source: Some(Location {
                        tier: Tier::L3 as i32,
                        node_id: "n".into(),
                        segment_id: src,
                        offset: 0,
                    }),
                    target_segment_id: dst,
                    target_offset: 0,
                    length: 4,
                }],
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn pull_wait_complete_and_publish_layers() {
        let transport = Arc::new(TcpTransport::new());
        let bytes = Arc::new(InMemoryByteStore::new());
        let b0 = block(b"h0");
        let b1 = block(b"h1");
        bytes.put(&b0, b"AAA".to_vec());
        bytes.put(&b1, b"BBBB".to_vec());
        let svc = TransferServer::new(transport, bytes);

        let resp = svc
            .pull(Request::new(PullRequest {
                ids: vec![b0.clone(), b1.clone()],
                policy: PullPolicy::PullWaitComplete as i32,
                budget_ms: 0,
                requester_node_id: "node-a".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.pulled_length, 2);
        assert!(resp.completed);
        assert!(resp.handle > 0);

        let ack = svc
            .publish(Request::new(PublishRequest {
                seq: 1,
                slices: vec![
                    LayerSlice {
                        id: Some(b0),
                        layer_idx: 0,
                        offset: 0,
                        length: 64,
                    },
                    LayerSlice {
                        id: Some(b1),
                        layer_idx: 1,
                        offset: 0,
                        length: 64,
                    },
                ],
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(ack.ok);

        // 同 seq 重试 OK;步进到 3 后再发 2 → fence 拒绝
        assert!(
            svc.publish(Request::new(PublishRequest {
                seq: 1,
                slices: vec![],
            }))
            .await
            .unwrap()
            .into_inner()
            .ok
        );
        svc.publish(Request::new(PublishRequest {
            seq: 3,
            slices: vec![],
        }))
        .await
        .unwrap();
        let back = svc
            .publish(Request::new(PublishRequest {
                seq: 2,
                slices: vec![],
            }))
            .await
            .unwrap()
            .into_inner();
        assert!(!back.ok);
    }

    #[tokio::test]
    async fn pull_timeout_partial() {
        let transport = Arc::new(TcpTransport::new());
        let bytes = Arc::new(InMemoryByteStore::new());
        for i in 0..5u8 {
            bytes.put(&block(&[i]), vec![i; 4]);
        }
        let svc = TransferServer::new(transport, bytes);
        let ids: Vec<_> = (0..5u8).map(|i| block(&[i])).collect();
        let resp = svc
            .pull(Request::new(PullRequest {
                ids,
                policy: PullPolicy::PullTimeout as i32,
                budget_ms: 2,
                requester_node_id: "n".into(),
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.pulled_length, 2);
        assert!(!resp.completed);
    }
}
