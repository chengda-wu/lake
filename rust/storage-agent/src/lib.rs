//! 存储池 agent(P3:边10 Dispatch 占位服务)。
//!
//! 生产:Dispatch → 组 batch → FFI 引擎;本进程只 ack,真实执行仍在 Python WorkerService。
//! 单 crate 双角色 feature:计算侧 / KV Node(见 kv-cache-pool.md)。
//! 参考:SGLang agent_hints / Dispatch 骨架;边6 FFI 留 P4+。

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

pub use lake_proto::lake::*;

use agent_service_server::AgentService;

/// 已接受的 Dispatch 计数(冒烟/观测用)。
static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

pub fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

#[derive(Default, Clone)]
pub struct Agent;

#[tonic::async_trait]
impl AgentService for Agent {
    async fn dispatch(
        &self,
        request: Request<DispatchRequest>,
    ) -> Result<Response<Ack>, Status> {
        let req = request.into_inner();
        if req.target_node_id.is_empty() {
            return Err(Status::invalid_argument("target_node_id required"));
        }
        DISPATCH_COUNT.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(Ack {
            ok: true,
            err: String::new(),
        }))
    }

    type ReportLoadStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<Ack, Status>> + Send + 'static>>;

    async fn report_load(
        &self,
        _request: Request<Streaming<LoadReport>>,
    ) -> Result<Response<Self::ReportLoadStream>, Status> {
        let (tx, rx) = mpsc::channel(1);
        drop(tx);
        let stream: Self::ReportLoadStream = Box::pin(ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }

    async fn place_blocks(
        &self,
        _request: Request<PlaceBlocksRequest>,
    ) -> Result<Response<Ack>, Status> {
        Ok(Response::new(Ack {
            ok: true,
            err: String::new(),
        }))
    }
}

/// 计算节点侧能力占位(FFI / mirror / block table / fence / slot)。
#[cfg(feature = "compute")]
pub mod compute {
    pub const ROLE: &str = "compute";
}

/// KV Node 侧能力占位(NVMe serve / bounce)。
#[cfg(feature = "kvnode")]
pub mod kvnode {
    pub const ROLE: &str = "kvnode";
}

// 编译期锚定:引用具体生成符号(与 PR #18 一致)。
#[allow(dead_code)]
type _AgentServer = lake_proto::lake::agent_service_server::AgentServiceServer<()>;
#[allow(dead_code)]
type _TransferServer = lake_proto::lake::transfer_service_server::TransferServiceServer<()>;
#[allow(dead_code)]
const _ANCHOR: fn() = || {
    let _ = DispatchRequest::default();
    let _ = PullRequest::default();
    let _ = PublishRequest::default();
};
