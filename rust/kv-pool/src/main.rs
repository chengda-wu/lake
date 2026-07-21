//! lake-kv-pool:P3 SkeletonKv gRPC 服务(内存 bytes)。
//!
//! 默认 `0.0.0.0:50052`。环境变量 `LAKE_KV_ADDR` 可覆盖。

use lake_kv_pool::KvPool;
use lake_proto::lake::skeleton_kv_service_server::SkeletonKvServiceServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: std::net::SocketAddr = std::env::var("LAKE_KV_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50052".into())
        .parse()?;
    println!("lake-kv-pool (SkeletonKv) listening on {addr}");
    Server::builder()
        .add_service(SkeletonKvServiceServer::new(KvPool::default()))
        .serve(addr)
        .await?;
    Ok(())
}
