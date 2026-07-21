//! lake-controlplane:P3 内存权威 gRPC 服务。
//!
//! 默认监听 `0.0.0.0:50051`。环境变量 `LAKE_CP_ADDR` 可覆盖。

use lake_controlplane::ControlPlane;
use lake_proto::lake::control_plane_service_server::ControlPlaneServiceServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: std::net::SocketAddr = std::env::var("LAKE_CP_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;
    println!("lake-controlplane listening on {addr}");
    Server::builder()
        .add_service(ControlPlaneServiceServer::new(ControlPlane::default()))
        .serve(addr)
        .await?;
    Ok(())
}
