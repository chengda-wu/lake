//! lake-storage-agent:P3 AgentService(边10 Dispatch)。
//!
//! 默认 `0.0.0.0:50054`。环境变量 `LAKE_AGENT_ADDR` 可覆盖。

use lake_proto::lake::agent_service_server::AgentServiceServer;
use lake_storage_agent::Agent;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: std::net::SocketAddr = std::env::var("LAKE_AGENT_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50054".into())
        .parse()?;
    println!("lake-storage-agent (AgentService) listening on {addr}");
    Server::builder()
        .add_service(AgentServiceServer::new(Agent::default()))
        .serve(addr)
        .await?;
    Ok(())
}
