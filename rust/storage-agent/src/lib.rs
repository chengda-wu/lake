// 存储池 agent 空壳(单 crate 双角色 feature:计算侧 / KV Node,见 kv-cache-pool.md)。
// 实现 AgentService(边10) + TransferService(边7/8) 控制信令。
// P2:仅验证能引用 lake-proto 生成的类型编译通过。
pub use lake_proto::lake::*;

pub const AGENT_SERVICE: &str = "lake.AgentService";
pub const TRANSFER_SERVICE: &str = "lake.TransferService";

// 编译期锚定:引用具体生成符号,防 proto 改名/删字段后 Rust 仍编译通过(Go/Python 已锚定)。
// 消息类型取 default();server struct 用类型别名引用(不构造实例,别名即编译期依赖)。
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
