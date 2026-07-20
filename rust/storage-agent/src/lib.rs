// 存储池 agent 空壳(单 crate 双角色 feature:计算侧 / KV Node,见 kv-cache-pool.md)。
// 实现 AgentService(边10) + TransferService(边7/8) 控制信令。
// P2:仅验证能引用 lake-proto 生成的类型编译通过。
pub use lake_proto::lake::*;

pub const AGENT_SERVICE: &str = "lake.AgentService";
pub const TRANSFER_SERVICE: &str = "lake.TransferService";
