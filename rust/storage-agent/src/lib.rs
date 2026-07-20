// 存储池 agent 空壳(单 crate 双 target,见 kv-cache-pool.md「KV Node 上的 agent」)。
// P2:仅验证能引用 lake-proto 生成的类型编译通过。
pub use lake_proto::lake::*;
