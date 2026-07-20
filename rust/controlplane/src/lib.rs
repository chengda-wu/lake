// 存储控制面空壳(位置视图权威进程)。
// P2:仅验证能引用 lake-proto 生成的 ControlPlaneService 类型编译通过。
// 参考:Dynamo protocols + Mooncake PutEnd/MountSegment 控制面侧(见 lake.proto 头注释)。
pub use lake_proto::lake::*;

/// 占位:后续实现 `ControlPlaneService` server。
pub const SERVICE: &str = "lake.ControlPlaneService";
