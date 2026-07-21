// 存储控制面空壳(位置视图权威进程)。
// P2:仅验证能引用 lake-proto 生成的 ControlPlaneService 类型编译通过。
// 参考:Dynamo protocols + Mooncake PutEnd/MountSegment 控制面侧(见 lake.proto 头注释)。
pub use lake_proto::lake::*;

/// 占位:后续实现 `ControlPlaneService` server。
pub const SERVICE: &str = "lake.ControlPlaneService";

// 编译期锚定:引用具体生成符号,防 proto 改名/删字段后 Rust 仍编译通过(Go/Python 已锚定)。
// 消息类型(glob re-export 到顶层)取 default();server struct 在子模块下,用类型别名引用
// (不构造实例,规避泛型构造约束;别名本身就是对生成符号的编译期依赖)。
#[allow(dead_code)]
type _CpServer =
    lake_proto::lake::control_plane_service_server::ControlPlaneServiceServer<()>;
#[allow(dead_code)]
const _ANCHOR: fn() = || {
    let _ = RegisterBlocksRequest::default();
    let _ = LookupPrefixRequest::default();
};
