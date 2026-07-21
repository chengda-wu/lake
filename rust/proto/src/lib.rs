// 共享 proto crate:把 tonic-build 生成的代码包成模块供其它 crate 依赖。
// lake.proto 与 schema.proto 的 package 都是 lake,生成到同一命名空间。
// lake.proto import 了 schema.proto,二者类型都在 lake:: 下。

pub mod lake {
    tonic::include_proto!("lake");
}
