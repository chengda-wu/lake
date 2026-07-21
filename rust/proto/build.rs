// 从 proto/schema.proto + proto/lake.proto 生成 tonic(prost) gRPC stub。
// 生成代码写到 OUT_DIR,经 src/lib.rs include! 作为本 crate 模块(不入仓)。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // proto crate 在 rust/proto,仓库根 proto/ 在上两级。
    let proto_dir = std::path::Path::new("../../proto");
    println!(
        "cargo:rerun-if-changed={}",
        proto_dir.join("schema.proto").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        proto_dir.join("lake.proto").display()
    );

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(
            &[proto_dir.join("lake.proto"), proto_dir.join("schema.proto")],
            &[proto_dir],
        )?;
    Ok(())
}
