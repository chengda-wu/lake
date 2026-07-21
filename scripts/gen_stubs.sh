#!/usr/bin/env bash
# 从 proto/ 重新生成 Go / Python gRPC stub(入仓)。
# Rust 走 tonic-build 在线生成,不入仓——改 proto 后 `cd rust && cargo build` 即可。
#
# 工具链版本(钉到生成当前入仓 stub 的版本,防他机重生成 drift,见 PR #17 review follow-up):
#   protoc              3.21.12
#   protoc-gen-go       1.36.11
#   protoc-gen-go-grpc  1.6.2
#   grpcio-tools        1.82.1   (生成 *_pb2.py;runtime 需 grpcio>=1.82.1 / protobuf>=7.35.0,见 python/setup.py)
# 建议用这些版本重生成;版本不符时生成物可能 diff,提交前 `git diff` 核对。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/proto"
GO_OUT="$ROOT/go"
PY_OUT="$ROOT/python"

echo "==> Go: protoc → go/pb/"
mkdir -p "$GO_OUT/pb"
protoc \
  -I "$PROTO_DIR" \
  --go_out="$GO_OUT" --go_opt=module=github.com/chengda-wu/lake/go \
  --go-grpc_out="$GO_OUT" --go-grpc_opt=module=github.com/chengda-wu/lake/go \
  "$PROTO_DIR/schema.proto" "$PROTO_DIR/lake.proto"

echo "==> Python: grpc_tools.protoc → python/lake_pb/"
mkdir -p "$PY_OUT/lake_pb"
python3 -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PY_OUT/lake_pb" \
  --grpc_python_out="$PY_OUT/lake_pb" \
  "$PROTO_DIR/schema.proto" "$PROTO_DIR/lake.proto"

# grpcio-tools 默认按 proto 文件名生成顶层 import;改成包内引用。
# schema_pb2 被 lake_pb2 import —— 改成 from lake_pb import schema_pb2
# 注意:patch 段用相对路径,须在仓库根执行(否则静默不 patch → 生成坏 import)。
# 幂等性:锚定行首 \nimport(只匹配生成器产出的裸 import 行),且替换串不含被匹配子串,
#   重跑不会叠加成 "from lake_pb from lake_pb ..."。
cd "$ROOT"
python3 - <<'PY'
from pathlib import Path
pb_dir = Path("python/lake_pb")
for path in pb_dir.glob("*_pb2*.py"):
    text = path.read_text()
    # 行首裸 import → 包内引用(每个模式只 replace 一次,勿重复)。
    text2 = text.replace(
        "\nimport schema_pb2 as schema__pb2",
        "\nfrom lake_pb import schema_pb2 as schema__pb2",
    )
    text2 = text2.replace(
        "\nimport lake_pb2 as lake__pb2",
        "\nfrom lake_pb import lake_pb2 as lake__pb2",
    )
    if text2 != text:
        path.write_text(text2)
        print(f"  patched imports: {path}")
PY

# 确保包标记存在
touch "$PY_OUT/lake_pb/__init__.py"

echo "==> done"
echo "  verify: cd go && go build ./..."
echo "  verify: PYTHONPATH=python python3 -c 'from lake_pb import lake_pb2, schema_pb2'"
echo "  verify: cd rust && cargo build"
