#!/usr/bin/env bash
# 从 proto/ 重新生成 Go / Python gRPC stub(入仓)。
# Rust 走 tonic-build 在线生成,不入仓——改 proto 后 `cd rust && cargo build` 即可。
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
python3 - <<'PY'
from pathlib import Path
pb_dir = Path("python/lake_pb")
for path in pb_dir.glob("*_pb2*.py"):
    text = path.read_text()
    # lake_pb2.py: import schema_pb2 as schema__pb2  → from lake_pb import schema_pb2 as schema__pb2
    text2 = text.replace("import schema_pb2 as schema__pb2", "from lake_pb import schema_pb2 as schema__pb2")
    # *_pb2_grpc.py: import lake_pb2 as lake__pb2 → from lake_pb import lake_pb2 as lake__pb2
    text2 = text2.replace("import lake_pb2 as lake__pb2", "from lake_pb import lake_pb2 as lake__pb2")
    text2 = text2.replace("import schema_pb2 as schema__pb2", "from lake_pb import schema_pb2 as schema__pb2")
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
