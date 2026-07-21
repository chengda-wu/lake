#!/usr/bin/env bash
# P3 编译门禁(不含起全栈冒烟)。仓库根执行:./scripts/verify-p3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> rust: bins + controlplane unit tests"
(cd rust && cargo build -q \
  -p lake-controlplane --bin lake-controlplane \
  -p lake-kv-pool --bin lake-kv-pool \
  -p lake-storage-agent --bin lake-storage-agent)
(cd rust && cargo test -q -p lake-controlplane)

echo "==> go: module"
(cd go && go build ./...)

echo "==> python: stub + runtime import"
PYTHONPATH=python python3 -c "
from lake_pb import lake_pb2, lake_pb2_grpc, schema_pb2
import runtime, prefill, decode
from runtime.worker import chain_block_hashes, mock_kv_bytes
assert hasattr(lake_pb2_grpc, 'WorkerServiceStub')
assert hasattr(lake_pb2_grpc, 'SkeletonKvServiceStub')
assert hasattr(lake_pb2_grpc, 'AgentServiceStub')
h = chain_block_hashes(list(range(24)))
assert len(h) == 3, h
assert mock_kv_bytes(h[0]).startswith(b'KV:')
print('py OK')
"

echo "OK: P3 compile gate"
