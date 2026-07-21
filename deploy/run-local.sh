#!/usr/bin/env bash
# P3 本地起全栈(无 Docker):controlplane + kv-pool + worker + router。
# 用法(仓库根):
#   ./deploy/run-local.sh          # 前台,Ctrl-C 全停
#   ./deploy/smoke.sh              # 另开终端打冒烟
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export LAKE_CP_ADDR="${LAKE_CP_ADDR:-127.0.0.1:50051}"
export LAKE_KV_ADDR="${LAKE_KV_ADDR:-127.0.0.1:50052}"
export LAKE_WORKER_BIND="${LAKE_WORKER_BIND:-[::]:50053}"
export LAKE_WORKER_ADDR="${LAKE_WORKER_ADDR:-127.0.0.1:50053}"
export LAKE_HTTP_ADDR="${LAKE_HTTP_ADDR:-:8080}"

# 兼容 CARGO_TARGET_DIR 覆盖(如 CI/sandbox);不要写死 rust/target/debug。
RUST_DEBUG="$(
  cd rust
  cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"] + "/debug")'
)"

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "==> build rust bins → $RUST_DEBUG"
(cd rust && cargo build -q -p lake-controlplane --bin lake-controlplane -p lake-kv-pool --bin lake-kv-pool)
test -x "$RUST_DEBUG/lake-controlplane"
test -x "$RUST_DEBUG/lake-kv-pool"

echo "==> start controlplane"
LAKE_CP_ADDR=0.0.0.0:50051 "$RUST_DEBUG/lake-controlplane" &
PIDS+=($!)

echo "==> start kv-pool"
LAKE_KV_ADDR=0.0.0.0:50052 "$RUST_DEBUG/lake-kv-pool" &
PIDS+=($!)

sleep 0.4

echo "==> start python worker"
(
  cd python
  PYTHONPATH=. python3 -m runtime
) &
PIDS+=($!)

sleep 0.4

echo "==> start go router"
(
  cd go
  go run ./router/cmd/router
) &
PIDS+=($!)

echo "stack up: HTTP http://127.0.0.1${LAKE_HTTP_ADDR}  (try ./deploy/smoke.sh)"
wait
