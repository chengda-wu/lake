#!/usr/bin/env bash
# P3 冒烟:两次共享前缀请求,第二次 reused_blocks > 0。
# 需先 ./deploy/run-local.sh
set -euo pipefail

URL="${LAKE_SMOKE_URL:-http://127.0.0.1:8080/v1/chat/completions}"
# 足够长且稳定的 system 前缀 → 多个 block;两次 user 后缀不同。
SYS='AAAAAAAA'  # tokenizeMock 按字符,补齐到 8 的倍数

post() {
  local user="$1"
  curl -sS "$URL" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - <<PY
import json
print(json.dumps({
  "model": "mock-llm",
  "messages": [
    {"role": "system", "content": "$SYS"},
    {"role": "user", "content": "$user"},
  ],
  "max_tokens": 4,
}))
PY
)"
}

echo "==> Req A"
A="$(post 'hello-AAA')"
echo "$A" | python3 -m json.tool
RA="$(echo "$A" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["reused_blocks"])')"
PA="$(echo "$A" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["prefill_blocks"])')"

echo "==> Req B (shared system prefix)"
B="$(post 'hello-BBB')"
echo "$B" | python3 -m json.tool
RB="$(echo "$B" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["reused_blocks"])')"
PB="$(echo "$B" | python3 -c 'import sys,json; print(json.load(sys.stdin)["lake"]["prefill_blocks"])')"

echo "Req A: reused=$RA prefill=$PA"
echo "Req B: reused=$RB prefill=$PB"

if [[ "$RB" -le 0 ]]; then
  echo "FAIL: expected Req B reused_blocks > 0 (prefix reuse via Rust ControlPlane+SkeletonKv)"
  exit 1
fi
if [[ "$PA" -le 0 ]]; then
  echo "FAIL: expected Req A to prefill at least one block"
  exit 1
fi
echo "OK: prefix reuse across languages"
