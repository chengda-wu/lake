"""端到端冒烟演示：两个请求共享前缀 → 第二个请求命中 KV Pool。

运行: python -m src
"""

from .compute import DecodePool, PrefillPool
from .kv_pool import KVPool
from .scheduler import Request, Router


def main() -> None:
    model_id = "demo-llm"
    kv_pool = KVPool()
    prefill_pool = PrefillPool(kv_pool, model_id=model_id, block_size=16)
    decode_pool = DecodePool(kv_pool, model_id=model_id)
    router = Router(kv_pool, prefill_pool, decode_pool)

    sys_prompt = list(range(48))  # 公共前缀
    req_a = Request(model_id=model_id, prompt_tokens=sys_prompt + [100, 101])
    req_b = Request(model_id=model_id, prompt_tokens=sys_prompt + [200, 201])

    d1 = router.route(req_a)
    d2 = router.route(req_b)

    print(f"Req A: reused={d1.reused_blocks} prefill={d1.prefill_blocks}")
    print(f"Req B: reused={d2.reused_blocks} prefill={d2.prefill_blocks}")
    print(f"KV Pool size: {len(kv_pool)} blocks")
    print("=> Req B 复用了 Req A 的公共前缀 KV，无需重新计算。")


if __name__ == "__main__":
    main()
