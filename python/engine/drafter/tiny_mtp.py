"""TinyMTPDrafter：C4 共置 MTP 骨架。

同一类上的 `post_forward`（target 后）/ `pre_forward`（下轮 target 前）。
参考:SGLang EAGLE/MTP worker 编排；lake 文档「投机解码」二阶段。
关键差异：draft KV 生产进池（PoolName.DRAFT）；C4 seed 仅 host 侧暂存。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from engine.models.tiny_lm import TinyLM


class TinyMTPDrafter:
    def __init__(
        self,
        num_draft_tokens: int = 2,
        *,
        vocab_size: int = 256,
        d_model: int = 32,
        n_heads: int = 4,
        seed: int = 99,
    ) -> None:
        self.num_draft_tokens = num_draft_tokens
        # 与 target 不同 seed → 故意制造部分拒绝，覆盖 reject 路径
        self._draft_lm = TinyLM(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, seed=seed
        )
        self._seed_ctx: Dict[str, List[int]] = {}

    def post_forward(self, req_id: str, context: Sequence[int]) -> None:
        """target 之后：消费本步上下文作 seed（生产为 hidden/KV）。"""
        self._seed_ctx[req_id] = list(context)

    def pre_forward(self, req_id: str) -> List[int]:
        """下轮 target 之前：自回归产 draft token。"""
        ctx = list(self._seed_ctx.get(req_id, []))
        drafts: List[int] = []
        for _ in range(self.num_draft_tokens):
            if not ctx:
                break
            tok = self._draft_lm.greedy_token(ctx)
            drafts.append(tok)
            ctx.append(tok)
        return drafts

    def clear(self, req_id: str) -> None:
        self._seed_ctx.pop(req_id, None)
