"""前缀命中提示——Router / probe 的输入，供节点组 batch（方案 Z 只读）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrefixHint:
    computed_tokens: int = 0  # 已可复用 token 数（半开上界语义：前 computed 已算）
    reused_blocks: int = 0
    local_hit: bool = False  # 前缀已在本机 L0（D-direct 条件）
    prebuilt: bool = False  # True：KV 已灌入，可走 PREBUILT 跳过 extend forward
