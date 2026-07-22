"""本步静态 / 批 buffer（非跨步请求权威）。C0 仅占位。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class InputBatch:
    req_ids: List[str] = field(default_factory=list)
    # 生产路径：固定 buffer + agent 写好的 block table 视图
