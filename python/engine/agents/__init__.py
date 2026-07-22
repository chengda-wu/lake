"""可替换 StorageAgent 实现：memory（单测）/ grpc_skeleton（P3）。"""

from engine.agents.grpc_skeleton import GrpcSkeletonAgent
from engine.agents.memory import InMemoryAgent

__all__ = ["GrpcSkeletonAgent", "InMemoryAgent"]
