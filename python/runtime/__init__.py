"""计算层 runtime 空壳(gRPC/RDMA client;节点级 scheduler 后续)。

P2:仅验证能 import lake_pb gRPC stub;worker↔agent 走 FFI 不进 proto。
参考:vLLM KVConnectorBase_V1(worker↔池接入点)。
"""

from lake_pb import lake_pb2_grpc

_ = lake_pb2_grpc.ControlPlaneServiceStub
_ = lake_pb2_grpc.AgentServiceStub
_ = lake_pb2_grpc.TransferServiceStub
