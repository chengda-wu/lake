"""计算层 runtime:P3 提供 WorkerService mock;生产含 gRPC/RDMA client + 节点级 scheduler。"""

from lake_pb import lake_pb2_grpc

_ = lake_pb2_grpc.ControlPlaneServiceStub
_ = lake_pb2_grpc.AgentServiceStub
_ = lake_pb2_grpc.TransferServiceStub
_ = lake_pb2_grpc.WorkerServiceStub
_ = lake_pb2_grpc.SkeletonKvServiceStub
