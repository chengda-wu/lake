"""计算层 runtime:WorkerService + node_scheduler（Host Req 权威）。"""

from lake_pb import lake_pb2_grpc

_ = lake_pb2_grpc.ControlPlaneServiceStub
_ = lake_pb2_grpc.AgentServiceStub
_ = lake_pb2_grpc.TransferServiceStub
_ = lake_pb2_grpc.WorkerServiceStub
_ = lake_pb2_grpc.SkeletonKvServiceStub
