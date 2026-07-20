"""Decode worker 空壳。P2:仅验证包可 import;无业务逻辑。"""

from lake_pb import lake_pb2, schema_pb2

_ = schema_pb2.Location
_ = lake_pb2.RequestBarrierRequest
