"""Prefill worker 空壳。P2:仅验证包可 import;无业务逻辑。"""

from lake_pb import lake_pb2, schema_pb2

# 锚定 stub 类型仍在生成物中(满块注册 / 位置)。
_ = schema_pb2.KVBlockID
_ = lake_pb2.RegisterBlocksRequest
