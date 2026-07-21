"""Draft worker 空壳(投机解码)。P2:仅验证包可 import;无业务逻辑。"""

from lake_pb import schema_pb2

# draft KV 进池统一管理(pool_kind);类型占位。
_ = schema_pb2.PoolKind
