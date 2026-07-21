from setuptools import setup, find_packages

setup(
    name="lake",
    version="0.0.0",
    packages=find_packages(),
    # 版本下限对齐生成 stub:lake_pb2.py 校验 protobuf runtime ≥ 7.35.0(major 7),
    # lake_pb2_grpc.py 校验 grpcio ≥ 1.82.1。低于此 import 即炸,故钉到生成版本。
    install_requires=["grpcio>=1.82.1", "protobuf>=7.35.0"],
)
