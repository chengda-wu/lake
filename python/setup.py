from setuptools import setup, find_packages

setup(
    name="lake-pb",
    version="0.0.0",
    packages=find_packages(),
    install_requires=["grpcio>=1.66", "protobuf>=5.0"],
)
