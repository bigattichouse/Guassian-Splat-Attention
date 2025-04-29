
from setuptools import setup, find_packages

setup(
    name="hsa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch>=1.7.0",
        "scikit-learn>=0.23.0",
    ],
    description="Hierarchical Splat Attention (HSA) for transformer models",
    author="HSA Team",
)
