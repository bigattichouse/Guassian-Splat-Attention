"""
Setup file for the Hierarchical Splat Attention (HSA) package.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hsa",
    version="0.1.0",
    author="HSA Team",
    author_email="info@hsa-team.org",
    description="Hierarchical Splat Attention - a more efficient attention mechanism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hsa-team/hsa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.10.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
