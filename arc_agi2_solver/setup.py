#!/usr/bin/env python3
"""
Setup script for ARC-AGI-2 Solver.

A comprehensive, modular solver for the ARC-AGI-2 benchmark.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="arc-agi2-solver",
    version="1.0.0",
    author="ARC-AGI-2 Team",
    description="A comprehensive, modular solver for the ARC-AGI-2 benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arc-agi2/solver",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "arc-solver=arc_solver.pipeline:main",
        ],
    },
) 