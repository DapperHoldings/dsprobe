"""
DSProbe System - Setup Script
For installation: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

setup(
    name="dsprobe",
    version="1.0.0",
    author="DapperHoldings",
    description="Autonomous beacon-based interstellar navigation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DapperHoldings/dsprobe",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "spiceypy>=2.0.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "ml": ["torch>=1.9.0", "gym>=0.18.0"],
        "gpu": ["cupy>=10.0.0", "numba>=0.54.0"],
        "cv": ["opencv-python>=4.5.0"],
        "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
        "ros2": ["rclpy>=0.7.0"],
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0", "black>=21.0", "mypy>=0.910"],
    },
    entry_points={
        "console_scripts": [
            "beacon-nav=main:cli",
        ],
    },
)