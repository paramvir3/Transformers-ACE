from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent

setup(
    name="transformers-ace",
    version="2.2.1",
    description="Local equivariant transformer potentials built on ACE descriptors",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/paramvir3/Transformers-ACE",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "e3nn>=0.6.0,<0.7.0",
        "ase>=3.22",
        "numpy>=1.23",
        "pyyaml>=6.0",
        "matplotlib>=3.6",
        "packaging>=23",
    ],
    extras_require={
        "accelerated": ["torch-scatter"],
        "test": ["pytest>=7"],
    },
)
