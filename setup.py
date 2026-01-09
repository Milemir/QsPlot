from skbuild import setup
from setuptools import find_packages

setup(
    name="qsplot",
    version="0.1.0",
    description="High-performance C++/OpenGL visualization engine specified for times series data",
    author="Milemir",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/qsplot",
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
    ],
    extras_require={
        "umap": ["umap-learn"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
