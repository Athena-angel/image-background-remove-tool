"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import os
import re

from setuptools import setup, find_packages

from carvekit import version

IS_COLAB_PACKAGE = os.getenv("COLAB_PACKAGE_RELEASE", None)


def read(filename: str):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    file = open(filepath, "r", encoding="utf-8")
    return file.read()


def req_file(filename: str, folder: str = "."):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters
    # Example: `\n` at the end of each line
    if os.getenv("COLAB_PACKAGE_RELEASE") is not None:
        return [re.sub("(~=.*)|(==.*)|(typing.*)", "", x.strip()) for x in content]
    return [x.strip() for x in content]


setup(
    name="carvekit" if IS_COLAB_PACKAGE is None else "carvekit_colab",
    version=version,
    author="Nikita Selin (Anodev)",
    author_email="farvard34@gmail.com",
    description="Open-Source background removal framework",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="Apache License v2.0",
    keywords=[
        "ml",
        "carvekit",
        "background removal",
        "neural networks",
        "machine learning",
        "remove bg",
    ],
    url="https://github.com/OPHoperHPO/image-background-remove-tool",
    packages=find_packages(),
    scripts=[],
    install_requires=req_file("requirements.txt"),
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "carvekit=carvekit:__main__.removebg",
        ],
    },
    python_requires=">=3.8" if IS_COLAB_PACKAGE is None else ">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
