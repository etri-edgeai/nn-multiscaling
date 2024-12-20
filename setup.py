"""Bespoke setup

"""

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bespoke",
    version="1.0.0",
    author="Jong-Ryul Lee",
    author_email="jongryul.lee@etri.re.kr",
    description="Bespoke Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=[
        "numpy"
    ]
)
