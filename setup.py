#  Copyright (C) Richard Vogel 2.4.2024
#  Author Richard Vogel <richard.vogel@gmx.com>
#  All Rights Reserved

import os
import re
from typing import List

from setuptools import setup, find_packages

# Read the version from the __init__.py file
with open("trex/__init__.py", "r") as f:
    #version = re.search(r'(?<=__version__ = ").*(?="$)', f.read()).group()
    version = "0.0.1"

# Read the README.md file
with open("readme.md", "r") as f:
    long_description = f.read()

# Read the requirements.txt file
with open("./trex/requirements.txt", "r") as f:
    requirements = f.read().splitlines()


def get_packages() -> List[str]:
    packages = find_packages()
    return packages


setup(
    name="Trex",
    license="MIT",
    version=version,
    author="Richard Vogel",
    author_email="richard.vogel@gmx.com",
    description="Ensemble Rule Learner Trusted Experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
