# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and
        sys.version_info[1] < 3
):
    raise RuntimeError("LiSE requires Python 3.3 or later")

from setuptools import setup


setup(
    name="LiSE",
    version="a5",
    description="Rules engine for life simulation games",
    author="Zachary Spector",
    author_email="zacharyspector@gmail.com",
    license="GPL3",
    keywords="game simulation",
    url="https://github.com/LogicalDash/LiSE",
    packages=[
        "LiSE",
    ],
    package_data={
        'LiSE': ['sqlite.json']
    },
    install_requires=[
        "gorm>=0.7.8",
    ],
)
