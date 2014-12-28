# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from setuptools import setup


setup(
    name="LiSE",
    description="Rules engine for life simulation games",
    author="Zachary Spector",
    author_email="zacharyspector@gmail.com",
    license="GPL3",
    keywords="game simulation IDE",
    url="https://github.com/LogicalDash/LiSE",
    packages=[
        "LiSE",
    ],
    package_data={
        'LiSE': ['sqlite.json']
    },
    install_requires=[
        "gorm>=0.5",
    ],
)
