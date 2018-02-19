# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] < 3
):
    raise RuntimeError("ELiDE requires Python 3.3 or later")
from setuptools import setup


setup(
    name="LiSE project",
    version="0.8.0a",
    packages=[
        "allegedb",
        "LiSE",
        "ELiDE",
        "ELiDE.board",
        "ELiDE.kivygarden.stiffscroll",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'allegedb': 'allegedb/allegedb',
        'LiSE': 'LiSE/LiSE',
        'ELiDE': 'ELiDE/ELiDE',
        'ELiDE.board': 'ELiDE/ELiDE/board',
        'ELiDE.kivygarden.stiffscroll': 'ELiDE/ELiDE/kivygarden/stiffscroll',
        'ELiDE.kivygarden.texturestack':
        'ELiDE/ELiDE/kivygarden/texturestack'
    },
    install_requires=[
        "numpy",
        "kivy>=1.10.0",
        "pygments",
        "networkx",
        "blinker"
    ],
    package_data={
        "ELiDE": [
            "assets/*.png",
            "assets/*.jpg",
            "assets/*.ttf",
            "assets/*.atlas",
            "assets/rltiles/*"
        ]
    },
    zip_safe=False
)
