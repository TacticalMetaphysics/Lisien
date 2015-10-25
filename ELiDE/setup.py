# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] < 3
):
    raise RuntimeError("ELiDE requires Python 3.3 or later")
from setuptools import setup, Extension
from Cython.Distutils import build_ext


setup(
    name="ELiDE",
    packages=[
        "ELiDE",
        "ELiDE.board",
        "ELiDE.kivygarden.collider",
        "ELiDE.kivygarden.stiffscroll",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'ELiDE.kivygarden.stiffscroll': 'ELiDE/kivygarden/stiffscroll',
        'ELiDE.kivygarden.collider': 'ELiDE/kivygarden/collider',
        'ELiDE.kivygarden.texturestack':
        'ELiDE/kivygarden/texturestack'
    },
    ext_modules=[
        Extension(
            name="collider",
            depends=["ELiDE/kivygarden/collider/__init__.py"],
            sources=["ELiDE/kivygarden/collider/collider.pyx"]
        )
    ],
    cmdclass={'build_ext': build_ext},
    install_requires=[
        "LiSE",
        "numpy",
        "kivy",
        "pygments"
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
