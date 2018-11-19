print("This is just for readthedocs, please don't install it for real")

from setuptools import setup
import os
# convince Kivy to do a Cythonless "build," so we can import it but it does nothing
os.environ["READTHEDOCS"] = "True"
setup(
    name="LiSE docs",
    version="0.9",
    license="AGPL3+",
    packages=[
        "allegedb",
        "LiSE",
        "ELiDE",
        "ELiDE.board",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'allegedb': 'allegedb/allegedb',
        'LiSE': 'LiSE/LiSE',
        'ELiDE': 'ELiDE/ELiDE',
        'ELiDE.board': 'ELiDE/ELiDE/board',
        'ELiDE.kivygarden.texturestack':
        'ELiDE/ELiDE/kivygarden/texturestack'
    },
    install_requires=[
        "networkx>=1.9",
        "astunparse>=1.5.0",
        "u-msgpack-python>=2.4.1",
        "blinker",
        "numpy",
        "kivy>=1.10.0",
        "pygments"
    ],
    package_data={
        "allegedb": [
            "sqlite.json"
        ],
        "LiSE": [
            "sqlite.json"
        ],
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
