from setuptools import setup, Extension
from Cython.Distutils import build_ext

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
    install_requires=[
        "gorm>=0.5",
        "numpy>=1.9"
    ],
)


setup(
    name="ELiDE.kivygarden.collider",
    ext_modules=[
        Extension(
            name="collider",
            depends=["ELiDE/kivygarden/collider/__init__.py"],
            sources=["ELiDE/kivygarden/collider/collider.pyx"]
        )
    ],
    cmdclass={'build_ext': build_ext},
)


setup(
    name="ELiDE",
    packages=[
        "ELiDE",
        "ELiDE.board",
        "ELiDE.kivygarden.stiffscroll",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'ELiDE.kivygarden.stiffscroll': 'ELiDE/kivygarden/stiffscroll',
        'ELiDE.kivygarden.texturestack':
        'ELiDE/kivygarden/texturestack'
    },
    install_requires=[
        "LiSE",
        "kivy",
        "ELiDE.kivygarden.collider"
    ],
    package_data={
        "ELiDE": [
            "elide.kv",
            "assets/*.png",
            "assets/*.jpg",
            "assets/*.ttf",
            "assets/*.atlas",
            "assets/rltiles/*"
        ]
    },
    zip_safe=False
)
