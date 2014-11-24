from setuptools import setup, Extension
from Cython.Distutils import build_ext

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
    name="LiSEkit",
    version="0.0",
    packages=[
        "LiSE",
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
        "kivy",
        "ELiDE.kivygarden.collider",
        "gorm>=0.5",
        "numpy>=1.9"
    ],
    author="Zachary Spector",
    author_email="zacharyspector@gmail.com",
    description="Life simulator construction kit with graphical frontend",
    license="GPL3",
    keywords="game simulation IDE",
    url="https://github.com/LogicalDash/LiSE",
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
