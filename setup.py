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
    package_data={
        'LiSE': ['sqlite.json']
    },
    install_requires=[
        "gorm>=0.5",
    ],
)


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
        "kivy"
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
