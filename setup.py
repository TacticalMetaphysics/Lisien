from setuptools import setup

setup(
    name="LiSEkit",
    version="0.0",
    packages=[
        "LiSE",
        "ELiDE",
        "ELiDE.board",
        "ELiDE.kivygarden.stiffscroll",
        "ELiDE.kivygarden.collider",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'ELiDE.kivygarden.stiffscroll': 'ELiDE/kivygarden/stiffscroll',
        'ELiDE.kivygarden.collider': 'ELiDE/kivygarden/collider',
        'ELiDE.kivygarden.texturestack':
        'ELiDE/kivygarden/texturestack'
    },
    install_requires=["kivy", "gorm>=0.5", "numpy>=1.9"],
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
