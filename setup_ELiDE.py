from setuptools import setup

setup(
    name="ELiDE",
    version="0.0",
    packages=["ELiDE", "ELiDE.board", "kivy.garden.stiffscroll"],
    package_dir={'kivy.garden.stiffscroll': 'ELiDE/libs/garden/garden.stiffscroll'},
    install_requires=["LiSE", "kivy"],
    author = "Zachary Spector",
    author_email = "zacharyspector@gmail.com",
    description = "Graphical developer's toolkit for LiSE",
    license = "GPL3",
    keywords = "game simulation IDE",
    url = "https://github.com/LogicalDash/LiSE",
    package_data = {
        "ELiDE": [
            "elide.kv",
            "assets/*.png",
            "assets/*.jpg",
            "assets/*.ttf",
            "assets/*.atlas",
            "assets/rltiles/*"
        ]
    }
)
