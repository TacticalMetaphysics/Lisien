from setuptools import setup
setup(
    name = "allegedb",
    version = "0.13.0",
    packages = ["allegedb"],
    install_requires = ['networkx>=1.9', 'blinker'],
    author = "Zachary Spector",
    author_email = "zacharyspector@gmail.com",
    description = "An object-relational mapper serving database-backed versions of the standard networkx graph classes.",
    license = "BSD",
    keywords = "orm graph networkx sql database",
    url = "https://github.com/LogicalDash/LiSE",
    package_dir={
      "allegedb": "allegedb"
    },
    package_data={
        "allegedb": [
            "sqlite.json"
        ]
    }
)
