from setuptools import setup
setup(
    name = "gorm",
    version = "0.6.0",
    packages = ["gorm"],
    install_requires = ['networkx>=1.9'],
    author = "Zachary Spector",
    author_email = "zacharyspector@gmail.com",
    description = "An object-relational mapper serving database-backed versions of the standard networkx graph classes.",
    license = "BSD",
    keywords = "orm graph networkx sql database",
    url = "https://github.com/LogicalDash/gorm",
    package_dir={
        "gorm": "gorm"
    },
    package_data={
        "gorm": [
            "sqlite.json"
        ]
    }
)
