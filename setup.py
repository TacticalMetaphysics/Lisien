"""
This script sets up the LiSE and ELiDE projects using setuptools.
It reads the version numbers and dependencies from the pyproject.toml files of both projects.
"""

import os

try:
    from setuptools import find_packages, setup
except ImportError:
    print("Please install setuptools before running setup.py")

here = os.path.dirname(__file__)

with open(os.path.join(here, "LiSE", "pyproject.toml"), "rt", encoding="utf-8") as inf:
    for line in inf:
        if line.startswith("version"):
            _, version, _ = line.split('"')
            break
    else:
        raise ValueError("Couldn't get version")

deps = {}
vers = {}
for subpkg in ["LiSE", "ELiDE"]:
    with open(os.path.join(here, subpkg, "pyproject.toml"), "rt", encoding="utf-8") as inf:
        for line in inf:
            if line.startswith("version"):
                _, version, _ = line.split('"')
                vers[subpkg] = tuple(map(int, version.split(".")))
            if line.startswith("dependencies"):
                break
        else:
            raise ValueError("Couldn't get %s dependencies" % subpkg)
        deps[subpkg] = []
        for line in inf:
            if line == "]\n":
                break
            _, dep, _ = line.split('"')
            if not dep.startswith("LiSE"):
                deps[subpkg].append(dep)
        else:
            raise ValueError("%s dependencies never ended" % subpkg)
datas = []
with open(os.path.join(here, "ELiDE", "MANIFEST.in"), "rt", encoding="utf-8") as inf:
    for line in inf:
        if line[: len("include")] == "include":
            line = line[len("include ") :]
        if line[-1] == "\n":
            line = line[:-1]
        datas.append(os.path.join(here, "ELiDE", line))

setup(
    name="LiSE_with_ELiDE",
    version=".".join(map(str, max((vers["LiSE"], vers["ELiDE"])))) + "dev",
    packages=find_packages(os.path.join(here, "LiSE"))
    + find_packages(os.path.join(here, "ELiDE")),
    package_dir={
        "LiSE": "LiSE/LiSE",
        "ELiDE": "ELiDE/ELiDE",
    },
    install_requires=deps["LiSE"] + deps["ELiDE"],
    package_data={"ELiDE": datas},
    include_package_data=True,
)




