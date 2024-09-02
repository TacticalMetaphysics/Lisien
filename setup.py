import os

from setuptools import setup, find_packages

here = os.path.dirname(__file__)

with open(os.path.join(here, "LiSE", "pyproject.toml"), "rt") as inf:
	for line in inf:
		if line.startswith("version"):
			_, version, _ = line.split('"')
			break
	else:
		raise ValueError("Couldn't get version")

deps = {}
vers = {}
for subpkg in ["LiSE", "ELiDE"]:
	with open(os.path.join(here, subpkg, "pyproject.toml"), "rt") as inf:
		for line in inf:
			if line.startswith("version"):
				_, version, _ = line.split('"')
				vers[subpkg] = version
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

setup(
	name="LiSE",
	version=vers["LiSE"],
	packages=find_packages(os.path.join(here, "LiSE")),
	package_dir={"LiSE": os.path.join(here, "LiSE", "LiSE")},
	install_requires=deps["LiSE"],
)
setup(
	name="ELiDE",
	version=vers["ELiDE"],
	packages=find_packages(os.path.join(here, "ELiDE")),
	package_dir={"ELiDE": os.path.join(here, "ELiDE", "ELiDE")},
	install_requires=deps["ELiDE"],
)
