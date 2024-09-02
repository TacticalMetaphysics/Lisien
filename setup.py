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

setup(
	name="ELiDE_bundle",
	version=version,
	packages=find_packages(os.path.join(here, "LiSE"))
	+ find_packages(os.path.join(here, "ELiDE")),
	package_dir={
		"LiSE": os.path.join(here, "LiSE", "LiSE"),
		"ELiDE": os.path.join(here, "ELiDE", "ELiDE"),
	},
)
