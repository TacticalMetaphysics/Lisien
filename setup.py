import os

from setuptools import setup

with open(
	os.path.join(os.path.dirname(__file__), "LiSE", "pyproject.toml"), "rt"
) as inf:
	for line in inf:
		if line.startswith("version"):
			_, version, _ = line.split('"')
			break
	else:
		raise ValueError("Couldn't get version")

setup(
	name="ELiDE_bundle",
	version=version,
	packages=["LiSE", "ELiDE", "LiSE.allegedb"],
	package_dir={
		"LiSE": os.path.join(os.path.dirname(__file__), "LiSE", "LiSE"),
		"ELiDE": os.path.join(os.path.dirname(__file__), "ELiDE", "ELiDE"),
	},
)
