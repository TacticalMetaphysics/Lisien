import sys
import tomllib

with open("LiSE/pyproject.toml", "rb") as inf:
	lise_version = tomllib.load(inf)["project"]["version"]
with open("ELiDE/pyproject.toml", "rb") as inf:
	elide_version = tomllib.load(inf)["project"]["version"]

if lise_version != elide_version:
	sys.exit(
		f"Version numbers differ. LiSE: {lise_version}, ELiDE: {elide_version}"
	)
