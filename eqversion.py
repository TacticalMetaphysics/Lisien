import sys
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	lise_version = tomllib.load(inf)["project"]["version"]
with open("elide/pyproject.toml", "rb") as inf:
	elide_version = tomllib.load(inf)["project"]["version"]

if lise_version != elide_version:
	sys.exit(
		f"Version numbers differ. lisien: {lise_version}, elide: {elide_version}"
	)
