import sys
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	lisien_version = tomllib.load(inf)["project"]["version"]
with open("elide/pyproject.toml", "rb") as inf:
	elide_version = tomllib.load(inf)["project"]["version"]
with open("buildozer.spec", "rb") as inf:
	buildozer_version = tomllib.load(inf)["app"]["version"]

if lisien_version != elide_version or lisien_version != buildozer_version:
	sys.exit(
		f"Version numbers differ. lisien: {lisien_version}, elide: {elide_version}, buildozer: {buildozer_version}"
	)
