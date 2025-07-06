import re
import sys
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	lisien_version = tomllib.load(inf)["project"]["version"]
with open("elide/pyproject.toml", "rb") as inf:
	elide_version = tomllib.load(inf)["project"]["version"]
with open("buildozer.spec", "rt") as inf:
	for line in inf:
		if line.startswith("version"):
			buildozer_version = line.split("=")[-1].strip()

pat = r"(.+?)(\.post[0-9]+)"

if (
	re.match(pat, lisien_version).group(1)
	!= re.match(pat, elide_version).group(1)
	or re.match(pat, lisien_version).group(1) != buildozer_version
):
	sys.exit(
		f"Version numbers differ. lisien: {lisien_version}, elide: {elide_version}, buildozer: {buildozer_version}"
	)
