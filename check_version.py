import re
import subprocess
import sys
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	lisien_version_str = tomllib.load(inf)["project"]["version"]
with open("elide/pyproject.toml", "rb") as inf:
	loaded = tomllib.load(inf)
	elide_version_str = loaded["project"]["version"]
	for dependency in loaded["project"]["dependencies"]:
		if not dependency.startswith("lisien"):
			continue
		_, ver = dependency.split("==")
		if ver != lisien_version_str:
			raise RuntimeError(
				f"Elide depends on Lisien version {ver}, not {lisien_version_str}"
			)
		break
	else:
		raise RuntimeError("Elide doesn't depend on Lisien")
with open("buildozer.spec", "rt") as inf:
	for line in inf:
		if line.startswith("version"):
			buildozer_version_str = line.split("=")[-1].strip()

pat = r"(\d+?\.\d+?\.\d+?)(\.post[0-9]+)?"
lisien_version = re.match(pat, lisien_version_str).group(1)
elide_version = re.match(pat, elide_version_str).group(1)
buildozer_version = re.match(pat, buildozer_version_str).group(1)

if not (lisien_version == elide_version == buildozer_version):
	sys.exit(
		f"Version numbers differ. lisien: {lisien_version}, elide: {elide_version}, buildozer: {buildozer_version}"
	)

output = subprocess.check_output(
	[sys.executable, "-m", "pip", "index", "versions", "lisien"], text=True
)
vers = ()
for line in output.split("\n"):
	if line.startswith("Available versions: "):
		vers = {
			ver.strip()
			for ver in line.removeprefix("Available versions: ").split(",")
		}
		break

if lisien_version_str in vers:
	sys.exit(f"Version {lisien_version_str} is already in PyPI.")
