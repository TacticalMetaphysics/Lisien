import re
import subprocess
import sys
import tomllib

SOFT_REQUIREMENTS = {"lxml", "parquetdb"}

DEP_NAME_PAT = r"([a-zA-Z0-9_-]+)([~!<>=]=)?.*"
REQUIREMENTS_PAT = r"requirements *= *(.+)"

with open("lisien/pyproject.toml", "rb") as inf:
	loaded = tomllib.load(inf)
lisien_version_str = loaded["project"]["version"]
deps = {
	re.match(DEP_NAME_PAT, dep).group(1)
	for dep in loaded["project"]["dependencies"]
} - SOFT_REQUIREMENTS
with open("elide/pyproject.toml", "rb") as inf:
	loaded = tomllib.load(inf)
	elide_version_str = loaded["project"]["version"]
	for dependency in loaded["project"]["dependencies"]:
		if not dependency.startswith("lisien"):
			deps.add(re.match(DEP_NAME_PAT, dependency).group(1))
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
		if reqs := re.match(REQUIREMENTS_PAT, line):
			deps.difference_update(reqs.group(1).split(","))
			break
	else:
		sys.exit("No requirements line in buildozer.spec")

if deps:
	sys.exit(f"Requirements missing from buildozer.spec: {', '.join(deps)}")

pat = r"(\d+?\.\d+?\.\d+?)(\.post[0-9]+)?"
lisien_version = re.match(pat, lisien_version_str).group(1)
elide_version = re.match(pat, elide_version_str).group(1)

if not (lisien_version == elide_version):
	sys.exit(
		f"Version numbers differ. lisien: {lisien_version}, elide: {elide_version}"
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
print(lisien_version_str)
