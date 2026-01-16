import os
import re
import shutil
import subprocess
import sys
import tomllib


lisien_version_str = sys.environ["CI_COMMIT_TAG"]
if not re.match(r"v\d+\.\d+\.\d+", lisien_version_str):
	sys.exit(f"Not a valid semantic version: {lisien_version_str}")

SOFT_REQUIREMENTS = {"lxml", "parquetdb"}

DEP_NAME_PAT = r"([a-zA-Z0-9_-]+)([~!<>=]=)?.*"
REQUIREMENTS_PAT = r"requirements *= *(.+)"

with open("lisien/pyproject.toml", "rb") as inf:
	loaded = tomllib.load(inf)
oldver = loaded["project"]["version"]
shutil.move("lisien/pyproject.toml", "lisien/.old.pyproject.toml")


def iter_toml_lines_replace_version(inf):
	for line in inf:
		if line.startswith("version"):
			yield line.replace(oldver, lisien_version_str)
		else:
			yield line


with (
	open("lisien/.old.pyproject.toml", "rt") as inf,
	open("lisien/pyproject.toml", "wt") as outf,
):
	outf.writelines(iter_toml_lines_replace_version(inf))
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
		_, oldver2 = dependency.split("==")
		break
	else:
		raise RuntimeError("Elide doesn't depend on Lisien")
shutil.move("elide/pyproject.toml", "elide/.old.pyproject.toml")
with (
	open("elide/.old.pyproject.toml", "rt") as inf,
	open("elide/pyproject.toml", "wt") as outf,
):
	outf.writelines(iter_toml_lines_replace_version(inf))


def put_files_back():
	os.remove("elide/pyproject.toml")
	os.remove("lisien/pyproject.toml")
	shutil.move("elide/.old.pyproject.toml", "elide/pyproject.toml")
	shutil.move("lisien/.old.pyproject.toml", "lisien/pyproject.toml")


with open("buildozer.spec", "rt") as inf:
	for line in inf:
		if reqs := re.match(REQUIREMENTS_PAT, line):
			deps.difference_update(reqs.group(1).split(","))
			break
	else:
		put_files_back()
		sys.exit("No requirements line in buildozer.spec")

if deps:
	put_files_back()
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

with open("CHANGES.txt", "rt") as inf:
	first_line = next(inf)
if lisien_version_str not in first_line:
	sys.exit(
		f"Version {lisien_version_str} is not the top entry in CHANGES.txt."
	)
print(lisien_version_str)
