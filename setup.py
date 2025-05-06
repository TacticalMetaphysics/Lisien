# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os

from setuptools import find_packages, setup

here = os.path.dirname(__file__)

with open(os.path.join(here, "lisien", "pyproject.toml"), "rt") as inf:
	for line in inf:
		if line.startswith("version"):
			_, version, _ = line.split('"')
			break
	else:
		raise ValueError("Couldn't get version")

deps = {}
vers = {}
for subpkg in ["lisien", "elide"]:
	with open(os.path.join(here, subpkg, "pyproject.toml"), "rt") as inf:
		for line in inf:
			if line.startswith("version"):
				_, version, _ = line.split('"')
				vers[subpkg] = tuple(map(int, version.split(".")))
			if line.startswith("dependencies"):
				break
		else:
			raise ValueError("Couldn't get %s dependencies" % subpkg)
		deps[subpkg] = []
		for line in inf:
			if line == "]\n":
				break
			_, dep, _ = line.split('"')
			if not dep.startswith("lisien"):
				deps[subpkg].append(dep)
		else:
			raise ValueError("%s dependencies never ended" % subpkg)
datas = []
with open(os.path.join(here, "elide", "MANIFEST.in"), "rt") as inf:
	for line in inf:
		if line[: len("include")] == "include":
			line = line[len("include ") :]
		if line[-1] == "\n":
			line = line[:-1]
		datas.append(os.path.join(here, "elide", line))

setup(
	name="lisien_with_elide",
	version=".".join(map(str, max((vers["lisien"], vers["elide"])))) + "dev",
	packages=find_packages(os.path.join(here, "lisien"))
	+ find_packages(os.path.join(here, "elide")),
	package_dir={
		"lisien": "lisien/lisien",
		"elide": "elide/elide",
	},
	install_requires=deps["lisien"] + deps["elide"],
	package_data={"elide": datas},
	include_package_data=True,
)
