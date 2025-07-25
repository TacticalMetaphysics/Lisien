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
"""Update the lisien and elide packages in .buildozer in-place"""

import os
import shutil

pardirs = []
prefix = ".buildozer/android/platform/"
for plat in os.listdir(prefix):
	if "dists" in os.listdir(os.path.join(prefix, plat)):
		elide_dist_dir = os.path.join(prefix, plat, "dists", "Elide")
		if not os.path.isdir(elide_dist_dir):
			continue
		for dist_dir in os.listdir(elide_dist_dir):
			if dist_dir.startswith("_python_bundle"):
				pardirs.append(
					os.path.join(
						prefix,
						plat,
						"dists",
						"Elide",
						dist_dir,
						"_python_bundle",
						"site-packages",
					)
				)
	if "build" not in os.listdir(os.path.join(prefix, plat)):
		continue
	for build_dir in os.listdir(os.path.join(prefix, plat, "build")):
		if build_dir not in {"other_builds", "python-installs"}:
			continue
		if build_dir == "other_builds":
			pardirs.append(os.path.join(prefix, plat, "build", build_dir))
		elif build_dir == "python-installs":
			pre = os.path.join(prefix, plat, "build", build_dir, "Elide")
			if os.path.exists(pre) and os.path.isdir(pre):
				pardirs.extend(
					os.path.join(pre, arch) for arch in os.listdir(pre)
				)
print("looking for built packages in", pardirs)
for pardir in pardirs:
	for package in os.listdir(pardir):
		if package.lower() == "lisien":
			abspath = os.path.join(pardir, package)
			print("replacing", abspath, "with current lisien")
			shutil.rmtree(abspath)
			shutil.copytree("lisien/lisien", abspath)
		elif package.lower() == "elide":
			abspath = os.path.join(pardir, package)
			print("replacing", abspath, "with current elide")
			shutil.rmtree(abspath)
			shutil.copytree("elide/elide", abspath)
