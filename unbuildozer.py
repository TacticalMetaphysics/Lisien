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
"""Utility to remove Lisien and Elide from Buildozer's workspace

You can use this instead of ``buildozer android clean``, as long as only
Lisien and Elide code has changed since the last time you ran
``buildozer android debug``.

"""

import os
import shutil

shutil.rmtree(".buildozer/android/app/elide", ignore_errors=True)
pardirs = []
prefix = ".buildozer/android/platform/"
for plat in os.listdir(prefix):
	if "dists" in os.listdir(os.path.join(prefix, plat)):
		shutil.rmtree(os.path.join(prefix, plat, "dists"))
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
				pardirs.extend(os.path.join(pre, arch) for arch in os.listdir(pre))
print("looking for built packages in", pardirs)
for pardir in pardirs:
	for package in os.listdir(pardir):
		if package.lower().startswith("lisien") or package.lower().startswith("elide"):
			abspath = os.path.join(pardir, package)
			if (
				not os.path.exists(abspath)
				or not os.path.isdir(abspath)
				or os.path.islink(abspath)
			):
				continue
			print("removing", abspath)
			shutil.rmtree(abspath)
