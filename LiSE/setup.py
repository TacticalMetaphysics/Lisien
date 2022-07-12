# This file is part of LiSE, a framework for life simulation games.
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
from setuptools import setup

shortdesc = "Rules engine for life simulation games"

readmepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
							'README.md')
if os.path.exists(readmepath):
	with open(readmepath, 'rt') as inf:
		longdesc = inf.read()
else:
	longdesc = shortdesc

setup(name="LiSE",
		version="0.13.0",
		description=shortdesc,
		author="Zachary Spector",
		author_email="public@zacharyspector.com",
		python_requires=">=3.7",
		license="AGPL3",
		keywords="game simulation",
		url="https://github.com/Tactical-Metaphysics/LiSE",
		packages=["LiSE", "LiSE.server", "LiSE.examples", "LiSE.allegedb"],
		package_data={'LiSE': ['sqlite.json']},
		project_urls={
			"Documentation": "https://tactical-metaphysics.github.io/LiSE/"
		},
		long_description=longdesc,
		long_description_content_type='text/markdown')
