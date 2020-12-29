# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and
        sys.version_info[1] < 6
):
    raise RuntimeError("LiSE requires Python 3.6 or later")

from setuptools import setup


setup(
    name="LiSE",
    version="0.11",
    description="Rules engine for life simulation games",
    author="Zachary Spector",
    author_email="public@zacharyspector.com",
    license="AGPL3+",
    keywords="game simulation",
    url="https://github.com/LogicalDash/LiSE",
    packages=[
        "LiSE",
        "LiSE.server",
        "LiSE.examples",
        "LiSE.allegedb"
    ],
    package_data={
        'LiSE': ['sqlite.json']
    },
    install_requires=[
        "astunparse>=1.6.3<2",
        "msgpack>=1.0.0<1.1",
        "blinker",
        "networkx==2.4"
    ],
    project_urls={
        "Documentation": "https://logicaldash.github.io/LiSE"
    }
)
