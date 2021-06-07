# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
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
import sys
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] < 6
):
    raise RuntimeError("ELiDE requires Python 3.6 or later")
import os
from setuptools import setup


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt'), 'rt') as inf:
    reqs = list(inf.readlines())


setup(
    name="ELiDE",
    version="0.11.2",
    license="AGPL3",
    packages=[
        "ELiDE",
        "ELiDE.graph",
        "ELiDE.grid",
        "ELiDE.kivygarden.texturestack"
    ],
    package_dir={
        'ELiDE.kivygarden.texturestack':
        'ELiDE/kivygarden/texturestack'
    },
    install_requires=reqs,
    package_data={
        "ELiDE": [
            "assets/*.png",
            "assets/*.jpg",
            "assets/*.ttf",
            "assets/*.atlas",
            "assets/rltiles/*"
        ]
    },
    zip_safe=False
)
