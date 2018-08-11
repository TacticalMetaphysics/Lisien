# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
from setuptools import setup
setup(
    name = "allegedb",
    version = "0.13.0",
    packages = ["allegedb"],
    install_requires = ['networkx>=1.9', 'blinker'],
    author = "Zachary Spector",
    author_email = "zacharyspector@gmail.com",
    description = "An object-relational mapper serving database-backed versions of the standard networkx graph classes.",
    license = "AGPL3+",
    keywords = "orm graph networkx sql database",
    url = "https://github.com/LogicalDash/LiSE",
    package_dir={
      "allegedb": "allegedb"
    },
    package_data={
        "allegedb": [
            "sqlite.json"
        ]
    }
)
