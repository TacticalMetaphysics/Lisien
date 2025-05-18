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
import sys

from elide.app import ElideApp


def elide():
	kwargs = {}
	if os.path.isdir(sys.argv[-1]):
		kwargs["prefix"] = sys.argv[-1]
		kwargs["immediate_start"] = True
	app = ElideApp(**kwargs)
	app.run()


if __name__ == "__main__":
	elide()
