# This file is part of Lisien, a framework for life simulation games.
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

from lisien.engine import Engine
from lisien.examples.sickle import install


def main():
	outpath = os.path.join(
		os.path.abspath(os.path.dirname(__file__)), "data", "sickle.lisien"
	)
	if os.path.exists(outpath):
		os.remove(outpath)
	with Engine(
		None,
		workers=0,
		keyframe_interval=None,
		keep_rules_journal=False,
		random_seed=69105,
	) as eng:
		install(eng)
		print("Installed. Exporting to", outpath)
		eng.export(path=outpath)
	print("All done")


if __name__ == "__main__":
	main()
