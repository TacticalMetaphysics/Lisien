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
import shutil
import tempfile

from lisien.engine import Engine
from lisien.examples.college import install


def main():
	outpath = os.path.join(
		os.path.abspath(os.path.dirname(__file__)),
		"data",
		"college{}.lisien",
	)
	if os.path.exists(outpath):
		os.remove(outpath)
	with Engine(
		None,
		workers=0,
		keyframe_interval=1,
		keep_rules_journal=False,
		random_seed=69105,
	) as eng:
		install(eng)
		for i in range(10):
			print(i)
			eng.next_turn()
		exto = outpath.format("10")
		print("Done simulating the 10 turn case. Exporting to " + exto)
		eng.export(path=exto)
		for i in range(10, 24):
			print(i)
			eng.next_turn()
		exto = outpath.format("24")
		print("Done simulating 24 turns. Exporting to " + exto)
		eng.export(path=exto)
	print("All done")


if __name__ == "__main__":
	main()
