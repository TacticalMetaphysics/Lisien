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

outpath = os.path.join(
	os.path.abspath(os.path.dirname(__file__)), "college24_premade.tar.xz"
)
if os.path.exists(outpath):
	os.remove(outpath)
with tempfile.TemporaryDirectory() as directory:
	with Engine(
		directory,
		workers=0,
		keep_rules_journal=False,
		commit_interval=1,
		connect_string=f"sqlite:///{directory}/world.sqlite3",
	) as eng:
		install(eng)
		for i in range(24):
			print(i)
			eng.next_turn()
		print("Done simulating.")
	print("Compressing...")
	shutil.make_archive(outpath[:-7], "xztar", directory, ".")
print("All done")
