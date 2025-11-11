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
import pytest

from lisien import Engine
from lisien.examples.polygons import install

from .util import make_test_engine_kwargs


# TODO: use a test sim that does everything in every cache
@pytest.mark.big
def test_resume(tmp_path, persistent_database, random_seed):
	ekwargs = make_test_engine_kwargs(
		tmp_path, "serial", persistent_database, random_seed
	)
	ekwargs["keyframe_on_close"] = False
	with Engine(**ekwargs) as eng:
		install(eng)
		eng.next_turn()
		last_branch, last_turn, last_tick = eng.time
	with Engine(**ekwargs) as eng:
		assert eng.time == (last_branch, last_turn, last_tick)
		curturn = eng.turn
		eng.next_turn()
		assert eng.turn == curturn + 1
