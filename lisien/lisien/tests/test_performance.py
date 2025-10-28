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
from time import monotonic

import networkx as nx
import pytest

from lisien import Engine
from lisien.proxy.manager import EngineProxyManager


@pytest.mark.parquetdb
def test_follow_path(tmp_path):
	big_grid = nx.grid_2d_graph(100, 100)
	big_grid.add_node("them", location=(0, 0))
	straightly = nx.shortest_path(big_grid, (0, 0), (99, 99))
	with Engine(tmp_path) as eng:
		eng.add_character("grid", big_grid)
	with EngineProxyManager(tmp_path, workers=0) as prox:
		them = prox.character["grid"].thing["them"]
		start = monotonic()
		them.follow_path(straightly)
		elapsed = monotonic() - start
		assert elapsed < 20, (
			f"Took too long to follow a path of length {len(straightly)}: {elapsed:.2} seconds"
		)
