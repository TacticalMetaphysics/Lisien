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
from time import monotonic

import pytest

from lisien.proxy.manager import EngineProxyManager

from .data import DATA_DIR


@pytest.fixture
def big_grid(tmp_path):
	shutil.unpack_archive(os.path.join(DATA_DIR, "pathfind.tar.gz"), tmp_path)
	yield tmp_path


@pytest.mark.parquetdb
def test_follow_path(big_grid):
	with EngineProxyManager(big_grid, workers=0) as prox:
		grid = prox.character["grid"]
		them = grid.thing["them"]
		straightly = grid.stat["straightly"]
		start = monotonic()
		them.follow_path(straightly)
		elapsed = monotonic() - start
		assert elapsed < 20, (
			f"Took too long to follow a path of length {len(straightly)}: {elapsed:.2} seconds"
		)
