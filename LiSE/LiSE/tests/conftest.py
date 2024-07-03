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
from LiSE import Engine
import pytest
import tempfile


@pytest.fixture(scope="function")
def tempdir():
	with tempfile.TemporaryDirectory() as d:
		yield d


@pytest.fixture(scope="function")
def engy(tempdir):
	with Engine(tempdir, random_seed=69105, enforce_end_of_time=False) as eng:
		yield eng
