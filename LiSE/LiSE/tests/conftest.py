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
import os
import shutil
import tempfile

import pytest

from LiSE import Engine
from ..examples import kobold


@pytest.fixture(scope="function")
def handle(tempdir):
	from LiSE.handle import EngineHandle

	hand = EngineHandle(
		tempdir, connect_string="sqlite:///:memory:", random_seed=69105
	)
	yield hand
	hand.close()


@pytest.fixture(
	scope="function",
	params=[
		lambda eng: kobold.inittest(
			eng, shrubberies=20, kobold_sprint_chance=0.9
		),
		# college.install,
		# sickle.install
	],
)
def handle_initialized(request, handle):
	with handle._real.advancing():
		request.param(handle._real)
	yield handle


@pytest.fixture(scope="function")
def tempdir():
	with tempfile.TemporaryDirectory() as d:
		yield d


@pytest.fixture(scope="function")
def engy(tempdir):
	with Engine(tempdir, random_seed=69105, enforce_end_of_time=False) as eng:
		yield eng


@pytest.fixture(scope="module")
def college24_premade():
	directory = tempfile.mkdtemp(".")
	shutil.unpack_archive(
		os.path.join(
			os.path.abspath(os.path.dirname(__file__)),
			"college24_premade.tar.xz",
		),
		directory,
	)
	with Engine(directory) as eng:
		yield eng
	shutil.rmtree(directory)
