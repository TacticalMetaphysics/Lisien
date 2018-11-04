# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
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
import pytest


@pytest.fixture(scope="function")
def someplace(engy):
    yield engy.new_character('physical').new_place('someplace')


def test_contents(someplace):
    stuff = [someplace.new_thing(i) for i in range(10)]
    assert len(someplace.content) == 10
    for i in range(10):
        assert i in someplace.content
        assert someplace.content[i] == stuff[i]
    for that in stuff:
        assert that in someplace.contents()