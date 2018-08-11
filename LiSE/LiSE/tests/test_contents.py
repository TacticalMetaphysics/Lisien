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
from LiSE.engine import Engine


@pytest.fixture(scope='function')
def chara():
    with Engine(":memory:") as eng:
        yield eng.new_character('chara')


def test_many_things_in_place(chara):
    place = chara.new_place(0)
    things = [place.new_thing(i) for i in range(1, 10)]
    for thing in things:
        assert thing in place.contents()
    for that in place.content:
        assert place.content[that].location == place
    things.sort(key=lambda th: th.name)
    contents = sorted(place.contents(), key=lambda th: th.name)
    assert things == contents


def test_many_things_in_portal(chara):
    chara.add_place(0)
    chara.add_place(1)
    port = chara.new_portal(0, 1)
    things = []
    for i in range(2, 10):
        th = chara.new_thing(i, location=0, next_location=1)
        things.append(th)
    for thing in things:
        assert thing in port.contents()
        assert thing.name in port.content
    things.sort(key=lambda th: th.name)
    contents = sorted(port.contents(), key=lambda th: th.name)
    assert things == contents