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


@pytest.fixture(scope='function')
def chara(engy):
    yield engy.new_character('chara')


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