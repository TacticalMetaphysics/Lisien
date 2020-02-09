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
from LiSE.exc import AmbiguousUserError


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
    fust = someplace.new_thing(11)
    assert fust not in someplace.contents()
    with pytest.raises(KeyError):
        someplace.content[11]


def test_portal(someplace):
    assert not someplace.portal
    assert 'there' not in someplace.portal
    there = someplace.character.new_place('there')
    assert not there.preportal
    assert 'someplace' not in there.preportal
    someplace.character.new_portal('someplace', 'there')
    assert someplace.portal
    assert 'there' in someplace.portal
    assert 'there' not in someplace.preportal
    assert there.preportal
    assert 'someplace' in there.preportal
    assert 'someplace' not in there.portal
    someplace.character.remove_edge('someplace', 'there')
    assert 'there' not in someplace.portal
    assert 'someplace' not in there.preportal


def test_user(someplace):
    with pytest.raises(AmbiguousUserError):
        someplace.user
    someone = someplace.engine.new_character('someone')
    someone.add_avatar(someplace)
    assert someplace.user is someone
    assert 'someone' in someplace.users
    assert someplace.users['someone'] is someone
    noone = someplace.engine.new_character('noone')
    assert 'noone' not in someplace.users
    noone.add_avatar(someplace)
    with pytest.raises(AmbiguousUserError):
        someplace.user
    assert 'noone' in someplace.users
    assert someplace.users['noone'] is noone