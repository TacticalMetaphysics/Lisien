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
import os
import tempfile
import pytest
import allegedb.tests.test_all
from LiSE.engine import Engine


class CharacterTest(allegedb.tests.test_all.AllegedTest):
    def setUp(self):
        self.engine = Engine("sqlite:///:memory:")
        self.graphmakers = (self.engine.new_character,)
        self.tempdir = tempfile.mkdtemp(dir='.')
        for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
        ):
            if os.path.exists(f):
                os.rename(f, os.path.join(self.tempdir, f))

    def tearDown(self):
        self.engine.close()
        for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
        ):
            if os.path.exists(f):
                os.remove(f)
            if os.path.exists(os.path.join(self.tempdir, f)):
                os.rename(os.path.join(self.tempdir, f), f)
        os.rmdir(self.tempdir)


class CharacterDictStorageTest(CharacterTest, allegedb.tests.test_all.DictStorageTest):
    pass


class CharacterListStorageTest(CharacterTest, allegedb.tests.test_all.ListStorageTest):
    pass


class CharacterSetStorageTest(CharacterTest, allegedb.tests.test_all.SetStorageTest):
    pass


def set_in_mapping(mapp, stat, v):
    """Sync a value in ``mapp``, having key ``stat``, with ``v``."""
    # Mutate the stuff in-place instead of simply replacing it,
    # because this could trigger side effects
    if stat == 'name':
        return
    if v is None:
        del mapp[stat]
        return
    if stat not in mapp:
        mapp[stat] = v
        return
    if isinstance(v, dict) or isinstance(v, set):
        mapp[stat].update(v)
        for item in list(mapp[stat]):
            if item not in v:
                try:
                    del mapp[stat][item]
                except TypeError:
                    mapp[stat].remove(item)
    elif isinstance(v, list):
        for item in list(mapp[stat]):
            if item not in v:
                mapp[stat].remove(item)
        for i, item in enumerate(v):
            if mapp[stat][i] != item:
                mapp[stat].insert(i, item)
    else:
        mapp[stat] = v


def update_char(char, *, stat=(), node=(), portal=()):
    """Make a bunch of changes to a character-like object"""
    def update(d, dd):
        for k, v in dd.items():
            if v is None and k in d:
                del d[k]
            else:
                d[k] = v
    end_stats = char.stat.unwrap()
    for stat, v in stat:
        set_in_mapping(char.stat, stat, v)
        if v is None and stat in end_stats:
            del end_stats[stat]
        else:
            end_stats[stat] = v
    end_places = char.place.unwrap()
    end_things = char.thing.unwrap()
    for node, v in node:
        if 'name' not in v:
            v['name'] = node
        if v is None:
            del char.node[node]
            if node in end_places:
                del end_places[node]
            if node in end_things:
                del end_things[node]
        elif node in char.place:
            if 'location' in v:
                del end_places[node]
                char.place2thing(node, v.pop('location'))
                if node in end_places:
                    me = end_things[node] = end_places.pop(node)
                else:
                    me = end_things[node] = dict(char.thing[node])
                update(me, v)
                for k, vv in v.items():
                    set_in_mapping(char.thing[node], k, vv)
            else:
                if node in end_places:
                    me = end_places[node]
                else:
                    me = end_places[node] = dict(char.place[node])
                update(me, v)
                for k, vv in v.items():
                    set_in_mapping(char.place[node], k, vv)
        elif node in char.thing:
            if 'location' in v and v['location'] is None:
                if node in end_things:
                    me = end_places[node] = end_things.pop(node)
                else:
                    me = end_places[node] = dict(char.thing[node])
                del me['location']
                del v['location']
                char.thing2place(node)
                update(me, v)
                for k, vv in v.items():
                    set_in_mapping(char.place[node], k, vv)
            else:
                me = end_things[node] = dict(char.thing[node])
                update(me, v)
                for k, vv in v.items():
                    set_in_mapping(char.thing[node], k, vv)
        elif 'location' in v:
            end_things[node] = v
            me = char.new_thing(node, v.pop('location'))
            for k, vv in v.items():
                set_in_mapping(me, k, vv)
        else:
            end_places[node] = v
            me = char.new_node(node)
            for k, vv in v.items():
                set_in_mapping(me, k, vv)
    end_edges = char.portal.unwrap()
    for o, d, v in portal:
        if v is None:
            del char.edge[o][d]
            del end_edges[o][d]
        else:
            me = end_edges.setdefault(o, {}).setdefault(d, {})
            update(me, v)
            e = char.new_portal(o, d)
            for k, vv in v.items():
                set_in_mapping(e, k, vv)
    return {
        'stat': end_stats,
        'place': end_places,
        'thing': end_things,
        'portal': end_edges
    }


# TODO parametrize bunch of characters
@pytest.fixture(scope="function", params=[
    ('empty', {}, {}, [], [], [], []),
    ('small',
     {0: [1], 1: [0], 'kobold': []},
     {'spam': 'eggs', 'ham': {'baked beans': 'delicious'}, 'qux': ['quux', 'quuux'],
                             'clothes': {'hats', 'shirts', 'pants'}},
     [('kobold', {'location': 0, 'evil': True}), (0, {'evil': False}), (1, {'evil': False})],
     [('spam', None), ('qux', ['quux']), ('clothes', 'no')],
     [(2, {'evil': False}), ('kobold', {'evil': False})],
     [(0, 1, None), (0, 2, {'hi': 'hello'})]
     )
])
def character_updates(request):
    tempdir = tempfile.mkdtemp(dir='.')
    for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
    ):
        if os.path.exists(f):
            os.rename(f, os.path.join(tempdir, f))
    name, data, stat, nodestat, statup, nodeup, edgeup = request.param
    engine = Engine("sqlite:///:memory:")
    char = engine.new_character(name, data, **stat)
    update_char(char, node=nodestat)
    yield char, statup, nodeup, edgeup
    engine.close()
    for f in (
        'trigger.py', 'prereq.py', 'action.py', 'function.py',
        'method.py', 'strings.json'
    ):
        if os.path.exists(f):
            os.remove(f)
        if os.path.exists(os.path.join(tempdir, f)):
            os.rename(os.path.join(tempdir, f), f)
    os.rmdir(tempdir)


def test_facade(character_updates):
    """Make sure you can alter a facade independent of the character it's from"""
    character, statup, nodeup, edgeup = character_updates
    start_stat = character.stat.unwrap()
    start_place = character.place.unwrap()
    start_thing = character.thing.unwrap()
    start_edge = {}
    for o in character.edge:
        for d in character.edge[o]:
            start_edge.setdefault(o, {})[d] = character.edge[o][d].unwrap()
    facade = character.facade()
    updated = update_char(
        facade, stat=statup, node=nodeup, portal=edgeup)
    assert facade.stat == updated['stat']
    assert facade.place == updated['place']
    assert facade.thing == updated['thing']
    assert facade.portal == updated['portal']
    # changes to a facade should not impact the underlying character
    assert start_stat == character.stat.unwrap()
    assert start_place == character.place.unwrap()
    assert start_thing == character.thing.unwrap()
    end_edge = {}
    for o in character.edge:
        for d in character.edge[o]:
            end_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
    assert start_edge == end_edge