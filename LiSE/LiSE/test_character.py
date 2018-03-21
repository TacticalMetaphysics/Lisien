import pytest
import allegedb.test
from .engine import Engine


class CharacterTest(allegedb.test.AllegedTest):
    def setUp(self):
        self.engine = Engine("sqlite:///:memory:")
        self.graphmakers = (self.engine.new_character,)

    def tearDown(self):
        self.engine.close()


class CharacterDictStorageTest(CharacterTest, allegedb.test.DictStorageTest):
    pass


class CharacterListStorageTest(CharacterTest, allegedb.test.ListStorageTest):
    pass


class CharacterSetStorageTest(CharacterTest, allegedb.test.SetStorageTest):
    pass


def set_in_mapping(mapp, stat, v):
    # Mutate the stuff in-place instead of simply replacing it,
    # because this could trigger side effects
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
    end_stats = {'name': char.name}
    for stat, v in stat:
        set_in_mapping(char.stat, stat, v)
        if v is None and stat in end_stats:
            del end_stats[stat]
        else:
            end_stats[stat] = v
    end_places = {}
    end_things = {}
    for node, v in node:
        if v is None:
            del char.node[node]
            if node in end_places:
                del end_places[node]
            if node in end_things:
                del end_things[node]
        elif node in char.place:
            me = end_places[node] = dict(char.place[node])
            if 'location' in v:
                del char.place[node]
                del end_places[node]
                char.thing[node] = me
                continue
            me.update(v)
            for k, vv in v.items():
                set_in_mapping(char.place[node], k, vv)
        elif node in char.thing:
            me = end_things[node] = dict(char.thing[node])
            if 'location' in v and v['location'] is None:
                del char.thing[node]
                del end_things[node]
                char.place[node] = me
                continue
            me.update(v)
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
    end_edges = {}
    for o, d, v in portal:
        if v is None:
            del char.edge[o][d]
        else:
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
@pytest.fixture(params=[
    ('empty', {}, {}, [], [], [], []),
    ('small',
     {0: [1], 1: [0], 'kobold': []},
     {'spam': 'eggs', 'ham': {'baked beans': 'delicious'}, 'qux': ['quux', 'quuux'],
                             'clothes': {'hats', 'shirts', 'pants'}},
     [('kobold', {'evil': True}), (0, {'evil': False}), (1, {'evil': False})],
     [('spam', None), ('qux', ['quux']), ('clothes', 'no')],
     [(2, {'evil': False}), ('kobold', {'evil': False})],
     [(0, 1, None), (0, 2, {'hi': 'hello'})]
     )
])
def character_updates(request):
    name, data, stat, nodestat, statup, nodeup, edgeup = request.param
    engine = Engine("sqlite:///:memory:")
    char = engine.new_character(name, data, **stat)
    update_char(char, node=nodestat)
    yield char, statup, nodeup, edgeup
    engine.close()


def test_facade(character_updates):
    character, statup, nodeup, edgeup = character_updates
    start_stat = dict(character.stat)
    start_place = dict(character.place)
    start_thing = dict(character.thing)
    start_edge = {}
    for o in character.edge:
        for d in character.edge[o]:
            start_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
    facade = character.facade()
    updated = update_char(
        facade, stat=statup, node=nodeup, portal=edgeup)
    assert facade.stat == updated['stat']
    assert facade.place == updated['place']
    assert facade.thing == updated['thing']
    assert facade.portal == updated['portal']
    # changes to a facade should not impact the underlying character
    assert start_stat == dict(character.stat)
    assert start_place == dict(character.place)
    assert start_thing == dict(character.place)
    end_edge = {}
    for o in character.edge:
        for d in character.edge[o]:
            end_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
    assert start_edge == end_edge