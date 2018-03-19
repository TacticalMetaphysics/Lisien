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


def update_char(char, *, stat=(), place=(), portal=(), thing=()):
    end_stats = {'name': char.name}
    for stat, v in stat:
        set_in_mapping(char.stat, stat, v)
        if v is None and stat in end_stats:
            del end_stats[stat]
        else:
            end_stats[stat] = v
    end_places = {}
    for node, v in place:
        if v is None:
            del char.node[node]
            if node in end_places:
                del end_places[node]
        else:
            end_places[node] = v
            me = char.new_node(node)
            for k, vv in v.items():
                set_in_mapping(me, k, vv)
    end_things = {}
    for thing, v in thing:
        if v is None:
            del char.thing[thing]
        else:
            end_things[thing] = dict(v)
            loc = v.pop('location')
            nxtloc = v.pop('next_location', None)
            me = char.new_thing(thing, loc, nxtloc)
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
@pytest.fixture(params=[('empty', {}, [], [], [], [])])
def character_updates(request):
    name, data, statup, placeup, thingup, edgeup = request.param
    engine = Engine("sqlite:///:memory:")
    yield engine.new_character(name, **data), statup, placeup, thingup, edgeup
    engine.close()


def test_facade(character_updates):
    character, statup, placeup, thingup, edgeup = character_updates
    start_stat = dict(character.stat)
    start_place = dict(character.place)
    start_thing = dict(character.thing)
    start_edge = {}
    for o in character.edge:
        for d in character.edge[o]:
            start_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
    facade = character.facade()
    updated = update_char(
        facade, stat=statup, place=placeup, thing=thingup, portal=edgeup)
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