from collections import Callable, MutableMapping, defaultdict
import os
from multiprocessing import Process, Queue, Lock
from multiprocessing.managers import BaseManager
from LiSE.core import Engine
from LiSE.util import dispatch, listen, listener


class EngineHandle(object):
    def reify(self, args):
        self._real = Engine(*args)

    def time_locked(self):
        return hasattr(self._real, 'locktime')

    def advance(self):
        print('advance')
        self._real.advance()

    def next_tick(self):
        print('next_tick')
        self._real.next_tick()

    def add_character(self, name, data, kwargs):
        print('add_character')
        self._real.add_character(name, data, **kwargs)

    def commit(self):
        print('commit')
        self._real.commit()

    def close(self):
        print('close')
        self._real.close()

    def get_branch(self):
        print('get_branch')
        return self._real.branch

    def set_branch(self, v):
        print('set_branch')
        self._real.branch = v

    def get_tick(self):
        print('get_tick')
        return self._real.tick

    def set_tick(self, v):
        print('set_tick')
        self._real.tick = v

    def get_time(self):
        print('get_time')
        return self._real.time

    def set_time(self, v):
        print('set_time')
        self._real.time = v

    def get_language(self):
        print('get_language')
        return self._real.string.language

    def set_language(self, v):
        print('set_language')
        self._real.string.language = v

    def get_string_ids(self):
        print('get_string_ids')
        return list(self._real.string)

    def count_strings(self):
        print('count_strings')
        return len(self._real.string)

    def get_string(self, k):
        print('get_string')
        return self._real.string[k]

    def set_string(self, k, v):
        print('set_string')
        self._real.string[k] = v

    def del_string(self, k):
        print('del_string')
        del self._real.string[k]

    def get_eternal(self, k):
        print('get_eternal')
        return self._real.eternal[k]

    def set_eternal(self, k, v):
        print('set_eternal')
        self._real.eternal[k] = v

    def del_eternal(self, k):
        print('del_eternal')
        del self._real.eternal[k]

    def eternal_keys(self):
        print('eternal_keys')
        return list(self._real.eternal.keys())

    def eternal_len(self):
        print('eternal_len')
        return len(self._real.eternal)

    def have_eternal(self, k):
        return k in self._real.eternal

    def get_universal(self, k):
        print('get_universal')
        return self._real.universal[k]

    def set_universal(self, k, v):
        print('set_universal')
        self._real.universal[k] = v

    def del_universal(self, k):
        print('del_universal')
        del self._real.universal[k]

    def universal_keys(self):
        print('universal_keys')
        return list(self._real.universal.keys())

    def universal_len(self):
        print('universal_len')
        return len(self._real.universal)

    def init_character(self, char, statdict={}):
        print('init_character')
        if char in self._real.character:
            raise KeyError("Already have character {}".format(char))
        self._real.character[char] = {}
        self._real.character[char].stat.update(statdict)

    def del_character(self, char):
        print('del_character')
        del self._real.character[char]

    def get_character_stat(self, char, k):
        print('get_character_stat')
        return self._real.character[char].stat[k]

    def set_character_stat(self, char, k, v):
        print('set_character_stat')
        self._real.character[char].stat[k] = v

    def del_character_stat(self, char, k):
        print('del_character_stat')
        del self._real.character[char].stat[k]

    def character_stats(self, char):
        print('character_stats')
        return list(self._real.character[char].stat.keys())

    def character_stats_len(self, char):
        print('character_stats_len')
        return len(self._real.character[char].stat)

    def characters(self):
        print('characters')
        return list(self._real.character.keys())

    def characters_len(self):
        print('characters_len')
        return len(self._real.character)

    def have_character(self, char):
        print('have_character')
        return char in self._real.character

    def set_character(self, char, v):
        print('set_character')
        self._real.character[char] = v

    def get_node_stat(self, char, node, k):
        print('get_node_stat')
        return self._real.character[char].node[node][k]

    def set_node_stat(self, char, node, k, v):
        print('set_node_stat')
        self._real.character[char].node[node][k] = v

    def del_node_stat(self, char, node, k):
        print('del_node_stat')
        del self._real.character[char].node[node][k]

    def node_stat_keys(self, char, node):
        print('node_stat_keys')
        return list(self._real.character[char].node[node])

    def node_stat_len(self, char, node):
        print('node_stat_len')
        return len(self._real.character[char].node[node])

    def del_node(self, char, node):
        print('del_node')
        del self._real.character[char].node[node]

    def character_things(self, char):
        print('character_things')
        return list(self._real.character[char].thing)

    def character_things_len(self, char):
        print('character_things_len')
        return len(self._real.character[char].thing)

    def character_has_thing(self, char, thing):
        print('character_has_thing')
        return thing in self._real.character[char].thing

    def character_places(self, char):
        print('character_places')
        return list(self._real.character[char].place)

    def character_places_len(self, char):
        print('character_places_len')
        return len(self._real.character[char].place)

    def character_has_place(self, char, place):
        print('character_has_place')
        return place in self._real.character[char].place

    def character_nodes(self, char):
        print('character_nodes')
        return list(self._real.character[char].node.keys())

    def character_predecessor_nodes(self, char):
        print('character_predecessor_nodes')
        return list(self._real.character[char].adj.keys())

    def node_has_predecessor(self, char, node):
        print('node_has_predecessor')
        return node in self._real.character[char].pred.keys()

    def node_predecessors_len(self, char, node):
        print('node_predecessors_len')
        return len(self._real.character[char].pred[node])

    def node_predecessors(self, char, node):
        print('node_predecessors')
        return list(self._real.character[char].pred[node].keys())

    def node_precedes(self, char, nodeB, nodeA):
        print('node_precedes')
        return nodeA in self._real.character[char].pred[nodeB]

    def character_nodes_with_predecessors(self, char):
        print('character_nodes_with_predecessors')
        return list(self._real.character[char].pred.keys())

    def character_nodes_with_predecessors_len(self, char):
        print('character_nodes_with_predecessors_len')
        return len(self._real.character[char].pred)

    def character_set_node_predecessors(self, char, node, preds):
        print('character_set_node_predecessors')
        self._real.character[char].pred[node] = preds

    def character_del_node_predecessors(self, char, node):
        print('character_del_node_predecessors')
        del self._real.character[char].pred[node]

    def character_nodes_len(self, char):
        print('character_nodes_len')
        return len(self._real.character[char].node)

    def character_has_node(self, char, node):
        print('character_has_node')
        return node in self._real.character[char].node

    def character_node_successors(self, char, node):
        print('character_node_successors')
        return list(self._real.character[char].adj[node].keys())

    def character_node_successors_len(self, char, node):
        print('character_node_successors_len')
        return len(self._real.character[char].adj[node])

    def character_set_node_successors(self, char, node, val):
        print('character_set_node_successors')
        self._real.character[char].adj[node] = val

    def character_del_node_successors(self, char, node):
        print('character_del_node_successors')
        del self._real.character[char].adj[node]

    def character_nodes_connected(self, char, nodeA, nodeB):
        print('character_nodes_connected')
        return nodeB in self._real.character[char].node[nodeA]

    def character_len_node_successors(self, char, nodeA):
        print('character_len_node_successors')
        return len(self._real.character[char].node[nodeA])

    def init_thing(self, char, thing, statdict={}):
        print('init_thing')
        if thing in self._real.character[char].thing:
            raise KeyError(
                'Already have thing in character {}: {}'.format(
                    char, thing
                )
            )
        self.set_thing(char, thing, statdict)

    def set_thing(self, char, thing, statdict):
        print('set_thing')
        self._real.character[char].thing[thing] = statdict

    def add_thing(self, char, thing, location, next_location, statdict):
        print('add_thing')
        self._real.character[char].add_thing(
            thing, location, next_location, **statdict
        )

    def place2thing(self, char, name, location, next_location=None):
        print('place2thing')
        self._real.character[char].place2thing(name, location, next_location)

    def thing2place(self, char, name):
        print('thing2place')
        self._real.character[char].thing2place(name)

    def add_things_from(self, char, seq):
        print('add_things_from')
        self._real.character[char].add_things_from(seq)

    def get_thing_location(self, char, th):
        print('get_thing_location')
        return self._real.character[char].thing[th]['location']

    def set_thing_location(self, char, th, loc):
        print('set_thing_location')
        self._real.character[char].thing[th]['location'] = loc

    def get_thing_next_location(self, char, th):
        print('get_thing_next_location')
        return self._real.character[char].thing[th]['next_location']

    def set_thing_next_location(self, char, th, loc):
        print('set_thing_next_location')
        self._real.character[char].thing[th]['next_location'] = loc

    def thing_follow_path(self, char, th, path, weight):
        print('thing_follow_path')
        self._real.character[char].thing[th].follow_path(path, weight)

    def thing_go_to_place(self, char, th, place, weight):
        print('thing_go_to_place')
        self._real.character[char].thing[th].go_to_place(place, weight)

    def thing_travel_to(self, char, th, dest, weight, graph):
        print('thing_travel_to')
        self._real.character[char].thing[th].travel_to(dest, weight, graph)

    def thing_travel_to_by(self, char, th, dest, arrival_tick, weight, graph):
        print('thing_travel_to_by')
        self._real.character[char].thing[th].travel_to_by(
            dest, arrival_tick, weight, graph
        )

    def init_place(self, char, place, statdict={}):
        print('init_place')
        if place in self._real.character[char].place:
            raise KeyError(
                'Already have place in character {}: {}'.format(
                    char, place
                )
            )
        self.set_place(char, place, statdict)

    def set_place(self, char, place, statdict):
        print('set_place')
        self._real.character[char].place[place] = statdict

    def add_places_from(self, char, seq):
        print('add_places_from')
        self._real.character[char].add_places_from(seq)

    def init_portal(self, char, o, d, statdict={}):
        print('init_portal')
        if (
                o in self._real.character[char].portal and
                d in self._real.character[char].portal[o]
        ):
            raise KeyError(
                'Already have portal in character {}: {}->{}'.format(
                    char, o, d
                )
            )
        self.set_portal(char, o, d, statdict)

    def set_portal(self, char, o, d, statdict):
        print('set_portal')
        self._real.character[char].portal[o][d] = statdict

    def character_portals(self, char):
        print('character_portals')
        return list(self._real.character[char].portals())

    def add_portal(self, char, o, d, symmetrical, statdict):
        print('add_portal')
        self._real.character[char].add_portal(o, d, symmetrical, **statdict)

    def add_portals_from(self, char, seq, symmetrical):
        print('add_portals_from')
        self._real.character[char].add_portals_from(seq, symmetrical)

    def del_portal(self, char, o, d):
        print('del_portal')
        del self._real.character[char].portal[o][d]

    def get_portal_stat(self, char, o, d, k):
        print('get_portal_stat')
        return self._real.character[char].portal[o][d][k]

    def set_portal_stat(self, char, o, d, k, v):
        print('set_portal_stat')
        self._real.character[char].portal[o][d][k] = v

    def del_portal_stat(self, char, o, d, k):
        print('del_portal_stat')
        del self._real.character[char][o][d][k]

    def portal_stats(self, char, o, d):
        print('portal_stats')
        return list(self._real.character[char][o][d].keys())

    def len_portal_stats(self, char, o, d):
        print('len_portal_stats')
        return len(self._real.character[char][o][d])

    def character_avatars(self, char):
        print('character_avatars')
        return list(self._real.character[char].avatars())

    def add_avatar(self, char, a, b):
        print('add_avatar')
        self._real.character[char].add_avatar(a, b)

    def del_avatar(self, char, a, b):
        print('del_avatar')
        self._real.character[char].del_avatar(a, b)

    def get_rule_actions(self, rule):
        print('get_rule_actions')
        return self._real.rule.db.rule_actions(rule)

    def set_rule_actions(self, rule, l):
        print('set_rule_actions')
        self._real.rule.db.set_rule_actions(rule, l)

    def get_rule_triggers(self, rule):
        print('get_rule_triggers')
        return self._real.rule.db.rule_triggers(rule)

    def set_rule_triggers(self, rule, l):
        print('set_rule_triggers')
        self._real.rule.db.set_rule_triggers(rule, l)

    def get_rule_prereqs(self, rule):
        print('get_rule_prereqs')
        return self._real.rule.db.rule_prereqs(rule)

    def set_rule_prereqs(self, rule, l):
        print('set_rule_prereqs')
        self._real.rule.db.set_rule_prereqs(rule, l)

    def get_rulebook_rules(self, rulebook):
        print('get_rulebook_rules')
        return self._real.rule.db.rulebook_rules(rulebook)


class EngineManager(BaseManager):
    pass


EngineManager.register(
    'EngineHandle',
    EngineHandle
)
EngineManager.register('Queue', Queue)
EngineManager.register('Lock', Lock)


class NodeProxy(MutableMapping):
    @property
    def character(self):
        return CharacterProxy(self._engine, self._charname)

    def __init__(self, engine_proxy, charname, nodename):
        self._engine = engine_proxy
        self._charname = charname
        self._name = nodename

    def __iter__(self):
        yield from self._engine.node_stat_keys(self._charname, self._name)

    def __len__(self):
        return self._engine.node_stat_len(self._charname, self._name)

    def __getitem__(self, k):
        return self._engine.get_node_stat(self._charname, self._name, k)

    def __setitem__(self, k, v):
        self._engine.set_node_stat(self._charname, self._name, k, v)

    def __delitem__(self, k):
        self._engine.del_node_stat(self._charname, self._name, k)


class PlaceProxy(NodeProxy):
    pass


class ThingProxy(NodeProxy):
    @property
    def location(self):
        ln = self['location']
        if ln in self._engine.character_things(self._charname):
            return ThingProxy(self._engine, self._charname, ln)
        else:
            return PlaceProxy(self._engine, self._charname, ln)

    @location.setter
    def location(self, v):
        self._engine.set_thing_location(self._charname, self._name, v._name)

    @property
    def next_location(self):
        ln = self['next_location']
        if ln in self._engine.character_things(self._charname):
            return ThingProxy(self._engine, self._charname, ln)
        else:
            return PlaceProxy(self._engine, self._charname, ln)

    @next_location.setter
    def next_location(self, v):
        self._engine.set_thing_next_location(
            self._charname, self._name, v._name
        )

    def follow_path(self, path, weight=None):
        self._engine.thing_follow_path(
            self._charname, self._name, path, weight
        )

    def go_to_place(self, place, weight=''):
        self._engine.thing_go_to_place(
            self._charname, self._name, place, weight
        )

    def travel_to(self, dest, weight=None, graph=None):
        self._engine.thing_travel_to(
            self._charname, self._name, dest, weight, graph
        )

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        self._engine.thing_travel_to_by(
            self._charname, self._name, dest, arrival_tick, weight, graph
        )


class PortalProxy(MutableMapping):
    def __init__(self, engine_proxy, charname, nodeAname, nodeBname):
        self._engine = engine_proxy
        self._charname = charname
        self._nodeA = nodeAname
        self._nodeB = nodeBname

    def __iter__(self):
        yield from self._engine.portal_stats(
            self._charname, self._nodeA, self._nodeB
        )

    def __len__(self):
        return self._engine.len_portal_stats(
            self._charname, self._nodeA, self._nodeB
        )

    def __getitem__(self, k):
        return self._engine.get_portal_stat(
            self._charname, self._nodeA, self._nodeB, k
        )

    def __setitem__(self, k, v):
        self._engine.set_portal_stat(
            self._charname, self._nodeA, self._nodeB, k, v
        )

    def __delitem__(self, k):
        self._engine.del_portal_stat(
            self._charname, self._nodeA, self._nodeB, k
        )


class NodeMapProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._charname = charname

    def __iter__(self):
        yield from self._engine.character_nodes(self._charname)

    def __len__(self):
        return self._engine.character_nodes_len(self._charname)

    def __getitem__(self, k):
        if not self._engine.character_has_node(self._name, k):
            raise KeyError("No such node: {}".format(k))
        if self._engine.character_has_thing(self._name, k):
            return ThingProxy(self._name, k)
        else:
            return PlaceProxy(self._name, k)

    def __setitem__(self, k, v):
        self._engine.set_place(self._charname, k, v)

    def __delitem__(self, k):
        self._engine.del_node(self._charname, k)


class ThingMapProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._name = charname

    def __iter__(self):
        yield from self._engine.character_things(self._name)

    def __len__(self):
        return self._engine.character_things_len(self._name)

    def __contains__(self, k):
        return self._engine.character_has_thing(self._name, k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such Thing: {}".format(k))
        return ThingProxy(self._engine, self._name, k)

    def __setitem__(self, k, v):
        self._engine.set_thing(self._name, k, v)

    def __delitem__(self, k):
        self._engine.del_node(self._name, k)


class PlaceMapProxy(MutableMapping):
    def __init__(self, engine_proxy, character):
        self._engine = engine_proxy
        self._name = character

    def __iter__(self):
        yield from self._engine.character_places(self._name)

    def __len__(self):
        return self._engine.character_places_len(self._name)

    def __getitem__(self, k):
        if not self._engine.character_has_place(self._name, k):
            raise KeyError("No such place: {}".format(k))
        return PlaceProxy(self._character, k)

    def __setitem__(self, k, v):
        self._engine.set_place(self._name, k, v)

    def __delitem__(self, k):
        self._engine.del_node(self._name, k)


class SuccessorsProxy(MutableMapping):
    def __init__(self, engine_proxy, charname, nodeAname):
        self._engine = engine_proxy
        self._charname = charname
        self._nodeA = nodeAname

    def __iter__(self):
        yield from self._engine.character_node_successors(
            self._charname, self._nodeAname
        )

    def __contains__(self, nodeB):
        return self._engine.character_nodes_connected(
            self._charname, self._nodeA, nodeB
        )

    def __len__(self):
        return self._engine.character_len_node_successors(
            self._charname, self._nodeA
        )

    def __getitem__(self, nodeB):
        return PortalProxy(self._engine, self._charname, self._nodeA, nodeB)

    def __setitem__(self, nodeB, value):
        self._engine.set_portal(
            self._character, self._nodeA, nodeB, value
        )

    def __delitem__(self, nodeB):
        self._engine.del_portal(self._character, self._nodeA, nodeB)


class CharSuccessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._charname = charname

    def __iter__(self):
        yield from self._engine.character_node_successors(
            self._charname
        )

    def __len__(self):
        return self._engine.character_node_successors_len(
            self._charname
        )

    def __getitem__(self, nodeA):
        if not self._engine.character_has_node(self._charname, nodeA):
            raise KeyError("No such node: {}".format(nodeA))
        return SuccessorsProxy(self._engine, self._charname, nodeA)

    def __setitem__(self, nodeA, val):
        self._engine.character_set_node_successors(
            self._charname, nodeA, val
        )

    def __delitem__(self, nodeA):
        self._engine.character_del_node_successors(
            self._charname, nodeA
        )


class PredecessorsProxy(MutableMapping):
    def __init__(self, engine_proxy, charname, nodeBname):
        self._engine = engine_proxy
        self._charname = charname
        self._name = nodeBname

    def __iter__(self):
        yield from self._engine.node_predecessors(
            self._charname, self._name
        )

    def __len__(self):
        return self._engine.node_predecessors_len(
            self._charname, self._name
        )

    def __getitem__(self, k):
        if not self._engine.node_precedes(self._charname, self._name, k):
            raise KeyError(
                "{} does not precede {}".format(k, self._name)
            )
        if self._engine.character_has_thing(self._charname, k):
            return ThingProxy(self._engine, self._charname, k)
        else:
            return PlaceProxy(self._engine, self._charname, k)

    def __setitem__(self, k, v):
        self._engine.set_place(self._charname, k, v)
        self._engine.set_portal(self._charname, k, self._name, v)

    def __delitem__(self, k):
        if not self._engine.node_precedes(self._charname, self._name, k):
            raise KeyError(
                "{} does not precede {}".format(k, self._name)
            )
        self._engine.del_portal(self._charname, k, self._name)


class CharPredecessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._name = charname

    def __contains__(self, k):
        return self._engine.node_has_predecessor(self._name, k)

    def __iter__(self):
        yield from self._engine.character_nodes_with_predecessors(
            self._name
        )

    def __len__(self):
        return self._engine.character_nodes_with_predecessors_len(
            self._name
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(
                "No predecessors to {} (if it even exists)".format(k)
            )
        return PredecessorsProxy(self._engine, self._name, k)

    def __setitem__(self, k, v):
        self._engine.character_set_node_predecessors(self._name, k, v)

    def __delitem__(self, k):
        self._engine.character_del_node_predecessors(self._name, k)


class CharStatProxy(MutableMapping):
    def __init__(self, engine_proxy, character):
        self._engine = engine_proxy
        self._name = character

    def __iter__(self):
        yield from self._engine.character_stats(self._name)

    def __len__(self):
        return self._engine.character_stats_len(self._name)

    def __getitem__(self, k):
        return self._engine.get_character_stat(self._name, k)

    def __setitem__(self, k, v):
        self._engine.set_character_stat(self._name, k, v)

    def __delitem__(self, k):
        self._engine.del_character_stat(self._name, k)


class CharacterProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._name = charname
        self.adj = self.succ = self.portal = CharSuccessorsMappingProxy(
            self._engine, self._name
        )
        self.pred = self.preportal = CharPredecessorsMappingProxy(
            self._engine, self._name
        )
        self.node = NodeMapProxy(self._engine, self._name)
        self.thing = ThingMapProxy(self._engine, self._name)
        self.place = PlaceMapProxy(self._engine, self._name)
        self.stat = CharStatProxy(self._engine, self._name)

    def __iter__(self):
        yield from self._engine.character_nodes(self._name)

    def __len__(self):
        return self._engine.character_nodes_len(self._name)

    def __getitem__(self, k):
        if not self._engine.character_has_node(self._name, k):
            raise KeyError("No such node: {}".format(k))
        if self._engine.character_has_thing(self._name, k):
            return ThingProxy(self._name, k)
        else:
            return PlaceProxy(self._name, k)

    def __setitem__(self, k, v):
        self._engine.set_place(self._name, k, v)

    def __delitem__(self, k):
        self._engine.del_place(self._name, k)

    def add_place(self, name, **kwargs):
        self[name] = kwargs

    def add_places_from(self, seq):
        self._engine.add_places_from(self._name, seq)

    def add_nodes_from(self, seq):
        self.add_places_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        self._engine.add_thing(
            self._name, name, location, next_location, kwargs
        )

    def add_things_from(self, seq):
        self._engine.add_things_from(self._name, seq)

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def new_thing(self, name, location, next_location=None, **kwargs):
        self.add_thing(name, location, next_location, **kwargs)
        return self.thing[name]

    def place2thing(self, name, location, next_location=None):
        self._engine.place2thing(self._name, name, location, next_location)

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        self._engine.add_portal(
            self._name, origin, destination, symmetrical, kwargs
        )

    def add_portals_from(self, seq, symmetrical=False):
        self._real.add_portals_from(self._name, seq, symmetrical)

    def portals(self):
        yield from self._real.character_portals(self._name)

    def add_avatar(self, a, b=None):
        self._real.add_avatar(self._name, a, b)

    def del_avatar(self, a, b=None):
        self._real.del_avatar(self._name, a, b)

    def avatars(self):
        yield from self._engine.character_avatars(self._name)


class CharacterMapProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._engine = engine_proxy

    def __iter__(self):
        yield from self._engine.characters()

    def __contains__(self, k):
        return self._engine.have_character(k)

    def __len__(self):
        return self._engine.characters_len()

    def __getitem__(self, k):
        if not self._engine.have_character(k):
            raise KeyError("No character: {}".format(k))
        return CharacterProxy(self._engine, k)

    def __setitem__(self, k, v):
        if isinstance(v, CharacterProxy):
            return
        self._engine.set_character(k, v)

    def __delitem__(self, k):
        self._engine.del_character(k)


class StringStoreProxy(MutableMapping):
    @property
    def language(self):
        return self._proxy.get_language()

    @language.setter
    def language(self, v):
        self._proxy.set_language(v)
        self._dispatch_lang(v)

    def __init__(self, engine_proxy):
        self._proxy = engine_proxy
        self._str_listeners = defaultdict(list)
        self._lang_listeners = []

    def __iter__(self):
        yield from self._proxy.get_string_ids()

    def __len__(self):
        return self._proxy.count_strings()

    def __getitem__(self, k):
        return self._proxy.get_string(k)

    def __setitem__(self, k, v):
        self._proxy.set_string(k, v)

    def __delitem__(self, k):
        self._proxy.del_string(k)

    def _dispatch_lang(self, v):
        for f in self._lang_listeners:
            f(self, v)

    def _dispatch_str(self, k, v):
        dispatch(self._str_listeners, k, self, k, v)

    def lang_listener(self, f):
        listen(self._lang_listeners, f)

    def listener(self, fun=None, string=None):
        return listener(self._str_listeners, fun, string)


class EternalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._engine = engine_proxy

    def __contains__(self, k):
        return self._engine.have_eternal(k)

    def __iter__(self):
        yield from self._engine.eternal_keys()

    def __len__(self):
        return self._engine.eternal_len()

    def __getitem__(self, k):
        return self._engine.get_eternal(k)

    def __setitem__(self, k, v):
        self._engine.set_eternal(k, v)

    def __delitem__(self, k):
        self._engine.del_eternal(k)


class GlobalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._proxy = engine_proxy
        self._listeners = defaultdict(list)

    def __iter__(self):
        yield from self._proxy.universal_keys()

    def __len__(self):
        return self._proxy.universal_len()

    def __getitem__(self, k):
        return self._proxy.get_universal(k)

    def __setitem__(self, k, v):
        self._proxy.set_universal(k, v)

    def __delitem__(self, k):
        self._proxy.del_universal(k)

    def _dispatch(self, k, v):
        dispatch(self._listeners, k, self, k, v)

    def listener(self, f=None, key=None):
        return listener(self._listeners, f, key)


class EngineProxy(object):
    def __init__(self, handle, *args):
        self._handle = handle
        self._handle.reify(args)
        self.eternal = EternalVarProxy(self._handle)
        self.universal = GlobalVarProxy(self._handle)
        self.character = CharacterMapProxy(self._handle)
        self._time_listeners = []

    @property
    def branch(self):
        return self._handle.get_branch()

    @branch.setter
    def branch(self, v):
        self._handle.set_branch(v)
        if not self._handle.time_locked():
            (branch, tick) = self.time
            for f in self._time_listeners:
                f(self, branch, tick, v, tick)

    @property
    def tick(self):
        return self._handle.get_tick()

    @tick.setter
    def tick(self, v):
        self._handle.set_tick(v)
        if not self._handle.time_locked():
            (b, t) = self.time
            for f in self._time_listeners:
                f(self, b, t, b, v)

    @property
    def time(self):
        return self._handle.get_time()

    @time.setter
    def time(self, v):
        self._handle.set_time(v)
        if not self._handle.time_locked():
            (b, t) = self.time
            (branch, tick) = v
            for f in self._time_listeners:
                f(self, b, t, branch, tick)

    def on_time(self, f):
        if not isinstance(f, Callable):
            raise TypeError('on_time is a decorator')
        if f not in self._time_listeners:
            self._time_listeners.append(f)

    def advance(self):
        self._handle.advance()

    def next_tick(self):
        self._handle.next_tick()

    def add_character(self, name, data=None, **kwargs):
        self._handle.add_character(name, data, kwargs)

    def new_character(self, name, **kwargs):
        self.add_character(name, **kwargs)
        return CharacterProxy(self._handle, name)

    def del_character(self, name):
        self._handle.del_character(name)

    def commit(self):
        self._handle.commit()

    def close(self):
        self._handle.close()
        self.manager.shutdown()


def create_handle(manager, queue):
    engine = manager.EngineHandle()
    print('engine handle created in process {}'.format(os.getpid()))
    queue.put(engine)
    lock.release()


class LiSERemoteControl(object):
    def __init__(self):
        self._manager = EngineManager()

    def start(
            self,
            worlddb,
            codedb,
            connect_args={},
            alchemy=False,
            caching=True,
            commit_modulus=None,
            random_seed=None
    ):
        self._manager.start()
        q = self._manager.Queue()
        self._p = Process(
            target=create_handle,
            args=(
                self._manager,
                q
            )
        )
        self._p.start()
        self._handle = q.get()
        print('got handle in process {}'.format(os.getpid()))
        return EngineProxy(
            self._handle,
            worlddb,
            codedb,
            connect_args,
            alchemy,
            caching,
            commit_modulus,
            random_seed
        )

    def shutdown(self):
        self.engine.close()
        self._p.join()
        self._manager.shutdown()


def test_eng_handle(engine):
    engine.init_character('FooChar', {'nice': True})
    assert(engine.get_character_stat('FooChar', 'nice') is True)
    print('character created in process {}'.format(os.getpid()))
    engine.set_character_stat('FooChar', 'boring', False)
    assert(engine.get_character_stat('FooChar', 'boring') is False)


if __name__ == "__main__":
    from examples.utiltest import clear_off
    clear_off()
    manager = EngineManager()
    manager.start()
    print('manager started in process {}'.format(os.getpid()))
    queue = manager.Queue()
    p = Process(target=create_engine, args=(manager, queue))
    p.start()
    engine = queue.get()
    test_eng_handle(engine)
    engine.close()
    manager.shutdown()
