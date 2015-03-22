# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    Callable,
    defaultdict,
    Mapping,
    MutableMapping,
    MutableSequence
)
from multiprocessing import Process, Pipe, Queue
from queue import Empty

from .core import Engine
from .character import Facade
from .util import JSONReWrapper, JSONListReWrapper


"""Proxy objects to make LiSE usable when launched in a subprocess,
and a manager class to launch it thus.

"""


class EngineHandle(object):
    """A wrapper for a :class:`LiSE.Engine` object that runs in the same
    process, but with an API built to be used in a command-processing
    loop that takes commands from another process.

    It's probably a bad idea to use this class unless you're
    developing your own API.

    """
    def __init__(self, args, kwargs, callbacq):
        """Instantiate an engine with the positional arguments ``args`` and
        the keyword arguments ``kwargs``.

        ``callbacq`` must be a :class:`Queue` object. I'll put tuples
        into it describing *apparent* changes to the world state.
        Changes are apparent if either (a) the world was changed at
        time ``(self.branch, self.tick)`` (default: ``('master',
        0)``), or (b) the user traveled from one point in time to
        another, and a watched entity's stats differ between those
        points.

        """
        self._real = Engine(*args, **kwargs)
        self._q = callbacq
        self._muted_chars = set()
        self.branch = self._real.branch
        self.tick = self._real.tick

    def mute_char(self, char):
        self._muted_chars.add(char)

    def unmute_char(self, char):
        self._muted_chars.discard(char)

    def listen_to_lang(self):
        """After calling this method, whenever the engine's language is
        changed, a tuple will be put into my queue of the
        form ``('language', v)``, where ``v`` is the new language.

        """
        @self._real.string.lang_listener
        def dispatch_lang(mapping, v):
            self._q.put(
                ('language', v)
            )

    def listen_to_strings(self):
        """After calling this method, whenever a string is set or
        deleted, a tuple will be put into my queue of the
        form ``('string', k, v)``, where ``k`` is the string's
        identifier and ``v`` is the string (or ``None`` if deleted).

        """
        @self._real.string.listener
        def dispatch_str(mapping, k, v):
            self._q.put(('string', k, v))

    def listen_to_string(self, k):
        """After calling this method, whenever a string named ``k`` is set or
        deleted, a tuple will be put into my queue of the
        form ``('string', k, v)``, where ``k`` is the same as you
        called this method with, and ``v`` is the string (or ``None``
        if deleted).

        """
        @self._real.string.listener(string=k)
        def dispatch_str(mapping, k, v):
            self._q.put(('string', k, v))

    def listen_to_universals(self):
        """After calling this method, whenever a universal (ie. "global," but
        sensitive to sim-time) variable appears to change from the
        host's perspective, a tuple will be put into my queue
        of the form ``('universal`, branch, tick, key, value)``, where
        ``(branch, tick)`` is the sim-time of the apparent change. If ``key``
        was deleted, ``value`` will be ``None``.

        """
        @self._real.universal
        def dispatch_var(branch, tick, mapping, key, value):
            if (branch, tick) == (self.branch, self.tick):
                self._q.put(('universal', branch, tick, key, value))

        @self._real.on_time
        def dispatch_vars(oldb, oldt, newb, newt):
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            old_universals = dict(self._real.universal)
            self._real.time = (newb, newt)
            del self._real.locktime
            for (k, oldv) in old_universals.items():
                newv = self._real.universal.get(k, None)
                if oldv != newv:
                    self._q.put(('universal', newb, newt, k, newv))

    def listen_to_universal(self, k):
        """After calling this method, whenever the given universal key appears
        to change its value from the host's perspective, a tuple will
        be put into my queue of the form ``('universal',
        branch, tick, key, value)``, where ``(branch, tick)`` is the
        sim-time of the apparent change. If ``key`` was deleted,
        ``value`` will be ``None``.

        """
        @self._real.universal(key=k)
        def dispatch_var(branch, tick, mapping, key, value):
            if (branch, tick) == (self.branch, self.tick):
                self._q.put(('universal', branch, tick, key, value))

        @self._real.on_time
        def dispatch_var_tt(oldb, oldt, newb, newt):
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            oldv = self._real.universal.get(k, None)
            self._real.time = (newb, newt)
            del self._real.locktime
            newv = self._real.universal.get(k, None)
            if oldv != newv:
                self._q.put(('universal', newb, newt, k, newv))

    def listen_to_character_map(self):
        @self._real.character.listener
        def charactered(charmap, k, v):
            if v is None:
                self._q.put(('character_map', k, False))
            else:
                self._q.put(('character_map', k, True))

    def listen_to_character(self, charn):
        character = self._real.character[charn]

        @character.listener
        def put_stat(b, t, char, k, v):
            if charn in self._muted_chars:
                return
            if (b, t) == (self.branch, self.tick):
                self._q.put(
                    (
                        'character',
                        b, t,
                        char.name,
                        k, self.get_character_stat(char.name, k)
                    )
                )

        @self._real.on_time
        def check_stats(oldb, oldt, newb, newt):
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            olds = dict(character.stat)
            self._real.time = (newb, newt)
            del self._real.locktime
            for (k, v) in olds.items():
                if character.stat[k] != v:
                    self._q.put(
                        (
                            'character',
                            newb, newt,
                            charn,
                            k, self.get_character_stat(charn, k)
                        )
                    )

    def listen_to_character_stat(self, charn, statn):
        character = self._real.character[charn]

        @character.listener(stat=statn)
        def put_stat(b, t, char, k, v):
            if charn in self._muted_chars:
                return
            if (b, t) == (self.branch, self.tick):
                self._q.put(
                    (
                        'character',
                        b, t,
                        char.name,
                        k, self.get_character_stat(char.name, k)
                    )
                )

        @self._real.on_time
        def check_stat(oldb, oldt, newb, newt):
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            oldv = character.stat[statn]
            self._real.time = (newb, newt)
            del self._real.locktime
            newv = character.stat[statn]
            if oldv != newv:
                self._q.put(
                    (
                        'character',
                        newb, newt,
                        charn,
                        statn, self.get_character_stat(charn, statn)
                    )
                )

    def listen_to_node(self, char, noden):
        node = self._real.character[char].node[noden]

        @node.listener
        def put_stat(b, t, node, k, v):
            if char in self._muted_chars:
                return
            if (b, t) == (self.branch, self.tick):
                self._q.put(
                    (
                        'node',
                        b, t,
                        char, noden,
                        k, self.get_node_stat(char, noden, k)
                    )
                )

        @self._real.on_time
        def check_stats(oldb, oldt, newb, newt):
            if char in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            olds = dict(node)
            self._real.time = (newb, newt)
            del self._real.locktime
            for (k, v) in olds.items():
                if node[k] != v:
                    self._q.put(
                        (
                            'node',
                            newb, newt,
                            char, noden,
                            k, self.get_node_stat(char, noden, k)
                        )
                    )

    def listen_to_thing_map(self, charn):
        char = self._real.character[charn]

        @char.thing.listener
        def put_thing(b, t, mapping, thingn, v):
            if charn in self._muted_chars:
                return
            if (b, t) != (self.branch, self.tick):
                return
            self._q.put(('thing_extant', b, t, charn, thingn, v is not None))

        @self._real.on_time
        def check_thing(oldb, oldt, newb, newt):
            if charn not in self._real.character:
                return
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            old_things = set(char.thing.keys())
            self._real.time = (newb, newt)
            del self._real.locktime
            new_things = set(char.thing.keys())
            for new_thing in new_things - old_things:
                self._q.put(
                    ('thing_extant', newb, newt, charn, new_thing, True)
                )
            for old_thing in old_things - new_things:
                self._q.put(
                    ('thing_extant', newb, newt, charn, old_thing, False)
                )

    def listen_to_place_map(self, charn):
        char = self._real.character[charn]

        @char.place.listener
        def put_place(b, t, mapping, placen, v):
            if charn in self._muted_chars:
                return
            if (b, t) != (self.branch, self.tick):
                return
            self._q.put(('place_extant', b, t, charn, placen, v is not None))

        @self._real.on_time
        def check_place(oldb, oldt, newb, newt):
            if charn not in self._real.character:
                return
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            old_places = set(char.place.keys())
            self._real.time = (newb, newt)
            del self._real.locktime
            new_places = set(char.place.keys())
            for new_place in new_places - old_places:
                self._q.put(
                    ('place_extant', newb, newt, charn, new_place, True)
                )
            for old_place in old_places - new_places:
                self._q.put(
                    ('place_extant', newb, newt, charn, old_place, False)
                )

    def listen_to_node_stat(self, charn, noden, statn):
        node = self._real.character[charn].node[noden]

        @node.listener(stat=statn)
        def put_stat(b, t, node, k, v):
            if charn in self._muted_chars:
                return
            if (b, t) == (self.branch, self.tick):
                self._q.put(
                    (
                        'node',
                        b, t,
                        charn, noden,
                        k, self.get_node_stat(charn, noden, k)
                    )
                )

        @self._real.on_time
        def check_stat(oldb, oldt, newb, newt):
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            oldv = node[statn]
            self._real.time = (newb, newt)
            del self._real.locktime
            newv = node[statn]
            if oldv != newv:
                self._q.put(
                    (
                        'node',
                        newb, newt,
                        charn, noden,
                        statn, self.get_node_stat(charn, noden, statn)
                    )
                )

    def listen_to_portal(self, char, a, b):
        port = self._real.character[char].portal[a][b]

        @port.listener
        def put_stat(b, t, portal, k, v):
            if char in self._muted_chars:
                return
            if (b, t) == (self.branch, self.tick):
                self._q.put(
                    (
                        'portal',
                        b, t,
                        char, a, b,
                        k, self.get_portal_stat(char, a, b, k)
                    )
                )

        @self._real.on_time
        def check_stats(oldb, oldt, newb, newt):
            if char not in self._real.character:
                return
            if char in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            olds = dict(port)
            self._real.time = (newb, newt)
            del self._real.locktime
            for (k, v) in olds.items():
                if port[k] != v:
                    self._q.put(
                        (
                            'portal',
                            newb, newt,
                            char, a, b,
                            k, self.get_portal_stat(char, a, b, k)
                        )
                    )

    def listen_to_portal_map(self, charn):
        char = self._real.character[charn]

        @char.portal.listener
        def put_port(b, t, mapping, onode, dnode, port):
            if charn in self._muted_chars:
                return
            if (b, t) != (self.branch, self.tick):
                return
            o = onode.name
            d = dnode.name
            self._q.put(
                ('portal_extant', b, t, charn, o, d, port is not None)
            )

        @self._real.on_time
        def check_ports(oldb, oldt, newb, newt):
            if charn not in self._real.character:
                return
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            old_ports = {}
            for o in char.portal.keys():
                old_ports[o] = set(char.portal[o].keys())
            self._real.time = (newb, newt)
            del self._real.locktime
            new_ports = {}
            for o in char.portal.keys():
                new_ports[o] = set(char.portal[o].keys())
            for (oldo, oldds) in old_ports.items():
                if oldo not in new_ports.keys():
                    for oldd in oldds:
                        self._q.put(
                            (
                                'portal_extant',
                                newb,
                                newt,
                                charn,
                                oldo,
                                oldd,
                                False
                            )
                        )
                else:
                    for createdd in new_ports[oldo] - oldds:
                        self._q.put(
                            (
                                'portal_extant',
                                newb,
                                newt,
                                charn,
                                oldo,
                                createdd,
                                True
                            )
                        )
            for (newo, newds) in new_ports.items():
                if newo not in old_ports.keys():
                    for newd in newds:
                        self._q.put(
                            (
                                'portal_extant',
                                newb,
                                newt,
                                charn,
                                newo,
                                newd,
                                True
                            )
                        )
                else:
                    for deletedd in old_ports[newo] - newds:
                        self._q.put(
                            (
                                'portal_extant',
                                newb,
                                newt,
                                charn,
                                newo,
                                deletedd,
                                False
                            )
                        )

    def listen_to_portal_stat(self, charn, a, b, statn):
        port = self._real.character[charn].portal[a][b]

        @port.listener(stat=statn)
        def put_stat(b, t, portal, k, v):
            if charn in self._muted_chars:
                return
            if (b, t) != (self.branch, self.tick):
                return
            self._q.put(
                (
                    'portal',
                    b, t,
                    charn, a, b,
                    statn, self.get_portal_stat(charn, a, b, statn)
                )
            )

        @self._real.on_time
        def check_stat(oldb, oldt, newb, newt):
            if charn not in self._real.character:
                return
            if charn in self._muted_chars:
                return
            self._real.locktime = True
            self._real.time = (oldb, oldt)
            oldv = port[statn]
            self._real.time = (newb, newt)
            del self._real.locktime
            newv = port[statn]
            if oldv != newv:
                self._q.put(
                    (
                        'portal',
                        newv, newt,
                        charn, a, b,
                        statn, self.get_portal_stat(charn, a, b, statn)
                    )
                )

    def time_locked(self):
        return hasattr(self._real, 'locktime')

    def advance(self):
        self._real.advance()

    def next_tick(self):
        self._real.next_tick()
        self.tick = self._real.tick
        self._q.put(('set_time', self.branch, self.tick))

    def add_character(self, name, data, kwargs):
        self._real.add_character(name, data, **kwargs)

    def commit(self):
        self._real.commit()

    def close(self):
        self._real.close()

    def get_branch(self):
        return self._real.branch

    def get_watched_branch(self):
        return self.branch

    def set_branch(self, v):
        self._real.branch = v
        self.branch = v
        self._q.put(('set_time', self.branch, self.tick))

    def get_tick(self):
        return self._real.tick

    def get_watched_tick(self):
        return self.tick

    def set_tick(self, v):
        self._real.tick = v
        self.tick = v
        self._q.put(('set_time', self.branch, self.tick))

    def get_time(self):
        return self._real.time

    def get_watched_time(self):
        return (self.branch, self.tick)

    def set_time(self, v):
        self._real.time = v
        (self.branch, self.tick) = v
        self._q.put(('set_time', self.branch, self.tick))

    def get_language(self):
        return self._real.string.language

    def set_language(self, v):
        self._real.string.language = v

    def get_string_ids(self):
        return list(self._real.string)

    def get_string_lang_items(self, lang):
        return list(self._real.string.lang_items(lang))

    def count_strings(self):
        return len(self._real.string)

    def get_string(self, k):
        return self._real.string[k]

    def set_string(self, k, v):
        self._real.string[k] = v

    def del_string(self, k):
        del self._real.string[k]

    def get_eternal(self, k):
        return self._real.eternal[k]

    def set_eternal(self, k, v):
        self._real.eternal[k] = v

    def del_eternal(self, k):
        del self._real.eternal[k]

    def eternal_keys(self):
        return list(self._real.eternal.keys())

    def eternal_len(self):
        return len(self._real.eternal)

    def have_eternal(self, k):
        return k in self._real.eternal

    def get_universal(self, k):
        return self._real.universal[k]

    def set_universal(self, k, v):
        self._real.universal[k] = v

    def del_universal(self, k):
        del self._real.universal[k]

    def universal_keys(self):
        return list(self._real.universal.keys())

    def universal_len(self):
        return len(self._real.universal)

    def init_character(self, char, statdict={}):
        if char in self._real.character:
            raise KeyError("Already have character {}".format(char))
        self._real.character[char] = {}
        self._real.character[char].stat.update(statdict)

    def del_character(self, char):
        del self._real.character[char]

    def character_has_stat(self, char, k):
        return k in self._real.character[char].stat

    def get_character_stat(self, char, k):
        r = self._real.character[char].stat[k]
        if isinstance(r, JSONReWrapper):
            return ('JSONReWrapper', 'character', char, k, dict(r))
        elif isinstance(r, JSONListReWrapper):
            return ('JSONListReWrapper', 'character', char, k, list(r))
        else:
            return r

    def set_character_stat(self, char, k, v):
        self._real.character[char].stat[k] = v

    def del_character_stat(self, char, k):
        del self._real.character[char].stat[k]

    def character_stats(self, char):
        return list(self._real.character[char].stat.keys())

    def character_stats_len(self, char):
        return len(self._real.character[char].stat)

    def characters(self):
        return list(self._real.character.keys())

    def characters_len(self):
        return len(self._real.character)

    def have_character(self, char):
        return char in self._real.character

    def set_character(self, char, v):
        self._real.character[char] = v

    def get_node_stat(self, char, node, k):
        r = self._real.character[char].node[node][k]
        if isinstance(r, JSONReWrapper):
            return ('JSONReWrapper', 'node', char, node, k, dict(r))
        elif isinstance(r, JSONListReWrapper):
            return ('JSONListReWrapper', 'node', char, node, k, list(r))
        else:
            return r

    def set_node_stat(self, char, node, k, v):
        self._real.character[char].node[node][k] = v

    def del_node_stat(self, char, node, k):
        del self._real.character[char].node[node][k]

    def on_node_stat(self, char, node, k, v):
        self._q.put(('on_node_stat', char, node, k, v))

    def node_stat_keys(self, char, node):
        return list(self._real.character[char].node[node])

    def node_stat_len(self, char, node):
        return len(self._real.character[char].node[node])

    def node_has_stat(self, char, node, k):
        return k in self._real.character[char].node[node]

    def del_node(self, char, node):
        del self._real.character[char].node[node]

    def character_things(self, char):
        return list(self._real.character[char].thing)

    def character_things_len(self, char):
        return len(self._real.character[char].thing)

    def character_has_thing(self, char, thing):
        return thing in self._real.character[char].thing

    def character_places(self, char):
        return list(self._real.character[char].place)

    def character_places_len(self, char):
        return len(self._real.character[char].place)

    def character_has_place(self, char, place):
        return place in self._real.character[char].place

    def character_nodes(self, char):
        return list(self._real.character[char].node.keys())

    def character_predecessor_nodes(self, char):
        return list(self._real.character[char].adj.keys())

    def node_has_predecessor(self, char, node):
        return node in self._real.character[char].pred.keys()

    def node_predecessors_len(self, char, node):
        return len(self._real.character[char].pred[node])

    def node_predecessors(self, char, node):
        return list(self._real.character[char].pred[node].keys())

    def node_precedes(self, char, nodeB, nodeA):
        return nodeA in self._real.character[char].pred[nodeB]

    def character_nodes_with_predecessors(self, char):
        return list(self._real.character[char].pred.keys())

    def character_nodes_with_predecessors_len(self, char):
        return len(self._real.character[char].pred)

    def character_set_node_predecessors(self, char, node, preds):
        self._real.character[char].pred[node] = preds

    def character_del_node_predecessors(self, char, node):
        del self._real.character[char].pred[node]

    def character_nodes_len(self, char):
        return len(self._real.character[char].node)

    def character_has_node(self, char, node):
        return node in self._real.character[char].node

    def character_nodes_with_successors(self, char):
        return list(self._real.character[char].adj.keys())

    def character_node_successors(self, char, node):
        return list(self._real.character[char].adj[node].keys())

    def character_node_successors_len(self, char, node):
        return len(self._real.character[char].adj[node])

    def character_set_node_successors(self, char, node, val):
        self._real.character[char].adj[node] = val

    def character_del_node_successors(self, char, node):
        del self._real.character[char].adj[node]

    def character_nodes_connected(self, char, nodeA, nodeB):
        return nodeB in self._real.character[char].portal[nodeA]

    def character_len_node_successors(self, char, nodeA):
        return len(self._real.character[char].node[nodeA])

    def init_thing(self, char, thing, statdict={}):
        if thing in self._real.character[char].thing:
            raise KeyError(
                'Already have thing in character {}: {}'.format(
                    char, thing
                )
            )
        self.set_thing(char, thing, statdict)

    def set_thing(self, char, thing, statdict):
        self._real.character[char].thing[thing] = statdict

    def add_thing(self, char, thing, location, next_location, statdict):
        self._real.character[char].add_thing(
            thing, location, next_location, **statdict
        )

    def place2thing(self, char, name, location, next_location=None):
        self._real.character[char].place2thing(name, location, next_location)

    def thing2place(self, char, name):
        self._real.character[char].thing2place(name)

    def add_things_from(self, char, seq):
        self._real.character[char].add_things_from(seq)

    def get_thing_location(self, char, th):
        return self._real.character[char].thing[th]['location']

    def set_thing_location(self, char, th, loc):
        self._real.character[char].thing[th]['location'] = loc

    def get_thing_next_location(self, char, th):
        return self._real.character[char].thing[th]['next_location']

    def set_thing_next_location(self, char, th, loc):
        self._real.character[char].thing[th]['next_location'] = loc

    def thing_follow_path(self, char, th, path, weight):
        self._real.character[char].thing[th].follow_path(path, weight)

    def thing_go_to_place(self, char, th, place, weight):
        self._real.character[char].thing[th].go_to_place(place, weight)

    def thing_travel_to(self, char, th, dest, weight, graph):
        self._real.character[char].thing[th].travel_to(dest, weight, graph)

    def thing_travel_to_by(self, char, th, dest, arrival_tick, weight, graph):
        self._real.character[char].thing[th].travel_to_by(
            dest, arrival_tick, weight, graph
        )

    def init_place(self, char, place, statdict={}):
        if place in self._real.character[char].place:
            raise KeyError(
                'Already have place in character {}: {}'.format(
                    char, place
                )
            )
        self.set_place(char, place, statdict)

    def set_place(self, char, place, statdict):
        self._real.character[char].place[place] = statdict

    def add_places_from(self, char, seq):
        self._real.character[char].add_places_from(seq)

    def init_portal(self, char, o, d, statdict={}):
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
        self._real.character[char].portal[o][d] = statdict

    def character_portals(self, char):
        return list(self._real.character[char].portals())

    def add_portal(self, char, o, d, symmetrical, statdict):
        self._real.character[char].add_portal(o, d, symmetrical, **statdict)

    def add_portals_from(self, char, seq, symmetrical):
        self._real.character[char].add_portals_from(seq, symmetrical)

    def del_portal(self, char, o, d):
        del self._real.character[char].portal[o][d]

    def get_portal_stat(self, char, o, d, k):
        r = self._real.character[char].portal[o][d][k]
        if isinstance(r, JSONReWrapper):
            return ('JSONReWrapper', 'portal', char, o, d, k, dict(r))
        elif isinstance(r, JSONListReWrapper):
            return ('JSONListReWrapper', 'portal', char, o, d, k, list(r))
        else:
            return r

    def set_portal_stat(self, char, o, d, k, v):
        self._real.character[char].portal[o][d][k] = v

    def del_portal_stat(self, char, o, d, k):
        del self._real.character[char][o][d][k]

    def portal_stats(self, char, o, d):
        return list(self._real.character[char][o][d].keys())

    def len_portal_stats(self, char, o, d):
        return len(self._real.character[char][o][d])

    def portal_has_stat(self, char, o, d, k):
        return k in self._real.character[char][o][d]

    def character_avatars(self, char):
        return list(self._real.character[char].avatars())

    def add_avatar(self, char, a, b):
        self._real.character[char].add_avatar(a, b)

    def del_avatar(self, char, a, b):
        self._real.character[char].del_avatar(a, b)

    def get_rule_actions(self, rule):
        return self._real.rule.db.rule_actions(rule)

    def set_rule_actions(self, rule, l):
        self._real.rule.db.set_rule_actions(rule, l)

    def get_rule_triggers(self, rule):
        return self._real.rule.db.rule_triggers(rule)

    def set_rule_triggers(self, rule, l):
        self._real.rule.db.set_rule_triggers(rule, l)

    def get_rule_prereqs(self, rule):
        return self._real.rule.db.rule_prereqs(rule)

    def set_rule_prereqs(self, rule, l):
        self._real.rule.db.set_rule_prereqs(rule, l)

    def get_rulebook_rules(self, rulebook):
        return list(self._real.db.rulebook_rules(rulebook))

    def set_rulebook_rule(self, rulebook, i, rule):
        self._real.db.rulebook_set(rulebook, i, rule)

    def ins_rulebook_rule(self, rulebook, i, rule):
        self._real.db.rulebook_decr(rulebook, i)
        self.set_rulebook_rule(rulebook, i, rule)

    def del_rulebook_rule(self, rulebook, i):
        self._real.db.rulebook_del(rulebook, i)

    def get_character_rulebook(self, character):
        return self._real.db.get_rulebook_char(
            "character",
            character
        )

    def get_node_rulebook(self, character, node):
        return self._real.db.node_rulebook(character, node)

    def get_portal_rulebook(self, char, nodeA, nodeB):
        return self._real.db.portal_rulebook(
            char, nodeA, nodeB, 0
        )

    def rulebooks(self):
        return list(self._real.rulebook.keys())

    def len_rulebooks(self):
        return len(self._real.rulebook)

    def have_rulebook(self, k):
        return k in self._real.rulebook

    def keys_in_store(self, store):
        return list(getattr(self._real, store).keys())

    def len_store(self, store):
        return len(getattr(self._real, store))

    def plain_items_in_store(self, store):
        return list(getattr(self._real, store).iterplain())

    def plain_source(self, store, k):
        return getattr(self._real, store).plain(k)

    def store_set_source(self, store, func_name, source):
        getattr(self._real, store).set_source(func_name, source)


class NodeProxy(MutableMapping):
    @property
    def character(self):
        return CharacterProxy(self._engine, self._charname)

    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = self._get_rulebook()
        return self._rulebook

    def _get_rulebook(self):
        return RuleBookProxy(self._engine, self._get_rulebook_name())

    def _get_rulebook_name(self):
        return self._engine.handle(
            'get_node_rulebook',
            (self._charname, self.name)
        )

    def __init__(self, engine_proxy, charname, nodename):
        assert(nodename is not None)
        self._engine = engine_proxy
        self._charname = charname
        self.name = nodename

    def __iter__(self):
        yield from self._engine.handle(
            'node_stat_keys',
            (self._charname, self.name)
        )

    def __len__(self):
        return self._engine.handle(
            'node_stat_len',
            (self._charname, self.name)
        )

    def __contains__(self, k):
        return self._engine.handle(
            'node_has_stat',
            (self._charname, self.name, k)
        )

    def __getitem__(self, k):
        if k == 'name':
            return self.name
        return self._engine.handle(
            'get_node_stat',
            (self._charname, self.name, k)
        )

    def __setitem__(self, k, v):
        if k == 'name':
            raise KeyError("Nodes can't be renamed")
        self._engine.handle(
            'set_node_stat',
            (self._charname, self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        if k == 'name':
            raise KeyError("Nodes need names")
        self._engine.handle(
            'del_node_stat',
            (self._charname, self.name, k),
            silent=True
        )

    def listener(self, fun=None, stat=None):
        if None not in (fun, stat):
            self._engine.node_stat_listener(
                self._charname, self.name, stat, fun
            )
        elif stat is None:
            self._engine.node_listener(
                self._charname, self.name, fun
            )
        else:
            return lambda f: self.listener(fun=f, stat=stat)


class PlaceProxy(NodeProxy):
    pass


class ThingProxy(NodeProxy):
    @property
    def location(self):
        ln = self['location']
        if ln in self._engine.handle(
                'character_things',
                (self._charname,)
        ):
            return ThingProxy(self._engine, self._charname, ln)
        else:
            return PlaceProxy(self._engine, self._charname, ln)

    @location.setter
    def location(self, v):
        self._engine.handle(
            'set_thing_location',
            (self._charname, self.name, v._name),
            silent=True
        )

    @property
    def next_location(self):
        ln = self['next_location']
        if ln is None:
            return None
        if ln in self._engine.handle(
                'character_things',
                (self._charname,)
        ):
            return ThingProxy(self._engine, self._charname, ln)
        else:
            return PlaceProxy(self._engine, self._charname, ln)

    @next_location.setter
    def next_location(self, v):
        self._engine.handle(
            'set_thing_next_location',
            (self._charname, self.name, v._name),
            silent=True
        )

    def follow_path(self, path, weight=None):
        self._engine.handle(
            'thing_follow_path',
            (self._charname, self.name, path, weight),
            silent=True
        )

    def go_to_place(self, place, weight=''):
        self._engine.handle(
            'thing_go_to_place',
            (self._charname, self.name, place, weight),
            silent=True
        )

    def travel_to(self, dest, weight=None, graph=None):
        self._engine.handle(
            'thing_travel_to',
            (self._charname, self.name, dest, weight, graph),
            silent=True
        )

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        self._engine.handle(
            'thing_travel_to_by',
            (self._charname, self.name, dest, arrival_tick, weight, graph),
            silent=True
        )


class PortalProxy(MutableMapping):
    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = self._get_rulebook()
        return self._rulebook

    def _get_rulebook_name(self):
        return self._engine.handle(
            'get_portal_rulebook',
            (self._charname, self._nodeA, self._nodeB)
        )

    def _get_rulebook(self):
        return RuleBookProxy(self._engine, self._get_rulebook_name())

    def __init__(self, engine_proxy, charname, nodeAname, nodeBname):
        self._engine = engine_proxy
        self._charname = charname
        self._nodeA = nodeAname
        self._nodeB = nodeBname
        self._stat_listeners = defaultdict(list)

    def __iter__(self):
        yield from self._engine.handle(
            'portal_stats',
            (self._charname, self._nodeA, self._nodeB)
        )

    def __len__(self):
        return self._engine.handle(
            'len_portal_stats',
            (self._charname, self._nodeA, self._nodeB)
        )

    def __contains__(self, k):
        return self._engine.handle(
            'portal_has_stat',
            (self._charname, self._nodeA, self._nodeB, k)
        )

    def __getitem__(self, k):
        if k == 'origin':
            return self._nodeA
        elif k == 'destination':
            return self._nodeB
        elif k == 'character':
            return self._charname
        elif k not in self:
            raise KeyError('Key unset: {}'.format(k))
        return self._engine.handle(
            'get_portal_stat',
            (self._charname, self._nodeA, self._nodeB, k)
        )

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_portal_stat',
            (self._charname, self._nodeA, self._nodeB, k, v),
            silent=True
        )

    def __delitem__(self, k):
        self._engine.handle(
            'del_portal_stat',
            (self._charname, self._nodeA, self._nodeB, k),
            silent=True
        )

    def listener(self, fun=None, stat=None):
        if None not in (fun, stat):
            self._engine.portal_stat_listener(
                self._charname, self._nodeA, self._nodeB, stat, fun
            )
        elif stat is None:
            self._engine.portal_listener(
                self._charname, self._nodeA, self._nodeB, fun
            )
        else:
            return lambda f: self.listener(fun=f, stat=stat)


class NodeMapProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._charname = charname
        self._cache = {}

    def __iter__(self):
        yield from self._engine.handle(
            'character_nodes',
            (self._charname,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_nodes_len',
            (self._charname,)
        )

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self._engine.handle(
            'character_has_node',
            (self._charname, k)
        )

    def __getitem__(self, k):
        if k in self._cache:
            return self._cache[k]
        if k not in self:
            raise KeyError("No such node: {}".format(k))
        if self._engine.handle(
                'character_has_thing',
                (self._charname, k)
        ):
            r = ThingProxy(self._engine, self._charname, k)
        else:
            r = PlaceProxy(self._engine, self._charname, k)
        self._cache[k] = r
        return r

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_place',
            (self._charname, k, v),
            silent=True
        )

    def __delitem__(self, k):
        self._engine.handle(
            'del_node',
            (self._charname, k),
            silent=True
        )
        if k in self._cache:
            del self._cache[k]


class ThingMapProxy(MutableMapping):
    @property
    def character(self):
        return self._engine.character[self.name]

    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self.name = charname

    def __iter__(self):
        yield from self._engine.handle(
            'character_things',
            (self.name,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_things_len',
            (self.name,)
        )

    def __contains__(self, k):
        if (
                k in self.character.node._cache and
                isinstance(
                    self.character.node._cache[k],
                    ThingProxy
                )
        ):
            return True
        return self._engine.handle(
            'character_has_thing',
            (self.name, k)
        )

    def __getitem__(self, k):
        if k in self.character.node._cache:
            r = self.character.node._cache[k]
            if isinstance(r, ThingProxy):
                return r
            else:
                raise TypeError(
                    '{} is a Place, not a Thing'.format(
                        k
                    )
                )
        if k not in self:
            raise KeyError("No such Thing: {}".format(k))
        r = ThingProxy(self._engine, self.name, k)
        self.character.node._cache[k] = r
        return r

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_thing',
            (self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        if k in self.character.node._cache:
            del self.character.node._cache[k]
        self._engine.handle(
            'del_node',
            (self.name, k),
            silent=True
        )


class PlaceMapProxy(MutableMapping):
    @property
    def character(self):
        return self._engine.character[self.name]

    def __init__(self, engine_proxy, character):
        self._engine = engine_proxy
        self.name = character

    def __iter__(self):
        yield from self._engine.handle(
            'character_places',
            (self.name,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_places_len',
            (self.name,)
        )

    def __contains__(self, k):
        if (
                k in self.character.node._cache and
                isinstance(
                    self.character.node._cache[k],
                    PlaceProxy
                )
        ):
            return True
        return self._engine.handle(
            'character_has_place',
            (self.name, k)
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such place: {}".format(k))
        if k in self.character.node._cache:
            r = self.character.node._cache[k]
            if isinstance(r, ThingProxy):
                raise TypeError(
                    '{} is a Thing, not a Place'.format(k)
                )
            return r
        r = PlaceProxy(self._engine, self.name, k)
        self.character.node._cache[k] = r
        return r

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_place',
            (self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        if k in self.character.node._cache:
            del self.character.node._cache[k]
        self._engine.handle(
            'del_node',
            (self.name, k),
            silent=True
        )


class SuccessorsProxy(MutableMapping):
    def __init__(self, engine_proxy, charname, nodeAname):
        self._engine = engine_proxy
        self._charname = charname
        self._nodeA = nodeAname
        self._cache = {}

    def __iter__(self):
        yield from self._engine.handle(
            'character_node_successors',
            (self._charname, self._nodeA)
        )

    def __contains__(self, nodeB):
        if nodeB in self._cache:
            return True
        return self._engine.handle(
            'character_nodes_connected',
            (self._charname, self._nodeA, nodeB)
        )

    def __len__(self):
        return self._engine.handle(
            'character_len_node_successors',
            (self._charname, self._nodeA)
        )

    def __getitem__(self, nodeB):
        if nodeB not in self:
            raise KeyError(
                'No portal from {} to {}'.format(
                    self._nodeA, nodeB
                )
            )
        if nodeB not in self._cache:
            self._cache[nodeB] = PortalProxy(
                self._engine, self._charname, self._nodeA, nodeB
            )
        return self._cache[nodeB]

    def __setitem__(self, nodeB, value):
        self._engine.handle(
            'set_portal',
            (self._character, self._nodeA, nodeB, value),
            silent=True
        )

    def __delitem__(self, nodeB):
        if nodeB in self._cache:
            del self._cache[nodeB]
        self._engine.handle(
            'del_portal',
            (self._character, self._nodeA, nodeB),
            silent=True
        )


class CharSuccessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self._charname = charname
        self._cache = {}

    def __iter__(self):
        yield from self._engine.handle(
            'character_nodes_with_successors',
            (self._charname,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_node_successors_len',
            (self._charname,)
        )

    def __contains__(self, nodeA):
        if nodeA in self._cache:
            return True
        return self._engine.handle(
            'character_has_node',
            (self._charname, nodeA)
        )

    def __getitem__(self, nodeA):
        if nodeA not in self:
            raise KeyError("No such node: {}".format(nodeA))
        if nodeA not in self._cache:
            self._cache[nodeA] = SuccessorsProxy(
                self._engine, self._charname, nodeA
            )
        return self._cache[nodeA]

    def __setitem__(self, nodeA, val):
        self._engine.handle(
            'character_set_node_successors',
            (self._charname, nodeA, val),
            silent=True
        )

    def __delitem__(self, nodeA):
        if nodeA in self._cache:
            del self._cache[nodeA]
        self._engine.handle(
            'character_del_node_successors',
            (self._charname, nodeA),
            silent=True
        )


class PredecessorsProxy(MutableMapping):
    @property
    def character(self):
        return self._engine.character[self._charname]

    def __init__(self, engine_proxy, charname, nodeBname):
        self._engine = engine_proxy
        self._charname = charname
        self.name = nodeBname

    def __iter__(self):
        yield from self._engine.handle(
            'node_predecessors',
            (self._charname, self.name)
        )

    def __len__(self):
        return self._engine.handle(
            'node_predecessors_len',
            (self._charname, self.name)
        )

    def __contains__(self, k):
        if (
            k in self.character.portal._cache and
            self.name in self.character.portal._cache[k]
        ):
            return True
        return self._engine.handle(
            'node_precedes',
            (self._charname, self.name, k)
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(
                "{} does not precede {}".format(k, self.name)
            )
        if k not in self.character.portal._cache:
            self.character.portal._cache[k] = {}
        if self.name not in self.character.portal._cache[k]:
            self.character.portal._cache[k][self.name] = PortalProxy(
                self._engine, self._charname, k, self.name
            )
        return self.character.portal._cache[k][self.name]

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_place',
            (self._charname, k, v),
            silent=True
        )
        self._engine.handle(
            'set_portal',
            (self._charname, k, self.name),
            silent=True
        )

    def __delitem__(self, k):
        if k not in self:
            raise KeyError(
                "{} does not precede {}".format(k, self.name)
            )
        if (
            k in self.character.portal._cache and
            self.name in self.character.portal._cache[k]
        ):
            del self.character.portal._cache[k][self.name]
        self._engine.handle(
            'del_portal',
            (self._charname, k, self.name),
            silent=True
        )


class CharPredecessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self.name = charname
        self._cache = {}

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self._engine.handle(
            'node_has_predecessor',
            (self.name, k)
        )

    def __iter__(self):
        yield from self._engine.handle(
            'character_nodes_with_predecessors',
            (self.name,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_nodes_with_predecessors_len',
            (self.name,)
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(
                "No predecessors to {} (if it even exists)".format(k)
            )
        if k not in self._cache:
            self._cache[k] = PredecessorsProxy(self._engine, self.name, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        self._engine.handle(
            'character_set_node_predecessors',
            (self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        if k not in self:
            raise KeyError(
                "No predecessors to {} (if it even exists)".format(k)
            )
        if k in self._cache:
            del self._cache[k]
        self._engine.handle(
            'character_del_node_predecessors',
            (self.name, k),
            silent=True
        )


class CharStatProxy(MutableMapping):
    def __init__(self, engine_proxy, character):
        self._engine = engine_proxy
        self.name = character

    def __iter__(self):
        yield from self._engine.handle(
            'character_stats',
            (self.name,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_stats_len',
            (self.name,)
        )

    def __contains__(self, k):
        return self._engine.handle(
            'character_has_stat', (self.name, k)
        )

    def __getitem__(self, k):
        return self._engine.handle(
            'get_character_stat',
            (self.name, k)
        )

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_character_stat',
            (self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        self._engine.handle(
            'del_character_stat',
            (self.name, k),
            silent=True
        )

    def listener(self, fun=None, stat=None):
        if stat is None:
            self._engine.char_listener(
                self.name, fun
            )
        elif fun is None:
            return lambda f: self._engine.char_stat_listener(
                self.name, stat, f
            )
        else:
            self._engine.char_stat_listener(
                self.name, stat, fun
            )


class RuleProxy(object):
    @property
    def triggers(self):
        return self._engine.handle(
            'get_rule_triggers',
            (self.name,)
        )

    @triggers.setter
    def triggers(self, v):
        self._engine.handle(
            'set_rule_triggers',
            (self.name, v),
            silent=True
        )

    @property
    def prereqs(self):
        return self._engine.handle(
            'get_rule_prereqs',
            (self.name,)
        )

    @prereqs.setter
    def prereqs(self, v):
        self._engine.handle(
            'set_rule_prereqs',
            (self.name, v),
            silent=True
        )

    @property
    def actions(self):
        return self._engine.handle(
            'get_rule_actions',
            (self.name,)
        )

    @actions.setter
    def actions(self, v):
        self._engine.handle(
            'set_rule_actions',
            (self.name, v),
            silent=True
        )

    def __init__(self, engine_proxy, rulename):
        self._engine = engine_proxy
        self.name = rulename


class RuleBookProxy(MutableSequence):
    def __init__(self, engine_proxy, bookname):
        self._engine = engine_proxy
        self.name = bookname
        self._cache = self._engine.handle(
            'get_rulebook_rules',
            (self.name,)
        )
        self._proxy_cache = {}

    def __iter__(self):
        for k in self._cache:
            if k not in self._proxy_cache:
                self._proxy_cache[k] = RuleProxy(self._engine, k)
            yield self._proxy_cache[k]

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, i):
        k = self._cache[i]
        if k not in self._proxy_cache:
            self._proxy_cache[k] = RuleProxy(self._engine, k)
        return self._proxy_cache[k]

    def __setitem__(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache[i] = v
        self._engine.handle(
            'set_rulebook_rule',
            (self.name, i, v),
            silent=True
        )

    def __delitem__(self, i):
        del self._cache[i]
        self._engine.handle(
            'del_rulebook_rule',
            (self.name, i),
            silent=True
        )

    def insert(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache.insert(i, v)
        self._engine.handle(
            'ins_rulebook_rule',
            (self.name, i, v),
            silent=True
        )

    def listener(self, fun):
        self._engine.rulebook_listener(self.name, fun)


class CharacterProxy(MutableMapping):
    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._upd_rulebook()
        return self._rulebook

    def _upd_rulebook(self):
        self._rulebook = self._get_rulebook()

    def _get_rulebook(self):
        return RuleBookProxy(
            self._engine,
            self._engine.handle(
                'get_character_rulebook',
                (self.name,)
            )
        )

    def __init__(self, engine_proxy, charname):
        self._engine = engine_proxy
        self.name = charname
        self.adj = self.succ = self.portal = CharSuccessorsMappingProxy(
            self._engine, self.name
        )
        self.pred = self.preportal = CharPredecessorsMappingProxy(
            self._engine, self.name
        )
        self.node = NodeMapProxy(self._engine, self.name)
        self.thing = ThingMapProxy(self._engine, self.name)
        self.place = PlaceMapProxy(self._engine, self.name)
        self.stat = CharStatProxy(self._engine, self.name)

    def __iter__(self):
        yield from self._engine.handle(
            'character_nodes',
            (self.name,)
        )

    def __len__(self):
        return self._engine.handle(
            'character_nodes_len',
            (self.name,)
        )

    def __contains__(self, k):
        return k in self.node

    def __getitem__(self, k):
        return self.node[k]

    def __setitem__(self, k, v):
        self.node[k] = v

    def __delitem__(self, k):
        del self.node[k]

    def add_place(self, name, **kwargs):
        self[name] = kwargs

    def add_places_from(self, seq):
        self._engine.handle(
            'add_places_from',
            (self.name, list(seq))
        )

    def add_nodes_from(self, seq):
        self.add_places_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        self._engine.handle(
            'add_thing',
            (self.name, name, location, next_location, kwargs)
        )

    def add_things_from(self, seq):
        self._engine.handle(
            'add_things_from',
            (self.name, seq)
        )

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def new_thing(self, name, location, next_location=None, **kwargs):
        self.add_thing(name, location, next_location, **kwargs)
        return self.thing[name]

    def place2thing(self, name, location, next_location=None):
        self._engine.handle(
            'place2thing',
            (self.name, name, location, next_location)
        )

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        self._engine.handle(
            'add_portal',
            (self.name, origin, destination, symmetrical, kwargs)
        )

    def add_portals_from(self, seq, symmetrical=False):
        self._engine.handle(
            'add_portals_from',
            (self.name, seq, symmetrical)
        )

    def new_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.add_portal(origin, destination, symmetrical, **kwargs)
        return self.portal[origin][destination]

    def portals(self):
        yield from self._engine.handle(
            'character_portals',
            (self.name,)
        )

    def add_avatar(self, a, b=None):
        self._engine.handle(
            'add_avatar',
            (self.name, a, b)
        )

    def del_avatar(self, a, b=None):
        self._engine.handle(
            'del_avatar',
            (self.name, a, b)
        )

    def avatars(self):
        yield from self._engine.handle(
            'character_avatars',
            (self.name,)
        )

    def facade(self):
        return Facade(self)

    def mute(self):
        self._engine.mute_character(self.name)

    def unmute(self):
        self._engine.unmute_character(self.name)


class CharacterMapProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._engine = engine_proxy
        self._cache = {}

    def __iter__(self):
        yield from self._engine.handle('characters')

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self._engine.handle(
            'have_character', (k,)
        )

    def __len__(self):
        return self._engine.handle('characters_len')

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No character: {}".format(k))
        if k not in self._cache:
            self._cache[k] = CharacterProxy(self._engine, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        if isinstance(v, CharacterProxy):
            return
        self._engine.handle(
            'set_character', (k, v), silent=True
        )
        self._cache[k] = CharacterProxy(self._engine, k)

    def __delitem__(self, k):
        self._engine.handle('del_character', (k,), silent=True)
        if k in self._cache:
            del self._cache[k]


class StringStoreProxy(MutableMapping):
    @property
    def language(self):
        return self._proxy.handle('get_language')

    @language.setter
    def language(self, v):
        self._proxy.handle('set_language', (v,))
        self._dispatch_lang(v)

    def __init__(self, engine_proxy):
        self._proxy = engine_proxy
        self._str_listeners = defaultdict(list)
        self._lang_listeners = []

    def __iter__(self):
        yield from self._proxy.handle('get_string_ids')

    def __len__(self):
        return self._proxy.handle('count_strings')

    def __getitem__(self, k):
        return self._proxy.handle('get_string', (k,))

    def __setitem__(self, k, v):
        self._proxy.handle('set_string', (k, v), silent=True)

    def __delitem__(self, k):
        self._proxy.handle('del_string', (k,), silent=True)

    def _dispatch_lang(self, v):
        for f in self._lang_listeners:
            f(self, v)

    def lang_listener(self, f):
        self._engine.lang_listener(f)

    def listener(self, fun=None, string=None):
        if None not in (fun, string):
            self._engine.string_listener(string, fun)
        elif string is None:
            self._engine.strings_listener(fun)
        else:
            return lambda f: self.listener(fun=f, string=string)

    def lang_items(self, lang=None):
        yield from self._proxy.handle(
            'get_string_lang_items', (lang,)
        )


class EternalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._engine = engine_proxy

    def __contains__(self, k):
        return self._engine.handle(
            'have_eternal', (k,)
        )

    def __iter__(self):
        yield from self._engine.handle(
            'eternal_keys'
        )

    def __len__(self):
        return self._engine.handle('eternal_len')

    def __getitem__(self, k):
        return self._engine.handle(
            'get_eternal', (k,)
        )

    def __setitem__(self, k, v):
        self._engine.handle(
            'set_eternal',
            (k, v),
            silent=True
        )

    def __delitem__(self, k):
        self._engine.handle(
            'del_eternal', (k,),
            silent=True
        )


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

    def listener(self, f=None, key=None):
        if None not in (f, key):
            self._proxy.universal_listener(key, f)
        elif key is None:
            self._proxy.universals_listener(f)
        else:
            return lambda fun: self.listener(f=fun, key=key)


class AllRuleBooksProxy(Mapping):
    def __init__(self, engine_proxy):
        self._engine = engine_proxy
        self._cache = {}

    def __iter__(self):
        yield from self._engine.handle('rulebooks')

    def __len__(self):
        return self._engine.handle('len_rulebooks')

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self._engine.handle('have_rulebook', (k,))

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No rulebook: {}".format(k))
        if k not in self._cache:
            self._cache[k] = RuleBookProxy(self._engine, k)
        return self._cache[k]


class FuncStoreProxy(object):
    def __init__(self, engine_proxy, store):
        self._engine = engine_proxy
        self._store = store

    def __iter__(self):
        yield from self._engine.handle(
            'keys_in_store', (self._store,)
        )

    def __len__(self):
        return self._engine.handle(
            'len_store',
            (self._store,)
        )

    def plain(self, k):
        return self._engine.handle(
            'plain_source',
            (self._store, k)
        )

    def iterplain(self):
        yield from self._engine.handle(
            'plain_items_in_store',
            (self._store,)
        )

    def set_source(self, func_name, source):
        self._engine.handle(
            'store_set_source',
            (self._store, func_name, source)
        )


class ChangeSignatureError(TypeError):
    pass


class EngineProxy(object):
    @property
    def branch(self):
        return self._branch

    @branch.setter
    def branch(self, v):
        self.handle('set_branch', (v,), silent=True)
        self._branch = v
        if not self.handle('time_locked'):
            (branch, tick) = self.time
            for f in self._time_listeners:
                f(self, branch, tick, v, tick)

    @property
    def tick(self):
        return self._tick

    @tick.setter
    def tick(self, v):
        self.handle('set_tick', (v,), silent=True)
        self._tick = v
        if not self.handle('time_locked'):
            (b, t) = self.time
            for f in self._time_listeners:
                f(self, b, t, b, v)

    @property
    def time(self):
        return (self._branch, self._tick)

    @time.setter
    def time(self, v):
        self.handle('set_time', (v,), silent=True)
        (self._branch, self._tick) = v
        if not self.handle('time_locked'):
            (b, t) = self.time
            (branch, tick) = v
            for f in self._time_listeners:
                f(b, t, branch, tick)

    def __init__(self, handle_out, handle_in, eventq):
        self._handle_out = handle_out
        self._handle_in = handle_in
        self._q = eventq
        self.eternal = EternalVarProxy(self)
        self.universal = GlobalVarProxy(self)
        self.character = CharacterMapProxy(self)
        self.string = StringStoreProxy(self)
        self.rulebook = AllRuleBooksProxy(self)
        for funstore in ('action', 'prereq', 'trigger', 'sense', 'function'):
            setattr(self, funstore, FuncStoreProxy(self, funstore))
        self._rulebook_listeners = defaultdict(list)
        self._time_listeners = []
        self._lang_listeners = []
        self._strings_listeners = []
        self._string_listeners = {}
        self._universals_listeners = []
        self._universal_listeners = {}
        self._char_listeners = {}
        self._char_map_listeners = []
        self._char_stat_listeners = {}
        self._node_listeners = {}
        self._node_stat_listeners = {}
        self._thing_map_listeners = {}
        self._place_map_listeners = {}
        self._portal_listeners = {}
        self._portal_stat_listeners = {}
        self._portal_map_listeners = {}
        (self._branch, self._tick) = self.handle('get_watched_time')

    def handle(self, func_name, args=[], silent=False):
        self._handle_out.send((silent, func_name, args))
        if not silent:
            r = self._handle_in.recv()
            return self.json_rewrap(r)

    def next_tick(self):
        self.handle('next_tick', silent=True)

    def char_listener(self, char, fun):
        if char not in self._char_listeners:
            self._char_listeners[char] = []
            self.handle('listen_to_character', (char,))
        if fun not in self._char_listeners[char]:
            self._char_listeners[char].append(fun)

    def char_map_listener(self, fun):
        if not self._char_map_listeners:
            self.handle('listen_to_character_map')
        if fun not in self._char_map_listeners:
            self._char_map_listeners.append(fun)

    def char_stat_listener(self, char, stat, fun):
        if char not in self._char_stat_listeners:
            self._char_stat_listeners[char] = {}
        if stat not in self._char_stat_listeners[char]:
            self._char_stat_listeners[char][stat] = []
            self.handle('listen_to_character_stat', (char, stat))
        if fun not in self._char_stat_listeners[char][stat]:
            self._char_stat_listeners[char][stat].append(fun)

    def node_listener(self, char, node, fun):
        if char not in self._node_listeners:
            self._node_listeners[char] = {}
        if node not in self._node_listeners[char]:
            self._node_listeners[char][node] = []
            self.handle('listen_to_node', (char, node))
        if fun not in self._node_listeners[char][node]:
            self._node_listeners[char][node].append(fun)

    def node_stat_listener(self, char, node, stat, fun):
        if char not in self._node_stat_listeners:
            self._node_stat_listeners[char] = {}
        if node not in self._node_stat_listeners[char]:
            self._node_stat_listeners[char][node] = {}
        if stat not in self._node_stat_listeners[char][node]:
            self._node_stat_listeners[char][node][stat] = []
            self.handle('listen_to_node_stat', (char, node, stat))
        if fun not in self._node_stat_listeners[char][node][stat]:
            self._node_stat_listeners[char][node][stat].append(fun)

    def thing_map_listener(self, char, fun):
        if char not in self._thing_map_listeners:
            self.handle('listen_to_thing_map', (char,))
            self._thing_map_listeners[char] = []
        if fun not in self._thing_map_listeners[char]:
            self._thing_map_listeners[char].append(fun)

    def place_map_listener(self, char, fun):
        if char not in self._place_map_listeners:
            self.handle('listen_to_place_map', (char,))
            self._place_map_listeners[char] = []
        if fun not in self._place_map_listeners[char]:
            self._place_map_listeners[char].append(fun)

    def portal_listener(self, char, orig, dest, fun):
        if char not in self._portal_listeners:
            self._portal_listeners[char] = {}
        if orig not in self._portal_listeners[char]:
            self._portal_listeners[char][orig] = {}
        if dest not in self._portal_listeners[char][orig]:
            self._portal_listeners[char][orig][dest] = []
            self.handle('listen_to_portal', (char, orig, dest))
        if fun not in self._portal_listeners[char][orig][dest]:
            self._portal_listeners[char][orig][dest].append(fun)

    def portal_stat_listener(self, char, orig, dest, stat, fun):
        if char not in self._portal_stat_listeners:
            self._portal_stat_listeners[char] = {}
        if orig not in self._portal_stat_listeners[char]:
            self._portal_stat_listeners[char][orig] = {}
        if dest not in self._portal_stat_listeners[char][orig]:
            self._portal_stat_listeners[char][orig][dest] = {}
        if stat not in self._portal_stat_listeners[char][orig][dest]:
            self._portal_stat_listeners[char][orig][dest][stat] = []
            self.handle('listen_to_portal_stat', (char, orig, dest, stat))
        if fun not in self._portal_stat_listeners[char][orig][dest][stat]:
            self._portal_stat_listeners[char][orig][dest][stat].append(fun)

    def portal_map_listener(self, char, fun):
        if char not in self._portal_map_listeners:
            self._portal_map_listeners[char] = []
            self.handle('listen_to_portal_map', (char,))
        if fun not in self._portal_map_listeners[char]:
            self._portal_map_listeners[char].append(fun)

    def lang_listener(self, fun):
        if not self._lang_listeners:
            self.handle('listen_to_lang')
        if fun not in self._lang_listeners:
            self._lang_listeners.append(fun)

    def strings_listener(self, fun):
        if not self._strings_listeners:
            self.handle('listening_to_strings')
        if fun not in self._strings_listeners:
            self._strings_listeners.append(fun)

    def string_listener(self, string, fun):
        if string not in self._string_listeners:
            self._string_listeners[string] = []
            self.handle('listen_to_string', (string,))
        if fun not in self._string_listeners[string]:
            self._string_listeners[string].append(fun)

    def universals_listener(self, fun):
        if not self._universals_listeners:
            self.handle('listen_to_universals')
        if fun not in self._universals_listeners:
            self._universals_listeners.append(fun)

    def universal_listener(self, k, fun):
        if k not in self._universal_listeners:
            self._universal_listeners[k] = []
            self.handle('listen_to_universal', (k,))
        if fun not in self._universal_listeners[k]:
            self._universal_listeners[k].append(fun)

    def json_rewrap(self, v):
        if not isinstance(v, tuple):
            return v
        if v[0] in ('JSONReWrapper', 'JSONListReWrapper'):
            cls = (
                JSONReWrapper if v[0] == 'JSONReWrapper'
                else JSONListReWrapper
            )
            if v[1] == 'character':
                (charn, k, initv) = v[2:]
                return cls(
                    self.character[charn], k, initv
                )
            elif v[1] == 'node':
                (charn, noden, k, initv) = v[2:]
                return cls(
                    self.character[charn].node[noden], k, initv
                )
            else:
                assert(v[1] == 'portal')
                (charn, o, d, k, initv) = v[2:]
                return cls(
                    self.character[charn].portal[o][d], k, initv
                )
        else:
            return v

    def poll_changes(self, num_changes=None):
        try:
            n = 0
            while num_changes is None or n < num_changes:
                self._process_change(self._q.get(False))
                n += 1
        except Empty:
            return

    def _process_change(self, v):
        assert(isinstance(v, tuple))
        typ = v[0]
        data = v[1:]
        if typ == 'language':
            (lang,) = data
            for fun in self._lang_listeners:
                fun(lang)
        elif typ == 'string':
            (k, v) = data
            for fun in self._strings_listeners:
                fun(k, v)
            if k in self._string_listeners:
                for fun in self._string_listeners[k]:
                    fun(k, v)
        elif typ == 'universal':
            (b, t, k, val) = data
            for fun in self._universals_listeners:
                fun(b, t, k, val)
            if k in self._universal_listeners:
                for fun in self._universal_listeners[k]:
                    fun(b, t, k, val)
        elif typ == 'set_time':
            (b, t) = self.time
            (self._branch, self._tick) = data
            for fun in self._time_listeners:
                fun(b, t, self._branch, self._tick)
        elif typ == 'character':
            (branch, tick, charn, stat, val) = data
            character = self.character[charn]
            if charn in self._char_listeners:
                for fun in self._char_listeners[charn]:
                    fun(branch, tick, charn, stat, self.json_rewrap(val))
            if (
                    charn in self._char_stat_listeners and
                    stat in self._char_stat_listeners[charn]
            ):
                for fun in self._char_stat_listeners[charn][stat]:
                    fun(branch, tick, character, stat, self.json_rewrap(val))
        elif typ == 'character_map':
            (charn, extant) = data
            for fun in self._char_map_listeners:
                fun(charn, extant)
        elif typ == 'node':
            (branch, tick, charn, noden, stat, val) = data
            node = self.character[charn].node[noden]
            if (
                    charn in self._node_listeners and
                    noden in self._node_listeners[charn]
            ):
                for fun in self._node_listeners[charn][noden]:
                    fun(branch, tick, node, stat, self.json_rewrap(val))
            if (
                    charn in self._node_stat_listeners and
                    noden in self._node_stat_listeners[charn] and
                    stat in self._node_stat_listeners
            ):
                for fun in self._node_stat_listeners[charn][noden][stat]:
                    fun(branch, tick, node, stat, self.json_rewrap(val))
        elif typ == 'thing_extant':
            (branch, tick, charn, thingn, extant) = data
            if charn in self._thing_map_listeners:
                for fun in self._thing_map_listeners[charn]:
                    fun(branch, tick, charn, thingn, extant)
        elif typ == 'place_extant':
            (branch, tick, charn, placen, extant) = data
            if charn in self._place_map_listeners:
                for fun in self._place_map_listeners[charn]:
                    fun(branch, tick, charn, placen, extant)
        elif typ == 'portal':
            (branch, tick, charn, a, b, stat, val) = data
            portal = self.character[charn].portal[a][b]
            if (
                    charn in self._portal_listeners and
                    a in self._portal_listeners[charn] and
                    b in self._portal_listeners[charn][a]
            ):
                for fun in self._portal_listeners[charn][a][b]:
                    fun(branch, tick, portal, stat, self.json_rewrap(val))
            if (
                    charn in self._portal_stat_listeners and
                    a in self._portal_stat_listeners[charn] and
                    b in self._portal_stat_listeners[charn][a] and
                    stat in self._portal_stat_listeners[charn][a][b]
            ):
                for fun in self._portal_stat_listeners[charn][a][b][stat]:
                    fun(branch, tick, portal, stat, self.json_rewrap(val))
        elif typ == 'portal_extant':
            (branch, tick, charn, o, d, extant) = data
            if charn in self._portal_map_listeners:
                for fun in self._portal_map_listeners[charn]:
                    fun(branch, tick, charn, o, d, extant)
        else:
            raise ChangeSignatureError(
                'Received a change notification from the LiSE core that '
                'did not match any of the known types: {}'.format(v)
            )

    def on_time(self, f):
        if not isinstance(f, Callable):
            raise TypeError('on_time is a decorator')
        if f not in self._time_listeners:
            self._time_listeners.append(f)

    def add_character(self, name, data=None, **kwargs):
        self.handle('add_character', (name, data, kwargs))

    def new_character(self, name, **kwargs):
        self.add_character(name, **kwargs)
        return CharacterProxy(self._handle, name)

    def del_character(self, name):
        self.handle('del_character', (name,))

    def mute_character(self, name):
        self.handle('mute_char', (name,), silent=True)

    def unmute_character(self, name):
        self.handle('unmute_char', (name,), silent=True)

    def commit(self):
        self.handle('commit')

    def close(self):
        self.handle('close')

    def rulebook_listener(self, rulebook, f):
        self._rulebook_listeners[rulebook].append(f)


def subprocess(
        args, kwargs, handle_out_pipe, handle_in_pipe, callbacq
):
    engine_handle = EngineHandle(args, kwargs, callbacq)
    while True:
        inst = handle_out_pipe.recv()
        if inst == 'shutdown':
            handle_out_pipe.close()
            handle_in_pipe.close()
            callbacq.close()
            return 0
        (silent, cmd, args) = inst
        if silent:
            r = getattr(engine_handle, cmd)(*args)
        else:
            r = getattr(engine_handle, cmd)(*args)
            handle_in_pipe.send(r)


class EngineProcessManager(object):
    def start(self, *args, **kwargs):
        (handle_out_pipe_recv, self._handle_out_pipe_send) = Pipe(duplex=False)
        (handle_in_pipe_recv, handle_in_pipe_send) = Pipe(duplex=False)
        callbacq = Queue()
        self._p = Process(
            name='LiSE Life Simulator Engine (core)',
            target=subprocess,
            args=(
                args,
                kwargs,
                handle_out_pipe_recv,
                handle_in_pipe_send,
                callbacq
            )
        )
        self._p.daemon = True
        self._p.start()
        self.engine_proxy = EngineProxy(
            self._handle_out_pipe_send,
            handle_in_pipe_recv,
            callbacq
        )
        return self.engine_proxy

    def shutdown(self):
        self.engine_proxy.close()
        self._handle_out_pipe_send.send('shutdown')
