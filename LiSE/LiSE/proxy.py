# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import sys
import logging
from collections import (
    Callable,
    defaultdict,
    Mapping,
    MutableMapping,
    MutableSequence
)
from threading import Thread
from multiprocessing import Process, Pipe, Queue
from queue import Empty

from .engine import AbstractEngine, Engine
from .character import Facade
from gorm.xjson import (
    JSONReWrapper,
    JSONListReWrapper,
    json_deepcopy
)
from gorm.reify import reify
from .util import (
    dict_diff,
    list_diff
)

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
    def __init__(self, args, kwargs, logq):
        """Instantiate an engine with the positional arguments ``args`` and
        the keyword arguments ``kwargs``.

        ``logq`` is a :class:`Queue` into which I'll put tuples of
        ``(loglevel, message)``.

        """
        self._real = Engine(*args, logfun=self.log, **kwargs)
        self._logq = logq
        self._muted_chars = set()
        self.branch = self._real.branch
        self.tick = self._real.tick
        self._node_stat_cache = defaultdict(dict)
        self._portal_stat_cache = defaultdict(
            lambda: defaultdict(dict)
        )
        self._char_stat_cache = {}
        self._char_things_cache = {}
        self._char_places_cache = {}
        self._char_portals_cache = {}
        self._char_nodes_with_successors = {}
        self._node_successors_cache = defaultdict(dict)

    def log(self, level, message):
        self._logq.put((level, message))

    def debug(self, message):
        self.log('debug', message)

    def info(self, message):
        self.log('info', message)

    def warning(self, message):
        self.log('warning', message)

    def error(self, message):
        self.log('error', message)

    def critical(self, message):
        self.log('critical', message)

    def unwrap_character_stat(self, char, k, v):
        if isinstance(v, JSONReWrapper):
            return ('JSONReWrapper', 'character', char, k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return ('JSONListReWrapper', 'character', char, k, v._v)
        else:
            return v

    def unwrap_node_stat(self, char, node, k, v):
        if isinstance(v, JSONReWrapper):
            return ('JSONReWrapper', 'node', char, node, k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return ('JSONListReWrapper', 'node', char, node, k, v._v)
        else:
            return v

    def unwrap_portal_stat(self, char, orig, dest, k, v):
        if isinstance(v, JSONReWrapper):
            return ('JSONReWrapper', 'portal', char, orig, dest, k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return ('JSONListReWrapper', 'portal', char, orig, dest, k, v._v)
        else:
            return v

    def time_locked(self):
        return hasattr(self._real, 'locktime')

    def advance(self):
        self._real.advance()

    def next_tick(self, char=None):
        self._real.next_tick()
        if char:
            return self.character_diff(char)

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

    def get_tick(self):
        return self._real.tick

    def get_watched_tick(self):
        return self.tick

    def set_tick(self, v):
        self._real.tick = v
        self.tick = v

    def get_time(self):
        return self._real.time

    def get_watched_time(self):
        return (self.branch, self.tick)

    def set_time(self, v):
        self._real.time = v
        (self.branch, self.tick) = v

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

    def have_string(self, k):
        return k in self._real.string

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
        try:
            return self.unwrap_character_stat(
                char, k,
                self._real.character[char].stat[k]
            )
        except KeyError:
            return None

    def character_stat_copy(self, char):
        return {
            k: self.unwrap_character_stat(char, k, v)
            for (k, v) in self._real.character[char].stat.items()
        }

    def character_stat_diff(self, char):
        try:
            old = self._char_stat_cache.get(char, {})
            new = self.character_stat_copy(char)
            self._char_stat_cache[char] = new
            return dict_diff(old, new)
        except KeyError:
            return None

    def character_diff(self, char):
        """Return a dictionary of changes to ``char`` since previous call."""
        return {
            'character_stat': self.character_stat_diff(char),
            'node_stat': self.character_nodes_stat_diff(char),
            'things': self.character_things_diff(char),
            'places': self.character_places_diff(char),
            'portal_stat': self.character_portals_stat_diff(char),
            'portals': self.character_portals_diff(char)
        }

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
        try:
            return self.unwrap_node_stat(
                char, node, k,
                self._real.character[char].node[node][k]
            )
        except KeyError:
            return None

    def set_node_stat(self, char, node, k, v):
        self._real.character[char].node[node][k] = v

    def del_node_stat(self, char, node, k):
        del self._real.character[char].node[node][k]

    def node_stat_keys(self, char, node):
        return list(self._real.character[char].node[node])

    def node_stat_copy(self, char, node):
        """Return a node's stats, prepared for pickling, in a dictionary."""
        return {
            k: self.unwrap_node_stat(char, node, k, v)
            for (k, v) in self._real.character[char].node[node].items()
        }

    def node_stat_diff(self, char, node):
        """Return a dictionary describing changes to a node's stats since the
        last time you looked at it.

        Deleted keys have the value ``None``.

        """
        try:
            old = self._node_stat_cache[char].get(node, {})
            new = self.node_stat_copy(char, node)
            self._node_stat_cache[char][node] = new
            r = dict_diff(old, new)
            return r
        except KeyError:
            return None

    def character_nodes_stat_diff(self, char):
        r = {}
        for node in self._real.character[char].node:
            diff = self.node_stat_diff(char, node)
            if diff:
                r[node] = diff
        return r

    def node_stat_len(self, char, node):
        return len(self._real.character[char].node[node])

    def node_has_stat(self, char, node, k):
        return k in self._real.character[char].node[node]

    def del_node(self, char, node):
        del self._real.character[char].node[node]

    def character_things(self, char):
        return list(self._real.character[char].thing)

    def character_things_diff(self, char):
        try:
            new = self.character_things(char)
            old = self._char_things_cache.get(char, [])
            self._char_things_cache[char] = new
            return list_diff(old, new)
        except KeyError:
            return None

    def character_things_len(self, char):
        return len(self._real.character[char].thing)

    def character_has_thing(self, char, thing):
        return thing in self._real.character[char].thing

    def character_places(self, char):
        return list(self._real.character[char].place)

    def character_places_diff(self, char):
        try:
            old = self._char_places_cache.get(char, [])
            new = self.character_places(char)
            self._char_places_cache[char] = new
            return list_diff(old, new)
        except KeyError:
            return None

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

    def character_nodes_with_successors_diff(self, char):
        try:
            old = self._char_nodes_with_successors.get(char, [])
            new = self.character_nodes_with_successors(char)
            self._char_nodes_with_successors[char] = new
            return list_diff(old, new)
        except KeyError:
            return None

    def node_successors(self, char, node):
        return list(self._real.character[char].portal[node].keys())

    def node_successors_diff(self, char, node):
        try:
            old = self._node_successors_cache[char].get(node, [])
            new = self.node_successors(char, node)
            self._node_successors_cache[char][node] = new
            return list_diff(old, new)
        except KeyError:
            return None

    def character_node_successors_len(self, char, node):
        return len(self._real.character[char].adj[node])

    def character_set_node_successors(self, char, node, val):
        self._real.character[char].adj[node] = val

    def character_del_node_successors(self, char, node):
        del self._real.character[char].adj[node]

    def nodes_connected(self, char, nodeA, nodeB):
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
        try:
            return self._real.character[char].thing[th]['location']
        except KeyError:
            return None

    def set_thing_location(self, char, th, loc):
        self._real.character[char].thing[th]['location'] = loc

    def get_thing_next_location(self, char, th):
        try:
            return self._real.character[char].thing[th]['next_location']
        except KeyError:
            return None

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
        r = []
        portal = self._real.character[char].portal
        for o in portal:
            for d in portal[o]:
                r.append((o, d))
        return r

    def character_portals_diff(self, char):
        try:
            old = self._char_portals_cache.get(char, {})
            new = self.character_portals(char)
            self._char_portals_cache[char] = new
            return list_diff(old, new)
        except KeyError:
            return None

    def add_portal(self, char, o, d, symmetrical, statdict):
        self._real.character[char].add_portal(o, d, symmetrical, **statdict)

    def add_portals_from(self, char, seq, symmetrical):
        self._real.character[char].add_portals_from(seq, symmetrical)

    def del_portal(self, char, o, d):
        del self._real.character[char].portal[o][d]

    def get_portal_stat(self, char, o, d, k):
        try:
            return self.unwrap_portal_stat(
                char, o, d, k,
                self._real.character[char].portal[o][d][k]
            )
        except KeyError:
            return None

    def set_portal_stat(self, char, o, d, k, v):
        self._real.character[char].portal[o][d][k] = v

    def del_portal_stat(self, char, o, d, k):
        del self._real.character[char][o][d][k]

    def portal_stat_copy(self, char, o, d):
        return {
            k: self.unwrap_portal_stat(char, o, d, k, v)
            for (k, v) in self._real.character[char].portal[o][d].items()
        }

    def portal_stat_diff(self, char, o, d):
        try:
            old = self._portal_stat_cache[char][o].get(d, {})
            new = self.portal_stat_copy(char, o, d)
            self._portal_stat_cache[char][o][d] = new
            return dict_diff(old, new)
        except KeyError:
            return None

    def character_portals_stat_diff(self, char):
        r = {}
        for orig in self._real.character[char].portal:
            for dest in self._real.character[char].portal[orig]:
                diff = self.portal_stat_diff(char, orig, dest)
                if diff:
                    if orig not in r:
                        r[orig] = {}
                    r[orig][dest] = diff
        return r

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

    def list_all_rules(self):
        return list(self._real.rule.keys())

    def count_all_rules(self):
        return len(self._real.rule)

    def have_rule(self, k):
        return k in self._real.rule

    def new_empty_rule(self, k):
        self._real.rule.new_empty(k)

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
        try:
            return self._real.db.node_rulebook(character, node)
        except KeyError:
            return None

    def set_node_rulebook(self, character, node, rulebook):
        self._real.db.set_node_rulebook(
            character, node, rulebook
        )

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


class CachingProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self.exists = True

    def __bool__(self):
        return bool(self.exists)

    def __iter__(self):
        yield from self._cache

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such key: {}".format(k))
        return self._cache[k]

    def __setitem__(self, k, v):
        self._set_item(k, v)
        self._cache[k] = self._cache_munge(k, v)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such key: {}".format(k))
        self._del_item(k)
        del self._cache[k]

    def _apply_diff(self, diff):
        for (k, v) in diff.items():
            if v is None:
                if k in self._cache:
                    del self._cache[k]
            elif k in self._cache and self._cache[k] != v:
                self._cache[k] = v

    def update_cache(self):
        diff = self._get_diff()
        self.exists = diff is not None
        if not self.exists:
            self._cache = {}
            return
        self._apply_diff(diff)

    def _get_diff(self):
        raise NotImplementedError("Abstract method")

    def _cache_munge(self, k, v):
        raise NotImplementedError("Abstract method")

    def _set_item(self, k, v):
        raise NotImplementedError("Abstract method")

    def _del_item(self, k):
        raise NotImplementedError("Abstract method")


class CachingEntityProxy(CachingProxy):
    def _cache_munge(self, k, v):
        return self.engine.json_rewrap(v)


class NodeProxy(CachingEntityProxy):
    @reify
    def character(self):
        return CharacterProxy(self.engine, self._charname)

    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = self._get_rulebook()
        return self._rulebook

    def _get_rulebook(self):
        return RuleBookProxy(self.engine, self._get_rulebook_name())

    def _get_rulebook_name(self):
        r = self.engine.handle(
            'get_node_rulebook',
            (self._charname, self.name)
        )
        if r is None:
            self.engine.handle(
                'set_node_rulebook',
                (
                    self._charname,
                    self.name,
                    (self._charname, self.name)
                ),
                silent=True
            )
            return (self._charname, self.name)
        return r

    @property
    def _cache(self):
        return self.engine._node_stat_cache[self._charname][self.name]

    def __init__(self, engine_proxy, charname, nodename):
        self._charname = charname
        self.name = nodename
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, NodeProxy) and
            self._charname == other._charname and
            self.name == other.name
        )

    def __contains__(self, k):
        if k in ('character', 'name'):
            return True
        return super().__contains__(k)

    def __getitem__(self, k):
        if k == 'character':
            return self._charname
        elif k == 'name':
            return self.name
        return super().__getitem__(k)

    def _get_state(self):
        return self.engine.handle(
            'node_stat_copy',
            (self._charname, self.name)
        )

    def _get_diff(self):
        return self.engine.handle(
            'node_stat_diff',
            (self._charname, self.name)
        )

    def _set_item(self, k, v):
        if k == 'name':
            raise KeyError("Nodes can't be renamed")
        self.engine.handle(
            'set_node_stat',
            (self._charname, self.name, k, v),
            silent=True
        )

    def _del_item(self, k):
        if k == 'name':
            raise KeyError("Nodes need names")
        self.engine.handle(
            'del_node_stat',
            (self._charname, self.name, k),
            silent=True
        )

    def delete(self):
        self.engine.del_node(self._charname, self.name)


class PlaceProxy(NodeProxy):
    def __repr__(self):
        return "proxy to {}.place[{}]".format(
            self._charname,
            self.name
        )


class ThingProxy(NodeProxy):
    @property
    def location(self):
        ln = self['location']
        if ln in self.engine.handle(
                'character_things',
                (self._charname,)
        ):
            return ThingProxy(self.engine, self._charname, ln)
        else:
            return PlaceProxy(self.engine, self._charname, ln)

    @location.setter
    def location(self, v):
        self['location'] = v

    @property
    def next_location(self):
        ln = self['next_location']
        if ln is None:
            return None
        if ln in self.engine.handle(
                'character_things',
                (self._charname,)
        ):
            return ThingProxy(self.engine, self._charname, ln)
        else:
            return PlaceProxy(self.engine, self._charname, ln)

    @next_location.setter
    def next_location(self, v):
        self.engine.handle(
            'set_thing_next_location',
            (self._charname, self.name, v._name),
            silent=True
        )

    def __iter__(self):
        already = set(k for k in super().__iter__())
        yield from already
        for k in {
                'name',
                'character',
                'location',
                'next_location',
                'arrival_time',
                'next_arrival_time'
        }:
            if k not in already:
                yield k

    def __getitem__(self, k):
        if k in {
                'location',
                'next_location',
                'arrival_time',
                'next_arrival_time'
        } and not super().__contains__(k):
            return None
        return super().__getitem__(k)

    def __repr__(self):
        if self['next_location'] is not None:
            return "proxy to {}.thing[{}]@{}->{}".format(
                self._charname,
                self.name,
                self['location'],
                self['next_location']
            )
        return "proxy to {}.thing[{}]@{}".format(
            self._charname,
            self.name,
            self['location'],
            self['next_location']
        )

    def update_cache(self):
        loc = self.engine.handle(
            'get_thing_location',
            (self._charname, self.name)
        )
        if loc is None:
            self.exists = False
            self._cache = {}
            return
        self._cache['location'] = loc
        self._cache['next_location'] = self.engine.handle(
            'get_thing_next_location',
            (self._charname, self.name)
        )
        super().update_cache()

    def follow_path(self, path, weight=None):
        self.engine.handle(
            'thing_follow_path',
            (self._charname, self.name, path, weight),
            silent=True
        )

    def go_to_place(self, place, weight=''):
        self.engine.handle(
            'thing_go_to_place',
            (self._charname, self.name, place, weight),
            silent=True
        )

    def travel_to(self, dest, weight=None, graph=None):
        self.engine.handle(
            'thing_travel_to',
            (self._charname, self.name, dest, weight, graph),
            silent=True
        )

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        self.engine.handle(
            'thing_travel_to_by',
            (self._charname, self.name, dest, arrival_tick, weight, graph),
            silent=True
        )


class PortalProxy(CachingEntityProxy):
    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = self._get_rulebook()
        return self._rulebook

    def _get_rulebook_name(self):
        return self.engine.handle(
            'get_portal_rulebook',
            (self._charname, self._nodeA, self._nodeB)
        )

    def _get_rulebook(self):
        return RuleBookProxy(self.engine, self._get_rulebook_name())

    @property
    def _cache(self):
        return self.engine._portal_stat_cache[self.charname][
            self._nodeA][self._nodeB]

    def _get_diff(self):
        return self.engine.handle(
            'portal_stat_diff',
            (self._charname, self._nodeA, self._nodeB)
        )

    def _set_item(self, k, v):
        self.engine.handle(
            'set_portal_stat',
            (self._charname, self._nodeA, self._nodeB, k, v),
            silent=True
        )

    def _del_item(self, k):
        self.engine_handle(
            'del_portal_stat',
            (self._charname, self._nodeA, self._nodeB, k),
            silent=True
        )

    def __init__(self, engine_proxy, charname, nodeAname, nodeBname):
        self._charname = charname
        self._nodeA = nodeAname
        self._nodeB = nodeBname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        if hasattr(other, 'engine'):
            oe = other.engine
        else:
            return False
        return (
            hasattr(other, 'character') and
            hasattr(other, 'origin') and
            hasattr(other, 'destination') and
            self.character == other.character and
            self.origin == other.origin and
            self.destination == other.destination
        )

    def __repr__(self):
        return "proxy to {}.portal[{}][{}]".format(
            self._charname,
            self._nodeA,
            self._nodeB
        )

    def __getitem__(self, k):
        if k == 'origin':
            return self._nodeA
        elif k == 'destination':
            return self._nodeB
        elif k == 'character':
            return self._charname
        return super().__getitem__(k)

    def delete(self):
        self.engine.del_portal(
            self._charname,
            self._nodeA,
            self._nodeB
        )


class NodeMapProxy(MutableMapping):
    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self._charname = charname

    def __iter__(self):
        yield from self.character.thing
        yield from self.character.place

    def __len__(self):
        return len(self.character.thing) + len(self.character.place)

    def __getitem__(self, k):
        if k in self.character.thing:
            return self.character.thing[k]
        else:
            return self.character.place[k]

    def __setitem__(self, k, v):
        self.character.place[k] = v

    def __delitem__(self, k):
        if k in self.character.thing:
            del self.character.thing[k]
        else:
            del self.character.place[k]


class ThingMapProxy(CachingProxy):
    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._character_things_cache[self.name]

    def __init__(self, engine_proxy, charname):
        self.name = charname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return self is other

    def _apply_diff(self, diff):
        for (thing, ex) in diff.items():
            if ex:
                if thing not in self._cache:
                    self._cache[thing] = ThingProxy(
                        self.engine,
                        self.name,
                        thing
                    )
            else:
                if thing in self._cache:
                    del self._cache[thing]

    def _get_diff(self):
        return self.engine.handle(
            'character_things_diff',
            (self.name,)
        )

    def _cache_munge(self, k, v):
        return ThingProxy(
            self.engine, self.name, k
        )

    def _set_item(self, k, v):
        self.engine.handle(
            'set_thing',
            (self.name, k, v),
            silent=True
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            'del_node',
            (self.name, k),
            silent=True
        )
        del self.engine._node_stat_cache[self.name][k]


class PlaceMapProxy(CachingProxy):
    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._character_places_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.name = character
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return self is other

    def _apply_diff(self, diff):
        for (place, ex) in diff.items():
            if ex:
                if place not in self._cache:
                    self._cache[place] = PlaceProxy(
                        self.engine,
                        self.name,
                        place
                    )
            else:
                if place in self._cache:
                    del self._cache[place]

    def _get_diff(self):
        return self.engine.handle(
            'character_places_diff',
            (self.name,)
        )

    def _cache_munge(self, k, v):
        return PlaceProxy(
            self.engine, self.name, k
        )

    def _set_item(self, k, v):
        self.engine.handle(
            'set_place',
            (self.name, k, v),
            silent=True
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            'del_node',
            (self.name, k),
            silent=True
        )
        del self.engine._node_stat_cache[self.name][k]


class SuccessorsProxy(CachingProxy):
    @property
    def _cache(self):
        return self.engine._character_portals_cache[
            self._charname][self._nodeA]

    def __init__(self, engine_proxy, charname, nodeAname):
        self._charname = charname
        self._nodeA = nodeAname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, SuccessorsProxy) and
            self.engine is other.engine and
            self._charname == other._charname and
            self._nodeA == other._nodeA
        )

    def _get_state(self):
        return {
            node: self._cache[node] if node in self._cache else
            PortalProxy(self.engine, self._charname, self._nodeA, node)
            for node in self.engine.handle(
                    'node_successors',
                    (self._charname, self._nodeA)
            )
        }

    def _apply_diff(self, diff):
        raise NotImplementedError(
            "Apply the diff on CharSuccessorsMappingProxy"
        )

    def _get_diff(self):
        return self.engine.handle(
            'node_successors_diff',
            (self._charname, self._nodeA)
        )

    def _cache_munge(self, k, v):
        return PortalProxy(
            self.engine,
            self._charname,
            self._nodeA,
            k
        )

    def _set_item(self, nodeB, value):
        self.engine.handle(
            'set_portal',
            (self._character, self._nodeA, nodeB, value),
            silent=True
        )

    def _del_item(self, nodeB):
        self.engine.del_portal(
            self._charname, self._nodeA, nodeB
        )


class CharSuccessorsMappingProxy(CachingProxy):
    @property
    def character(self):
        return self.engine.character[self._charname]

    @property
    def _cache(self):
        return self.engine._character_portals_cache[self.name]

    def __init__(self, engine_proxy, charname):
        self.name = charname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, CharSuccessorsMappingProxy) and
            other.engine is self.engine and
            other.name == self.name
        )

    def _cache_munge(self, k, v):
        return {
            vk: PortalProxy(self.engine, self.name, vk, vv)
            for (vk, vv) in v.items()
        }

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No portals from {}".format(k))
        return SuccessorsProxy(
            self.engine,
            self.name,
            k
        )

    def _apply_diff(self, diff):
        for ((o, d), ex) in diff.items():
            if ex:
                if d not in self._cache[o]:
                    self._cache[o][d] = PortalProxy(
                        self.engine,
                        self.name,
                        o, d
                    )
            else:
                if o in self._cache and d in self._cache[o]._cache:
                    del self._cache[o]._cache[d]
                    if not self._cache[o]._cache:
                        del self._cache[o]

    def _get_diff(self):
        return self.engine.handle(
            'character_nodes_with_successors_diff',
            (self.name,)
        )

    def _set_item(self, nodeA, val):
        self.engine.handle(
            'character_set_node_successors',
            (self.name, nodeA, val),
            silent=True
        )

    def _del_item(self, nodeA):
        for nodeB in self[nodeA]:
            self.engine.del_portal(
                self.name, nodeA, nodeB
            )


class PredecessorsProxy(MutableMapping):
    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname, nodeBname):
        self.engine = engine_proxy
        self._charname = charname
        self.name = nodeBname

    def __iter__(self):
        yield from self.engine.handle(
            'node_predecessors',
            (self._charname, self.name)
        )

    def __len__(self):
        return self.engine.handle(
            'node_predecessors_len',
            (self._charname, self.name)
        )

    def __contains__(self, k):
        if (
            k in self.character.portal._cache and
            self.name in self.character.portal._cache[k]
        ):
            return True
        return self.engine.handle(
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
                self.engine, self._charname, k, self.name
            )
        return self.character.portal._cache[k][self.name]

    def __setitem__(self, k, v):
        self.engine.handle(
            'set_place',
            (self._charname, k, v),
            silent=True
        )
        self.engine.handle(
            'set_portal',
            (self._charname, k, self.name),
            silent=True
        )

    def __delitem__(self, k):
        self.engine.del_portal(
            self._charname, k, self.name
        )


class CharPredecessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self.engine.handle(
            'node_has_predecessor',
            (self.name, k)
        )

    def __iter__(self):
        yield from self.engine.handle(
            'character_nodes_with_predecessors',
            (self.name,)
        )

    def __len__(self):
        return self.engine.handle(
            'character_nodes_with_predecessors_len',
            (self.name,)
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(
                "No predecessors to {} (if it even exists)".format(k)
            )
        if k not in self._cache:
            self._cache[k] = PredecessorsProxy(self.engine, self.name, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        self.engine.handle(
            'character_set_node_predecessors',
            (self.name, k, v),
            silent=True
        )

    def __delitem__(self, k):
        for v in self[k]:
            self.engine.del_portal(
                self.name, k, v
            )


class CharStatProxy(CachingEntityProxy):
    @property
    def _cache(self):
        return self.engine._char_stat_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.name = character
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, CharStatProxy) and
            self.engine is other.engine and
            self.name == other.name
        )

    def _get_state(self):
        return self.engine.handle(
            'character_stat_copy',
            (self.name,)
        )

    def _get_diff(self):
        return self.engine.handle(
            'character_stat_diff',
            (self.name,)
        )

    def _set_item(self, k, v):
        self.engine.handle(
            'set_character_stat',
            (self.name, k, v),
            silent=True
        )

    def _del_item(self, k):
        self.engine.handle(
            'del_character_stat',
            (self.name, k),
            silent=True
        )


class RuleProxy(object):
    @property
    def triggers(self):
        return self.engine.handle(
            'get_rule_triggers',
            (self.name,)
        )

    @triggers.setter
    def triggers(self, v):
        self.engine.handle(
            'set_rule_triggers',
            (self.name, v),
            silent=True
        )

    @property
    def prereqs(self):
        return self.engine.handle(
            'get_rule_prereqs',
            (self.name,)
        )

    @prereqs.setter
    def prereqs(self, v):
        self.engine.handle(
            'set_rule_prereqs',
            (self.name, v),
            silent=True
        )

    @property
    def actions(self):
        return self.engine.handle(
            'get_rule_actions',
            (self.name,)
        )

    @actions.setter
    def actions(self, v):
        self.engine.handle(
            'set_rule_actions',
            (self.name, v),
            silent=True
        )

    def __init__(self, engine_proxy, rulename):
        self.engine = engine_proxy
        self.name = self._name = rulename

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            self.name == other.name
        )


class RuleBookProxy(MutableSequence):
    def __init__(self, engine_proxy, bookname):
        self.engine = engine_proxy
        self.name = bookname
        self._cache = self.engine.handle(
            'get_rulebook_rules',
            (self.name,)
        )
        self._proxy_cache = {}

    def __iter__(self):
        for k in self._cache:
            if k not in self._proxy_cache:
                self._proxy_cache[k] = RuleProxy(self.engine, k)
            yield self._proxy_cache[k]

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, i):
        k = self._cache[i]
        if k not in self._proxy_cache:
            self._proxy_cache[k] = RuleProxy(self.engine, k)
        return self._proxy_cache[k]

    def __setitem__(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache[i] = v
        self.engine.handle(
            'set_rulebook_rule',
            (self.name, i, v),
            silent=True
        )

    def __delitem__(self, i):
        del self._cache[i]
        self.engine.handle(
            'del_rulebook_rule',
            (self.name, i),
            silent=True
        )

    def insert(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache.insert(i, v)
        self.engine.handle(
            'ins_rulebook_rule',
            (self.name, i, v),
            silent=True
        )


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
            self.engine,
            self.engine.handle(
                'get_character_rulebook',
                (self.name,)
            )
        )

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        self.adj = self.succ = self.portal = CharSuccessorsMappingProxy(
            self.engine, self.name
        )
        self.pred = self.preportal = CharPredecessorsMappingProxy(
            self.engine, self.name
        )
        self.thing = ThingMapProxy(self.engine, self.name)
        self.place = PlaceMapProxy(self.engine, self.name)
        self.node = NodeMapProxy(self.engine, self.name)
        self.stat = CharStatProxy(self.engine, self.name)

    def __bool__(self):
        """It means something that I exist, even if I don't have any data yet."""
        return True

    def __eq__(self, other):
        if hasattr(other, 'engine'):
            oe = other.engine
        else:
            return False
        return (
            self.engine is oe and
            hasattr(other, 'name') and
            self.name == other.name
        )

    def __iter__(self):
        yield from self.engine.handle(
            'character_nodes',
            (self.name,)
        )

    def __len__(self):
        return self.engine.handle(
            'character_nodes_len',
            (self.name,)
        )

    def __contains__(self, k):
        if k == 'name':
            return True
        return k in self.node

    def __getitem__(self, k):
        if k == 'name':
            return self.name
        return self.node[k]

    def __setitem__(self, k, v):
        self.node[k] = v

    def __delitem__(self, k):
        del self.node[k]

    def _apply_diff(self, diff):
        self.stat._apply_diff(diff['character_stat'])
        self.thing._apply_diff(diff['things'])
        self.place._apply_diff(diff['places'])
        self.portal._apply_diff(diff['portals'])
        for (node, nodediff) in diff['node_stat'].items():
            if node in self.node:
                # if the node had its stats changed, and THEN
                # got deleted...
                # really it shouldn't send the stat changes anyway,
                # but let's be safe
                self.node[node]._apply_diff(nodediff)
        for ((o, d), portdiff) in diff['portal_stat'].items():
            if o in self.portal and d in self.portal[o]:
                self.portal[o][d]._apply_diff(diff['portal_stat'])

    def add_place(self, name, **kwargs):
        self[name] = kwargs

    def add_places_from(self, seq):
        self.engine.handle(
            'add_places_from',
            (self.name, list(seq))
        )
        for pln in seq:
            self.place._cache[pln] = PlaceProxy(
                self.engine, self.name, pln
            )

    def add_nodes_from(self, seq):
        self.add_places_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        self.engine.handle(
            'add_thing',
            (self.name, name, location, next_location, kwargs)
        )
        self.thing._cache[name] = ThingProxy(
            self.engine, self.name, name
        )

    def add_things_from(self, seq):
        self.engine.handle(
            'add_things_from',
            (self.name, seq)
        )
        for thn in seq:
            self.thing._cache[thn] = ThingProxy(
                self.engine, self.name, thn
            )

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def new_thing(self, name, location, next_location=None, **kwargs):
        self.add_thing(name, location, next_location, **kwargs)
        return self.thing[name]

    def place2thing(self, name, location, next_location=None):
        self.engine.handle(
            'place2thing',
            (self.name, name, location, next_location)
        )

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.engine.handle(
            'add_portal',
            (self.name, origin, destination, symmetrical, kwargs)
        )
        if origin not in self.portal._cache:
            self.portal._cache[origin] = SuccessorsProxy(
                self.engine,
                self.name,
                origin
            )
        self.portal[origin]._cache[destination] = PortalProxy(
            self.engine,
            self.name,
            origin,
            destination
        )

    def add_portals_from(self, seq, symmetrical=False):
        l = list(seq)
        self.engine.handle(
            'add_portals_from',
            (self.name, l, symmetrical)
        )
        for (origin, destination) in l:
            if origin not in self.portal._cache:
                self.portal._cache[origin] = SuccessorsProxy(
                    self.engine,
                    self.name,
                    origin
                )
            self.portal[origin]._cache[destination] = PortalProxy(
                self.engine,
                self.name,
                origin,
                destination
            )

    def new_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.add_portal(origin, destination, symmetrical, **kwargs)
        return self.portal[origin][destination]

    def portals(self):
        yield from self.engine.handle(
            'character_portals',
            (self.name,)
        )

    def add_avatar(self, a, b=None):
        self.engine.handle(
            'add_avatar',
            (self.name, a, b)
        )

    def del_avatar(self, a, b=None):
        self.engine.handle(
            'del_avatar',
            (self.name, a, b)
        )

    def avatars(self):
        yield from self.engine.handle(
            'character_avatars',
            (self.name,)
        )

    def facade(self):
        return Facade(self)


class CharacterMapProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = {}

    def __iter__(self):
        yield from self.engine.handle('characters')

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self.engine.handle(
            'have_character', (k,)
        )

    def __len__(self):
        return self.engine.handle('characters_len')

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No character: {}".format(k))
        if k not in self._cache:
            self._cache[k] = CharacterProxy(self.engine, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        if isinstance(v, CharacterProxy):
            return
        self.engine.handle(
            'set_character', (k, v), silent=True
        )
        self._cache[k] = CharacterProxy(self.engine, k)

    def __delitem__(self, k):
        self.engine.handle('del_character', (k,), silent=True)
        if k in self._cache:
            del self._cache[k]


class StringStoreProxy(MutableMapping):
    @property
    def language(self):
        return self._proxy.handle('get_language')

    @language.setter
    def language(self, v):
        self._proxy.handle('set_language', (v,))

    def __init__(self, engine_proxy):
        self._proxy = engine_proxy

    def __iter__(self):
        yield from self._proxy.handle('get_string_ids')

    def __contains__(self, k):
        return self._proxy.handle('have_string', (k,))

    def __len__(self):
        return self._proxy.handle('count_strings')

    def __getitem__(self, k):
        return self._proxy.handle('get_string', (k,))

    def __setitem__(self, k, v):
        self._proxy.handle('set_string', (k, v), silent=True)

    def __delitem__(self, k):
        self._proxy.handle('del_string', (k,), silent=True)

    def lang_items(self, lang=None):
        yield from self._proxy.handle(
            'get_string_lang_items', (lang,)
        )


class EternalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy

    def __contains__(self, k):
        return self.engine.handle(
            'have_eternal', (k,)
        )

    def __iter__(self):
        yield from self.engine.handle(
            'eternal_keys'
        )

    def __len__(self):
        return self.engine.handle('eternal_len')

    def __getitem__(self, k):
        return self.engine.handle(
            'get_eternal', (k,)
        )

    def __setitem__(self, k, v):
        self.engine.handle(
            'set_eternal',
            (k, v),
            silent=True
        )

    def __delitem__(self, k):
        self.engine.handle(
            'del_eternal', (k,),
            silent=True
        )


class GlobalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self._proxy = engine_proxy

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


class AllRuleBooksProxy(Mapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = {}

    def __iter__(self):
        yield from self.engine.handle('rulebooks')

    def __len__(self):
        return self.engine.handle('len_rulebooks')

    def __contains__(self, k):
        if k in self._cache:
            return True
        return self.engine.handle('have_rulebook', (k,))

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No rulebook: {}".format(k))
        if k not in self._cache:
            self._cache[k] = RuleBookProxy(self.engine, k)
        return self._cache[k]


class AllRulesProxy(Mapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = {}

    def __iter__(self):
        yield from self.engine.handle('list_all_rules')

    def __len__(self):
        return self.engine.handle('count_all_rules')

    def __contains__(self, k):
        return self.engine.handle('have_rule', (k,))

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No rule: {}".format(k))
        if k not in self._cache:
            self._cache[k] = RuleProxy(self.engine, k)
        return self._cache[k]

    def new_empty(self, k):
        self.engine.handle('new_empty_rule', (k,), silent=True)
        self._cache[k] = RuleProxy(self.engine, k)
        return self._cache[k]


class FuncStoreProxy(object):
    def __init__(self, engine_proxy, store):
        self.engine = engine_proxy
        self._store = store

    def __iter__(self):
        yield from self.engine.handle(
            'keys_in_store', (self._store,)
        )

    def __len__(self):
        return self.engine.handle(
            'len_store',
            (self._store,)
        )

    def plain(self, k):
        return self.engine.handle(
            'plain_source',
            (self._store, k)
        )

    def iterplain(self):
        yield from self.engine.handle(
            'plain_items_in_store',
            (self._store,)
        )

    def set_source(self, func_name, source):
        self.engine.handle(
            'store_set_source',
            (self._store, func_name, source)
        )


class ChangeSignatureError(TypeError):
    pass


class EngineProxy(AbstractEngine):
    char_cls = CharacterProxy
    node_cls = NodeProxy
    portal_cls = PortalProxy

    @property
    def branch(self):
        return self._branch

    @branch.setter
    def branch(self, v):
        self.handle('set_branch', (v,), silent=True)
        self._branch = v

    @property
    def tick(self):
        return self._tick

    @tick.setter
    def tick(self, v):
        self.handle('set_tick', (v,), silent=True)
        self._tick = v

    @property
    def time(self):
        return (self._branch, self._tick)

    @time.setter
    def time(self, v):
        self.handle('set_time', (v,), silent=True)
        (self._branch, self._tick) = v

    @reify
    def eternal(self):
        return EternalVarProxy(self)

    @reify
    def universal(self):
        return GlobalVarProxy(self)

    @reify
    def character(self):
        return CharacterMapProxy(self)

    @reify
    def string(self):
        return StringStoreProxy(self)

    @reify
    def rulebook(self):
        return AllRuleBooksProxy(self)

    @reify
    def rule(self):
        return AllRulesProxy(self)

    @reify
    def action(self):
        return FuncStoreProxy(self, 'action')

    @reify
    def prereq(self):
        return FuncStoreProxy(self, 'prereq')

    @reify
    def trigger(self):
        return FuncStoreProxy(self, 'trigger')

    @reify
    def sense(self):
        return FuncStoreProxy(self, 'sense')

    @reify
    def function(self):
        return FuncStoreProxy(self, 'function')

    @reify
    def _node_stat_cache(self):
        r = defaultdict(  # character
            lambda: defaultdict(  # node
                dict  # stat: value
            )
        )
        for char in self.character:
            for (node, stats) in self.handle(
                    'character_nodes_stat_diff',
                    (char,)
            ).items():
                r[char][node] = stats
        return r

    @reify
    def _portal_stat_cache(self):
        r = defaultdict(  # character
            lambda: defaultdict(  # origin
                lambda: defaultdict(  # destination
                    dict  # stat: value
                )
            )
        )
        for char in self.character:
            diff = self.handle(
                'character_portals_stat_diff',
                (char,)
            )
            for orig in diff:
                for dest in diff[orig]:
                    r[char][orig][dest] = diff[orig][dest]
        return r

    @reify
    def _char_stat_cache(self):
        r = defaultdict(dict)
        for char in self.character:
            r[char] = self.handle(
                'character_stat_diff',
                (char,)
            )
        return r

    @reify
    def _character_things_cache(self):
        r = defaultdict(dict)
        for char in self.character:
            for (thing, ex) in self.handle(
                    'character_things_diff',
                    (char,)
            ).items():
                if ex:
                    r[char][thing] = ThingProxy(
                        self, char, thing
                    )
        return r

    @reify
    def _character_places_cache(self):
        r = defaultdict(dict)
        for char in self.character:
            for (place, ex) in self.handle(
                    'character_places_diff',
                    (char,)
            ).items():
                if ex:
                    r[char][place] = PlaceProxy(
                        self, char, place
                    )
        return r

    @reify
    def _character_portals_cache(self):
        r = defaultdict(
            lambda: defaultdict(dict)
        )
        for char in self.character:
            for ((orig, dest), ex) in self.handle(
                'character_portals_diff',
                (char,)
            ).items():
                if ex:
                    r[char][orig][dest] = PortalProxy(
                        self,
                        char,
                        orig,
                        dest
                    )
        return r

    def __init__(self, handle_out, handle_in, logger):
        self._handle_out = handle_out
        self._handle_in = handle_in
        self.logger = logger
        (self._branch, self._tick) = self.handle('get_watched_time')

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def handle(self, func_name, args=[], silent=False):
        self._handle_out.send(self.json_dump((silent, func_name, args)))
        if not silent:
            return self.json_load(self._handle_in.recv())

    def json_rewrap(self, r):
        if isinstance(r, tuple):
            if r[0] in ('JSONListReWrapper', 'JSONReWrapper'):
                cls = JSONReWrapper if r[0] == 'JSONReWrapper' else JSONListReWrapper
                if r[1] == 'character':
                    (charn, k, v) = r[2:]
                    return cls(CharacterProxy(self, charn), k, v)
                elif r[1] == 'node':
                    (char, node, k, v) = r[2:]
                    try:
                        node = self.character[char].node[node]
                    except KeyError:
                        node = PlaceProxy(self, char, node)
                    return cls(node, k, v)
                else:
                    assert (r[1] == 'portal')
                    (char, nodeA, nodeB, k, v) = r[2:]
                    return cls(PortalProxy(self, char, nodeA, nodeB), k, v)
            else:
                return tuple(self.json_rewrap(v) for v in r)
        elif isinstance(r, dict):
            # These can't have been stored in a stat
            return {k: self.json_rewrap(v) for (k, v) in r.items()}
        elif isinstance(r, list):
            return [self.json_rewrap(v) for v in r]
        return r

    def json_load(self, s):
        return self.json_rewrap(super().json_load(s))

    def _call_with_recv(self, char, *cbs):
        received = self.json_load(self._handle_in.recv())
        for cb in cbs:
            cb(char, received)

    def _upd_char_cache(self, char, chardiff):
        self.character[char]._apply_diff(chardiff)

    def next_tick(self, char=None, cb=None):
        if cb and not char:
            raise TypeError("Callbacks require char name")
        if char:
            self._handle_out.send(self.json_dump((False, 'next_tick', [char])))
            Thread(
                target=self._call_with_recv,
                args=(char, self._upd_char_cache, cb) if cb else
                (char, self._upd_char_cache,)
            ).start()
        else:
            self.handle('next_tick', (char,), silent=True)

    def add_character(self, name, data=None, **kwargs):
        self.handle('add_character', (name, data, kwargs))

    def new_character(self, name, **kwargs):
        self.add_character(name, **kwargs)
        return CharacterProxy(self, name)

    def del_character(self, name):
        self.handle('del_character', (name,))

    def del_node(self, charname, name):
        self.handle(
            'del_node', (charname, name), silent=True
        )
        if charname in self.character._cache:
            char = self.character[charname]
            if name in char.thing._cache:
                del char.thing._cache[name]
            if name in char.place._cache:
                del char.place._cache[name]

    def del_portal(self, charname, orig, dest):
        self.handle(
            'del_portal', (charname, orig, dest), silent=True
        )
        if (
                charname in self.character._cache and
                orig in self.character[charname].portal._cache and
                dest in self.character[charname].portal[orig]._cache
        ):
            del self.character[charname].portal[orig]._cache[dest]

    def commit(self):
        self.handle('commit')

    def close(self):
        self.handle('close')


def subprocess(
    args, kwargs, handle_out_pipe, handle_in_pipe, logq
):
    def log(typ, data):
        if typ == 'command':
            (cmd, args) = data
            logq.put((
                'debug',
                "calling {}{}".format(
                    cmd,
                    tuple(args)
                )
            ))
        else:
            logq.put((
                'debug',
                "returning {} (of type {})".format(data, repr(type(data)))
            ))
    engine_handle = EngineHandle(args, kwargs, callbacq, logq)

    while True:
        inst = handle_out_pipe.recv()
        if inst == 'shutdown':
            handle_out_pipe.close()
            handle_in_pipe.close()
            logq.close()
            return 0
        (silent, cmd, args) = engine_handle._real.json_load(inst)
        log('command', (cmd, args))
        r = getattr(engine_handle, cmd)(*args)
        if silent:
            continue
        log('result', r)
        handle_in_pipe.send(engine_handle._real.json_dump(r))


class EngineProcessManager(object):
    def start(self, *args, **kwargs):
        (handle_out_pipe_recv, self._handle_out_pipe_send) = Pipe(duplex=False)
        (handle_in_pipe_recv, handle_in_pipe_send) = Pipe(duplex=False)
        self.logq = Queue()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='[{levelname}] LiSE.proxy({process})\t{message}',
            style='{'
        )
        handlers = []
        logl = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        loglevel = None
        if 'loglevel' in kwargs:
            if kwargs['loglevel'] in logl:
                loglevel = logl[kwargs['loglevel']]
            del kwargs['loglevel']
        stdout = logging.StreamHandler(sys.stdout)
        stdout.set_name('stdout')
        handlers.append(stdout)
        handlers[0].setLevel(loglevel or logging.DEBUG)
        if 'logfile' in kwargs:
            try:
                fh = logging.FileHandler(kwargs['logfile'])
                handlers.append(fh)
                handlers[-1].setLevel(loglevel or logging.INFO)
            except OSError:
                pass
            del kwargs['logfile']

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self._p = Process(
            name='LiSE Life Simulator Engine (core)',
            target=subprocess,
            args=(
                args,
                kwargs,
                handle_out_pipe_recv,
                handle_in_pipe_send,
                self.logq
            )
        )
        self._p.daemon = True
        self._p.start()
        self.engine_proxy = EngineProxy(
            self._handle_out_pipe_send,
            handle_in_pipe_recv,
            self.logger,
        )
        return self.engine_proxy

    def sync_log(self, limit=None):
        def log_once():
            (level, message) = self.logq.get(block=False)
            getattr(self.logger, level)(message)
        n = 0
        while limit is None or n < limit:
            try:
                log_once()
                n += 1
            except Empty:
                return

    def shutdown(self):
        self.engine_proxy.close()
        self._handle_out_pipe_send.send('shutdown')
