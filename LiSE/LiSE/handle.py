# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Wrap a LiSE engine so you can access and control it using only
ordinary method calls.

"""
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from re import match
from collections import defaultdict
from functools import partial
from importlib import import_module
from allegedb.xjson import (
    JSONReWrapper,
    JSONListReWrapper
)
from .engine import Engine
from .util import dict_diff, set_diff


class EngineHandle(object):
    """A wrapper for a :class:`LiSE.Engine` object that runs in the same
    process, but with an API built to be used in a command-processing
    loop that takes commands from another process.

    It's probably a bad idea to use this class unless you're
    developing your own API.

    """
    def __init__(self, args, kwargs={}, logq=None, logfile=None, loglevel=None):
        """Instantiate an engine with the positional arguments ``args`` and
        the keyword arguments ``kwargs``.

        ``logq`` is a :class:`Queue` into which I'll put tuples of
        ``(loglevel, message)``.

        """
        kwargs.setdefault('logfun', self.log)
        self._real = Engine(*args, **kwargs)
        self._logq = logq
        self._loglevel = loglevel
        self._muted_chars = set()
        self.branch = self._real.branch
        self.turn = self._real.turn
        self.tick = self._real.tick
        self._node_stat_cache = defaultdict(dict)
        self._portal_stat_cache = defaultdict(
            lambda: defaultdict(dict)
        )
        self._char_stat_cache = {}
        self._char_av_cache = defaultdict(lambda: defaultdict(set))
        self._char_rulebooks_cache = {}
        self._char_nodes_rulebooks_cache = defaultdict(dict)
        self._char_portals_rulebooks_cache = defaultdict(
            lambda: defaultdict(dict)
        )
        self._char_nodes_cache = {}
        self._char_portals_cache = {}
        self._node_successors_cache = defaultdict(dict)
        self._strings_cache = {}
        self._eternal_cache = {}
        self._universal_cache = {}
        self._rule_cache = {}
        self._rulebook_cache = defaultdict(list)
        self._stores_cache = defaultdict(dict)

    def log(self, level, message):
        if isinstance(level, str):
            level = {
                'debug': 10,
                'info': 20,
                'warning': 30,
                'error': 40,
                'critical': 50
            }[level]
        if self._logq and level >= self._loglevel:
            self._logq.put((level, message))

    def debug(self, message):
        self.log(DEBUG, message)

    def info(self, message):
        self.log(INFO, message)

    def warning(self, message):
        self.log(WARNING, message)

    def error(self, message):
        self.log(ERROR, message)

    def critical(self, message):
        self.log(CRITICAL, message)

    def json_load(self, s):
        return self._real.json_load(s)

    def json_dump(self, o):
        return self._real.json_dump(o)

    def unwrap_character_stat(self, char, k, v):
        if isinstance(v, JSONReWrapper):
            return ('JSONReWrapper', 'character', char, k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return ('JSONListReWrapper', 'character', char, k, v._v)
        else:
            return v

    def unwrap_place_stat(self, place, k, v):
        if isinstance(v, JSONReWrapper):
            return ('JSONReWrapper', 'place', place.character.name, place.name, k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return ('JSONListReWrapper', 'place', place.character.name, place.name, k, v._v)
        else:
            return v

    def unwrap_thing_stat(self, thing, k, v):
        if isinstance(v, JSONReWrapper):
            return('JSONReWrapper', 'thing', thing.character.name, thing.name, thing['location'], thing['next_location'], thing['arrival_time'], thing['next_arrival_time'], k, v._v)
        elif isinstance(v, JSONListReWrapper):
            return('JSONListReWrapper', 'thing', thing.character.name, thing.name, thing['location'], thing['next_location'], thing['arrival_time'], thing['next_arrival_time'], k, v._v)
        else:
            return v

    def unwrap_node_stat(self, node, k, v):
        if hasattr(node, 'location'):
            return self.unwrap_thing_stat(node, k, v)
        else:
            return self.unwrap_place_stat(node, k, v)

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

    def get_chardiffs(self, chars, *, store=True):
        if chars == 'all':
            return {
                char: self.character_diff(char, store=store)
                for char in self._real.character.keys()
            }
        else:
            return {
                char: self.character_diff(char, store=store)
                for char in chars
            }

    def _upd_local_caches(self, diff=None):
        if diff is None:
            self._eternal_cache = dict(self._real.eternal)
            self._universal_cache = dict(self._real.universal)
            self._rulebook_cache = {
                rb: self.rulebook_copy(rb) for rb in self._real.rulebook
            }
            self._rule_cache = {
                r: self.rule_copy(r) for r in self._real.rule
            }
            self._strings_cache = self.strings_copy()
            # stores?
            for charn, char in self._real.character.items():
                self._char_stat_cache[charn] = self.character_stat_copy(charn)
                self._node_stat_cache[charn] = {
                    node: self.node_stat_copy(charn, node) for node in char.node
                }
                portst = self._portal_stat_cache[charn]
                for port in char.portals():
                    portst.setdefault(port.orig, {})[port.dest] = self.portal_stat_copy(charn, port.orig, port.dest)
                self._char_things_cache[charn] = self.character_things(char)
                self._char_places_cache[charn] = self.character_places(char)
                self._char_portals_cache[charn] = self.character_portals(char)
                self._char_rulebooks_cache[charn] = self.character_rulebooks_copy(char)
            return
        assert diff == self.get_slow_diff(store=False), "Incorrect world-state delta"

        def updd(d0, d1):
            for k, v in d1.items():
                if k not in ('location', 'next_location') and v is None:
                    del d0[k]
                else:
                    d0[k] = v
        updd(self._eternal_cache, diff.pop('eternal', {}))
        updd(self._universal_cache, diff.pop('universal', {}))
        updd(self._rulebook_cache, diff.pop('rulebooks', {}))
        updd(self._strings_cache, diff.pop('strings', {}))
        for rule, d in diff.pop('rules', {}).items():
            updd(self._rulebook_cache.setdefault(rule, {}), d)
        for char, d in diff.items():
            updd(self._char_nodes_cache.setdefault(char, {}), d.pop('nodes', {}))
            nodevd = self._node_stat_cache.setdefault(char, {})
            for node, val in d.pop('node_val', {}).items():
                updd(nodevd.setdefault(node, {}), val)
            edges = self._char_portals_cache.setdefault(char, set())
            for orig, dests in d.pop('edges', {}).items():
                for dest, exists in dests.items():
                    if exists:
                        edges.add((orig, dest))
                    else:
                        edges.remove((orig, dest))
            edgevd = self._portal_stat_cache.setdefault(char, {})
            for orig, dests in d.pop('edge_val', {}).items():
                for dest, val in dests.items():
                    updd(edgevd.setdefault(orig, {}).setdefault(dest, {}), val)

    def next_turn(self):
        ret, diff = self._real.next_turn()
        self.branch, self.turn, self.tick = self._real.btt()
        self._after_ret = partial(self._upd_local_caches, diff)
        return ret, diff

    def get_slow_diff(self, chars='all', store=True):
        diff = {}
        if chars:
            diff = self.get_chardiffs(chars, store=store)
        diff['eternal'] = self.eternal_diff(store=store)
        diff['universal'] = self.universal_diff(store=store)
        diff['rules'] = self.all_rules_diff(store=store)
        diff['rulebooks'] = self.all_rulebooks_diff(store=store)
        return diff

    def time_travel(self, branch, turn, tick=None, chars='all'):
        branch_from, turn_from, tick_from = self._real.btt()
        slow_diff = branch != branch_from
        self._real.time = (branch, turn)
        if tick is None:
            self.tick = tick = self._real.tick
        else:
            self._real.tick = tick
        self.branch = branch
        self.turn = turn
        if slow_diff:
            diff = self.get_slow_diff(chars)
        else:
            diff = self._real.get_delta(branch, turn_from, tick_from, self.turn, self.tick)
            self._after_ret = partial(self._upd_local_caches, diff)
        return None, diff

    def increment_branch(self, chars=[]):
        branch = self._real.branch
        m = match('(.*)([0-9]+)', branch)
        if m:
            stem, n = m.groups()
            branch = stem + str(int(n)+1)
        else:
            branch += '1'
        if branch in self._real._branches:
            if m:
                n = int(n)
            else:
                stem = branch[:-1]
                n = 1
            while stem + str(n) in self._real._branches:
                n += 1
            branch = stem + str(n)
        ret = {'branch': branch}
        self._real.branch = self.branch = branch
        if chars:
            ret.update(self.get_chardiffs(chars))
        return ret

    def add_character(self, char, data, attr):
        character = self._real.new_character(char, **attr)
        placedata = data.get('place', data.get('node', {}))
        for place, stats in placedata.items():
            character.add_place(place, **stats)
        thingdata = data.get('thing',  {})
        for thing, stats in thingdata.items():
            character.add_thing(thing, **stats)
        portdata = data.get('edge', data.get('portal', data.get('adj',  {})))
        for orig, dests in portdata.items():
            for dest, stats in dests.items():
                character.add_portal(orig, dest, **stats)

    def commit(self):
        self._real.commit()

    def close(self):
        self._real.close()

    def get_branch(self):
        return self._real.branch

    def get_watched_branch(self):
        return self.branch

    def set_branch(self, branch):
        self._real.branch = branch
        self.branch = branch

    def get_tick(self):
        return self._real.tick

    def get_watched_tick(self):
        return self.tick

    def set_tick(self, tick):
        self._real.tick = tick
        self.tick = tick

    def get_time(self):
        return self._real.time

    def get_watched_btt(self):
        return (self.branch, self.turn, self.tick)

    def get_language(self):
        return self._real.string.language

    def set_language(self, lang):
        self._real.string.language = lang
        return self.strings_diff()

    def get_string_ids(self):
        return list(self._real.string)

    def get_string_lang_items(self, lang):
        ret = list(self._real.string.lang_items(lang))
        for lang, (k, v) in ret:
            self._strings_cache[lang][k] = v
        return ret

    def strings_copy(self, lang=None):
        if lang is None:
            lang = self._real.string.language
        return dict(
            self._real.string.lang_items(lang)
        )

    def strings_diff(self):
        old = self._strings_cache
        new = dict(self._real.string)
        ret = dict_diff(old, new)
        self._strings_cache = new
        return ret

    def get_string(self, k):
        return self._real.string[k]

    def have_string(self, k):
        return k in self._real.string

    def set_string(self, k, v):
        self._real.string[k] = v
        self._strings_cache[k] = v

    def del_string(self, k):
        del self._real.string[k]
        del self._strings_cache[k]

    def get_eternal(self, k):
        ret = self._eternal_cache[k] = self._real.eternal[k]
        return ret

    def set_eternal(self, k, v):
        self._real.eternal[k] = v
        self._eternal_cache[k] = v

    def del_eternal(self, k):
        del self._real.eternal[k]
        del self._eternal_cache[k]

    def have_eternal(self, k):
        return k in self._real.eternal

    def eternal_copy(self):
        return dict(self._real.eternal)

    def eternal_diff(self, *, store=True):
        old = self._eternal_cache
        new = self.eternal_copy()
        if store:
            self._eternal_cache = new
        return dict_diff(old, new)

    def get_universal(self, k):
        ret = self._universal_cache[k] = self._real.universal[k]
        return ret

    def set_universal(self, k, v):
        self._real.universal[k] = v
        self._universal_cache[k] = v

    def del_universal(self, k):
        del self._real.universal[k]
        del self._universal_cache[k]

    def universal_copy(self):
        return dict(self._real.universal)

    def universal_diff(self, *, store=True):
        old = self._universal_cache
        new = self.universal_copy()
        if store:
            self._universal_cache = new
        return dict_diff(old, new)

    def init_character(self, char, statdict={}):
        if char in self._real.character:
            raise KeyError("Already have character {}".format(char))
        self._real.character[char] = {}
        self._real.character[char].stat.update(statdict)

    def del_character(self, char):
        del self._real.character[char]
        for cache in (
                self._char_stat_cache,
                self._node_stat_cache,
                self._portal_stat_cache,
                self._char_av_cache,
                self._char_rulebooks_cache,
                self._char_nodes_rulebooks_cache,
                self._char_portals_rulebooks_cache,
                self._char_things_cache,
                self._char_places_cache,
                self._char_portals_cache
        ):
            if char in cache:
                del cache[char]

    def character_stat_copy(self, char):
        return {
            k: self.unwrap_character_stat(char, k, v)
            for (k, v) in self._real.character[char].stat.items()
        }

    @staticmethod
    def _character_something_diff(char, cache, copier, *args, store=True):
        old = cache.get(char, {})
        new = copier(char, *args)
        if store:
            cache[char] = new
        return dict_diff(old, new)

    def character_stat_diff(self, char, *, store=True):
        return self._character_something_diff(
            char, self._char_stat_cache, self.character_stat_copy, store=store
        )

    def character_avatars_copy(self, char):
        return {
            graph: list(nodes.keys()) for (graph, nodes) in
            self._real.character[char].avatar.items()
        }

    def character_avatars_diff(self, char, *, store=True):
        return self._character_something_diff(
            char, self._char_av_cache, self.character_avatars_copy, store=store
        )

    def character_rulebooks_copy(self, char):
        chara = self._real.character[char]
        return {
            'character': chara.rulebook.name,
            'avatar': chara.avatar.rulebook.name,
            'thing': chara.thing.rulebook.name,
            'place': chara.place.rulebook.name,
            'portal': chara.portal.rulebook.name,
            'node': chara.node.rulebook.name
        }

    def character_rulebooks_diff(self, char, *, store=True):
        return self._character_something_diff(
            char, self._char_rulebooks_cache, self.character_rulebooks_copy, store=store
        )

    def character_nodes_rulebooks_copy(self, char, nodes='all'):
        chara = self._real.character[char]
        if nodes == 'all':
            nodeiter = iter(chara.node.values())
        else:
            nodeiter = (chara.node[k] for k in nodes)
        return {node.name: node.rulebook.name for node in nodeiter}

    def character_nodes_rulebooks_diff(self, char, nodes='all', *, store=True):
        return self._character_something_diff(
            char, self._char_nodes_rulebooks_cache,
            self.character_nodes_rulebooks_copy, nodes, store=store
        )

    def character_portals_rulebooks_copy(self, char, portals='all'):
        chara = self._real.character[char]
        result = defaultdict(dict)
        if portals == 'all':
            portiter = chara.portals()
        else:
            portiter = (chara.portal[orig][dest] for (orig, dest) in portals)
        for portal in portiter:
            result[portal['origin']][portal['destination']] \
                = portal.rulebook.name
        return result

    def character_portals_rulebooks_diff(self, char, portals='all', *, store=True):
        try:
            old = self._char_portals_rulebooks_cache.get(
                char, defaultdict(dict)
            )
            new = self.character_portals_rulebooks_copy(char, portals)
            if store:
                self._char_portals_rulebooks_cache[char] = new
            result = {}
            for origin in old:
                if origin in new:
                    result[origin] = dict_diff(old[origin], new[origin])
                else:
                    result[origin] = None
            for origin in new:
                if origin not in result:
                    result[origin] = new[origin]
            return result
        except KeyError:
            return None

    def character_diff(self, char, *, store=True):
        """Return a dictionary of changes to ``char`` since previous call."""
        ret = self.character_stat_diff(char, store=store)
        ret['nodes'] = self.character_nodes_diff(char, store=store)
        ret['edges'] = self.character_portals_diff(char, store=store)
        ret['avatars'] = self.character_avatars_diff(char, store=store)
        ret['rulebooks'] = self.character_rulebooks_diff(char, store=store)
        ret['node_rulebooks'] = self.character_nodes_rulebooks_diff(char, store=store)
        ret['portal_rulebooks'] = self.character_portals_rulebooks_diff(char, store=store)
        ret['node_val'] = self.character_nodes_stat_diff(char, store=store)
        ret['edge_val'] = self.character_portals_stat_diff(char, store=store)
        return ret

    def set_character_stat(self, char, k, v):
        self._real.character[char].stat[k] = v
        self._char_stat_cache.setdefault(char, {})[k] = v

    def del_character_stat(self, char, k):
        del self._real.character[char].stat[k]
        del self._char_stat_cache[char][k]

    def update_character_stats(self, char, patch):
        self._real.character[char].stat.update(patch)
        self._char_stat_cache.setdefault(char, {}).update(patch)

    def update_character(self, char, patch):
        self.update_character_stats(char, patch['character'])
        self.update_nodes(char, patch['node'])
        self.update_portals(char, patch['portal'])

    def characters(self):
        return list(self._real.character.keys())

    def set_character(self, char, v):
        self._real.character[char] = v

    def set_node_stat(self, char, node, k, v):
        self._real.character[char].node[node][k] = v
        self._node_stat_cache.setdefault(char, {})[node][k] = v

    def del_node_stat(self, char, node, k):
        del self._real.character[char].node[node][k]
        del self._node_stat_cache[char][node][k]

    def node_stat_copy(self, node_or_char, node=None):
        """Return a node's stats, prepared for pickling, in a dictionary."""
        if node is None:
            node = node_or_char
        else:
            node = self._real.character[node_or_char].node[node]
        unwrapper = self.unwrap_thing_stat if isinstance(node, self._real.thing_cls) else self.unwrap_place_stat
        return {
            k: unwrapper(node, k, v)
            for (k, v) in node.items()
            if k not in {
                    'arrival_time',
                    'next_arrival_time'
            }
        }

    def node_stat_diff(self, char, node, *, store=True):
        """Return a dictionary describing changes to a node's stats since the
        last time you looked at it.

        Deleted keys have the value ``None``. If the node's been deleted, this
        returns ``None``.

        """
        try:
            old = self._node_stat_cache[char].get(node, {})
            new = self.node_stat_copy(self._real.character[char].node[node])
            if store:
                self._node_stat_cache[char][node] = new
            r = dict_diff(old, new)
            return r
        except KeyError:
            return None

    def character_nodes_stat_diff(self, char, *, store=True):
        """Return a dictionary of ``node_stat_diff`` output for each node in a
        character.

        """
        r = {}
        for node in self._real.character[char].node:
            diff = self.node_stat_diff(char, node, store=store)
            if diff:
                r[node] = diff
        return r

    def update_node(self, char, node, patch):
        """Change a node's stats according to a dictionary.

        The ``patch`` dictionary should hold the new values of stats,
        keyed by the stats' names; a value of ``None`` deletes the
        stat.

        """
        character = self._real.character[char]
        if patch is None:
            del character.node[node]
        elif node not in character.node:
            character.node[node] = patch
            return
        else:
            character.node[node].update(patch)

    def update_nodes(self, char, patch, backdate=False):
        """Change the stats of nodes in a character according to a
        dictionary.

        """
        if backdate:
            parbranch, parrev = self._real._parentbranch_rev.get(
                self._real.branch, ('trunk', 0)
            )
            tick_now = self._real.tick
            self._real.tick = parrev
        for (n, npatch) in patch.items():
            self.update_node(char, n, npatch)
        if backdate:
            self._real.tick = tick_now

    def del_node(self, char, node):
        """Remove a node from a character."""
        del self._real.character[char].node[node]
        for cache in (
                self._char_nodes_rulebooks_cache,
                self._node_stat_cache,
                self._node_successors_cache
        ):
            try:
                del cache[char][node]
            except KeyError:
                pass
        if char in self._char_things_cache and node in self._char_things_cache[char]:
            del self._char_things_cache[char][node]
        if char in self._char_places_cache and node in self._char_places_cache[char]:
            self._char_places_cache[char].remove(node)
        if char in self._portal_stat_cache:
            portal_stat_cache_char = self._portal_stat_cache[char]
            if node in portal_stat_cache_char:
                del portal_stat_cache_char[node]
            for charo in portal_stat_cache_char.values():
                if node in charo:
                    del charo[node]
        if char in self._char_portals_rulebooks_cache:
            portal_rulebook_cache_char = self._char_portals_rulebooks_cache[char]
            if node in portal_rulebook_cache_char:
                del portal_rulebook_cache_char[node]
            for porto in portal_rulebook_cache_char.values():
                if node in porto:
                    del porto[node]

    def character_nodes(self, char):
        return list(self._real.character[char].node)

    def character_nodes_diff(self, char, *, store=True):
        try:
            old = self._char_nodes_cache.get(char, [])
            new = self.character_nodes(char)
            if store:
                self._char_nodes_cache[char] = new
            return set_diff(old, new)
        except KeyError:
            return None

    def node_predecessors(self, char, node):
        return list(self._real.character[char].pred[node].keys())

    def character_set_node_predecessors(self, char, node, preds):
        self._real.character[char].pred[node] = preds

    def character_del_node_predecessors(self, char, node):
        del self._real.character[char].pred[node]

    def node_successors(self, char, node):
        return list(self._real.character[char].portal[node].keys())

    def node_successors_diff(self, char, node):
        try:
            old = self._node_successors_cache[char].get(node, [])
            new = self.node_successors(char, node)
            self._node_successors_cache[char][node] = new
            return set_diff(old, new)
        except KeyError:
            return None

    def character_set_node_successors(self, char, node, val):
        self._real.character[char].adj[node] = val

    def character_del_node_successors(self, char, node):
        del self._real.character[char].adj[node]

    def nodes_connected(self, char, orig, dest):
        return dest in self._real.character[char].portal[orig]

    def init_thing(self, char, thing, statdict={}):
        if thing in self._real.character[char].thing:
            raise KeyError(
                'Already have thing in character {}: {}'.format(
                    char, thing
                )
            )
        return self.set_thing(char, thing, statdict)

    def set_thing(self, char, thing, statdict):
        self._real.character[char].thing[thing] = statdict
        self._node_stat_cache.setdefault(char, {})[thing] = statdict
        loc = statdict.pop('location')
        nxtloc = statdict.pop('next_location', None)
        arrt = statdict.pop('arrival_time', self.tick)
        nxtarrt = statdict.pop('next_arrival_time', None)
        self._char_things_cache.setdefault(char, {})[thing] = (loc, nxtloc, arrt, nxtarrt)

    def add_thing(self, char, thing, loc, next_loc, statdict):
        self._real.character[char].add_thing(
            thing, loc, next_loc, **statdict
        )
        self._node_stat_cache.setdefault(char, {})[thing] = statdict
        self._char_things_cache.setdefault(char, {})[thing] = (loc, next_loc, self.tick, None)

    def place2thing(self, char, node, loc):
        self._real.character[char].place2thing(node, loc)

    def thing2place(self, char, node):
        self._real.character[char].thing2place(node)

    def add_things_from(self, char, seq):
        self._real.character[char].add_things_from(seq)

    def get_thing_location(self, char, thing):
        try:
            return self._real.character[char].thing[thing]['location']
        except KeyError:
            return None

    def set_thing_location(self, char, thing, loc):
        self._real.character[char].thing[thing]['location'] = loc
        _, nxtloc, arrt, nxtarrt = self._char_things_cache.setdefault(char, {}).get(
            thing, (loc, None, self.tick, None)
        )
        self._char_things_cache[char][thing] = (loc, nxtloc, arrt, nxtarrt)

    def get_thing_special_stats(self, char, thing):
        try:
            thing = self._real.character[char].thing[thing]
        except KeyError:
            return (None, None, None, None)
        return (
            thing['location'],
            thing['next_location'],
            thing['arrival_time'],
            thing['next_arrival_time']
        )

    def thing_follow_path(self, char, thing, path, weight):
        return self._real.character[char].thing[thing].follow_path(path, weight)

    def thing_go_to_place(self, char, thing, place, weight):
        return self._real.character[char].thing[thing].go_to_place(place, weight)

    def thing_travel_to(self, char, thing, dest, weight, graph):
        return self._real.character[char].thing[thing].travel_to(dest, weight, graph)

    def thing_travel_to_by(
            self, char, thing, dest, arrival_tick, weight, graph
    ):
        return self._real.character[char].thing[thing].travel_to_by(
            dest, arrival_tick, weight, graph
        )

    def init_place(self, char, place, statdict={}):
        if place in self._real.character[char].place:
            raise KeyError(
                'Already have place in character {}: {}'.format(
                    char, place
                )
            )
        return self.set_place(char, place, statdict)

    def set_place(self, char, place, statdict):
        self._real.character[char].place[place] = statdict
        self._node_stat_cache.setdefault(char, {})[place] = statdict

    def add_places_from(self, char, seq):
        self._real.character[char].add_places_from(seq)

    def init_portal(self, char, orig, dest, statdict={}):
        if (
                orig in self._real.character[char].portal and
                dest in self._real.character[char].portal[orig]
        ):
            raise KeyError(
                'Already have portal in character {}: {}->{}'.format(
                    char, orig, dest
                )
            )
        return self.set_portal(char, orig, dest, statdict)

    def set_portal(self, char, orig, dest, statdict):
        self._real.character[char].portal[orig][dest] = statdict
        self._portal_stat_cache.setdefault(char, {})[orig][dest] = statdict

    def character_portals(self, char):
        r = set()
        portal = self._real.character[char].portal
        for o in portal:
            for d in portal[o]:
                r.add((o, d))
        return r

    def character_portals_diff(self, char, *, store=True):
        try:
            old = self._char_portals_cache.get(char, {})
            new = self.character_portals(char)
            if store:
                self._char_portals_cache[char] = new
            return set_diff(old, new)
        except KeyError:
            return None

    def add_portal(self, char, orig, dest, symmetrical, statdict):
        self._real.character[char].add_portal(
            orig, dest, symmetrical, **statdict
        )

    def add_portals_from(self, char, seq, symmetrical):
        self._real.character[char].add_portals_from(seq, symmetrical)

    def del_portal(self, char, orig, dest):
        del self._real.character[char].portal[orig][dest]
        try:
            del self._portal_stat_cache[char][orig][dest]
        except KeyError:
            pass

    def set_portal_stat(self, char, orig, dest, k, v):
        self._real.character[char].portal[orig][dest][k] = v
        self._portal_stat_cache.setdefault(char, {}).setdefault(orig, {})[dest][k] = v

    def del_portal_stat(self, char, orig, dest, k):
        del self._real.character[char][orig][dest][k]
        del self._portal_stat_cache[char][orig][dest][k]

    def portal_stat_copy(self, char, orig, dest):
        return {
            k: self.unwrap_portal_stat(char, orig, dest, k, v)
            for (k, v) in self._real.character[char].portal[orig][dest].items()
        }

    def portal_stat_diff(self, char, orig, dest, *, store=True):
        try:
            old = self._portal_stat_cache[char][orig].get(dest, {})
            new = self.portal_stat_copy(char, orig, dest)
            if store:
                self._portal_stat_cache[char][orig][dest] = new
            return dict_diff(old, new)
        except KeyError:
            return None

    def character_portals_stat_diff(self, char, *, store=True):
        r = {}
        for orig in self._real.character[char].portal:
            for dest in self._real.character[char].portal[orig]:
                diff = self.portal_stat_diff(char, orig, dest, store=store)
                if diff:
                    if orig not in r:
                        r[orig] = {}
                    r[orig][dest] = diff
        return r

    def update_portal(self, char, orig, dest, patch):
        character = self._real.character[char]
        if patch is None:
            del character.portal[orig][dest]
        elif orig not in character.portal \
             or dest not in character.portal[orig]:
            character.portal[orig][dest] = patch
        else:
            character.portal[orig][dest].update(patch)

    def update_portals(self, char, patch):
        branch = self.branch
        for ((orig, dest), ppatch) in patch.items():
            branch = self.update_portal(char, orig, dest, ppatch)
        return branch

    def add_avatar(self, char, graph, node):
        self._real.character[char].add_avatar(graph, node)
        self._char_av_cache.setdefault(char, {})[graph].add(node)

    def del_avatar(self, char, graph, node):
        self._real.character[char].del_avatar(graph, node)
        self._char_av_cache.setdefault(char, {})[graph].remove(node)

    def new_empty_rule(self, rule):
        self._real.rule.new_empty(rule)

    def new_empty_rulebook(self, rulebook):
        self._real.rulebook[rulebook]

    def rulebook_copy(self, rulebook):
        return list(self._real.rulebook[rulebook]._get_cache(*self._real.btt()))

    def rulebook_diff(self, rulebook, *, store=True):
        # TODO: do actual diffing
        old = self._rulebook_cache[rulebook]
        new = self.rulebook_copy(rulebook)
        if store:
            self._rulebook_cache[rulebook] = new
        if old == new:
            return
        return new

    def all_rulebooks_diff(self, *, store=True):
        ret = {}
        for rulebook in self._real.rulebook.keys():
            diff = self.rulebook_diff(rulebook, store=store)
            if diff:
                ret[rulebook] = diff
        return ret

    def set_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook][i] = rule

    def ins_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook].insert(i, rule)

    def del_rulebook_rule(self, rulebook, i):
        del self._real.rulebook[rulebook][i]
        del self._rulebook_cache[rulebook][i]

    def set_rule_triggers(self, rule, triggers):
        self._real.rule[rule].triggers = triggers

    def set_rule_prereqs(self, rule, prereqs):
        self._real.rule[rule].prereqs = prereqs

    def set_rule_actions(self, rule, actions):
        self._real.rule[rule].actions = actions

    def set_character_rulebook(self, char, rulebook):
        self._real.character[char].rulebook = rulebook

    def set_avatar_rulebook(self, char, rulebook):
        self._real.character[char].avatar.rulebook = rulebook

    def set_character_thing_rulebook(self, char, rulebook):
        self._real.character[char].thing.rulebook = rulebook

    def set_character_place_rulebook(self, char, rulebook):
        self._real.character[char].place.rulebook = rulebook

    def set_character_node_rulebook(self, char, rulebook):
        self._real.character[char].node.rulebook = rulebook

    def set_character_portal_rulebook(self, char, rulebook):
        self._real.character[char].portal.rulebook = rulebook

    def set_node_rulebook(self, char, node, rulebook):
        self._real.character[char].node[node].rulebook = rulebook

    def set_portal_rulebook(self, char, orig, dest, rulebook):
        self._real.character[char].portal[orig][dest].rulebook = rulebook

    def rule_copy(self, rule):
        branch, turn, tick = self.branch, self.turn, self.tick
        return {
            'triggers': list(self._real._triggers_cache.retrieve(rule, branch, turn, tick)),
            'prereqs': list(self._real._prereqs_cache.retrieve(rule, branch, turn, tick)),
            'actions': list(self._real._actions_cache.retrieve(rule, branch, turn, tick))
        }

    def rule_diff(self, rule, *, store=True):
        old = self._rule_cache.get(rule, {'triggers': [], 'prereqs': [], 'actions': []})
        new = self.rule_copy(rule)
        if store:
            self._rule_cache[rule] = new
        ret = {}
        if new['triggers'] != old['triggers']:
            ret['triggers'] = new['triggers']
        if new['prereqs'] != old['prereqs']:
            ret['prereqs'] = new['prereqs']
        if new['actions'] != old['actions']:
            ret['actions'] = new['actions']
        return ret

    def all_rules_diff(self, *, store=True):
        ret = {}
        for rule in self._real.rule.keys():
            diff = self.rule_diff(rule, store=store)
            if diff:
                ret[rule] = diff
        return ret

    def source_copy(self, store):
        return dict(getattr(self._real, store).iterplain())

    def source_diff(self, store):
        old = self._stores_cache.get(store, {})
        new = self._stores_cache[store] = self.source_copy(store)
        return dict_diff(old, new)

    def get_source(self, store, name):
        return getattr(self._real, store).get_source(name)

    def store_source(self, store, v, name=None):
        getattr(self._real, store).store_source(v, name)
        self._stores_cache.setdefault(store, {})[name or v.__name__] = v

    def del_source(self, store, k):
        delattr(getattr(self._real, store), k)
        try:
            del self._stores_cache[store][k]
        except KeyError:
            pass

    def call_stored_function(self, store, func, args, kwargs):
        if store == 'method':
            args = (self._real,) + tuple(args)
        store = getattr(self._real, store)
        if store not in self._real.stores:
            raise ValueError("{} is not a function store".format(store))
        callme = getattr(store, func)
        try:
            return callme(*args, **kwargs)
        except Exception as ex:
            raise

    def install_module(self, module):
        import_module(module).install(self._real)

    def do_game_start(self):
        self._real.game_start()
