# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Wrap a LiSE engine so you can access and control it using only
ordinary method calls.

"""
from collections import defaultdict
from gorm.xjson import (
    JSONReWrapper,
    JSONListReWrapper
)
from .engine import Engine
from .util import (
    dict_diff,
    list_diff
)


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
        self.tick += 1
        if char:
            return self.character_diff(char)

    def time_travel(self, branch, tick, char=None):
        self._real.time = (branch, tick)
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

    def update_character_stats(self, char, patch):
        self._real.character[char].stat.update(patch)

    def update_character(self, char, patch):
        self.update_character_stats(char, patch['character'])
        self.update_nodes(char, patch['node'])
        self.update_portals(char, patch['portal'])

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

    def update_node(self, char, node, patch):
        character = self._real.character[char]
        if patch is None:
            del character.node[node]
        elif node not in character.node:
            character.node[node] = patch
            return
        else:
            character.node[node].update(patch)

    def update_nodes(self, char, patch):
        for (n, npatch) in patch.items():
            self.update_node(char, n, npatch)

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

    def update_portal(self, char, o, d, patch):
        character = self._real.character[char]
        if patch is None:
            del character.portal[o][d]
        elif o not in character.portal or d not in character.portal[o]:
            character.portal[o][d] = patch
        else:
            character.portal[o][d].update(patch)

    def update_portals(self, char, patch):
        for ((o, d), ppatch) in patch.items():
            self.update_portal(char, o, d, ppatch)

    def character_avatars(self, char):
        return list(self._real.character[char].avatars())

    def add_avatar(self, char, a, b):
        self._real.character[char].add_avatar(a, b)

    def del_avatar(self, char, a, b):
        self._real.character[char].del_avatar(a, b)

    def get_rule_actions(self, rule):
        return self._real.rule[rule].actions._cache

    def set_rule_actions(self, rule, l):
        self._real.rule[rule].actions = l

    def get_rule_triggers(self, rule):
        return self._real.rule[rule].triggers._cache

    def set_rule_triggers(self, rule, l):
        self._real.rule[rule].triggers = l

    def get_rule_prereqs(self, rule):
        return self._real.rule[rule].prereqs._cache

    def set_rule_prereqs(self, rule, l):
        self._real.rule[rule].prereqs = l

    def list_all_rules(self):
        return list(self._real.rule.keys())

    def count_all_rules(self):
        return len(self._real.rule)

    def have_rule(self, k):
        return k in self._real.rule

    def new_empty_rule(self, k):
        self._real.rule.new_empty(k)

    def get_rulebook_rules(self, rulebook):
        return self._real.rulebook[rulebook]._cache

    def set_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook][i] = rule

    def ins_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook].insert(i, rule)

    def del_rulebook_rule(self, rulebook, i):
        del self._real.rulebook[rulebook][i]

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