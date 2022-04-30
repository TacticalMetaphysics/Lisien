# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Wrap a LiSE engine so you can access and control it using only
ordinary method calls.

"""
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from re import match
from collections import defaultdict
from functools import partial, wraps
from importlib import import_module
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, Tuple, Set, KeysView, Callable, Union

import numpy as np
import msgpack

from .engine import Engine, MSGPACK_SET


def _dict_delta_added(oldkeys: KeysView[bytes], new: Dict[bytes, bytes],
                      d: Dict[bytes, bytes]):
    for k in new.keys() - oldkeys:
        d[k] = new[k]


TRUE = msgpack.packb(True)
FALSE = msgpack.packb(False)
NONE = msgpack.packb(None)
NODES = msgpack.packb('nodes')
EDGES = msgpack.packb('edges')
UNITS = msgpack.packb('units')
RULEBOOK = msgpack.packb('rulebook')
RULEBOOKS = msgpack.packb('rulebooks')
NODE_VAL = msgpack.packb('node_val')
EDGE_VAL = msgpack.packb('edge_val')
ETERNAL = msgpack.packb('eternal')
UNIVERSAL = msgpack.packb('universal')
STRINGS = msgpack.packb('strings')
RULES = msgpack.packb('rules')
LOCATION = msgpack.packb('location')


def _dict_delta_removed(oldkeys: KeysView[bytes], newkeys: KeysView[bytes],
                        d: Dict[bytes, bytes]):
    for k in oldkeys - newkeys:
        d[k] = NONE


def _set_delta_added(old: Set[bytes], new: Set[bytes], d: Dict[bytes, bytes]):
    d.update((item, TRUE) for item in new.difference(old))


def set_delta(old: Set[bytes], new: Set[bytes]):
    r = {}
    added_thread = Thread(target=_set_delta_added, args=(old, new, r))
    added_thread.start()
    r.update((item, FALSE) for item in old.difference(new))
    added_thread.join()
    return r


def concat_d(r: Dict[bytes, bytes]) -> bytes:
    """Pack a dictionary of msgpack-encoded keys and values into msgpack bytes"""
    resp = msgpack.Packer().pack_map_header(len(r))
    for k, v in r.items():
        resp += k + v
    return resp


def concat_s(s: Set[bytes]) -> bytes:
    """Pack a set of msgpack-encoded values into a msgpack array with ext code"""
    resp = msgpack.Packer().pack_array_header(len(s))
    for v in s:
        resp += v
    data_len = len(resp)
    set_code = MSGPACK_SET.to_bytes(1, "big", signed=False)
    if data_len == 1:
        return b"\xd4" + set_code + resp
    elif data_len == 2:
        return b"\xd5" + set_code + resp
    elif data_len == 4:
        return b"\xd6" + set_code + resp
    elif data_len == 8:
        return b"\xd7" + set_code + resp
    elif data_len == 16:
        return b"\xd8" + set_code + resp
    elif data_len < 2**8:
        return b"\xc7" + data_len.to_bytes(1, "big",
                                           signed=False) + set_code + resp
    elif data_len < 2**16:
        return b"\xc8" + data_len.to_bytes(2, "big",
                                           signed=False) + set_code + resp
    elif data_len < 2**32:
        return b"\xc9" + data_len.to_bytes(4, "big",
                                           signed=False) + set_code + resp
    else:
        raise ValueError("Too long")


def timely(fun: Callable) -> Callable:

    @wraps(fun)
    def run_timely(self, *args, **kwargs):
        ret = fun(self, *args, **kwargs)
        self.branch, self.turn, self.tick = self._real._btt()
        return ret

    run_timely.timely = True
    return run_timely


def prepacked(fun: Callable) -> Callable:
    fun.prepacked = True
    return fun


def _packed_dict_delta(old: Dict[bytes, bytes],
                       new: Dict[bytes, bytes]) -> Dict[bytes, bytes]:
    """Describe changes from one shallow dictionary of msgpack data to another

    The returned dictionary indicates deleted keys with the value \xc0.
    Added or changed keys have their actual value.

    """

    r = {}
    added_thread = Thread(target=_dict_delta_added, args=(old.keys(), new, r))
    removed_thread = Thread(target=_dict_delta_removed,
                            args=(old.keys(), new.keys(), r))
    added_thread.start()
    removed_thread.start()
    ks = old.keys() & new.keys()
    oldv_l = []
    newv_l = []
    k_l = []
    for k in ks:
        # \xc1 is unused by msgpack.
        # I append it here so that the last byte will never be \x00,
        # which numpy uses as a terminator, but msgpack uses for
        # the integer zero.
        oldv_l.append(old[k] + b'\xc1')
        newv_l.append(new[k] + b'\xc1')
        k_l.append(k + b'\xc1')
    oldvs = np.array(oldv_l)
    newvs = np.array(newv_l)
    ks = np.array(k_l)
    changes = oldvs != newvs
    added_thread.join()
    removed_thread.join()
    if not (changes.any() or r):
        return {}
    if changes.any():
        changed_keys = ks[changes]
        changed_values = newvs[changes]
        for (k, v) in zip(changed_keys, changed_values):
            r[k[:-1]] = v[:-1]
    return r


class BytesDict(dict):

    def __getitem__(self, item: bytes) -> bytes:
        return super().__getitem__(item)

    def __setitem__(self, key: bytes, value: bytes):
        assert isinstance(key, bytes)
        assert isinstance(value, bytes)
        super().__setitem__(key, value)


class EngineHandle(object):
    """A wrapper for a :class:`LiSE.Engine` object that runs in the same
    process, but with an API built to be used in a command-processing
    loop that takes commands from another process.

    It's probably a bad idea to use this class unless you're
    developing your own API.

    This holds a cache of the last observed state of each
    :class:`LiSE.character.Character`. When updating the view on the
    simulation from your API, use the method `character_delta`
    to get a dictionary describing the changes since last you observed
    the :class:`LiSE.character.Character`.

    """

    def __init__(self, args=(), kwargs=None, logq=None, loglevel=None):
        """Instantiate an engine with the positional arguments ``args`` and
        the keyword arguments ``kwargs``.

        ``logq`` is a :class:`Queue` into which I'll put tuples of
        ``(loglevel, message)``. ``loglevel`` is one of
        `'debug'`, `'info'`, `'warning'`, `'error'`, or `'critical'`
        (or the constants from the `logging` module, or an integer)
        and controls what messages will be logged.

        """
        if kwargs is None:
            kwargs = {}
        kwargs.setdefault('logfun', self.log)
        self._real = Engine(*args, cache_arranger=False, **kwargs)
        self.pack = pack = self._real.pack

        def pack_pair(pair):
            k, v = pair
            return pack(k), pack(v)

        self.pack_pair = pack_pair
        self.unpack = unpack = self._real.unpack

        def unpack_pair(pair):
            k, v = pair
            return unpack(k), unpack(v)

        def unpack_dict(d: dict):
            return dict(map(unpack_pair, d.items()))

        self.unpack_dict = unpack_dict
        self._logq = logq
        self._loglevel = loglevel
        self._muted_chars = set()
        self.branch = self._real.branch
        self.turn = self._real.turn
        self.tick = self._real.tick
        # It's possible that the memoization I've done could break if this handle observes the world while it's in
        # a planning context. This shouldn't happen, I think
        self._node_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
        self._portal_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
        self._char_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
        self._char_units_copy_memo: Dict[tuple, Dict[bytes, Set[bytes]]] = {}
        self._char_rulebooks_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
        self._char_nodes_rulebooks_copy_memo: Dict[tuple, Dict[bytes,
                                                               bytes]] = {}
        self._char_portals_rulebooks_copy_memo: Dict[tuple, Dict[bytes,
                                                                 bytes]] = {}
        self._char_nodes_copy_memo: Dict[tuple, Set[bytes]] = {}
        self._char_portals_copy_memo: Dict[tuple, Set[bytes]] = {}
        self._node_successors_copy_memo: Dict[tuple, Set[bytes]] = {}
        self._strings_cache = {}
        self._strings_copy_memo = {}
        self._eternal_cache = BytesDict()
        self._eternal_copy_memo = {}
        self._universal_cache = BytesDict()
        self._universal_copy_memo = {}
        self._rule_cache = defaultdict(dict)
        self._rule_copy_memo = {}
        self._rulebook_copy_memo = {}
        self._stores_cache = defaultdict(BytesDict)
        self._character_delta_memo = defaultdict(
            lambda: defaultdict(lambda: defaultdict(BytesDict)))
        self._real.arrange_cache_signal.connect(self._precopy_at_time)
        self.threadpool = ThreadPoolExecutor(cpu_count())
        self._cache_arranger_started = False

    def _precopy_at_time(self, sender, *, branch, turn, tick):
        locks = self._real.locks
        btt = (branch, turn, tick)
        with locks:
            chars = list(self._real.character)
        for char in chars:
            with locks:
                self.character_stat_copy(char, btt=btt)
            with locks:
                self.character_nodes(char, btt=btt)
            with locks:
                self._character_nodes_stat_copy(char, btt=btt)
            with locks:
                self.character_portals(char, btt=btt)
            with locks:
                self._character_portals_stat_copy(char, btt=btt)
            with locks:
                self._character_units_copy(char, btt=btt)

    def log(self, level, message):
        if isinstance(level, str):
            level = {
                'debug': 10,
                'info': 20,
                'warning': 30,
                'error': 40,
                'critical': 50
            }[level.lower()]
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

    def time_locked(self):
        """Return whether the sim-time has been prevented from advancing"""
        return hasattr(self._real, 'locktime')

    @timely
    def advance(self):
        """Run one rule"""
        self._real.advance()

    def _get_char_deltas(self,
                         chars,
                         *,
                         btt_from: Tuple[str, int, int] = None,
                         btt_to: Tuple[str, int, int] = None):
        """Return a dict describing changes to characters since last call"""
        pack = self._real.pack
        ret = {}
        if chars == 'all':
            it = iter(self._real.character.keys())
        else:
            it = iter(chars)
        for char in it:
            delt = self._character_delta(char,
                                         btt_from=btt_from,
                                         btt_to=btt_to)
            if delt:
                ret[pack(char)] = delt
        return ret

    @prepacked
    def copy_character(self, char, *, btt=None) -> Dict[bytes, bytes]:
        units = self._character_units_copy(char, btt=btt)
        ports = self._character_portals_stat_copy(char, btt=btt)
        ported = {}
        for orig, dests in ports.items():
            dest_stats = {}
            for dest, stats in dests.items():
                dest_stats[dest] = concat_d(stats)
            ported[orig] = concat_d(dest_stats)
        return {
            NODES:
            concat_d(
                {node: TRUE
                 for node in self.character_nodes(char, btt=btt)}),
            EDGES:
            concat_d({
                origdest: TRUE
                for origdest in self.character_portals(char, btt=btt)
            }),
            UNITS:
            concat_d({k: concat_s(v)
                      for (k, v) in units.items()}),
            RULEBOOKS:
            concat_d(self.character_rulebooks_copy(char, btt=btt)),
            NODE_VAL:
            concat_d(self._character_nodes_stat_copy(char, btt=btt)),
            EDGE_VAL:
            concat_d(ported)
        }

    @prepacked
    def copy_chars(self, chars: Union[str, list]):
        if chars == 'all':
            it = iter(self._real.character.keys())
        else:
            it = iter(chars)
        return {
            self.pack(char): concat_d(self.copy_character(char))
            for char in it
        }

    @prepacked
    def get_char_deltas(self,
                        chars,
                        *,
                        btt_from: Tuple[str, int, int] = None,
                        btt_to: Tuple[str, int, int] = None):
        delt = self._get_char_deltas(chars, btt_from=btt_from, btt_to=btt_to)
        ret = {}
        for char, delta in delt.items():
            charret = ret[char] = {}
            if NODES in delta:
                charret[NODES] = concat_d(delta[NODES])
            if EDGES in delta:
                charret[EDGES] = concat_d(delta[EDGES])
            if UNITS in delta:
                graph_units = {}
                for graph, unitss in delta[UNITS].items():
                    graph_units[graph] = concat_d(unitss)
                charret[UNITS] = concat_d(graph_units)
            if RULEBOOKS in delta:
                charret[RULEBOOKS] = concat_d(delta[RULEBOOKS])
            if NODE_VAL in delta:
                charret[NODE_VAL] = concat_d({
                    node: concat_d(vals)
                    for node, vals in delta[NODE_VAL].items()
                })
        if not self._cache_arranger_started:
            self._real._start_cache_arranger()
            self._cache_arranger_started = True
        return {char: concat_d(charret) for char, charret in ret.items()}

    def _pack_delta(self, delta):
        pack = self.pack
        slightly_packed_delta = {}
        mostly_packed_delta = {}
        for char, chardelta in delta.items():
            pchar = pack(char)
            chard = slightly_packed_delta[pchar] = {}
            packd = mostly_packed_delta[pchar] = {}
            if 'nodes' in chardelta:
                nd = chard[NODES] = {
                    pack(node): pack(ex)
                    for node, ex in chardelta['nodes']
                }
                packd[NODES] = concat_d(nd)
            if 'node_val' in chardelta:
                slightnoded = chard[NODE_VAL] = {}
                packnodevd = {}
                for node, vals in chardelta['node_val'].items():
                    pnode = pack(node)
                    pvals = dict(map(self.pack_pair, vals.items()))
                    slightnoded[pnode] = pvals
                    packnodevd[pnode] = concat_d(pvals)
                packd[NODE_VAL] = concat_d(packnodevd)
            if 'edges' in chardelta:
                ed = chard[EDGES] = {
                    pack(origdest): pack(ex)
                    for origdest, ex in chardelta['edges']
                }
                packd[EDGES] = concat_d(ed)
            if 'edge_val' in chardelta:
                slightorigd = chard[EDGE_VAL] = {}
                packorigd = {}
                for orig, dests in chardelta['edge_val'].items():
                    porig = pack(orig)
                    slightdestd = slightorigd[porig] = {}
                    packdestd = {}
                    for dest, port in dests.items():
                        pdest = pack(dest)
                        slightportd = slightdestd[pdest] = dict(
                            map(self.pack_pair, port.items()))
                        packdestd[pdest] = concat_d(slightportd)
                    packorigd[porig] = concat_d(packdestd)
                packd[EDGE_VAL] = concat_d(packorigd)
            if 'units' in chardelta:
                slightgraphd = chard[UNITS] = {}
                packunitd = {}
                for graph, unitss in chardelta['units'].items():
                    pgraph = pack(graph)
                    slightunitd = slightgraphd[pgraph] = dict(
                        map(self.pack_pair, unitss.items()))
                    packunitd[pgraph] = concat_d(slightunitd)
                packd[UNITS] = concat_d(packunitd)
            if 'rulebooks' in chardelta:
                chard[RULEBOOKS] = slightrbd = dict(
                    map(self.pack_pair, chardelta['rulebooks'].items()))
                packd[RULEBOOKS] = concat_d(slightrbd)
        return slightly_packed_delta, concat_d({
            charn: concat_d(stuff)
            for charn, stuff in mostly_packed_delta.items()
        })

    @timely
    @prepacked
    def next_turn(self):
        pack = self.pack
        self.debug(
            'calling next_turn at {}, {}, {}'.format(*self._real._btt()))
        ret, delta = self._real.next_turn()
        slightly_packed_delta, packed_delta = self._pack_delta(delta)
        self.debug(
            'got results from next_turn at {}, {}, {}. Packing...'.format(
                *self._real._btt()))
        return pack(ret), packed_delta

    def _get_slow_delta(self,
                        chars='all',
                        btt_from: Tuple[str, int, int] = None,
                        btt_to: Tuple[str, int, int] = None):
        pack = self._real.pack
        delta = {}
        if chars:
            delta = self._get_char_deltas(chars,
                                          btt_from=btt_from,
                                          btt_to=btt_to)
        unid = self.universal_delta(btt_from=btt_from, btt_to=btt_to)
        if unid:
            delta[UNIVERSAL] = concat_d(unid)
        rud = self.all_rules_delta(btt_from=btt_from, btt_to=btt_to)
        if rud:
            delta[RULES] = pack(rud)
        rbd = self.all_rulebooks_delta(btt_from=btt_from, btt_to=btt_to)
        if rbd:
            delta[RULEBOOKS] = pack(rbd)
        return delta

    @timely
    @prepacked
    def time_travel(self, branch, turn, tick=None, chars='all'):
        # TODO: detect if you're headed to sometime outside of the already simulated past, and respond appropriately
        #       - refuse to time travel to a plan
        #       - refuse to go too far outside the past (I think no more than one turn)
        #       That last would also make a lot of sense as a restriction of the LiSE core...
        branch_from, turn_from, tick_from = self._real._btt()
        slow_delta = branch != branch_from
        self._real.time = (branch, turn)
        if tick is None:
            self.tick = tick = self._real.tick
        else:
            self._real.tick = tick
        self.branch = branch
        self.turn = turn
        if slow_delta:
            delta = self._get_slow_delta(chars)
            slightly_packed_delta, packed_delta = self._pack_delta(delta)
        else:
            delta = self._real.get_delta(branch, turn_from, tick_from, turn,
                                         tick)
            slightly_packed_delta, packed_delta = self._pack_delta(delta)
        return NONE, packed_delta

    @timely
    def increment_branch(self, chars: list = None):
        if chars is None:
            chars = []
        branch = self._real.branch
        m = match('(.*)([0-9]+)', branch)
        if m:
            stem, n = m.groups()
            branch = stem + str(int(n) + 1)
        else:
            stem = branch
            n = 1
            branch = stem + str(n)
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
            ret.update(self._get_char_deltas(chars))
        return ret

    @timely
    def add_character(self, char, data, attr):
        # Probably not great that I am unpacking and then repacking the stats
        pack_pair = self.pack_pair
        character = self._real.new_character(char, **attr)
        branch, turn, tick = self._get_btt()
        self._char_stat_copy_memo[char, branch, turn, tick] = BytesDict(
            map(pack_pair, attr.items()))
        placedata = data.get('place', data.get('node', {}))
        node_stat_memo = self._node_stat_copy_memo
        for place, stats in placedata.items():
            character.add_place(place, **stats)
            branch, turn, tick = self._get_btt()
            node_stat_memo[place, branch, turn,
                           tick] = dict(map(self.pack_pair, stats.items()))
        thingdata = data.get('thing', {})
        for thing, stats in thingdata.items():
            character.add_thing(thing, **stats)
            node_stat_memo[thing, branch, turn,
                           tick] = dict(map(self.pack_pair, stats.items()))
        portdata = data.get('edge', data.get('portal', data.get('adj', {})))
        port_stat_memo = self._portal_stat_copy_memo
        for orig, dests in portdata.items():
            for dest, stats in dests.items():
                character.add_portal(orig, dest, **stats)
                branch, turn, tick = self._get_btt()
                port_stat_memo[char, orig, dest, branch, turn,
                               tick] = dict(map(self.pack_pair, stats.items()))

    def commit(self):
        self._real.commit()

    def close(self):
        self._real.close()
        self.threadpool.shutdown()

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
        return self.branch, self.turn, self.tick

    def get_language(self):
        return str(self._real.string.language)

    def set_language(self, lang):
        self._real.string.language = lang
        return self.strings_copy(lang)

    def get_string_ids(self):
        return list(self._real.string)

    def get_string_lang_items(self, lang):
        return list(self._real.string.lang_items(lang))

    def strings_copy(self, lang=None):
        if lang is None:
            lang = self._real.string.language
        return dict(self._real.string.lang_items(lang))

    def get_string(self, k):
        return self._real.string[k]

    def have_string(self, k):
        return k in self._real.string

    def set_string(self, k, v):
        self._real.string[k] = v

    def del_string(self, k):
        del self._real.string[k]

    @prepacked
    def get_eternal(self, k):
        return self.pack(self._real.eternal[k])

    def set_eternal(self, k, v):
        self._real.eternal[k] = v
        self._eternal_cache[self.pack(k)] = self.pack(v)

    def del_eternal(self, k):
        del self._real.eternal[k]

    def have_eternal(self, k):
        return k in self._real.eternal

    @prepacked
    def eternal_copy(self):
        return dict(map(self.pack_pair, self._real.eternal.items()))

    def get_universal(self, k):
        return self._real.universal[k]

    @timely
    def set_universal(self, k, v):
        self._real.universal[k] = v

    @timely
    def del_universal(self, k):
        del self._real.universal[k]

    @prepacked
    def universal_copy(self, btt: Tuple[str, int, int] = None):
        btt = self._get_btt(btt)
        if btt in self._universal_copy_memo:
            return self._universal_copy_memo[btt]
        ret = self._universal_copy_memo[btt] = dict(
            map(self.pack_pair, self._real.universal.items()))
        return ret

    @prepacked
    def universal_delta(self, *, store=True):
        old = self._universal_cache
        new = self.universal_copy()
        if store:
            self._universal_cache = new
        return _packed_dict_delta(old, new)

    @timely
    def init_character(self, char, statdict: dict = None):
        if char in self._real.character:
            raise KeyError("Already have character {}".format(char))
        if statdict is None:
            statdict = {}
        self._real.character[char] = {}
        self._real.character[char].stat.update(statdict)
        self.character_stat_copy(char)

    @timely
    def del_character(self, char):
        del self._real.character[char]

    @prepacked
    def character_stat_copy(self, char, *, btt: Tuple[str, int, int] = None):
        if char not in self._real.character:
            raise KeyError("no such character")
        pack = self.pack
        key = self._get_btt(btt) + (char, )
        if key in self._char_stat_copy_memo:
            return self._char_stat_copy_memo[key]
        ret = self._char_stat_copy_memo[key] = {
            pack(k): pack(v.unwrap()) if hasattr(v, 'unwrap')
            and not hasattr(v, 'no_unwrap') else pack(v)
            for (k, v) in self._real.character[char].stat.items()
        }
        return ret

    @prepacked
    def _character_something_delta(self,
                                   char,
                                   copier,
                                   *args,
                                   btt_from: Tuple[str, int, int] = None,
                                   btt_to: Tuple[str, int, int] = None):
        old = copier(char, *args, btt=self._get_watched_btt(btt_from))
        new = copier(char, *args, btt=btt_to)
        return _packed_dict_delta(old, new)

    @prepacked
    def character_stat_delta(self,
                             char,
                             *,
                             btt_from: Tuple[str, int, int] = None,
                             btt_to: Tuple[str, int, int] = None):
        return self._character_something_delta(char,
                                               self.character_stat_copy,
                                               btt_from=btt_from,
                                               btt_to=btt_to)

    def _character_units_copy(
            self,
            char,
            *,
            btt: Tuple[str, int, int] = None) -> Dict[bytes, Set[bytes]]:
        pack = self._real.pack
        key = self._get_btt(btt) + (char, )
        memo = self._char_units_copy_memo
        if key in memo:
            return memo[key]
        ret = memo[key] = {
            pack(graph): set(map(pack, nodes.keys()))
            for (graph, nodes) in self._real.character[char].unit.items()
        }
        return ret

    def _character_units_delta(self,
                               char,
                               *,
                               btt_from: Tuple[str, int, int] = None,
                               btt_to: Tuple[str, int, int] = None):
        old = self._character_units_copy(char,
                                         btt=self._get_watched_btt(btt_from))
        new = defaultdict(set)
        new.update(self._character_units_copy(char, btt=btt_to))
        ret = {}
        for graph in old.keys() - new.keys():
            ret[graph] = {node: FALSE for node in old[graph]}
        for graph in new.keys() - old.keys():
            ret[graph] = {node: TRUE for node in new[graph]}
        for graph in old.keys() & new.keys():
            graph_nodes = {}
            for node in old[graph].difference(new[graph]):
                graph_nodes[node] = FALSE
            for node in new[graph].difference(old[graph]):
                graph_nodes[node] = TRUE
            if graph_nodes:
                ret[graph] = graph_nodes
        return ret

    @prepacked
    def character_rulebooks_copy(self, char, btt: Tuple[str, int, int] = None):
        chara = self._real.character[char]
        key = self._get_btt(btt) + (char, )
        if key in self._char_rulebooks_copy_memo:
            return self._char_rulebooks_copy_memo[key]
        ret = self._char_rulebooks_copy_memo[key] = dict(
            map(self.pack_pair, [('character', chara.rulebook.name),
                                 ('unit', chara.unit.rulebook.name),
                                 ('thing', chara.thing.rulebook.name),
                                 ('place', chara.place.rulebook.name),
                                 ('portal', chara.portal.rulebook.name)]))
        return ret

    @prepacked
    def character_rulebooks_delta(self,
                                  char,
                                  *,
                                  btt_from: Tuple[str, int, int] = None,
                                  btt_to: Tuple[str, int, int] = None):
        return self._character_something_delta(char,
                                               self.character_rulebooks_copy,
                                               btt_from=btt_from,
                                               btt_to=btt_to)

    @prepacked
    def character_nodes_rulebooks_copy(self,
                                       char,
                                       nodes='all',
                                       btt: Tuple[str, int, int] = None):
        key = self._get_btt(btt) + (char, nodes)
        if key in self._char_nodes_rulebooks_copy_memo:
            return self._char_nodes_rulebooks_copy_memo[key]
        chara = self._real.character[char]
        if nodes == 'all':
            nodeiter = iter(chara.node.values())
        else:
            nodeiter = (chara.node[k] for k in nodes)
        pack = self.pack
        ret = self._char_nodes_rulebooks_copy_memo[key] = {
            pack(node.name): pack(node.rulebook.name)
            for node in nodeiter
        }
        return ret

    @prepacked
    def character_nodes_rulebooks_delta(self,
                                        char,
                                        nodes='all',
                                        *,
                                        btt_from: Tuple[str, int, int] = None,
                                        btt_to: Tuple[str, int, int] = None):
        return self._character_something_delta(
            char,
            self.character_nodes_rulebooks_copy,
            nodes,
            btt_from=btt_from,
            btt_to=btt_to)

    def character_portals_rulebooks_copy(self,
                                         char,
                                         portals='all',
                                         btt: Tuple[str, int, int] = None):
        chara = self._real.character[char]
        result = defaultdict(dict)
        branch, turn, tick = self._get_btt(btt)
        origtime = self._get_btt()
        if (branch, turn, tick) != origtime:
            self._real._set_btt(branch, turn, tick)
        if portals == 'all':
            portiter = chara.portals()
        else:
            portiter = (chara.portal[orig][dest] for (orig, dest) in portals)
        for portal in portiter:
            result[portal['origin']][portal['destination']] \
                = portal.rulebook.name
        if (branch, turn, tick) != origtime:
            self._real._set_btt(*origtime)
        return result

    def _character_portals_rulebooks_delta(self,
                                           char,
                                           portals='all',
                                           *,
                                           btt_from: Tuple[str, int,
                                                           int] = None,
                                           btt_to: Tuple[str, int,
                                                         int] = None):
        old = self.character_portals_rulebooks_copy(
            char, portals, self._get_watched_btt(btt_from))
        new = {}
        for orig, dests in self.character_portals_rulebooks_copy(char,
                                                                 portals,
                                                                 btt=btt_to):
            new[orig] = dict(map(self.pack_pair, dests.items()))
        result = {}
        for origin in old:
            if origin in new:
                result[origin] = _packed_dict_delta(old[origin], new[origin])
            else:
                result[origin] = None
        for origin in new:
            if origin not in result:
                result[origin] = new[origin]
        return result

    def _character_delta(self,
                         char,
                         *,
                         btt_from: Tuple[str, int, int] = None,
                         btt_to: Tuple[str, int, int] = None) -> dict:
        observed_btt = self._get_watched_btt(btt_from)
        actual_btt = self._get_btt(btt_to)
        memo = self._character_delta_memo[char]
        if observed_btt in memo and actual_btt in memo[observed_btt]:
            return self._character_delta_memo[observed_btt][actual_btt]
        nodes_fut = self.threadpool.submit(self.character_nodes_delta,
                                           char,
                                           btt_from=observed_btt,
                                           btt_to=actual_btt)
        edges_fut = self.threadpool.submit(self.character_portals_delta,
                                           char,
                                           btt_from=observed_btt,
                                           btt_to=actual_btt)
        units_fut = self.threadpool.submit(self._character_units_delta,
                                           char,
                                           btt_from=observed_btt,
                                           btt_to=actual_btt)
        rbs_fut = self.threadpool.submit(self.character_rulebooks_delta,
                                         char,
                                         btt_from=observed_btt,
                                         btt_to=actual_btt)
        nrbs_fut = self.threadpool.submit(self.character_nodes_rulebooks_delta,
                                          char,
                                          btt_from=observed_btt,
                                          btt_to=actual_btt)
        porbs_fut = self.threadpool.submit(
            self._character_portals_rulebooks_delta,
            char,
            btt_from=observed_btt,
            btt_to=actual_btt)
        nv_fut = self.threadpool.submit(self._character_nodes_stat_delta,
                                        char,
                                        btt_from=observed_btt,
                                        btt_to=actual_btt)
        ev_fut = self.threadpool.submit(self._character_portals_stat_delta,
                                        char,
                                        btt_from=observed_btt,
                                        btt_to=actual_btt)
        chara = self._real.character[char]
        ret = self.character_stat_delta(char,
                                        btt_from=observed_btt,
                                        btt_to=actual_btt)
        nodes_res = nodes_fut.result()
        if nodes_res:
            ret[NODES] = nodes_res
        edges_res = edges_fut.result()
        if edges_res:
            ret[EDGES] = edges_res
        units_res = units_fut.result()
        if units_res:
            ret[UNITS] = units_res
        rbs = rbs_fut.result()
        if rbs:
            ret[RULEBOOKS] = rbs
        nv = nv_fut.result()
        nrbs = nrbs_fut.result()
        if nrbs:
            for node, rb in nrbs.items():
                if node not in chara.node:
                    continue
                if node in nv:
                    nv[node][RULEBOOK] = rb
                else:
                    nv[node] = {RULEBOOK: rb}
        if nv:
            ret[NODE_VAL] = nv
        ev = ev_fut.result()
        porbs = porbs_fut.result()
        if porbs:
            for orig, dests in porbs.items():
                if orig not in chara.portal:
                    continue
                portals = chara.portal[orig]
                for dest, rb in dests.items():
                    if dest not in portals:
                        continue
                    ev.setdefault(orig, {}).setdefault(dest, {})[RULEBOOK] = rb
        if ev:
            ret[EDGE_VAL] = ev
        self._character_delta_memo[char][observed_btt][
            actual_btt] = ret  # canon
        return ret

    @timely
    def set_character_stat(self, char, k, v):
        self._real.character[char].stat[k] = v

    @timely
    def del_character_stat(self, char, k):
        del self._real.character[char].stat[k]

    @timely
    def update_character_stats(self, char, patch):
        self._real.character[char].stat.update(patch)

    @timely
    def update_character(self, char, patch):
        self.update_character_stats(char, patch['character'])
        self.update_nodes(char, patch['node'])
        self.update_portals(char, patch['portal'])

    def characters(self):
        return list(self._real.character.keys())

    @timely
    def set_node_stat(self, char, node, k, v):
        self._real.character[char].node[node][k] = v

    @timely
    def del_node_stat(self, char, node, k):
        del self._real.character[char].node[node][k]

    def _get_btt(self,
                 btt: Tuple[str, int, int] = None) -> Tuple[str, int, int]:
        if btt is None:
            return self._real._btt()
        return btt

    def _get_watched_btt(self,
                         btt: Tuple[str, int,
                                    int] = None) -> Tuple[str, int, int]:
        if btt is None:
            return self.branch, self.turn, self.tick
        return btt

    @prepacked
    def node_stat_copy(self,
                       char,
                       node,
                       btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
        branch, turn, tick = self._get_btt(btt)
        memo = self._node_stat_copy_memo
        if (char, node, branch, turn, tick) in memo:
            return memo[char, node, branch, turn, tick]
        pack = self._real.pack
        origtime = self._real._btt()
        if (branch, turn, tick) != origtime:
            self._real._set_btt(branch, turn, tick)
        noden = node
        node = self._real.character[char].node[noden]
        ret = memo[char, noden, branch, turn, tick] = {
            pack(k): pack(v.unwrap()) if hasattr(v, 'unwrap')
            and not hasattr(v, 'no_unwrap') else pack(v)
            for (k, v) in node.items() if k not in
            {'character', 'name', 'arrival_time', 'next_arrival_time'}
        }
        if (branch, turn, tick) != origtime:
            self._real._set_btt(*origtime)
        return ret

    @prepacked
    def node_stat_delta(self,
                        char,
                        node,
                        *,
                        btt_from: Tuple[str, int, int] = None,
                        btt_to: Tuple[str, int, int] = None):
        """Return a dictionary describing changes to a node's stats since the
        last time you looked at it.

        """
        old = self.node_stat_copy(char,
                                  node,
                                  btt=self._get_watched_btt(btt_from))
        new = self.node_stat_copy(char, node, btt=btt_to)
        return _packed_dict_delta(old, new)

    def _character_nodes_stat_delta(self,
                                    char,
                                    *,
                                    btt_from: Tuple[str, int, int] = None,
                                    btt_to: Tuple[str, int, int] = None):
        """Return a dictionary of ``node_stat_delta`` output for each node in a
        character.

        """
        pack = self._real.pack
        r = {}
        nodes = set(self._real.character[char].node.keys())
        futs = []
        for node in nodes:
            fut = self.threadpool.submit(self.node_stat_delta,
                                         char,
                                         node,
                                         btt_from=btt_from,
                                         btt_to=btt_to)
            fut.node = node
            futs.append(fut)
        for fut in as_completed(futs):
            delta = fut.result()
            if delta:
                r[pack(fut.node)] = delta
        nsc = self._character_nodes_stat_copy(
            char, btt=self._get_watched_btt(btt_from))
        for node in list(nsc.keys()):
            if node not in nodes:
                del nsc[node]
        return r

    def _character_nodes_stat_copy(
            self,
            char,
            btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
        pack = self._real.pack
        return {
            pack(node): concat_d(self.node_stat_copy(char, node, btt=btt))
            for node in self._real.character[char].node
        }

    @timely
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

    @timely
    def update_nodes(self, char, patch, backdate=False):
        """Change the stats of nodes in a character according to a
        dictionary.

        """
        # Performance could be improved by preserving the packed values
        if backdate:
            parbranch, parrev = self._real._parentbranch_rev.get(
                self._real.branch, ('trunk', 0))
            tick_now = self._real.tick
            self._real.tick = parrev
        for i, (n, npatch) in enumerate(patch.items(), 1):
            self.update_node(char, n, npatch)
        if backdate:
            self._real.tick = tick_now

    @timely
    def del_node(self, char, node):
        """Remove a node from a character."""
        del self._real.character[char].node[node]

    @prepacked
    def character_nodes(self, char, btt: Tuple[str, int, int] = None):
        pack = self.pack
        memo = self._char_nodes_copy_memo
        branch, turn, tick = self._get_btt(btt)
        if (char, branch, turn, tick) in memo:
            return memo[char, branch, turn, tick]
        origtime = self._real._btt()
        if (branch, turn, tick) != origtime:
            self._real.time = branch, turn
            self._real.tick = tick
        ret = memo[char, branch, turn,
                   tick] = set(map(pack, self._real.character[char].node))
        if (branch, turn, tick) != origtime:
            self._real.time = origtime[:2]
            self._real.tick = origtime[2]
        return ret

    @prepacked
    def character_nodes_delta(self,
                              char,
                              *,
                              btt_from: Tuple[str, int, int] = None,
                              btt_to: Tuple[str, int, int] = None):
        old = self.character_nodes(char, btt=self._get_watched_btt(btt_from))
        new = self.character_nodes(char, btt=btt_to)
        return set_delta(old, new)

    def node_predecessors(self, char, node):
        return list(self._real.character[char].pred[node].keys())

    def character_set_node_predecessors(self, char, node, preds):
        self._real.character[char].pred[node] = preds

    def character_del_node_predecessors(self, char, node):
        del self._real.character[char].pred[node]

    @prepacked
    def node_successors(self, char, node, btt: Tuple[str, int, int] = None):
        memo = self._node_successors_copy_memo
        branch, turn, tick = self._get_btt(btt)
        if (char, node, branch, turn, tick) in memo:
            return memo[char, node, branch, turn, tick]
        origtime = self._real._btt()
        if (branch, turn, tick) != origtime:
            self._real._set_btt(branch, turn, tick)
        ret = memo[char, node, branch, turn, tick] = set(
            map(self.pack, self._real.character[char].portal[node].keys()))
        if (branch, turn, tick) != origtime:
            self._real._set_btt(*origtime)
        return ret

    def nodes_connected(self, char, orig, dest):
        return dest in self._real.character[char].portal[orig]

    @timely
    def init_thing(self, char, thing, statdict: dict = None):
        if thing in self._real.character[char].thing:
            raise KeyError('Already have thing in character {}: {}'.format(
                char, thing))
        if statdict is None:
            statdict = {}
        return self.set_thing(char, thing, statdict)

    @timely
    def set_thing(self, char, thing, statdict):
        self._real.character[char].thing[thing] = statdict
        statdict = dict(map(self.pack_pair, statdict.items()))
        branch, turn, tick = self._get_btt()

    @timely
    def add_thing(self, char, thing, loc, statdict):
        self._real.character[char].add_thing(thing, loc, **statdict)

    @timely
    def place2thing(self, char, node, loc):
        self._real.character[char].place2thing(node, loc)

    @timely
    def thing2place(self, char, node):
        self._real.character[char].thing2place(node)

    @timely
    def add_things_from(self, char, seq):
        for thing in seq:
            self.add_thing(char, *thing)

    def get_thing_location(self, char, thing):
        try:
            return self._real.character[char].thing[thing]['location']
        except KeyError:
            return None

    @timely
    def set_thing_location(self, char, thing, loc):
        self._real.character[char].thing[thing]['location'] = loc

    @timely
    def thing_follow_path(self, char, thing, path, weight):
        return self._real.character[char].thing[thing].follow_path(
            path, weight)

    @timely
    def thing_go_to_place(self, char, thing, place, weight):
        return self._real.character[char].thing[thing].go_to_place(
            place, weight)

    @timely
    def thing_travel_to(self, char, thing, dest, weight=None, graph=None):
        """Make something find a path to ``dest`` and follow it.

        Optional argument ``weight`` is the portal stat to use to schedule movement times.

        Optional argument ``graph`` is an alternative graph to use for pathfinding.
        Should resemble a networkx DiGraph.

        """
        return self._real.character[char].thing[thing].travel_to(
            dest, weight, graph)

    @timely
    def init_place(self, char, place, statdict=None):
        if place in self._real.character[char].place:
            raise KeyError('Already have place in character {}: {}'.format(
                char, place))
        if statdict is None:
            statdict = {}
        return self.set_place(char, place, statdict)

    @timely
    def set_place(self, char, place, statdict):
        self._real.character[char].place[place] = statdict
        self._after_ret = partial(self.node_stat_copy, char, place)

    @timely
    def add_places_from(self, char, seq):
        self._real.character[char].add_places_from(seq)

    @prepacked
    def character_portals(self, char, btt: Tuple[str, int, int] = None):
        pack = self._real.pack
        branch, turn, tick = self._get_btt(btt)
        r = set()
        portal = self._real.character[char].portal
        origtime = self._real._btt()
        if (branch, turn, tick) != origtime:
            self._real._set_btt(branch, turn, tick)
        for o in portal:
            for d in portal[o]:
                r.add(pack((o, d)))
        if (branch, turn, tick) != origtime:
            self._real._set_btt(*origtime)
        return r

    @prepacked
    def character_portals_delta(self,
                                char,
                                *,
                                btt_from: Tuple[str, int, int] = None,
                                btt_to: Tuple[str, int, int] = None):
        old = self.character_portals(char, btt=self._get_watched_btt(btt_from))
        new = self.character_portals(char, btt=btt_to)
        return set_delta(old, new)

    @timely
    def add_portal(self, char, orig, dest, symmetrical, statdict):
        self._real.character[char].add_portal(orig, dest, symmetrical,
                                              **statdict)

    @timely
    def add_portals_from(self, char, seq, symmetrical):
        self._real.character[char].add_portals_from(seq, symmetrical)

    def del_portal(self, char, orig, dest):
        del self._real.character[char].portal[orig][dest]

    def set_portal_stat(self, char, orig, dest, k, v):
        self._real.character[char].portal[orig][dest][k] = v

    def del_portal_stat(self, char, orig, dest, k):
        del self._real.character[char][orig][dest][k]

    def portal_stat_copy(self,
                         char,
                         orig,
                         dest,
                         btt: Tuple[str, int, int] = None):
        pack = self._real.pack
        branch, turn, tick = self._get_btt(btt)
        memo = self._portal_stat_copy_memo
        if (char, orig, dest, branch, turn, tick) in memo:
            return memo[char, orig, dest, branch, turn, tick]
        origtime = self._real._btt()
        if (branch, turn, tick) != origtime:
            self._real._set_btt(branch, turn, tick)
        ret = memo[char, orig, dest, branch, turn, tick] = {
            pack(k): pack(v.unwrap())
            if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v
            for (k,
                 v) in self._real.character[char].portal[orig][dest].items()
        }
        if (branch, turn, tick) != origtime:
            self._real._set_btt(*origtime)
        return ret

    @prepacked
    def portal_stat_delta(self,
                          char,
                          orig,
                          dest,
                          *,
                          btt_from: Tuple[str, int, int] = None,
                          btt_to: Tuple[str, int, int] = None):
        old = self.portal_stat_copy(char,
                                    orig,
                                    dest,
                                    btt=self._get_watched_btt(btt_from))
        new = self.portal_stat_copy(char, orig, dest, btt=btt_to)
        return _packed_dict_delta(old, new)

    def _character_portals_stat_copy(
        self,
        char,
        btt: Tuple[str, int, int] = None
    ) -> Dict[bytes, Dict[bytes, Dict[bytes, bytes]]]:
        pack = self.pack
        r = {}
        chara = self._real.character[char]
        for orig, dests in chara.portal.items():
            porig = pack(orig)
            for dest in dests:
                pdest = pack(dest)
                if porig not in r:
                    r[porig] = {}
                r[porig][pdest] = self.portal_stat_copy(char,
                                                        orig,
                                                        dest,
                                                        btt=btt)
        return r

    def _character_portals_stat_delta(self,
                                      char,
                                      *,
                                      btt_from: Tuple[str, int, int] = None,
                                      btt_to: Tuple[str, int, int] = None):
        r = {}
        futs = []
        btt_from = self._get_watched_btt(btt_from)
        btt_to = self._get_btt(btt_to)
        for orig in self._real.character[char].portal:
            for dest in self._real.character[char].portal[orig]:
                fut = self.threadpool.submit(self.portal_stat_delta,
                                             char,
                                             orig,
                                             dest,
                                             btt_from=btt_from,
                                             btt_to=btt_to)
                fut.orig = orig
                fut.dest = dest
                futs.append(fut)
        for fut in as_completed(futs):
            delta = fut.result()
            orig = fut.orig
            dest = fut.dest
            if delta:
                if orig not in r:
                    r[orig] = {}
                r[orig][dest] = delta
        return r

    @timely
    def update_portal(self, char, orig, dest, patch):
        character = self._real.character[char]
        if patch is None:
            del character.portal[orig][dest]
            self._char_portals_cache[char].discard(self.pack((orig, dest)))
        elif orig not in character.portal \
                or dest not in character.portal[orig]:
            character.portal[orig][dest] = patch
            self._char_portals_cache[char].add(self.pack((orig, dest)))
            self._portal_stat_cache[char][orig][dest] = BytesDict(
                map(self.pack_pair, patch.items()))
        else:
            character.portal[orig][dest].update(patch)
            self._char_portals_cache[char].add(self.pack((orig, dest)))
            self._portal_stat_cache[char][orig][dest].update(
                map(self.pack_pair, patch.items()))

    @timely
    def update_portals(self, char, patch):
        for ((orig, dest), ppatch) in patch.items():
            self.update_portal(char, orig, dest, ppatch)

    @timely
    def add_unit(self, char, graph, node):
        self._real.character[char].add_unit(graph, node)
        self._char_av_cache[char][graph].add(self.pack(node))

    @timely
    def remove_unit(self, char, graph, node):
        self._real.character[char].remove_unit(graph, node)
        self._char_av_cache[char][graph].remove(self.pack(node))

    @timely
    def new_empty_rule(self, rule):
        self._real.rule.new_empty(rule)

    @timely
    def new_empty_rulebook(self, rulebook, btt=None):
        branch, turn, tick = self._get_btt(btt)
        self._rulebook_copy_memo[rulebook, branch, turn, tick] = []
        self._real.rulebook[rulebook]
        return []

    def rulebook_copy(self, rulebook, btt: Tuple[str, int, int] = None):
        branch, turn, tick = self._get_btt(btt)
        memo = self._rulebook_copy_memo
        if (rulebook, branch, turn, tick) in memo:
            return memo[rulebook, branch, turn, tick]
        ret = memo[rulebook, branch, turn,
                   tick] = list(self._real.rulebook[rulebook]._get_cache(
                       branch, turn, tick))
        return ret

    def rulebook_delta(self,
                       rulebook,
                       *,
                       btt_from: Tuple[str, int, int] = None,
                       btt_to: Tuple[str, int, int] = None):
        old = self.rulebook_copy(rulebook, btt=self._get_watched_btt(btt_from))
        new = self.rulebook_copy(rulebook, btt=btt_to)
        if old == new:
            return
        return new

    def all_rulebooks_delta(self,
                            *,
                            btt_from: Tuple[str, int, int] = None,
                            btt_to: Tuple[str, int, int] = None):
        ret = {}
        for rulebook in self._real.rulebook.keys():
            delta = self.rulebook_delta(rulebook,
                                        btt_from=btt_from,
                                        btt_to=btt_to)
            if delta:
                ret[rulebook] = delta
        return ret

    @timely
    def set_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook][i] = rule
        branch, turn, tick = self._real._btt()
        memo = self._rulebook_copy_memo
        if (rulebook, branch, turn, tick) in memo:
            memo[rulebook, branch, turn, tick][i] = rule

    @timely
    def ins_rulebook_rule(self, rulebook, i, rule):
        self._real.rulebook[rulebook].insert(i, rule)
        branch, turn, tick = self._real._btt()
        memo = self._rulebook_copy_memo
        if (rulebook, branch, turn, tick) in memo:
            memo[rulebook, branch, turn, tick].insert(i, rule)

    @timely
    def del_rulebook_rule(self, rulebook, i):
        del self._real.rulebook[rulebook][i]
        del self._rulebook_cache[rulebook][i]
        branch, turn, tick = self._real._btt()
        memo = self._rulebook_copy_memo
        if (rulebook, branch, turn, tick) in memo:
            del memo[rulebook, branch, turn, tick][i]

    @timely
    def set_rule_triggers(self, rule, triggers):
        self._real.rule[rule].triggers = triggers
        branch, turn, tick = self._real._btt()
        memo = self._rule_copy_memo
        if (rule, branch, turn, tick) in memo:
            memo[rule, branch, turn, tick]['triggers'] = triggers

    @timely
    def set_rule_prereqs(self, rule, prereqs):
        self._real.rule[rule].prereqs = prereqs
        branch, turn, tick = self._real._btt()
        memo = self._rule_copy_memo
        if (rule, branch, turn, tick) in memo:
            memo[rule, branch, turn, tick]['prereqs'] = prereqs

    @timely
    def set_rule_actions(self, rule, actions):
        self._real.rule[rule].actions = actions
        branch, turn, tick = self._real._btt()
        memo = self._rule_copy_memo
        if (rule, branch, turn, tick) in memo:
            memo[rule, branch, turn, tick]['actions'] = actions

    @timely
    def set_character_rulebook(self, char, rulebook):
        self._real.character[char].rulebook = rulebook

    @timely
    def set_unit_rulebook(self, char, rulebook):
        self._real.character[char].unit.rulebook = rulebook

    @timely
    def set_character_thing_rulebook(self, char, rulebook):
        self._real.character[char].thing.rulebook = rulebook

    @timely
    def set_character_place_rulebook(self, char, rulebook):
        self._real.character[char].place.rulebook = rulebook

    @timely
    def set_character_node_rulebook(self, char, rulebook):
        self._real.character[char].node.rulebook = rulebook

    @timely
    def set_character_portal_rulebook(self, char, rulebook):
        self._real.character[char].portal.rulebook = rulebook

    @timely
    def set_node_rulebook(self, char, node, rulebook):
        self._real.character[char].node[node].rulebook = rulebook

    @timely
    def set_portal_rulebook(self, char, orig, dest, rulebook):
        self._real.character[char].portal[orig][dest].rulebook = rulebook

    def rule_copy(self, rule, btt: Tuple[str, int, int] = None):
        branch, turn, tick = self._get_btt(btt)
        memo = self._rule_copy_memo
        if (rule, branch, turn, tick) in memo:
            return memo[rule, branch, turn, tick]
        ret = memo[rule, branch, turn, tick] = {
            'triggers':
            list(self._real._triggers_cache.retrieve(rule, branch, turn,
                                                     tick)),
            'prereqs':
            list(self._real._prereqs_cache.retrieve(rule, branch, turn, tick)),
            'actions':
            list(self._real._actions_cache.retrieve(rule, branch, turn, tick))
        }
        return ret

    def rule_delta(self,
                   rule,
                   *,
                   btt_from: Tuple[str, int, int] = None,
                   btt_to: Tuple[str, int, int] = None):
        old = self.rule_copy(rule, btt=self._get_watched_btt(btt_from))
        new = self.rule_copy(rule, btt=btt_to)
        ret = {}
        if new['triggers'] != old['triggers']:
            ret['triggers'] = new['triggers']
        if new['prereqs'] != old['prereqs']:
            ret['prereqs'] = new['prereqs']
        if new['actions'] != old['actions']:
            ret['actions'] = new['actions']
        return ret

    def all_rules_delta(self,
                        *,
                        btt_from: Tuple[str, int, int] = None,
                        btt_to: Tuple[str, int, int] = None):
        ret = {}
        for rule in self._real.rule.keys():
            delta = self.rule_delta(rule, btt_from=btt_from, btt_to=btt_to)
            if delta:
                ret[rule] = delta
        return ret

    @prepacked
    def source_copy(self, store):
        return dict(map(self.pack_pair,
                        getattr(self._real, store).iterplain()))

    def get_source(self, store, name):
        return getattr(self._real, store).get_source(name)

    def store_source(self, store, v, name=None):
        getattr(self._real, store).store_source(v, name)

    def del_source(self, store, k):
        delattr(getattr(self._real, store), k)

    @timely
    def call_stored_function(self, store, func, args, kwargs):
        if store == 'method':
            args = (self._real, ) + tuple(args)
        store = getattr(self._real, store)
        if store not in self._real.stores:
            raise ValueError("{} is not a function store".format(store))
        callme = getattr(store, func)
        try:
            return callme(*args, **kwargs)
        except Exception as ex:
            raise

    @timely
    def call_randomizer(self, method, *args, **kwargs):
        return getattr(self._real._rando, method)(*args, **kwargs)

    @timely
    def install_module(self, module):
        import_module(module).install(self._real)

    @timely
    def do_game_start(self):
        self._real.game_start()

    def is_parent_of(self, parent, child):
        return self._real.is_parent_of(parent, child)

    def apply_choices(self, choices, dry_run=False, perfectionist=False):
        return self._real.apply_choices(choices, dry_run, perfectionist)

    @staticmethod
    def get_schedule(entity, stats, beginning, end):
        ret = {}
        for stat in stats:
            ret[stat] = list(
                entity.historical(stat).iter_history(beginning, end))
        return ret

    @timely
    def grid_2d_8graph(self, character, m, n):
        self._real.character[character].grid_2d_8graph(m, n)
        return self._get_char_deltas([character])

    @timely
    def grid_2d_graph(self, character, m, n, periodic):
        self._real.character[character].grid_2d_graph(m, n, periodic)
        return self._get_char_deltas([character])

    def rules_handled_turn(self, branch=None, turn=None):
        if branch is None:
            branch = self.branch
        if turn is None:
            turn = self.turn
        eng = self._real
        # assume the caches are all sync'd
        return {
            'character':
            eng._character_rules_handled_cache.handled_deep[branch][turn],
            'unit':
            eng._unit_rules_handled_cache.handled_deep[branch][turn],
            'character_thing':
            eng._character_thing_rules_handled_cache.handled_deep[branch]
            [turn],
            'character_place':
            eng._character_place_rules_handled_cache.handled_deep[branch]
            [turn],
            'character_portal':
            eng._character_portal_rules_handled_cache.handled_deep[branch]
            [turn],
            'node':
            eng._node_rules_handled_cache.handled_deep[branch][turn],
            'portal':
            eng._portal_rules_handled_cache.handled_deep[branch][turn]
        }

    def branch_lineage(self):
        return self._real._branches
