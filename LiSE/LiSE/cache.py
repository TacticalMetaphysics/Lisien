from collections import defaultdict
from allegedb.cache import (
    Cache,
    NodesCache,
    PickyDefaultDict,
    StructuredDefaultDict,
    FuturistWindowDict,
    HistoryError
)
from .util import singleton_get


class UniversalCache(Cache):
    def store(self, key, branch, tick, value):
        super().store(None, key, branch, tick, value)


class AvatarnessCache(Cache):
    """A cache for remembering when a node is an avatar of a character."""
    def __init__(self, engine):
        Cache.__init__(self, engine)
        self.user_order = StructuredDefaultDict(3, FuturistWindowDict)
        self.user_shallow = PickyDefaultDict(FuturistWindowDict)
        self.graphs = StructuredDefaultDict(1, FuturistWindowDict)
        self.graphavs = StructuredDefaultDict(1, FuturistWindowDict)
        self.charavs = StructuredDefaultDict(1, FuturistWindowDict)
        self.soloav = StructuredDefaultDict(1, FuturistWindowDict)
        self.uniqav = StructuredDefaultDict(1, FuturistWindowDict)
        self.uniqgraph = StructuredDefaultDict(1, FuturistWindowDict)

    def store(self, character, graph, node, branch, tick, is_avatar):
        if not is_avatar:
            is_avatar = None
        Cache.store(self, character, graph, node, branch, tick, is_avatar)
        self.user_order[graph][node][character][branch][tick] = is_avatar
        self.user_shallow[(graph, node, character, branch)][tick] = is_avatar
        self._forward_valcache(self.charavs[character], branch, tick)
        self._forward_valcache(
            self.graphavs[(character, graph)], branch, tick
        )
        self._forward_valcache(self.graphs[character], branch, tick)
        self._forward_valcache(
            self.soloav[(character, graph)],
            branch, tick, copy=False
        )
        self._forward_valcache(
            self.uniqav[character], branch, tick, copy=False
        )
        self._forward_valcache(
            self.uniqgraph[character], branch, tick, copy=False
        )
        charavs = self.charavs[character][branch]
        graphavs = self.graphavs[(character, graph)][branch]
        graphs = self.graphs[character][branch]
        uniqgraph = self.uniqgraph[character][branch]
        soloav = self.soloav[(character, graph)][branch]
        uniqav = self.uniqav[character][branch]
        for avmap in (charavs, graphavs, graphs):
            if not avmap.has_exact_rev(tick):
                try:
                    avmap[tick] = avmap[tick].copy()
                except HistoryError:
                    avmap[tick] = set()
        if is_avatar:
            if graphavs[tick]:
                soloav[tick] = None
            else:
                soloav[tick] = node
            if charavs[tick]:
                uniqav[tick] = None
            else:
                uniqav[tick] = (graph, node)
            if graphs[tick]:
                uniqgraph[tick] = None
            else:
                uniqgraph[tick] = graph
            graphavs[tick].add(node)
            charavs[tick].add((graph, node))
            graphs[tick].add(graph)
        else:
            graphavs[tick].remove(node)
            charavs[tick].remove((graph, node))
            soloav[tick] = singleton_get(graphavs[tick])
            uniqav[tick] = singleton_get(charavs[tick])
            if not graphavs[tick]:
                graphs[tick].remove(graph)
                if len(graphs[tick]) == 1:
                    uniqgraph[tick] = next(iter(graphs[tick]))
                else:
                    uniqgraph[tick] = None

    def get_char_graph_avs(self, char, graph, branch, tick):
        return self._forward_valcache(
            self.graphavs[(char, graph)], branch, tick
        ) or set()

    def get_char_graph_solo_av(self, char, graph, branch, tick):
        return self._forward_valcache(
            self.soloav[(char, graph)], branch, tick, copy=False
        )

    def get_char_only_av(self, char, branch, tick):
        return self._forward_valcache(
            self.uniqav[char], branch, tick, copy=False
        )

    def get_char_only_graph(self, char, branch, tick):
        return self._forward_valcache(
            self.uniqgraph[char], branch, tick, copy=False
        )

    def get_char_graphs(self, char, branch, tick):
        return self._forward_valcache(
            self.graphs[char], branch, tick
        ) or set()

    def iter_node_users(self, graph, node, branch, tick):
        if graph not in self.user_order:
            return
        for character in self.user_order[graph][node]:
            if (graph, node, character, branch) not in self.user_shallow:
                for (b, t) in self.allegedb._active_branches(branch, tick):
                    if b in self.user_order[graph][node][character]:
                        isav = self.user_order[graph][node][character][b][t]
                        self.store(character, graph, node, b, t, isav)
                        self.store(character, graph, node, branch, tick, isav)
                        break
                else:
                    self.store(character, graph, node, branch, tick, None)
            if self.user_shallow[(graph, node, character, branch)][tick]:
                yield character


class RulebooksCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = defaultdict(list)

    def store(self, rulebook, rule):
        self._data[rulebook].append(rule)

    def retrieve(self, rulebook):
        return self._data[rulebook]


class CharacterRulebooksCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = {}

    def store(
            self, char,
            character=None,
            avatar=None,
            character_thing=None,
            character_place=None,
            character_node=None,
            character_portal=None
    ):
        if char in self._data:
            old = self._data[char]
            character = character or old['character']
            avatar = avatar or old['avatar']
            character_thing = character_thing or old['character_thing']
            character_place = character_place or old['character_place']
            character_portal = character_portal or old['character_portal']
            character_node = character_node or old['character_node']
        self._data[char] = {
            'character': character or (char, 'character'),
            'avatar': avatar or (char, 'avatar'),
            'character_thing': character_thing or (char, 'character_thing'),
            'character_place': character_place or (char, 'character_place'),
            'character_portal': character_portal or (char, 'character_portal'),
            'character_node': character_node or (char, 'character_node')
        }

    def retrieve(self, char):
        if char not in self._data:
            self.store(char)
        return self._data[char]


class NodeRulesHandledCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = StructuredDefaultDict(4, set)
        self.shallow = {}
        self.unhandled = StructuredDefaultDict(1, dict)

    def store(self, character, node, rulebook, rule, branch, tick):
        the_set = self.shallow[(character, node, rulebook, rule, branch)] \
                  = self._data[character][node][rulebook][rule][branch]
        the_set.add(tick)
        if tick not in self.unhandled[(character, node)][branch]:
            self.unhandled[(character, node)][branch][tick] = set(
                self.engine._active_rules_cache.active_sets[
                    rulebook][branch][tick]
            )
        self.unhandled[(character, node)][branch][tick].remove(rule)

    def retrieve(self, character, node, rulebook, rule, branch):
        return self.shallow[(character, node, rulebook, rule, branch)]

    def check_handled(self, character, node, rulebook, rule, branch, tick):
        try:
            ret = tick in self.shallow[
                (character, node, rulebook, rule, branch)]
        except KeyError:
            ret = False
        assert ret is rule not in self.unhandled[
            (character, node)][branch][tick]
        return ret

    def iter_unhandled_rules(self, character, node, rulebook, branch, tick):
        try:
            unhandl = self.unhandled[(character, node)][branch][tick]
        except KeyError:
            try:
                unhandl = self.unhandled[(character, node)][branch][tick] \
                          = self.engine._active_rules_cache.active_sets[
                              rulebook][branch][tick].copy()
            except KeyError:
                return
        yield from unhandl


class PortalRulesHandledCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = StructuredDefaultDict(5, set)
        self.shallow = {}
        self.unhandled = StructuredDefaultDict(1, dict)

    def store(self, character, orig, dest, rulebook, rule, branch, tick):
        the_set = self.shallow[
            (character, orig, dest, rulebook, rule, branch)
        ] = self._data[character][orig][dest][rulebook][rule][branch]
        the_set.add(tick)
        if tick not in self.unhandled[(character, orig, dest)][branch]:
            self.unhandled[(character, orig, dest)][branch][tick] = set(
                self.engine._active_rules_cache[rulebook][branch][tick]
            )
        self.unhandled[(character, orig, dest)][branch][tick].remove(rule)

    def retrieve(self, character, orig, dest, rulebook, rule, branch):
        return self.shallow[(character, orig, dest, rulebook, rule, branch)]

    def check_handled(
            self, character, orig, dest, rulebook, rule, branch, tick
    ):
        try:
            ret = tick in self.shallow[
                (character, orig, dest, rulebook, rule, branch)]
        except KeyError:
            ret = False
        assert ret is rule not in self.unhandled[
            (character, orig, dest)][branch][tick]
        return ret

    def iter_unhandled_rules(
            self, character, orig, dest, rulebook, branch, tick
    ):
        try:
            unhandl = self.unhandled[(character, orig, dest)][branch][tick]
        except KeyError:
            try:
                unhandl = self.unhandled[
                    (character, orig, dest)][branch][tick] \
                    = self.engine._active_rules_cache.retrieve(
                        rulebook, branch, tick
                    ).copy()
            except KeyError:
                return
        yield from unhandl


class NodeRulebookCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = defaultdict(dict)
        self.shallow = {}

    def store(self, character, node, rulebook):
        self._data[character][node] \
            = self.shallow[(character, node)] \
            = rulebook

    def retrieve(self, character, node):
        return self.shallow[(character, node)]


class PortalRulebookCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = defaultdict(lambda: defaultdict(dict))
        self.shallow = {}

    def store(self, character, orig, dest, rulebook):
        self._data[character][orig][dest] \
            = self.shallow[(character, orig, dest)] = rulebook

    def retrieve(self, character, orig, dest):
        return self.shallow[(character, orig, dest)]


class ActiveRulesCache(Cache):
    iter_rules = iter_active_rules = Cache.iter_keys
    
    def __init__(self, engine):
        Cache.__init__(self, engine)
        self.active_sets = StructuredDefaultDict(1, FuturistWindowDict)

    def store(self, rulebook, rule, branch, tick, active):
        if not active:
            active = None
        Cache.store(self, rulebook, rule, branch, tick, active)
        auh = self.active_sets[rulebook][branch].setdefault(tick, set())
        if self.active_sets[rulebook][branch].rev_before(tick) != tick:
            auh = self.active_sets[rulebook][branch][tick] = auh.copy()
        if active:
            auh.add(rule)
        else:
            auh.discard(rule)

    def retrieve(self, rulebook, branch, tick):
        return self.active_sets[rulebook][branch][tick]


class CharacterRulesHandledCache(object):
    def __init__(self, engine):
        self.engine = engine
        self._data = StructuredDefaultDict(3, set)
        self.shallow = {}
        self.unhandled = StructuredDefaultDict(2, dict)

    def store(self, character, ruletype, rulebook, rule, branch, tick):
        the_set = self.shallow[
            (character, ruletype, rule, branch)
        ] = self._data[character][ruletype][rule][branch]
        the_set.add(tick)
        if tick not in self.unhandled[character][ruletype][branch]:
            self.unhandled[character][ruletype][branch][tick] = set(
                self.engine._active_rules_cache.active_sets[
                    rulebook][branch][tick]
            )
        self.unhandled[character][ruletype][branch][tick].remove(rule)

    def retrieve(self, character, ruletype, rulebook, rule, branch):
        return self.shallow[(character, ruletype, rule, branch)]

    def check_rule_handled(
            self, character, ruletype, rulebook, rule, branch, tick
    ):
        try:
            ret = tick in self.shallow[(character, ruletype, rule, branch)]
        except KeyError:
            ret = False
        assert ret is rule not in self.unhandled[
            character][ruletype][branch][tick]
        return ret

    def iter_unhandled_rules(
            self, character, ruletype, rulebook, branch, tick
    ):
        try:
            unhandl = self.unhandled[character][ruletype][branch][tick]
        except KeyError:
            try:
                unhandl \
                    = self.unhandled[character][ruletype][branch][tick] \
                    = self.engine._active_rules_cache.retrieve(
                        rulebook, branch, tick
                    ).copy()
            except KeyError:
                return
        yield from unhandl


class ThingsCache(NodesCache):
    def __init__(self, db):
        Cache.__init__(self, db)
        self._make_node = db.thing_cls

    def store(self, character, thing, branch, tick, loc, nextloc=None):
        super().store(character, thing, branch, tick, (loc, nextloc))

    def tick_before(self, character, thing, branch, tick):
        self.retrieve(character, thing, branch, tick)
        return self.keys[(character,)][thing][branch].rev_before(tick)

    def tick_after(self, character, thing, branch, tick):
        self.retrieve(character, thing, branch, tick)
        return self.keys[(character,)][thing][branch].rev_after(tick)
