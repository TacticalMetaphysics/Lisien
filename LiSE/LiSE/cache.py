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


class EntitylessCache(Cache):
    def store(self, key, branch, tick, value):
        super().store(None, key, branch, tick, value)

    def retrieve(self, key, branch, tick):
        return super().retrieve(None, key, branch, tick)

    def iter_entities_or_keys(self, branch, tick):
        return super().iter_entities_or_keys(None, branch, tick)
    iter_entities = iter_keys = iter_entities_or_keys


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


class RulesHandledCache(object):
    depth = 1

    def __init__(self, engine):
        self.engine = engine
        self.shallow = {}
        self.unhandled = StructuredDefaultDict(self.depth, dict)

    def store(self, *args):
        entity = args[:-4]
        rulebook, rule, branch, tick = args[-4:]
        shalo = self.shallow.setdefault(entity + (rulebook, rule, branch), set())
        unhandl = self.unhandled
        for spot in entity:
            unhandl = unhandl[spot]
        if tick not in unhandl.setdefault(branch, {}):
            itargs = entity + (branch, tick)
            unhandl[branch][tick] = list(self._iter_rulebook(*itargs))
        unhandl[branch][tick].remove(entity + (rulebook, rule))
        shalo.add(rule)

    def retrieve(self, *args):
        return self.shallow[args]

    def check_handled(self, *args):
        rule = args[-1]
        return rule in self.shallow.get(args[:-1], [])

    def iter_unhandled_rules(self, *args):
        try:
            unhandl = self.unhandled
            for spot in args:
                unhandl = ret = unhandl[spot]
        except KeyError:
            try:
                unhandl[spot] = ret = list(self._iter_rulebook(*args))
            except KeyError:
                return
        yield from ret

    def _iter_rulebook(self, *args):
        raise NotImplementedError


class CharacterRulesHandledCache(RulesHandledCache):
    def _iter_rulebook(self, character, branch, tick):
        rulebook = self.engine._characters_rulebooks_cache.retrieve(character, branch, tick)
        for rule in self.engine._rulebooks_cache.retrieve(rulebook, branch, tick):
            yield character, rulebook, rule


class AvatarRulesHandledCache(RulesHandledCache):
    depth = 3

    def _iter_rulebook(self, character, branch, tick):
        rulebook = self.engine._avatars_rulebooks_cache.retrieve(character, branch, tick)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, tick)
        for graph in self.engine.character[character].avatar:
            for avatar in self.engine.character[character].avatar[graph]:
                for rule in rules:
                    yield character, graph, avatar, rulebook, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
    depth = 2

    def _iter_rulebook(self, character, thing, branch, tick):
        rulebook = self.engine._characters_things_rulebooks_cache.retrieve(character, branch, tick)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, tick)
        for thing in self.engine.character[character].thing:
            for rule in rules:
                yield character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):
    depth = 2

    def _iter_rulebook(self, character, branch, tick):
        rulebook = self.engine._characters_places_rulebooks_cache.retrieve(character, branch, tick)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, tick)
        for place in self.engine.character[character].place:
            for rule in rules:
                yield character, place, rulebook, rule


class CharacterPortalRulesHandledCache(RulesHandledCache):
    depth = 4

    def _iter_rulebook(self, character, branch, tick):
        rulebook = self.engine._characters_portals_rulebooks_cache.retrieve(character, branch, tick)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, tick)
        for orig in self.engine.character[character].portal:
            for dest in self.engine.character[character].portal[orig]:
                for rule in rules:
                    yield character, orig, dest, rulebook, rule


class NodeRulesHandledCache(RulesHandledCache):
    depth = 2

    def _iter_rulebook(self, character, node, branch, tick):
        rulebook = self.engine._nodes_rulebooks_cache.retrieve(character, node, branch, tick)
        for rule in self.engine._rulebooks_cache.retrieve(rulebook, branch, tick):
            yield character, node, rulebook, rule


class PortalRulesHandledCache(RulesHandledCache):
    depth = 3

    def _iter_rulebook(self, character, orig, dest, branch, tick):
        rulebook = self.engine._portals_rulebooks_cache.retrieve(character, orig, dest, branch, tick)
        for rule in self.engine._rulebooks_cache.retrieve(rulebook, branch, tick):
            yield character, orig, dest, rulebook, rule


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
