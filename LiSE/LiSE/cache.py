from allegedb.cache import (
    Cache,
    PickyDefaultDict,
    StructuredDefaultDict,
    TurnDict,
    HistoryError
)
from .util import singleton_get


class EntitylessCache(Cache):
    def store(self, key, branch, turn, tick, value):
        super().store(None, key, branch, turn, tick, value)

    def retrieve(self, key, branch, turn, tick):
        return super().retrieve(None, key, branch, turn, tick)

    def iter_entities_or_keys(self, branch, turn, tick):
        return super().iter_entities_or_keys(None, branch, turn, tick)
    iter_entities = iter_keys = iter_entities_or_keys

    def contains_entity_or_key(self, ke, branch, turn, tick):
        return super().contains_entity_or_key(None, ke, branch, turn, tick)
    contains_entity = contains_key = contains_entity_or_key


class AvatarnessCache(Cache):
    """A cache for remembering when a node is an avatar of a character."""
    def __init__(self, engine):
        Cache.__init__(self, engine)
        self.user_order = StructuredDefaultDict(3, TurnDict)
        self.user_shallow = PickyDefaultDict(TurnDict)
        self.graphs = StructuredDefaultDict(1, TurnDict)
        self.graphavs = StructuredDefaultDict(1, TurnDict)
        self.charavs = StructuredDefaultDict(1, TurnDict)
        self.soloav = StructuredDefaultDict(1, TurnDict)
        self.uniqav = StructuredDefaultDict(1, TurnDict)
        self.uniqgraph = StructuredDefaultDict(1, TurnDict)

    def store(self, character, graph, node, branch, turn, tick, is_avatar):
        if not is_avatar:
            is_avatar = None
        Cache.store(self, character, graph, node, branch, turn, tick, is_avatar)
        self.user_order[graph][node][character][branch][turn][tick] = is_avatar
        self.user_shallow[(graph, node, character, branch)][turn][tick] = is_avatar
        self._forward_valcache(self.charavs[character], branch, turn, tick)
        self._forward_valcache(
            self.graphavs[(character, graph)], branch, turn, tick
        )
        self._forward_valcache(self.graphs[character], branch, turn, tick)
        self._forward_valcache(
            self.soloav[(character, graph)],
            branch, turn, tick, copy=False
        )
        self._forward_valcache(
            self.uniqav[character], branch, turn, copy=False
        )
        self._forward_valcache(
            self.uniqgraph[character], branch, turn, copy=False
        )
        charavs = self.charavs[character][branch]
        graphavs = self.graphavs[(character, graph)][branch]
        graphs = self.graphs[character][branch]
        uniqgraph = self.uniqgraph[character][branch]
        soloav = self.soloav[(character, graph)][branch]
        uniqav = self.uniqav[character][branch]
        for avmap in (charavs, graphavs, graphs):
            if not avmap.has_exact_rev(turn):
                try:
                    avmap[turn][tick] = avmap[turn][tick].copy()
                except HistoryError:
                    avmap[turn][tick] = set()
        if is_avatar:
            if turn in graphavs and graphavs[turn][tick]:
                soloav[turn][tick] = None
            else:
                soloav[turn][tick] = node
            if turn in charavs and charavs[turn][tick]:
                uniqav[turn][tick] = None
            else:
                uniqav[turn][tick] = (graph, node)
            if turn in graphs and graphs[turn][tick]:
                uniqgraph[turn][tick] = None
            else:
                uniqgraph[turn][tick] = graph
            graphavs[turn][tick].add(node)
            charavs[turn][tick].add((graph, node))
            graphs[turn][tick].add(graph)
        else:
            graphavs[turn][tick].remove(node)
            charavs[turn][tick].remove((graph, node))
            soloav[turn][tick] = singleton_get(graphavs[turn][tick])
            uniqav[turn][tick] = singleton_get(charavs[turn][tick])
            if not graphavs[turn][tick]:
                graphs[turn][tick].remove(graph)
                if len(graphs[turn][tick]) == 1:
                    uniqgraph[turn][tick] = next(iter(graphs[turn][tick]))
                else:
                    uniqgraph[turn][tick] = None

    def get_char_graph_avs(self, char, graph, branch, turn, tick):
        return self._forward_valcache(
            self.graphavs[(char, graph)], branch, turn, tick
        ) or set()

    def get_char_graph_solo_av(self, char, graph, branch, turn, tick):
        return self._forward_valcache(
            self.soloav[(char, graph)], branch, turn, tick, copy=False
        )

    def get_char_only_av(self, char, branch, turn, tick):
        return self._forward_valcache(
            self.uniqav[char], branch, turn, tick, copy=False
        )

    def get_char_only_graph(self, char, branch, turn, tick):
        return self._forward_valcache(
            self.uniqgraph[char], branch, turn, tick, copy=False
        )

    def get_char_graphs(self, char, branch, turn, tick):
        return self._forward_valcache(
            self.graphs[char], branch, turn, tick
        ) or set()

    def iter_node_users(self, graph, node, branch, turn, tick):
        if graph not in self.user_order:
            return
        for character in self.user_order[graph][node]:
            if (graph, node, character, branch) not in self.user_shallow:
                for (b, t) in self.db._active_branches(branch, turn):
                    if b in self.user_order[graph][node][character]:
                        isav = self.user_order[graph][node][character][b][t]
                        self.store(character, graph, node, branch, turn, tick, isav[isav.end])
                        break
                else:
                    self.store(character, graph, node, branch, turn, tick, None)
            if self.user_shallow[(graph, node, character, branch)][turn][tick]:
                yield character


class RulesHandledCache(object):
    depth = 1

    def __init__(self, engine):
        self.engine = engine
        self.shallow = {}
        self.unhandled = StructuredDefaultDict(self.depth, dict)

    def store(self, *args):
        entity = args[:-4]
        rulebook, rule, branch, turn = args[-4:]
        if turn >= self.engine._branch_end[branch]:
            self.engine._branch_end[branch] = turn
        else:
            raise HistoryError(
                "Tried to cache a value at {}, "
                "but the branch {} already has history up to {}".format(
                    turn, branch, self.engine._branch_end[branch]
                )
            )
        shalo = self.shallow.setdefault(entity + (rulebook, rule, branch), set())
        unhandl = self.unhandled
        for spot in entity:
            unhandl = unhandl[spot]
        if turn not in unhandl.setdefault(branch, {}):
            itargs = entity + (branch, turn)
            unhandl[branch][turn] = list(self._iter_rulebook(*itargs))
        unhandl[branch][turn].remove(entity + (rulebook, rule))
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
    def _iter_rulebook(self, character, branch, turn):
        rulebook = self.engine._characters_rulebooks_cache.retrieve(character, branch, turn, 0)
        for rule in self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, 0):
            yield character, rulebook, rule


class AvatarRulesHandledCache(RulesHandledCache):
    depth = 3

    def _iter_rulebook(self, character, branch, turn):
        rulebook = self.engine._avatars_rulebooks_cache.retrieve(character, branch, turn, 0)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, 0)
        for graph in self.engine.character[character].avatar:
            for avatar in self.engine.character[character].avatar[graph]:
                for rule in rules:
                    yield character, graph, avatar, rulebook, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
    depth = 2

    def _iter_rulebook(self, character, thing, branch, turn):
        rulebook = self.engine._characters_things_rulebooks_cache.retrieve(character, branch, turn, 0)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, 0)
        for thing in self.engine.character[character].thing:
            for rule in rules:
                yield character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):
    depth = 2

    def _iter_rulebook(self, character, branch, turn):
        rulebook = self.engine._characters_places_rulebooks_cache.retrieve(character, branch, turn, 0)
        rules = self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, 0)
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

    def _iter_rulebook(self, character, orig, dest, branch, turn):
        rulebook = self.engine._portals_rulebooks_cache.retrieve(character, orig, dest, branch, turn, 0)
        for rule in self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, 0):
            yield character, orig, dest, rulebook, rule


class ThingsCache(Cache):
    def __init__(self, db):
        Cache.__init__(self, db)
        self._make_node = db.thing_cls

    def turn_before(self, character, thing, branch, turn):
        self.retrieve(character, thing, branch, turn)
        return self.keys[(character,)][thing][branch].rev_before(turn)

    def turn_after(self, character, thing, branch, turn):
        self.retrieve(character, thing, branch, turn)
        return self.keys[(character,)][thing][branch].rev_after(turn)
