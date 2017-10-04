from allegedb.cache import (
    Cache,
    PickyDefaultDict,
    StructuredDefaultDict,
    TurnDict,
    HistoryError
)
from .util import singleton_get


class EntitylessCache(Cache):
    def store(self, key, branch, turn, tick, value, *, planning=False):
        super().store(None, key, branch, turn, tick, value, planning=planning)

    def load(self, data, validate=False):
        return super().load(((None,) + row for row in data), validate)

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

    def store(self, character, graph, node, branch, turn, tick, is_avatar, *, planning=False):
        if not is_avatar:
            is_avatar = None
        Cache.store(self, character, graph, node, branch, turn, tick, is_avatar, planning=False)
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
            self.uniqav[character], branch, turn, tick, copy=False
        )
        self._forward_valcache(
            self.uniqgraph[character], branch, turn, tick, copy=False
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
                for (b, t, tc) in self.db._iter_parent_btt(branch, turn, tick):
                    if b in self.user_order[graph][node][character]:
                        isav = self.user_order[graph][node][character][b][t]
                        self.store(character, graph, node, branch, turn, tick, isav[tc])
                        break
                else:
                    self.store(character, graph, node, branch, turn, tick, None)
            if self.user_shallow[(graph, node, character, branch)][turn][tick]:
                yield character


class RulesHandledCache(object):
    def __init__(self, engine):
        self.engine = engine
        self.handled = {}
        self.unhandled = StructuredDefaultDict(3, list)

    def get_rulebook(self, *args):
        raise NotImplementedError

    def iter_unhandled_rules(self, branch, turn, tick):
        raise NotImplementedError

    def store(self, *args):
        entity = args[:-5]
        rulebook, rule, branch, turn, tick = args[-5:]
        shalo = self.handled.setdefault(entity + (rulebook, branch, turn), set())
        unhandl = self.unhandled[entity]
        if turn not in unhandl.setdefault(branch, {}):
            unhandl[branch][turn] = list(self.iter_unhandled_rules(branch, turn, tick))
        unhandl[branch][turn].remove(entity + (rulebook, rule))
        shalo.add(rule)

    def fork(self, branch, turn, tick):
        parent_branch, parent_turn, parent_tick = self.engine._parent_btt[branch]
        unhandl = self.unhandled
        for entity in unhandl:
            rulebook = self.get_rulebook(*entity + (branch, turn, tick))
            if entity + (rulebook, branch, turn) in self.handled:
                raise HistoryError(
                    "Tried to fork history in a RulesHandledCache, "
                    "but it seems like rules have already been run where we're "
                    "forking to"
                )
            unhandled_rules = unhandl[entity][parent_branch][parent_turn].copy()
            unhandled_rules_set = set(unhandled_rules)
            self.handled[entity + (rulebook, branch, turn)] = {
                rule for rule in self.iter_unhandled_rules(branch, turn, tick)
                if rule not in unhandled_rules_set
            }
            unhandl[entity][branch][turn] = unhandled_rules

    def retrieve(self, *args):
        return self.handled[args]

    def unhandled_rulebook_rules(self, entity, rulebook, branch, turn, tick):
        if (
            entity in self.unhandled and
            rulebook in self.unhandled[entity] and
            branch in self.unhandled[entity][rulebook] and
            turn in self.unhandled[entity][rulebook][branch]
        ):
            ret = self.unhandled[entity][rulebook][branch][turn]
        else:
            try:
                self.unhandled[entity][rulebook][branch][turn] = ret = [
                    rule for rule in
                    self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, tick)
                    if rule not in self.handled.setdefault(entity + (rulebook, branch, turn), set())
                ]
            except KeyError:
                return []
        return ret


class CharacterRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        return self.engine._characters_rulebooks_cache.retrieve(character, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character in self.engine.character:
            for rule in self.unhandled_rulebook_rules(
                character,
                self.get_rulebook(character, branch, turn, tick),
                branch, turn, tick
            ):
                yield character, rule


class AvatarRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        return self.engine._avatars_rulebooks_cache.retrieve(character, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            rulebook = self.get_rulebook(character, branch, turn, tick)
            for graph in char.avatar.values():
                for avatar in graph:
                    try:
                        rules = self.unhandled_rulebook_rules((graph, avatar), rulebook, branch, turn, tick)
                    except KeyError:
                        continue
                    for rule in rules:
                        yield character, graph, avatar, rulebook, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        self.engine._characters_things_rulebooks_cache.retrieve(character, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            rulebook = self.get_rulebook(character, branch, turn, tick)
            for thing in char.thing:
                try:
                    rules = self.unhandled_rulebook_rules((character, thing), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        return self.engine._characters_places_rulebooks_cache.retrieve(character, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            rulebook = self.get_rulebook(character, branch, turn, tick)
            for place in char.place:
                try:
                    rules = self.unhandled_rulebook_rules((character, place), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, place, rulebook, rule


class CharacterPortalRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        return self.engine._characters_portals_rulebooks_cache.retrieve(character, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            rulebook = self.get_rulebook(character, branch, turn, tick)
            for orig in char.portal:
                for dest in char.portal[orig]:
                    try:
                        rules = self.unhandled_rulebook_rules((character, orig, dest), rulebook, branch, turn, tick)
                    except KeyError:
                        continue
                    for rule in rules:
                        yield character, orig, dest, rulebook, rule


class NodeRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, node, branch, turn, tick):
        return self.engine._nodes_rulebooks_cache.retrieve(character, node, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            for node in char.node:
                try:
                    rulebook = self.get_rulebook(character, node, branch, turn, tick)
                    rules = self.unhandled_rulebook_rules((character, node), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, node, rulebook, rule


class PortalRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, orig, dest, branch, turn, tick):
        return self.engine._portals_rulebooks_cache.retrieve(character, orig, dest, branch, turn, tick)

    def iter_unhandled_rules(self, branch, turn, tick):
        for character, char in self.engine.character.items():
            for orig, dests in char.portal.items():
                for dest in dests:
                    try:
                        rulebook = self.get_rulebook(character, orig, dest, branch, turn, tick)
                        rules = self.unhandled_rulebook_rules((character, orig, dest), rulebook, branch, turn, tick)
                    except KeyError:
                        continue
                    for rule in rules:
                        yield character, orig, dest, rulebook, rule


class ThingsCache(Cache):
    def __init__(self, db):
        Cache.__init__(self, db)
        self._make_node = db.thing_cls

    def turn_before(self, character, thing, branch, turn):
        try:
            self.retrieve(character, thing, branch, turn, 0)
        except KeyError:
            pass
        return self.keys[(character,)][thing][branch].rev_before(turn)

    def turn_after(self, character, thing, branch, turn):
        try:
            self.retrieve(character, thing, branch, turn, 0)
        except KeyError:
            pass
        return self.keys[(character,)][thing][branch].rev_after(turn)
