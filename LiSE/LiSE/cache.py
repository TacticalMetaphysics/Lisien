# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
from collections import defaultdict
from allegedb.cache import (
    Cache,
    PickyDefaultDict,
    StructuredDefaultDict,
    TurnDict,
    HistoryError,
    JournalContainer
)
from .util import singleton_get, sort_set


class InitializedJournalContainer(JournalContainer):
    def store(self, *args):
        entity, key, branch, turn, tick, prev, value = args[-7:]
        if prev is None or prev == value:
            return
        parent = args[:-7]
        settings_turns = self.settings[branch]
        presettings_turns = self.presettings[branch]
        if turn in settings_turns or turn in settings_turns.future():
            assert turn in presettings_turns or turn in presettings_turns.future()
            setticks = settings_turns[turn]
            presetticks = presettings_turns[turn]
            presetticks[tick] = parent + (entity, key, prev)
            setticks[tick] = parent + (entity, key, value)
        else:
            presettings_turns[turn] = {tick: parent + (entity, key, prev)}
            settings_turns[turn] = {tick: parent + (entity, key, value)}


class InitializedCache(Cache):
    journal_container_cls = InitializedJournalContainer


class EntitylessCache(Cache):
    def store(self, key, branch, turn, tick, prev, value, *, planning=None):
        super().store(None, key, branch, turn, tick, prev, value, planning=planning)

    def load(self, data, validate=False):
        return super().load(((None,) + row for row in data), validate)

    def retrieve(self, key, branch, turn, tick):
        return super().retrieve(None, key, branch, turn, tick)

    def iter_entities_or_keys(self, branch, turn, tick, *, forward=None):
        return super().iter_entities_or_keys(None, branch, turn, tick, forward=forward)
    iter_entities = iter_keys = iter_entities_or_keys

    def contains_entity_or_key(self, ke, branch, turn, tick):
        return super().contains_entity_or_key(None, ke, branch, turn, tick)
    contains_entity = contains_key = contains_entity_or_key

    def retrieve(self, *args):
        return super().retrieve(*(None,)+args)


class InitializedEntitylessCache(EntitylessCache, InitializedCache):
    pass


class AvatarnessCache(Cache):
    """A cache for remembering when a node is an avatar of a character."""
    def __init__(self, engine):
        Cache.__init__(self, engine)
        self.user_order = StructuredDefaultDict(3, TurnDict)
        self.user_shallow = PickyDefaultDict(TurnDict)
        self.graphs = StructuredDefaultDict(1, TurnDict)
        self.graphs_kf = PickyDefaultDict(TurnDict)
        self.graphavs = StructuredDefaultDict(1, TurnDict)
        self.graphavs_kf = PickyDefaultDict(TurnDict)
        self.charavs = StructuredDefaultDict(1, TurnDict)
        self.charavs_kf = PickyDefaultDict(TurnDict)
        self.soloav = StructuredDefaultDict(1, TurnDict)
        self.soloav_kf = PickyDefaultDict(TurnDict)
        self.uniqav = StructuredDefaultDict(1, TurnDict)
        self.uniqav_kf = PickyDefaultDict(TurnDict)
        self.uniqgraph = StructuredDefaultDict(1, TurnDict)
        self.uniqgraph_kf = PickyDefaultDict(TurnDict)
        self.users = StructuredDefaultDict(1, TurnDict)
        self.users_kf = PickyDefaultDict(TurnDict)

    def load(self, data, validate=False, keyframe=False, cb=None):
        kf = super().load(data, validate, keyframe, cb)
        if keyframe:
            soloav_kf = {}
            uniqav_kf = {}
            uniqgraph_kf = {}
            chargraphs = defaultdict(set)
            graphavs = defaultdict(set)
            avuser = defaultdict(set)
            charavs = defaultdict(set)

            for (character, graph, avatar), extant in kf.items():
                if extant:
                    charavs[character].add((graph, avatar))
                    chargraphs[character].add(graph)
                    graphavs[character, graph].add(avatar)
                    avuser[graph, avatar].add(character)
            for character, graphav_s in charavs.items():
                if len(graphav_s) == 1:
                    soloav_kf[character] = uniqav_kf[character] = next(iter(graphav_s))
            for character, graph_s in chargraphs.items():
                if len(graph_s) == 1:
                    uniqgraph_kf[character] = next(iter(graph_s))

    def _store(self, character, graph, node, branch, turn, tick, was_avatar, is_avatar, *, planning, journal):
        if was_avatar == is_avatar:
            return
        is_avatar = True if is_avatar else None
        super()._store(character, graph, node, branch, turn, tick, was_avatar, is_avatar, planning=planning, journal=journal)
        userturns = self.user_order[graph][node][character][branch]
        if turn in userturns:
            userturns[turn][tick] = is_avatar
        else:
            userturns[turn] = {tick: is_avatar}
        usershal = self.user_shallow[(graph, node, character, branch)]
        if turn in usershal:
            usershal[turn][tick] = is_avatar
        else:
            usershal[turn] = {tick: is_avatar}
        charavs = self.charavs[character][branch]
        graphavs = self.graphavs[(character, graph)][branch]
        graphs = self.graphs[character][branch]
        uniqgraph = self.uniqgraph[character][branch]
        soloav = self.soloav[(character, graph)][branch]
        uniqav = self.uniqav[character][branch]
        users = self.users[graph, node][branch]

        def add_something(cache, what):
            if turn in cache:
                nucache = cache[turn][tick].copy()
                nucache.add(what)
                cache[turn][tick] = nucache
            elif cache.rev_gettable(turn):
                cacheturn = cache[turn]
                nucache = cacheturn[cacheturn.end].copy()
                nucache.add(what)
                cache[turn] = {tick: nucache}
            else:
                cache[turn] = {tick: {what}}

        def remove_something(cache, what):
            if turn in cache:
                nucache = cache[turn][tick].copy()
                nucache.remove(what)
                cache[turn][tick] = nucache
            elif cache.rev_gettable(turn):
                cacheturn = cache[turn]
                nucache = cacheturn[cacheturn.end].copy()
                nucache.remove(what)
                cache[turn] = {tick: nucache}
            else:
                raise ValueError
        if is_avatar:
            add_something(graphavs, node)
            add_something(charavs, (graph, node))
            add_something(graphs, graph)
            add_something(users, character)
        else:
            remove_something(graphavs, node)
            remove_something(charavs, (graph, node))
            if not graphavs[turn][tick]:
                remove_something(graphs, graph)
            if not charavs[turn][tick]:
                remove_something(users, character)
        graphav = singleton_get(graphavs[turn][tick])
        if turn in soloav:
            soloav[turn][tick] = graphav
        else:
            soloav[turn] = {tick: graphav}
        charav = singleton_get(charavs[turn][tick])
        if turn in uniqav:
            uniqav[turn][tick] = charav
        else:
            uniqav[turn] = {tick: charav}
        if not graphavs[turn][tick]:
            graphs[turn][tick].remove(graph)
            if len(graphs[turn][tick]) == 1:
                uniqgraph[turn][tick] = next(iter(graphs[turn][tick]))
            else:
                uniqgraph[turn][tick] = None
        if turn in graphavs and tick in graphavs[turn] and len(graphavs[turn][tick]) != 1:
            if turn in soloav:
                soloav[turn][tick] = None
            else:
                soloav[turn] = soloav.cls({tick: None})
        else:
            if turn in soloav:
                soloav[turn][tick] = node
            else:
                soloav[turn] = soloav.cls({tick: None})
        if turn in charavs and charavs[turn].rev_gettable(tick) and len(charavs[turn][tick]) != 1:
            if turn in uniqav:
                uniqav[turn][tick] = None
            else:
                uniqav[turn] = uniqav.cls({tick: None})
        elif turn in uniqav:
            uniqav[turn][tick] = (graph, node)
        else:
            uniqav[turn] = uniqav.cls({tick: (graph, node)})
        if turn in graphs and graphs[turn].rev_gettable(tick) and len(graphs[turn][tick]) != 1:
            if turn in uniqgraph:
                uniqgraph[turn][tick] = None
            else:
                uniqgraph[turn] = uniqgraph.cls({tick: None})
        elif turn in uniqgraph:
            uniqgraph[turn][tick] = graph
        else:
            uniqgraph[turn] = uniqgraph.cls({tick: graph})

    def get_char_graph_avs(self, char, graph, branch, turn, tick):
        return self._valcache_lookup(
            self.graphavs[(char, graph)], self.graphavs_kf, branch, turn, tick
        ) or set()

    def get_char_graph_solo_av(self, char, graph, branch, turn, tick):
        return self._valcache_lookup(
            self.soloav[(char, graph)], self.soloav_kf, branch, turn, tick
        )

    def get_char_only_av(self, char, branch, turn, tick):
        return self._valcache_lookup(
            self.uniqav[char], self.uniqav_kf, branch, turn, tick
        )

    def get_char_only_graph(self, char, branch, turn, tick):
        return self._valcache_lookup(
            self.uniqgraph[char], self.uniqgraph_kf, branch, turn, tick
        )

    def get_char_graphs(self, char, branch, turn, tick):
        return self._valcache_lookup(
            self.graphs[char], self.graphs_kf, branch, turn, tick
        ) or set()

    def _slow_iter_character_avatars(self, character, branch, turn, tick, *, forward):
        for graph in self.iter_entities(character, branch, turn, tick, forward=forward):
            for node in self.iter_entities(character, graph, branch, turn, tick, forward=forward):
                yield graph, node

    def _slow_iter_users(self, graph, node, branch, turn, tick):
        if graph not in self.user_order:
            return
        for character in self.user_order[graph][node]:
            if (graph, node, character, branch) not in self.user_shallow:
                for (b, t, tc) in self.db._iter_parent_btt(branch, turn, tick):
                    if b in self.user_order[graph][node][character]:
                        isav = self.user_order[graph][node][character][b][t]
                        # side effect!! bad!
                        self.store(character, graph, node, branch, turn, tick, isav[tc])
                        break
                else:
                    self.store(character, graph, node, branch, turn, tick, None)
            try:
                if self.user_shallow[(graph, node, character, branch)][turn][tick]:
                   yield character
            except HistoryError:
                continue


class RulesHandledCache(object):
    def __init__(self, engine):
        self.engine = engine
        self.handled = {}
        self.unhandled = {}

    def get_rulebook(self, *args):
        raise NotImplementedError

    def iter_unhandled_rules(self, branch, turn, tick):
        raise NotImplementedError

    def store(self, *args, loading=False):
        entity = args[:-5]
        rulebook, rule, branch, turn, tick = args[-5:]
        self.handled.setdefault(entity + (rulebook, branch, turn), set()).add(rule)
        unhandl = self.unhandled.setdefault(entity, {}).setdefault(rulebook, {}).setdefault(branch, {})
        if turn not in unhandl:
            unhandl[turn] = list(self.unhandled_rulebook_rules(entity, rulebook, branch, turn, tick))
        try:
            unhandl[turn].remove(rule)
        except ValueError:
            if not loading:
                raise

    def fork(self, branch, turn, tick):
        parent_branch, parent_turn, parent_tick, end_turn, end_tick = self.engine._branches[branch]
        unhandl = self.unhandled
        handl = self.handled
        rbcache = self.engine._rulebooks_cache
        unhandrr = self.unhandled_rulebook_rules
        getrb = self.get_rulebook
        for entity in unhandl:
            rulebook = getrb(*entity + (branch, turn, tick))
            if entity + (rulebook, branch, turn) in handl:
                raise HistoryError(
                    "Tried to fork history in a RulesHandledCache, "
                    "but it seems like rules have already been run where we're "
                    "forking to"
                )
            rules = set(rbcache.retrieve(rulebook, branch, turn, tick))
            unhandled_rules = unhandrr(entity, rulebook, parent_branch, parent_turn, tick)
            unhandled_rules_set = set(unhandled_rules)
            handl[entity + (rulebook, branch, turn)] = rules.difference(unhandled_rules_set)
            unhandl[entity][rulebook][branch][turn] = unhandled_rules

    def retrieve(self, *args):
        return self.handled[args]

    def unhandled_rulebook_rules(self, *args):
        entity = args[:-4]
        rulebook, branch, turn, tick = args[-4:]
        if (
            entity in self.unhandled and
            rulebook in self.unhandled[entity] and
            branch in self.unhandled[entity][rulebook] and
            turn in self.unhandled[entity][rulebook][branch]
        ):
            ret = self.unhandled[entity][rulebook][branch][turn]
        else:
            try:
                return [
                    rule for rule in
                    self.engine._rulebooks_cache.retrieve(rulebook, branch, turn, tick)
                    if rule not in self.handled.setdefault(entity + (rulebook, branch, turn), set())
                ]
            except KeyError:
                return []
        return ret


class CharacterRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        try:
            return self.engine._characters_rulebooks_cache.retrieve(character, branch, turn, tick)
        except KeyError:
            return character, 'character'

    def iter_unhandled_rules(self, branch, turn, tick):
        for character in sort_set(self.engine.character.keys()):
            rb = self.get_rulebook(character, branch, turn, tick)
            for rule in self.unhandled_rulebook_rules(
                character, rb, branch, turn, tick
            ):
                yield character, rb, rule


class AvatarRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        try:
            return self.engine._avatars_rulebooks_cache.retrieve(character, branch, turn, tick)
        except KeyError:
            return character, 'avatar'

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            rulebook = self.get_rulebook(character, branch, turn, tick)
            charavm = charm[character].avatar
            for graph in sort_set(charavm.keys()):
                for avatar in sort_set(charavm[graph].keys()):
                    try:
                        rules = self.unhandled_rulebook_rules((graph, avatar), rulebook, branch, turn, tick)
                    except KeyError:
                        continue
                    for rule in rules:
                        yield character, rulebook, graph, avatar, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        try:
            return self.engine._characters_things_rulebooks_cache.retrieve(character, branch, turn, tick)
        except KeyError:
            return character, 'character_thing'

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            rulebook = self.get_rulebook(character, branch, turn, tick)
            things = sort_set(charm[character].thing.keys())
            pass
            for thing in things:
                try:
                    rules = self.unhandled_rulebook_rules((character, thing), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        try:
            return self.engine._characters_places_rulebooks_cache.retrieve(character, branch, turn, tick)
        except KeyError:
            return character, 'character_place'

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            rulebook = self.get_rulebook(character, branch, turn, tick)
            for place in sort_set(charm[character].place.keys()):
                try:
                    rules = self.unhandled_rulebook_rules((character, place), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, place, rulebook, rule


class CharacterPortalRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, branch, turn, tick):
        try:
            return self.engine._characters_portals_rulebooks_cache.retrieve(character, branch, turn, tick)
        except KeyError:
            return character, 'character_portal'

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            try:
                rulebook = self.get_rulebook(character, branch, turn, tick)
            except KeyError:
                continue
            charp = charm[character].portal
            for orig in sort_set(charp.keys()):
                for dest in sort_set(charp[orig].keys()):
                    try:
                        rules = self.unhandled_rulebook_rules((character, orig, dest), rulebook, branch, turn, tick)
                    except KeyError:
                        continue
                    for rule in rules:
                        yield character, orig, dest, rulebook, rule


class NodeRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, node, branch, turn, tick):
        try:
            return self.engine._nodes_rulebooks_cache.retrieve(character, node, branch, turn, tick)
        except KeyError:
            return character, node

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            for node in sort_set(charm[character].node.keys()):
                try:
                    rulebook = self.get_rulebook(character, node, branch, turn, tick)
                    rules = self.unhandled_rulebook_rules((character, node), rulebook, branch, turn, tick)
                except KeyError:
                    continue
                for rule in rules:
                    yield character, node, rulebook, rule


class PortalRulesHandledCache(RulesHandledCache):
    def get_rulebook(self, character, orig, dest, branch, turn, tick):
        try:
            return self.engine._portals_rulebooks_cache.retrieve(character, orig, dest, branch, turn, tick)
        except KeyError:
            return character, orig, dest

    def iter_unhandled_rules(self, branch, turn, tick):
        charm = self.engine.character
        for character in sort_set(charm.keys()):
            charp = charm[character].portal
            for orig in sort_set(charp.keys()):
                dests = charp[orig]
                for dest in sort_set(dests.keys()):
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

    def _store(self, *args, planning, journal=True):
        character, thing, branch, turn, tick, (oldloc, oldnxtloc), (location, next_location) = args
        super()._store(*args, planning=planning, journal=journal)
        if oldloc is not None:
            try:
                oldnodecont = self.db._node_contents_cache.retrieve(
                    character, oldloc, branch, turn, tick
                )
            except KeyError:
                oldnodecont = frozenset()
            self.db._node_contents_cache.store(
                character, thing, branch, turn, tick, oldnodecont, oldnodecont.difference((thing,))
            )
        if oldnxtloc is not None:
            try:
                oldedgecont = self.db._portal_contents_cache.retrieve(
                    character, oldloc, oldnxtloc, branch, turn, tick
                )
            except KeyError:
                oldedgecont = frozenset()
            self.db._portal_contents_cache.store(
                character, oldloc, oldnxtloc, branch, turn, tick,
                oldedgecont, oldedgecont.difference((thing,))
            )
        try:
            newnodecont = self.db._node_contents_cache.retrieve(
                character, location, branch, turn, tick
            )
        except KeyError:
            newnodecont = frozenset()
        self.db._node_contents_cache.store(
            character, location, branch, turn, tick,
            newnodecont, newnodecont.union((thing,))
        )
        if next_location is not None:
            try:
                newedgecont = self.db._portal_contents_cache.retrieve(
                    character, location, next_location,
                    branch, turn, tick
                )
            except KeyError:
                newedgecont = frozenset()
            self.db._portal_contents_cache.store(
                character, location, next_location,
                branch, turn, tick,
                newedgecont, newedgecont.union((thing,))
            )

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