# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The "engine" of LiSE is an object relational mapper with special
stores for game data and entities, as well as properties for manipulating the
flow of time.

"""
from random import Random
from functools import partial, partialmethod
from types import FunctionType
from json import dumps, loads, JSONEncoder
from operator import gt, lt, ge, le, eq, ne
from blinker import Signal
from allegedb import ORM as gORM, update_window, update_backward_window
from allegedb.xjson import JSONReWrapper, JSONListReWrapper
from .xcollections import (
    StringStore,
    FunctionStore,
    CharacterMapping,
    UniversalMapping
)
from .character import Character
from .thing import Thing
from .place import Place
from .portal import Portal
from .rule import AllRuleBooks, AllRules, Rule
from .query import Query, QueryEngine
from .util import getatt, reify, EntityStatAccessor
from .cache import (
    Cache,
    EntitylessCache,
    AvatarnessCache,
    AvatarRulesHandledCache,
    CharacterThingRulesHandledCache,
    CharacterPlaceRulesHandledCache,
    CharacterPortalRulesHandledCache,
    NodeRulesHandledCache,
    PortalRulesHandledCache,
    CharacterRulesHandledCache,
    ThingsCache
)


class NextTurn(Signal):
    """Make time move forward in the simulation.

    Calls ``advance`` repeatedly, returning a list of the rules' return values.

    I am also a ``Signal``, so you can register functions to be
    called when the simulation runs. Pass them to my ``connect``
    method.

    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __call__(self):
        engine = self.engine
        start_branch, start_turn, start_tick = engine.btt()
        with engine.advancing:
            done = False
            for res in iter(engine.advance, final_rule):
                if res:
                    branch, turn, tick = engine.btt()
                    engine.universal['last_result'] = res
                    engine.universal['last_result_idx'] = 0
                    engine.universal['rando_state'] = engine.rando.getstate()
                    self.send(
                        engine,
                        branch=branch,
                        turn=turn,
                        tick=tick
                    )
                    break
            else:
                done = True
        if not done:
            return [], engine.get_delta(
                branch=start_branch,
                turn_from=start_turn,
                turn_to=engine.turn,
                tick_from=start_tick,
                tick_to=engine.tick
            )
        branch, turn = engine.time
        turn += 1
        # As a side effect, the following assignment sets the tick to
        # the latest in the new turn, which will be 0 if that turn has not
        # yet been simulated.
        engine.time = branch, turn
        if engine.tick == 0:
            engine.universal['rando_state'] = engine.rando.getstate()
        else:
            engine.rando.setstate(engine.universal['rando_state'])
        self.send(
            self.engine,
            branch=branch,
            turn=turn,
            tick=engine.tick
        )
        return [], engine.get_delta(
            branch=branch,
            turn_from=start_turn,
            turn_to=turn,
            tick_from=start_tick,
            tick_to=engine.tick
        )


class DummyEntity(dict):
    """Something to use in place of a node or edge"""
    __slots__ = ['engine']

    def __init__(self, engine):
        self.engine = engine


class FinalRule:
    """A singleton sentinel for the rule iterator"""
    __slots__ = []

    def __hash__(self):
        # completely random integer
        return 6448962173793096248


final_rule = FinalRule()


json_dump_hints = {final_rule: 'final_rule'}
json_load_hints = {'final_rule': final_rule}


class Encoder(JSONEncoder):
    """Extend the base JSON encoder to handle a couple of numpy types I might need"""
    def encode(self, o):
        if type(o) in (str, int, float):
            return super().encode(o)
        return super().encode(self.listify(o))

    def default(self, o):
        t = type(o)
        if t is JSONReWrapper:
            return dict(o)
        elif t is JSONListReWrapper:
            return list(o)
        try:
            from numpy import sctypes
        except ImportError:
            return super().default(o)
        if t in sctypes['int']:
            return int(o)
        elif t in sctypes['float']:
            return float(o)
        else:
            return super().default(o)


class AbstractEngine(object):
    """Parent class to the real Engine as well as EngineProxy.

    Implements serialization methods and the __getattr__ for stored methods.

    """
    def __getattr__(self, att):
        if hasattr(super(), 'method') and hasattr(self.method, att):
            return partial(getattr(self.method, att), self)
        raise AttributeError('No attribute or stored method: {}'.format(att))

    def _listify_function(self, obj):
        if not hasattr(getattr(self, obj.__module__), obj.__name__):
            raise ValueError("Function {} is not in my function stores".format(obj.__name__))
        return [obj.__module__, obj.__name__]

    def listify(self, obj):
        """Turn a LiSE object into a list for easier serialization"""
        listify_dispatch = {
            list: lambda obj: ["list"] + [self.listify(v) for v in obj],
            tuple: lambda obj: ["tuple"] + [self.listify(v) for v in obj],
            dict: lambda obj: ["dict"] + [
                [self.listify(k), self.listify(v)]
                for (k, v) in obj.items()
            ],
            self.char_cls: lambda obj: ["character", obj.name],
            self.thing_cls: lambda obj: ["thing", obj.character.name, obj.name, self.listify(obj.location.name), self.listify(obj.next_location.name), obj['arrival_time'], obj['next_arrival_time']],
            self.place_cls: lambda obj: ["place", obj.character.name, obj.name],
            self.portal_cls: lambda obj: [
                "portal", obj.character.name, obj.orig, obj.dest],
            FunctionType: self._listify_function
        }
        try:
            listifier = listify_dispatch[type(obj)]
            return listifier(obj)
        except KeyError:
            return obj


    def delistify(self, obj):
        """Turn a list describing a LiSE object into that object

        If this is impossible, return the argument.

        """
        def nodeget(obj):
            return self._node_objs[(
                self.delistify(obj[1]), self.delistify(obj[2])
            )]
        delistify_dispatch = {
            'list': lambda obj: [self.delistify(v) for v in obj[1:]],
            'tuple': lambda obj: tuple(self.delistify(v) for v in obj[1:]),
            'dict': lambda obj: {
                self.delistify(k): self.delistify(v)
                for (k, v) in obj[1:]
            },
            'character': lambda obj: self.character[self.delistify(obj[1])],
            'place': nodeget,
            'thing': nodeget,
            'node': nodeget,
            'portal': lambda obj: self._portal_objs[(
                self.delistify(obj[1]),
                self.delistify(obj[2]),
                self.delistify(obj[3])
            )],
            'function': lambda obj: getattr(self.function, obj[1]),
            'method': lambda obj: getattr(self.method, obj[1]),
            'prereq': lambda obj: getattr(self.prereq, obj[1]),
            'trigger': lambda obj: getattr(self.trigger, obj[1]),
            'action': lambda obj: getattr(self.action, obj[1])
        }
        if isinstance(obj, list) or isinstance(obj, tuple):
            return delistify_dispatch[obj[0]](obj)
        else:
            return obj

    @reify
    def json_encoder(self):
        class EngEncoder(Encoder):
            listify = self.listify
        return EngEncoder

    def json_dump(self, obj):
        global json_dump_hints, json_load_hints
        try:
            if obj not in json_dump_hints:
                dumped = json_dump_hints[obj] = dumps(
                    obj, cls=self.json_encoder
                )
                json_load_hints[dumped] = obj
            return json_dump_hints[obj]
        except TypeError:
            return dumps(obj, cls=self.json_encoder)

    def json_load(self, s):
        global json_dump_hints, json_load_hints
        if s in json_load_hints:
            return json_load_hints[s]
        return self.delistify(loads(s))


class Engine(AbstractEngine, gORM):
    """LiSE, the Life Simulator Engine.

    Each instance of LiSE maintains a connection to a database
    representing the state of a simulated world. Simulation rules
    within this world are described by lists of Python functions, some
    of which make changes to the world.

    The top-level data structure within LiSE is the character. Most
    data within the world model is kept in some character or other;
    these will quite frequently represent people, but can be readily
    adapted to represent any kind of data that can be comfortably
    described as a graph or a JSON object. Every change to a character
    will be written to the database.

    LiSE tracks history as a series of turns. In each turn, each
    simulation rule is evaluated once for each of the simulated
    entities it's been applied to. World changes in a given turn are
    remembered together, such that the whole world state can be
    rewound: simply set the properties ``branch`` and ``turn`` back to
    what they were just before the change you want to undo.

    Properties:

    - ``branch``: The fork of the timestream that we're on.
    - ``turn``: Units of time that have passed since the sim started.
    - ``time``: ``(branch, turn)``
    - ``tick``: A counter of how many changes have occurred this turn
    - ``character``: A mapping of :class:`Character` objects by name.
    - ``rule``: A mapping of all rules that have been made.
    - ``rulebook``: A mapping of lists of rules. They are followed in
      their order.  A whole rulebook full of rules may be assigned to
      an entity at once.
    - ``trigger``: A mapping of functions that might trigger a rule.
    - ``prereq``: A mapping of functions a rule might require to return
      ``True`` for it to run.
    - ``action``: A mapping of functions that might manipulate the world
      state as a result of a rule running.
    - ``function``: A mapping of generic functions stored in the same
      database as the previous.
    - ``string``: A mapping of strings, probably shown to the player
      at some point.
    - ``eternal``: Mapping of arbitrary serializable objects. It isn't
      sensitive to sim-time. A good place to keep game settings.
    - ``universal``: Another mapping of arbitrary serializable
      objects, but this one *is* sensitive to sim-time. Each turn, the
      state of the randomizer is saved here under the key
      ``'rando_state'``.
    - ``rando``: The randomizer used by all of the rules.

    """
    char_cls = Character
    thing_cls = Thing
    place_cls = node_cls = Place
    portal_cls = edge_cls = _make_edge = Portal
    query_engine_cls = QueryEngine
    illegal_graph_names = ['global', 'eternal', 'universal', 'rulebooks', 'rules']
    illegal_node_names = ['nodes', 'node_val', 'edges', 'edge_val', 'things']

    def _make_node(self, char, node):
        if self._is_thing(char.name, node):
            return Thing(char, node)
        else:
            return Place(char, node)

    def get_delta(self, branch, turn_from, tick_from, turn_to, tick_to):
        """Get a dictionary describing changes to the world.

        Most keys will be character names, and their values will be dictionaries of
        the character's stats' new values, with ``None`` for deleted keys. Characters'
        dictionaries have special keys 'nodes' and 'edges' which contain booleans indicating
        whether the node or edge exists at the moment, and 'node_val' and 'edge_val' for
        the stats of those entities. For edges (also called portals) these dictionaries
        are two layers deep, keyed first by the origin, then by the destination.

        Characters also have special keys for the various rulebooks they have:

        * 'character_rulebook'
        * 'avatar_rulebook'
        * 'character_thing_rulebook'
        * 'character_place_rulebook'
        * 'character_portal_rulebook'

        And each node and edge may have a 'rulebook' stat of its own. If a node is a thing,
        it gets a 'location' and possibly 'next_location'; when the 'location' is deleted,
        that means it's back to being a place.

        Keys at the top level that are not character names:

        * 'rulebooks', a dictionary keyed by the name of each changed rulebook, the value
        being a list of rule names
        * 'rules', a dictionary keyed by the name of each changed rule, containing any
        of the lists 'triggers', 'prereqs', and 'actions'

        """
        if turn_from == turn_to:
            return self.get_turn_delta(branch, turn_to, tick_to, start_tick=tick_from)
        delta = super().get_delta(branch, turn_from, tick_from, turn_to, tick_to)
        if turn_from < turn_to:
            updater = partial(update_window, turn_from, tick_from, turn_to, tick_to)
            thbranches = self._things_cache.settings
            rbbranches = self._rulebooks_cache.settings
            trigbranches = self._triggers_cache.settings
            preqbranches = self._prereqs_cache.settings
            actbranches = self._actions_cache.settings
            charrbbranches = self._characters_rulebooks_cache.settings
            avrbbranches = self._avatars_rulebooks_cache.settings
            charthrbbranches = self._characters_things_rulebooks_cache.settings
            charplrbbranches = self._characters_places_rulebooks_cache.settings
            charporbbranches = self._characters_portals_rulebooks_cache.settings
            noderbbranches = self._nodes_rulebooks_cache.settings
            edgerbbranches = self._portals_rulebooks_cache.settings
        else:
            updater = partial(update_backward_window, turn_from, tick_from, turn_to, tick_to)
            thbranches = self._things_cache.presettings
            rbbranches = self._rulebooks_cache.presettings
            trigbranches = self._triggers_cache.presettings
            preqbranches = self._prereqs_cache.presettings
            actbranches = self._actions_cache.presettings
            charrbbranches = self._characters_rulebooks_cache.presettings
            avrbbranches = self._avatars_rulebooks_cache.presettings
            charthrbbranches = self._characters_things_rulebooks_cache.presettings
            charplrbbranches = self._characters_places_rulebooks_cache.presettings
            charporbbranches = self._characters_portals_rulebooks_cache.presettings
            noderbbranches = self._nodes_rulebooks_cache.presettings
            edgerbbranches = self._portals_rulebooks_cache.presettings

        def updthing(char, thing, locs):
            if locs is None:
                loc = nxtloc = None
            else:
                loc, nxtloc = locs
            thingd = delta.setdefault(char, {}).setdefault('node_val', {}).setdefault(thing, {})
            thingd['location'] = loc
            thingd['next_location'] = nxtloc
        if branch in thbranches:
            updater(updthing, thbranches[branch])
        # TODO handle arrival_time and next_arrival_time stats of things

        delta['rulebooks'] = {}
        def updrb(whatev, rulebook, rules):
            delta['rulebooks'][rulebook] = rules

        if branch in rbbranches:
            updater(updrb, rbbranches[branch])

        delta['rules'] = {}

        def updru(key, _, rule, funs):
            delta['rules'].setdefault(rule, {})[key] = funs

        if branch in trigbranches:
            updater(partial(updru, 'triggers'), trigbranches[branch])

        if branch in preqbranches:
            updater(partial(updru, 'prereqs'), preqbranches[branch])

        if branch in actbranches:
            updater(partial(updru, 'actions'), actbranches[branch])

        def updcrb(key, character, rulebook):
            delta.setdefault(character, {})[key] = rulebook

        if branch in charrbbranches:
            updater(partial(updcrb, 'character_rulebook'), charrbbranches[branch])

        if branch in avrbbranches:
            updater(partial(updcrb, 'avatar_rulebook'), avrbbranches[branch])

        if branch in charthrbbranches:
            updater(partial(updcrb, 'character_thing_rulebook'), charthrbbranches[branch])

        if branch in charplrbbranches:
            updater(partial(updcrb, 'character_place_rulebook'), charplrbbranches[branch])

        if branch in charporbbranches:
            updater(partial(updcrb, 'character_portal_rulebook'), charporbbranches[branch])

        def updnoderb(character, node, rulebook):
            delta.setdefault(character, {}).setdefault('node_val', {}).setdefault(node, {})['rulebook'] = rulebook

        if branch in noderbbranches:
            updater(updnoderb, noderbbranches[branch])

        def updedgerb(character, orig, dest, rulebook):
            delta.setdefault(character, {}).setdefault('edge_val', {}).setdefault(
                orig, {}).setdefault(dest, {})['rulebook'] = rulebook

        if branch in edgerbbranches:
            updater(updedgerb, edgerbbranches[branch])

        return delta

    def get_turn_delta(self, branch=None, turn=None, tick=None, start_tick=0):
        """Get a dictionary describing changes to the world within a given turn

        Defaults to the present turn, and stops at the present tick unless specified.

        See the documentation for ``get_delta`` for a detailed description of the
        delta format.

        """
        branch = branch or self.branch
        turn = turn or self.turn
        tick = tick or self.tick
        delta = super().get_turn_delta(branch, turn, start_tick, tick)
        if branch in self._things_cache.settings and self._things_cache.settings[branch].has_exact_rev(turn):
            for chara, thing, (location, next_location) in self._things_cache.settings[branch][turn][start_tick:tick]:
                thingd = delta.setdefault(chara, {}).setdefault('node_val', {}).setdefault(thing, {})
                thingd['location'] = location
                thingd['next_location'] = next_location
        delta['rulebooks'] = rbdif = {}
        if branch in self._rulebooks_cache.settings and self._rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, rulebook, rules in self._rulebooks_cache.settings[branch][turn][start_tick:tick]:
                rbdif[rulebook] = rules
        delta['rules'] = rdif = {}
        if branch in self._triggers_cache.settings and self._triggers_cache.settings[branch].has_exact_rev(turn):
            for _, rule, funs in self._triggers_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['triggers'] = funs
        if branch in self._prereqs_cache.settings and self._prereqs_cache.settings[branch].has_exact_rev(turn):
            for _, rule, funs in self._prereqs_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['prereqs'] = funs
        if branch in self._actions_cache.settings and self._triggers_cache.settings[branch].has_exact_rev(turn):
            for _, rule, funs in self._triggers_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['actions'] = funs

        if branch in self._characters_rulebooks_cache.settings and self._characters_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, character, rulebook in self._characters_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_rulebook'] = rulebook
        if branch in self._avatars_rulebooks_cache.settings and self._avatars_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, character, rulebook in self._avatars_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['avatar_rulebook'] = rulebook
        if branch in self._characters_things_rulebooks_cache.settings and self._characters_things_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, character, rulebook in self._characters_things_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_thing_rulebook'] = rulebook
        if branch in self._characters_places_rulebooks_cache.settings and self._characters_places_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, character, rulebook in self._characters_places_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_place_rulebook'] = rulebook
        if branch in self._characters_portals_rulebooks_cache.settings and self._characters_portals_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for _, character, rulebook in self._characters_portals_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_portal_rulebook'] = rulebook

        if branch in self._nodes_rulebooks_cache.settings and self._nodes_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for character, node, rulebook in self._nodes_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {}).setdefault('node_val', {}).setdefault(node, {})['rulebook'] = rulebook
        if branch in self._portals_rulebooks_cache.settings and self._portals_rulebooks_cache.settings[branch].has_exact_rev(turn):
            for character, orig, dest, rulebook in self._portals_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {}).setdefault('edge_val', {})\
                    .setdefault(orig, {}).setdefault(dest, {})['rulebook'] = rulebook
        return delta

    def _del_rulebook(self, rulebook):
        for (character, character_rulebooks) in \
             self._characters_rulebooks_cache.items():
            if rulebook not in character_rulebooks.values():
                continue
            for (which, rb) in character_rulebooks.items():
                if rb == rulebook:
                    raise ValueError(
                        "Rulebook still in use by {} as {}".format(
                            character, which
                        ))
        for (character, nodes) in self._nodes_rulebooks_cache.items():
            if rulebook not in nodes.values():
                continue
            for (node, rb) in nodes.items():
                if rb == rulebook:
                    raise ValueError(
                        "Rulebook still in use by node "
                        "{} in character {}".format(
                            node, character
                        ))
        for (character, origins) in self._portals_rulebooks_cache.items():
            for (origin, destinations) in origins.items():
                if rulebook not in destinations.values():
                    continue
                for (destination, rb) in destinations:
                    if rb == rulebook:
                        raise ValueError(
                            "Rulebook still in use by portal "
                            "{}->{} in character {}".format(
                                origin, destination, character
                            ))
        self.rule.query.rulebook_del_all(rulebook)
        del self._rulebooks_cache._data[rulebook]

    def _set_node_rulebook(self, character, node, rulebook):
        branch, turn, tick = self.engine.nbtt()
        self._nodes_rulebooks_cache.store(character, node, branch, turn, tick, rulebook)
        self.engine.query.set_node_rulebook(character, node, branch, turn, tick, rulebook)

    def _set_portal_rulebook(self, character, orig, dest, rulebook):
        branch, turn, tick = self.engine.nbtt()
        self._portals_rulebooks_cache.store(character, orig, dest, branch, turn, tick, rulebook)
        self.query.set_portal_rulebook(character, orig, dest, branch, turn, tick, rulebook)

    def _remember_avatarness(
            self, character, graph, node,
            is_avatar=True, branch=None, turn=None,
            tick=None
    ):
        """Use this to record a change in avatarness.

        Should be called whenever a node that wasn't an avatar of a
        character now is, and whenever a node that was an avatar of a
        character now isn't.

        ``character`` is the one using the node as an avatar,
        ``graph`` is the character the node is in.

        """
        branch = branch or self.branch
        turn = turn or self.turn
        tick = tick or self.tick
        self._avatarness_cache.store(
            character,
            graph,
            node,
            branch,
            turn,
            tick,
            is_avatar,
            planning=self.planning
        )
        self.query.avatar_set(
            character,
            graph,
            node,
            branch,
            turn,
            tick,
            is_avatar
        )

    def _init_caches(self):
        super()._init_caches()
        self._portal_objs = {}
        self._things_cache = ThingsCache(self)
        self.character = self.graph = CharacterMapping(self)
        self._universal_cache = EntitylessCache(self)
        self._rulebooks_cache = EntitylessCache(self)
        self._characters_rulebooks_cache = EntitylessCache(self)
        self._avatars_rulebooks_cache = EntitylessCache(self)
        self._characters_things_rulebooks_cache = EntitylessCache(self)
        self._characters_places_rulebooks_cache = EntitylessCache(self)
        self._characters_portals_rulebooks_cache = EntitylessCache(self)
        self._nodes_rulebooks_cache = Cache(self)
        self._portals_rulebooks_cache = Cache(self)
        self._triggers_cache = EntitylessCache(self)
        self._prereqs_cache = EntitylessCache(self)
        self._actions_cache = EntitylessCache(self)
        self._node_rules_handled_cache = NodeRulesHandledCache(self)
        self._portal_rules_handled_cache = PortalRulesHandledCache(self)
        self._character_rules_handled_cache = CharacterRulesHandledCache(self)
        self._avatar_rules_handled_cache = AvatarRulesHandledCache(self)
        self._character_thing_rules_handled_cache \
            = CharacterThingRulesHandledCache(self)
        self._character_place_rules_handled_cache \
            = CharacterPlaceRulesHandledCache(self)
        self._character_portal_rules_handled_cache \
            = CharacterPortalRulesHandledCache(self)
        self._avatarness_cache = AvatarnessCache(self)
        self.eternal = self.query.globl
        self.universal = UniversalMapping(self)
        if hasattr(self, '_action_file'):
            self.action = FunctionStore(self._action_file)
        if hasattr(self, '_prereq_file'):
            self.prereq = FunctionStore(self._prereq_file)
        if hasattr(self, '_trigger_file'):
            self.trigger = FunctionStore(self._trigger_file)
        if hasattr(self, '_function_file'):
            self.function = FunctionStore(self._function_file)
        if hasattr(self, '_method_file'):
            self.method = FunctionStore(self._method_file)
        self.rule = AllRules(self)
        self.rulebook = AllRuleBooks(self)
        if hasattr(self, '_string_file'):
            self.string = StringStore(
                self.query,
                self._string_file,
                self.eternal.setdefault('language', 'eng')
            )

    def _load_graphs(self):
        for charn in self.query.characters():
            self._graph_objs[charn] = Character(self, charn, init_rulebooks=False)

    def __init__(
            self,
            worlddb,
            *,
            string='strings.json',
            function='function.py',
            method='method.py',
            trigger='trigger.py',
            prereq='prereq.py',
            action='action.py',
            connect_args={},
            alchemy=False,
            commit_modulus=None,
            random_seed=None,
            logfun=None,
            validate=False
    ):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        if isinstance(string, str):
            self._string_file = string
        else:
            self.string = string
        if isinstance(function, str):
            self._function_file = function
        else:
            self.function = function
        if isinstance(method, str):
            self._method_file = method
        else:
            self.method = method
        if isinstance(trigger, str):
            self._trigger_file = trigger
        else:
            self.trigger = trigger
        if isinstance(prereq, str):
            self._prereq_file = prereq
        else:
            self.prereq = prereq
        if isinstance(action, str):
            self._action_file = action
        else:
            self.action = action
        super().__init__(
            worlddb,
            connect_args=connect_args,
            alchemy=alchemy,
            validate=validate
        )
        self.next_turn = NextTurn(self)
        if logfun is None:
            from logging import getLogger
            logger = getLogger(__name__)

            def logfun(level, msg):
                getattr(logger, level)(msg)
        self.log = logfun
        self.commit_modulus = commit_modulus
        self.random_seed = random_seed
        self._rules_iter = self._follow_rules()
        # set up the randomizer
        self.rando = Random()
        if 'rando_state' in self.universal:
            self.rando.setstate(self.universal['rando_state'])
        else:
            self.rando.seed(self.random_seed)
            self.universal['rando_state'] = self.rando.getstate()
        if hasattr(self.method, 'init'):
            self.method.init(self)

    def _init_load(self, validate=False):
        q = self.query
        self._things_cache.load((
            (character, thing, branch, turn, tick, (location, next_location))
            for character, thing, branch, turn, tick, location, next_location
            in q.things_dump()
        ), validate)
        super()._init_load(validate=validate)
        self._avatarness_cache.load(q.avatars_dump(), validate)
        self._universal_cache.load(q.universals_dump(), validate)
        self._rulebooks_cache.load(q.rulebooks_dump(), validate)
        self._characters_rulebooks_cache.load(q.character_rulebook_dump(), validate)
        self._avatars_rulebooks_cache.load(q.avatar_rulebook_dump(), validate)
        self._characters_things_rulebooks_cache.load(q.character_thing_rulebook_dump(), validate)
        self._characters_places_rulebooks_cache.load(q.character_place_rulebook_dump(), validate)
        self._characters_portals_rulebooks_cache.load(q.character_portal_rulebook_dump(), validate)
        self._nodes_rulebooks_cache.load(q.node_rulebook_dump(), validate)
        self._portals_rulebooks_cache.load(q.portal_rulebook_dump(), validate)
        self._triggers_cache.load(q.rule_triggers_dump(), validate)
        self._prereqs_cache.load(q.rule_prereqs_dump(), validate)
        self._actions_cache.load(q.rule_actions_dump(), validate)
        # I'm throwing out the ticks here, but I think I might want to use them
        # to map handled rules to changes made by those rules
        for character, rulebook, rule, branch, turn, tick in q.character_rules_handled_dump():
            self._character_rules_handled_cache.store(
                character, rulebook, rule, branch, turn, loading=True
            )
        for character, rulebook, rule, graph, avatar, branch, turn, tick in \
                q.avatar_rules_handled_dump():
            self._avatar_rules_handled_cache.store(
                character, rulebook, rule, graph, avatar, branch, turn, loading=True
            )
        for character, rulebook, rule, thing, branch, turn, tick in \
                q.character_thing_rules_handled_dump():
            self._character_thing_rules_handled_cache.store(
                character, rulebook, rule, thing, branch, turn, loading=True
            )
        for character, rulebook, rule, place, branch, turn, tick in \
                q.character_place_rules_handled_dump():
            self._character_place_rules_handled_cache.store(
                character, rulebook, rule, place, branch, turn, loading=True
            )
        for character, rulebook, rule, orig, dest, branch, turn, tick in \
                q.character_portal_rules_handled_dump():
            self._character_portal_rules_handled_cache.store(
                character, rulebook, rule, orig, dest, branch, turn, loading=True
            )
        for character, node, rulebook, rule, branch, turn, tick in q.node_rules_handled_dump():
            self._node_rules_handled_cache.store(character, node, rulebook, rule, branch, turn, tick, loading=True)
        for character, orig, dest, rulebook, rule, branch, turn, tick in q.portal_rules_handled_dump():
            self._portal_rules_handled_cache.store(character, orig, dest, rulebook, rule, branch, turn, tick)
        self._rules_cache = {name: Rule(self, name, create=False) for name in q.rules_dump()}

    betavariate = getatt('rando.betavariate')
    choice = getatt('rando.choice')
    expovariate = getatt('rando.expovariate')
    gammavariate = getatt('rando.gammavariate')
    gauss = getatt('rando.gauss')
    getrandbits = getatt('rando.getrandbits')
    lognormvariate = getatt('rando.lognormvariate')
    normalvariate = getatt('rando.normalvariate')
    paretovariate = getatt('rando.paretovariate')
    randint = getatt('rando.randint')
    random = getatt('rando.random')
    randrange = getatt('rando.randrange')
    sample = getatt('rando.sample')
    shuffle = getatt('rando.shuffle')
    triangular = getatt('rando.triangular')
    uniform = getatt('rando.uniform')
    vonmisesvariate = getatt('rando.vonmisesvariate')
    weibullvariate = getatt('rando.weibullvariate')

    @property
    def stores(self):
        return (
            self.action,
            self.prereq,
            self.trigger,
            self.function,
            self.method,
            self.string
        )

    def debug(self, msg):
        """Log a message at level 'debug'"""
        self.log('debug', msg)

    def info(self, msg):
        """Log a message at level 'info'"""
        self.log('info', msg)

    def warning(self, msg):
        """Log a message at level 'warning'"""
        self.log('warning', msg)

    def error(self, msg):
        """Log a message at level 'error'"""
        self.log('error', msg)

    def critical(self, msg):
        """Log a message at level 'critical'"""
        self.log('critical', msg)

    def coinflip(self):
        """Return True or False with equal probability."""
        return self.choice((True, False))

    def roll_die(self, d):
        """Roll a die with ``d`` faces. Return the result."""
        return self.randint(1, d)

    def dice(self, n, d):
        """Roll ``n`` dice with ``d`` faces, and yield the results.

        This is an iterator. You'll get the result of each die in
        successon.

        """
        for i in range(0, n):
            yield self.roll_die(d)

    def dice_check(self, n, d, target, comparator=le):
        """Roll ``n`` dice with ``d`` sides, sum them, and return whether they
        are <= ``target``.

        If ``comparator`` is provided, use it instead of <=. You may
        use a string like '<' or '>='.

        """
        comps = {
            '>': gt,
            '<': lt,
            '>=': ge,
            '<=': le,
            '=': eq,
            '==': eq,
            '!=': ne
        }
        try:
            comparator = comps.get(comparator, comparator)
        except TypeError:
            pass
        return comparator(sum(self.dice(n, d)), target)

    def percent_chance(self, pct):
        """Given a ``pct``% chance of something happening right now, decide at
        random whether it actually happens, and return ``True`` or
        ``False`` as appropriate.

        Values not between 0 and 100 are treated as though they
        were 0 or 100, whichever is nearer.

        """
        if pct <= 0:
            return False
        if pct >= 100:
            return True
        return pct / 100 < self.random()

    def close(self):
        """Commit changes and close the database."""
        for store in self.stores:
            if hasattr(store, 'save'):
                store.save()
        super().close()

    def __enter__(self):
        """Return myself. For compatibility with ``with`` semantics."""
        return self

    def __exit__(self, *args):
        """Close on exit."""
        self.close()

    def _set_branch(self, v):
        super()._set_branch(v)
        self.time.send(self.time, branch=self._obranch, turn=self._oturn)

    def _set_turn(self, v):
        super()._set_turn(v)
        self.time.send(self.time, branch=self._obranch, turn=self._oturn)

    def _handled_char(self, charn, rulebook, rulen, branch, turn, tick):
        try:
            self._character_rules_handled_cache.store(
                charn, rulebook, rulen, branch, turn, tick
            )
        except ValueError:
            assert rulen in self._character_rules_handled_cache.handled[
                charn, rulebook, branch, turn
            ]
            return
        self.query.handled_character_rule(
            charn, rulebook, rulen, branch, turn, tick
        )

    def _handled_av(self, character, graph, avatar, rulebook, rule, branch, turn, tick):
        try:
            self._avatar_rules_handled_cache.store(
                character, graph, avatar, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._avatar_rules_handled_cache.handled[
                character, graph, avatar, rulebook, branch, turn
            ]
            return
        self.query.handled_avatar_rule(
            character, rulebook, rule, graph, avatar, branch, turn, tick
        )

    def _handled_char_thing(self, character, thing, rulebook, rule, branch, turn, tick):
        try:
            self._character_thing_rules_handled_cache.store(
                character, thing, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._character_thing_rules_handled_cache.handled[
                character, thing, rulebook, branch, turn
            ]
            return
        self.query.handled_character_thing_rule(
            character, rulebook, rule, thing, branch, turn, tick
        )

    def _handled_char_place(self, character, place, rulebook, rule, branch, turn, tick):
        try:
            self._character_place_rules_handled_cache.store(
                character, place, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._character_place_rules_handled_cache.handled[
                character, place, rulebook, branch, turn
            ]
            return
        self.query.handled_character_place_rule(
            character, rulebook, rule, place, branch, turn, tick
        )

    def _handled_char_port(self, character, orig, dest, rulebook, rule, branch, turn, tick):
        try:
            self._character_portal_rules_handled_cache.store(
                character, orig, dest, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._character_portal_rules_handled_cache.handled[
                character, orig, dest, rulebook, branch, turn
            ]
            return
        self.query.handled_character_portal_rule(
            character, orig, dest, rulebook, rule, branch, turn, tick
        )

    def _handled_node(self, character, node, rulebook, rule, branch, turn, tick):
        try:
            self._node_rules_handled_cache.store(
                character, node, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._node_rules_handled_cache.handled[
                character, node, rulebook, branch, turn
            ]
            return
        self.query.handled_node_rule(
            character, node, rulebook, rule, branch, turn, tick
        )

    def _handled_portal(self, character, orig, dest, rulebook, rule, branch, turn, tick):
        try:
            self._portal_rules_handled_cache.store(
                character, orig, dest, rulebook, rule, branch, turn, tick
            )
        except ValueError:
            assert rule in self._portal_rules_handled_cache.handled[
                character, orig, dest, rulebook, branch, turn
            ]
            return
        self.query.handled_portal_rule(
            character, orig, dest, rulebook, rule, branch, turn, tick
        )

    def _follow_rule(self, rule, handled_fun, branch, turn, *args):
        satisfied = True
        for prereq in rule.prereqs:
            res = prereq(*args)
            if not res:
                satisfied = False
                break
        if not satisfied:
            return handled_fun()
        for trigger in rule.triggers:
            res = trigger(*args)
            if res:
                break
        else:
            return handled_fun()
        actres = []
        for action in rule.actions:
            res = action(*args)
            if res:
                actres.append(res)
        handled_fun()
        return actres

    def _follow_rules(self):
        # TODO: rulebook priorities (not individual rule priorities, just follow the order of the rulebook)
        # TODO: apply changes to a facade first, and commit it when you're done. Then report changes to the facade
        branch, turn, tick = self.btt()
        charmap = self.character
        rulemap = self.rule

        # TODO: if there's a paradox while following some rule, start a new branch, copying handled rules
        for (
            charactername, rulebook, rulename
        ) in list(
            self._character_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            if charactername not in charmap:
                continue
            yield self._follow_rule(
                rulemap[rulename],
                partial(self._handled_char, charactername, rulebook, rulename, branch, turn, tick),
                branch, turn,
                charmap[charactername]
            )
        for (
            charn, rulebook, graphn, avn, rulen
        ) in list(
            self._avatar_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            if charn not in charmap:
                continue
            char = charmap[charn]
            if graphn not in char.avatar or avn not in char.avatar[graphn]:
                continue
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_av, charn, graphn, avn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[graphn].node[avn]
            )
        for (
            charn, rulebook, rulen, thingn
        ) in list(
            self._character_thing_rules_handled_cache.iter_unhandled_rules(branch, turn, tick)
        ):
            if charn not in charmap or thingn not in charmap[charn].thing:
                continue
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_thing, charn, thingn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].thing[thingn]
            )
        for (
            charn, rulebook, rulen, placen
        ) in list(
            self._character_place_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            if charn not in charmap or placen not in charmap[charn].place:
                continue
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_place, charn, placen, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].place[placen]
            )
        for (
            charn, rulebook, rulen, orign, destn
        ) in list(
            self._character_portal_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_port, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].portal[orign][destn]
            )
        for (
                charn, noden, rulebook, rulen
        ) in list(
            self._node_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            if charn not in charmap or noden not in charmap[charn]:
                continue
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_node, charn, noden, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].node[noden]
            )
        for (
                charn, orign, destn, rulebook, rulen
        ) in list(
            self._portal_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
            )
        ):
            if charn not in charmap:
                continue
            char = charmap[charn]
            if orign not in char.portal or destn not in char.portal[orign]:
                continue
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_port, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].portal[orign][destn]
            )

    def advance(self):
        """Follow the next rule if available, or advance to the next turn."""
        try:
            return next(self._rules_iter)
        except StopIteration:
            self._rules_iter = self._follow_rules()
            return final_rule

    def new_character(self, name, **kwargs):
        """Create and return a new :class:`Character`."""
        self.add_character(name, **kwargs)
        return self.character[name]

    def add_character(self, name, data=None, **kwargs):
        """Create a new character.

        You'll be able to access it as a :class:`Character` object by
        looking up ``name`` in my ``character`` property.

        ``data``, if provided, should be a networkx-compatible graph
        object. Your new character will be a copy of it.

        Any keyword arguments will be set as stats of the new character.

        """
        self._init_graph(name, 'DiGraph')
        self._graph_objs[name] = Character(self, name, data, **kwargs)

    def del_character(self, name):
        """Remove the Character from the database entirely.

        This also deletes all its history. You'd better be sure.

        """
        self.query.del_character(name)
        self.del_graph(name)
        del self.character[name]

    def _is_thing(self, character, node):
        return self._things_cache.contains_entity(character, node, *self.btt())

    def _set_thing_loc_and_next(
            self, character, node, loc, nextloc=None
    ):
        branch, turn, tick = self.nbtt()
        self._things_cache.store(character, node, branch, turn, tick, (loc, nextloc))
        self.query.thing_loc_and_next_set(
            character,
            node,
            branch,
            turn,
            tick,
            loc,
            nextloc
        )

    def _node_exists(self, character, node):
        return self._nodes_cache.contains_entity(character, node, *self.btt())

    def _exist_node(self, character, node):
        branch, turn, tick = self.nbtt()
        self.query.exist_node(
            character,
            node,
            branch,
            turn,
            tick,
            True
        )
        self._nodes_cache.store(character, node, branch, turn, tick, True)
        self._nodes_rulebooks_cache.store(character, node, branch, turn, tick, (character, node))

    def _exist_edge(
            self, character, orig, dest, exist=True
    ):
        branch, turn, tick = self.nbtt()
        planning = self.planning
        self.query.exist_edge(
            character,
            orig,
            dest,
            0,
            branch,
            turn,
            tick,
            exist
        )
        self._edges_cache.store(
            character, orig, dest, 0, branch, turn, tick, exist, planning=planning
        )
        assert self._edges_cache.contains_entity(
            character, orig, dest, 0, branch, turn, tick
        )

    def alias(self, v, stat='dummy'):
        r = DummyEntity(self)
        r[stat] = v
        return EntityStatAccessor(r, stat, engine=self)

    def entityfy(self, v, stat='dummy'):
        if (
                isinstance(v, Thing) or
                isinstance(v, Place) or
                isinstance(v, Portal) or
                isinstance(v, Query) or
                isinstance(v, EntityStatAccessor)
        ):
            return v
        return self.alias(v, stat)

    def ticks_when(self, query):
        return query()
