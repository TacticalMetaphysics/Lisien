# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
"""The "engine" of LiSE is an object relational mapper with special
stores for game data and entities, as well as properties for manipulating the
flow of time.

"""
from functools import partial
from collections import defaultdict
from operator import attrgetter
from types import FunctionType, MethodType
from abc import ABC, abstractmethod

import msgpack
from blinker import Signal
from .allegedb import ORM as gORM
from .allegedb import HistoryError
from .allegedb.window import SettingsTurnDict, WindowDict
from .reify import reify
from .util import sort_set

from . import exc


class getnoplan:
    """Attribute getter that raises an exception if in planning mode"""
    __slots__ = ('_getter',)

    def __init__(self, attr, *attrs):
        self._getter = attrgetter(attr, *attrs)

    def __get__(self, instance, owner):
        if instance._planning:
            raise exc.PlanError("Don't use randomization in a plan")
        return self._getter(instance)


class InnerStopIteration(StopIteration):
    pass


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
        start_branch, start_turn, start_tick = engine._btt()
        latest_turn = engine._turns_completed[start_branch]
        if start_turn < latest_turn:
            engine.turn += 1
            self.send(
                engine,
                branch=engine.branch,
                turn=engine.turn,
                tick=engine.tick
            )
            return [], engine.get_delta(
                branch=start_branch,
                turn_from=start_turn,
                turn_to=engine.turn,
                tick_from=start_tick,
                tick_to=engine.tick
            )
        elif start_turn > latest_turn + 1:
            raise exc.RulesEngineError(
                "Can't run the rules engine on any turn but the latest")
        if start_turn == latest_turn:
            # As a side effect, the following assignment sets the tick
            # to the latest in the new turn, which will be 0 if that
            # turn has not yet been simulated.
            engine.turn += 1
        with engine.advancing():
            for res in iter(engine.advance, final_rule):
                if res:
                    engine.universal['last_result'] = res
                    engine.universal['last_result_idx'] = 0
                    branch, turn, tick = engine._btt()
                    self.send(
                        engine,
                        branch=branch,
                        turn=turn,
                        tick=tick
                    )
                    return res, engine.get_delta(
                        branch=start_branch,
                        turn_from=start_turn,
                        turn_to=turn,
                        tick_from=start_tick,
                        tick_to=tick
                    )
        engine._turns_completed[start_branch] = engine.turn
        engine.query.complete_turn(start_branch, engine.turn)
        self.send(
            self.engine,
            branch=engine.branch,
            turn=engine.turn,
            tick=engine.tick
        )
        return [], engine.get_delta(
            branch=engine.branch,
            turn_from=start_turn,
            turn_to=engine.turn,
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


MSGPACK_TUPLE = 0x00
MSGPACK_FROZENSET = 0x01
MSGPACK_SET = 0x02
MSGPACK_EXCEPTION = 0x03
MSGPACK_CHARACTER = 0x7f
MSGPACK_PLACE = 0x7e
MSGPACK_THING = 0x7d
MSGPACK_PORTAL = 0x7c
MSGPACK_FINAL_RULE = 0x7b
MSGPACK_FUNCTION = 0x7a
MSGPACK_METHOD = 0x79
MSGPACK_TRIGGER = 0x78
MSGPACK_PREREQ = 0x77
MSGPACK_ACTION = 0x76


class AbstractEngine(object):
    """Parent class to the real Engine as well as EngineProxy.

    Implements serialization and the __getattr__ for stored methods.

    By default, the deserializers will refuse to create LiSE entities.
    If you want them to, use my ``loading`` property to open a ``with``
    block, in which deserialized entities will be created as needed.

    """

    def __getattr__(self, item):
        meth = super().__getattribute__('method').__getattr__(item)
        return MethodType(meth, self)

    @reify
    def pack(self):
        handlers = {
            self.char_cls: lambda char: msgpack.ExtType(
                MSGPACK_CHARACTER, packer(char.name)
                ),
            self.place_cls: lambda place: msgpack.ExtType(
                MSGPACK_PLACE, packer(
                    (place.character.name, place.name)
                )),
            self.thing_cls: lambda thing: msgpack.ExtType(
                MSGPACK_THING, packer(
                    (thing.character.name, thing.name)
                )),
            self.portal_cls: lambda port: msgpack.ExtType(
                MSGPACK_PORTAL, packer(
                    (port.character.name, port.origin.name, port.destination.name)
                )),
            tuple: lambda tup: msgpack.ExtType(
                MSGPACK_TUPLE, packer(list(tup))),
            frozenset: lambda frozs: msgpack.ExtType(
                MSGPACK_FROZENSET, packer(list(frozs))),
            set: lambda s: msgpack.ExtType(MSGPACK_SET, packer(
                list(s))),
            FinalRule: lambda obj: msgpack.ExtType(
                MSGPACK_FINAL_RULE, b""
            ),
            FunctionType: lambda func: msgpack.ExtType({
                'method': MSGPACK_METHOD,
                'function': MSGPACK_FUNCTION,
                'trigger': MSGPACK_TRIGGER,
                'prereq': MSGPACK_PREREQ,
                'action': MSGPACK_ACTION
            }[func.__module__], packer(func.__name__)),
            MethodType: lambda meth: msgpack.ExtType(
                MSGPACK_METHOD, packer(meth.__name__)),
            Exception: lambda exc: msgpack.ExtType(
                MSGPACK_EXCEPTION, packer(
                    [exc.__class__.__name__] + list(exc.args)
                ))
        }

        def pack_handler(obj):
            if isinstance(obj, Exception):
                typ = Exception
            else:
                typ = type(obj)
            if typ in handlers:
                return handlers[typ](obj)
            raise TypeError("Can't pack {}".format(typ))
        packer = partial(
            msgpack.packb,
            default=pack_handler, strict_types=True,
            use_bin_type=True
        )
        return packer

    @reify
    def unpack(self):
        charmap = self.character
        char_cls = self.char_cls
        place_cls = self.place_cls
        thing_cls = self.thing_cls
        portal_cls = self.portal_cls
        trigger = self.trigger
        prereq = self.prereq
        action = self.action
        function = self.function
        method = self.method
        excs = {
            # builtin exceptions
            'AssertionError': AssertionError,
            'AttributeError': AttributeError,
            'EOFError': EOFError,
            'FloatingPointError': FloatingPointError,
            'GeneratorExit': GeneratorExit,
            'ImportError': ImportError,
            'IndexError': IndexError,
            'KeyError': KeyError,
            'KeyboardInterrupt': KeyboardInterrupt,
            'MemoryError': MemoryError,
            'NameError': NameError,
            'NotImplementedError': NotImplementedError,
            'OSError': OSError,
            'OverflowError': OverflowError,
            'RecursionError': RecursionError,
            'ReferenceError': ReferenceError,
            'RuntimeError': RuntimeError,
            'StopIteration': StopIteration,
            'IndentationError': IndentationError,
            'TabError': TabError,
            'SystemError': SystemError,
            'SystemExit': SystemExit,
            'TypeError': TypeError,
            'UnboundLocalError': UnboundLocalError,
            'UnicodeError': UnicodeError,
            'UnicodeEncodeError': UnicodeEncodeError,
            'UnicodeDecodeError': UnicodeDecodeError,
            'UnicodeTranslateError': UnicodeTranslateError,
            'ValueError': ValueError,
            'ZeroDivisionError': ZeroDivisionError,
            # LiSE exceptions
            'NonUniqueError': exc.NonUniqueError,
            'AmbiguousAvatarError': exc.AmbiguousAvatarError,
            'AmbiguousUserError': exc.AmbiguousUserError,
            'RulesEngineError': exc.RulesEngineError,
            'RuleError': exc.RuleError,
            'RedundantRuleError': exc.RedundantRuleError,
            'UserFunctionError': exc.UserFunctionError,
            'WorldIntegrityError': exc.WorldIntegrityError,
            'CacheError': exc.CacheError,
            'TravelException': exc.TravelException
        }

        def unpack_exception(ext):
            data = unpacker(ext)
            if data[0] not in excs:
                return Exception(*data)
            return excs[data[0]](*data[1:])


        def unpack_char(ext):
            charn = unpacker(ext)
            try:
                return charmap[charn]
            except KeyError:
                return char_cls(self, charn)

        def unpack_place(ext):
            charn, placen = unpacker(ext)
            try:
                char = charmap[charn]
            except KeyError:
                return place_cls(char_cls(self, charn), placen)
            try:
                return char.place[placen]
            except KeyError:
                return place_cls(char, placen)

        def unpack_thing(ext):
            charn, thingn = unpacker(ext)
            try:
                char = charmap[charn]
            except KeyError:
                return thing_cls(char_cls(self, charn), thingn)
            try:
                return char.thing[thingn]
            except KeyError:
                return thing_cls(char, thingn)

        def unpack_portal(ext):
            charn, orign, destn = unpacker(ext)
            try:
                char = charmap[charn]
            except KeyError:
                char = char_cls(self, charn)
            try:
                return char.portal[orign][destn]
            except KeyError:
                return portal_cls(char, orign, destn)

        handlers = {
            MSGPACK_CHARACTER: unpack_char,
            MSGPACK_PLACE: unpack_place,
            MSGPACK_THING: unpack_thing,
            MSGPACK_PORTAL: unpack_portal,
            MSGPACK_FINAL_RULE: lambda obj: final_rule,
            MSGPACK_TUPLE: lambda ext: tuple(unpacker(ext)),
            MSGPACK_FROZENSET: lambda ext: frozenset(unpacker(ext)),
            MSGPACK_SET: lambda ext: set(unpacker(ext)),
            MSGPACK_TRIGGER: lambda ext: getattr(trigger, unpacker(ext)),
            MSGPACK_PREREQ: lambda ext: getattr(prereq, unpacker(ext)),
            MSGPACK_ACTION: lambda ext: getattr(action, unpacker(ext)),
            MSGPACK_FUNCTION: lambda ext: getattr(function, unpacker(ext)),
            MSGPACK_METHOD: lambda ext: getattr(method, unpacker(ext)),
            MSGPACK_EXCEPTION: unpack_exception
        }

        def unpack_handler(code, data):
            if code in handlers:
                return handlers[code](data)
            return msgpack.ExtType(code, data)
        unpacker = partial(
            msgpack.unpackb,
            ext_hook=unpack_handler,
            raw=False, strict_map_key=False
        )
        return unpacker

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

    def dice_check(self, n, d, target, comparator='<='):
        """Roll ``n`` dice with ``d`` sides, sum them, and compare

        If ``comparator`` is provided, use it instead of the default <=.
        You may use a string like '<' or '>='.

        """
        from operator import gt, lt, ge, le, eq, ne

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
        """Return True or False with a given percentile probability

        Values not between 0 and 100 are treated as though they
        were 0 or 100, whichever is nearer.

        """
        if pct <= 0:
            return False
        if pct >= 100:
            return True
        return pct / 100 < self.random()

    betavariate = getnoplan('_rando.betavariate')
    choice = getnoplan('_rando.choice')
    expovariate = getnoplan('_rando.expovariate')
    gammavariate = getnoplan('_rando.gammavariate')
    gauss = getnoplan('_rando.gauss')
    getrandbits = getnoplan('_rando.getrandbits')
    lognormvariate = getnoplan('_rando.lognormvariate')
    normalvariate = getnoplan('_rando.normalvariate')
    paretovariate = getnoplan('_rando.paretovariate')
    randint = getnoplan('_rando.randint')
    random = getnoplan('_rando.random')
    randrange = getnoplan('_rando.randrange')
    sample = getnoplan('_rando.sample')
    shuffle = getnoplan('_rando.shuffle')
    triangular = getnoplan('_rando.triangular')
    uniform = getnoplan('_rando.uniform')
    vonmisesvariate = getnoplan('_rando.vonmisesvariate')
    weibullvariate = getnoplan('_rando.weibullvariate')


class AbstractSchema(ABC):
    def __init__(self, engine):
        self.engine = engine

    @abstractmethod
    def entity_permitted(self, entity):
        raise NotImplementedError

    @abstractmethod
    def validate(self, turn, entity, key, value):
        raise NotImplementedError

    def get_not_permitted_entity_message(self, entity):
        return


class NullSchema(AbstractSchema):
    def entity_permitted(self, entity):
        return True

    def validate(self, turn, entity, key, value):
        return True


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
    - ``function``: A mapping of generic functions.
    - ``string``: A mapping of strings, probably shown to the player
      at some point.
    - ``eternal``: Mapping of arbitrary serializable objects. It isn't
      sensitive to sim-time. A good place to keep game settings.
    - ``universal``: Another mapping of arbitrary serializable
      objects, but this one *is* sensitive to sim-time. Each turn, the
      state of the randomizer is saved here under the key
      ``'rando_state'``.

    """
    from .character import Character
    from .thing import Thing
    from .place import Place
    from .portal import Portal
    from .query import QueryEngine
    char_cls = Character
    thing_cls = Thing
    place_cls = node_cls = Place
    portal_cls = edge_cls = Portal
    query_engine_cls = QueryEngine
    illegal_graph_names = [
        'global', 'eternal', 'universal', 'rulebooks', 'rules']
    illegal_node_names = ['nodes', 'node_val', 'edges', 'edge_val', 'things']

    def _make_node(self, graph, node):
        if self._is_thing(graph.name, node):
            return self.thing_cls(graph, node)
        else:
            return self.place_cls(graph, node)

    def _make_edge(self, graph, orig, dest, idx=0):
        return self.portal_cls(graph, orig, dest)

    def _load_graphs(self):
        for charn in self.query.characters():
            self._graph_objs[charn] = self.char_cls(self, charn, init_rulebooks=False)

    def get_delta(self, branch, turn_from, tick_from, turn_to, tick_to):
        """Get a dictionary describing changes to the world.

        Most keys will be character names, and their values will be
        dictionaries of the character's stats' new values, with ``None``
        for deleted keys. Characters' dictionaries have special keys
        'nodes' and 'edges' which contain booleans indicating whether
        the node or edge exists at the moment, and 'node_val' and
        'edge_val' for the stats of those entities. For edges (also
        called portals) these dictionaries are two layers deep, keyed
        first by the origin, then by the destination.

        Characters also have special keys for the various rulebooks
        they have:

        * 'character_rulebook'
        * 'avatar_rulebook'
        * 'character_thing_rulebook'
        * 'character_place_rulebook'
        * 'character_portal_rulebook'

        And each node and edge may have a 'rulebook' stat of its own.
        If a node is a thing, it gets a 'location'; when the 'location'
        is deleted, that means it's back to being a place.

        Keys at the top level that are not character names:

        * 'rulebooks', a dictionary keyed by the name of each changed
        rulebook, the value being a list of rule names
        * 'rules', a dictionary keyed by the name of each changed rule,
        containing any of the lists 'triggers', 'prereqs', and 'actions'

        """
        from LiSE.allegedb.window import update_window, update_backward_window
        if turn_from == turn_to:
            return self.get_turn_delta(
                branch, turn_to, tick_to,start_tick=tick_from)
        delta = super().get_delta(
            branch, turn_from, tick_from, turn_to, tick_to)
        if turn_from < turn_to:
            updater = partial(
                update_window, turn_from, tick_from, turn_to, tick_to)
            univbranches = self._universal_cache.settings
            avbranches = self._avatarness_cache.settings
            thbranches = self._things_cache.settings
            rbbranches = self._rulebooks_cache.settings
            trigbranches = self._triggers_cache.settings
            preqbranches = self._prereqs_cache.settings
            actbranches = self._actions_cache.settings
            charrbbranches = self._characters_rulebooks_cache.settings
            avrbbranches = self._avatars_rulebooks_cache.settings
            charthrbbranches = self._characters_things_rulebooks_cache.settings
            charplrbbranches = self._characters_places_rulebooks_cache.settings
            charporbbranches = \
                self._characters_portals_rulebooks_cache.settings
            noderbbranches = self._nodes_rulebooks_cache.settings
            edgerbbranches = self._portals_rulebooks_cache.settings
        else:
            updater = partial(
                update_backward_window, turn_from, tick_from, turn_to, tick_to)
            univbranches = self._universal_cache.presettings
            avbranches = self._avatarness_cache.presettings
            thbranches = self._things_cache.presettings
            rbbranches = self._rulebooks_cache.presettings
            trigbranches = self._triggers_cache.presettings
            preqbranches = self._prereqs_cache.presettings
            actbranches = self._actions_cache.presettings
            charrbbranches = self._characters_rulebooks_cache.presettings
            avrbbranches = self._avatars_rulebooks_cache.presettings
            charthrbbranches = \
                self._characters_things_rulebooks_cache.presettings
            charplrbbranches = \
                self._characters_places_rulebooks_cache.presettings
            charporbbranches = \
                self._characters_portals_rulebooks_cache.presettings
            noderbbranches = self._nodes_rulebooks_cache.presettings
            edgerbbranches = self._portals_rulebooks_cache.presettings

        def upduniv(_, key, val):
            delta.setdefault('universal', {})[key] = val
        if branch in univbranches:
            updater(upduniv, univbranches[branch])

        def updav(char, graph, node, av):
            delta.setdefault(char, {}).setdefault(
                'avatars', {}).setdefault(graph, {})[node] = bool(av)
        if branch in avbranches:
            updater(updav, avbranches[branch])

        def updthing(char, thing, loc):
            if (
                char in delta and 'nodes' in delta[char]
                and thing in delta[char]['nodes'] and not
                delta[char]['nodes'][thing]
            ):
                return
            thingd = delta.setdefault(char, {}).setdefault(
                'node_val', {}).setdefault(thing, {})
            thingd['location'] = loc
        if branch in thbranches:
            updater(updthing, thbranches[branch])
        # TODO handle arrival_time and next_arrival_time stats of things

        def updrb(whatev, rulebook, rules):
            delta.setdefault('rulebooks', {})[rulebook] = rules

        if branch in rbbranches:
            updater(updrb, rbbranches[branch])

        def updru(key, _, rule, funs):
            delta.setdefault('rules', {}).setdefault(rule, {})[key] = funs

        if branch in trigbranches:
            updater(partial(updru, 'triggers'), trigbranches[branch])

        if branch in preqbranches:
            updater(partial(updru, 'prereqs'), preqbranches[branch])

        if branch in actbranches:
            updater(partial(updru, 'actions'), actbranches[branch])

        def updcrb(key, _, character, rulebook):
            delta.setdefault(character, {})[key] = rulebook

        if branch in charrbbranches:
            updater(partial(
                updcrb, 'character_rulebook'), charrbbranches[branch])

        if branch in avrbbranches:
            updater(partial(
                updcrb, 'avatar_rulebook'), avrbbranches[branch])

        if branch in charthrbbranches:
            updater(partial(
                updcrb, 'character_thing_rulebook'), charthrbbranches[branch])

        if branch in charplrbbranches:
            updater(partial(
                updcrb, 'character_place_rulebook'), charplrbbranches[branch])

        if branch in charporbbranches:
            updater(partial(
                updcrb, 'character_portal_rulebook'), charporbbranches[branch])

        def updnoderb(character, node, rulebook):
            if (
                character in delta and 'nodes' in delta[character]
                and node in delta[character]['nodes']
                and not delta[character]['nodes'][node]
            ):
                return
            delta.setdefault(character, {}).setdefault(
                'node_val', {}).setdefault(node, {})['rulebook'] = rulebook

        if branch in noderbbranches:
            updater(updnoderb, noderbbranches[branch])

        def updedgerb(character, orig, dest, rulebook):
            if (
                character in delta and 'edges' in delta[character]
                and orig in delta[character]['edges']
                and dest in delta[character]['edges'][orig]
                and not delta[character]['edges'][orig][dest]
            ):
                return
            delta.setdefault(character, {}).setdefault(
                'edge_val', {}).setdefault(
                orig, {}).setdefault(dest, {})['rulebook'] = rulebook

        if branch in edgerbbranches:
            updater(updedgerb, edgerbbranches[branch])

        return delta

    def get_turn_delta(self, branch=None, turn=None, tick=None, start_tick=0):
        """Get a dictionary of changes to the world within a given turn

        Defaults to the present turn, and stops at the present tick
        unless specified.

        See the documentation for ``get_delta`` for a detailed
        description of the delta format.

        :arg branch: branch of history, defaulting to the present branch
        :arg turn: turn within the branch, defaulting to the present
        turn
        :arg tick: tick at which to stop the delta, defaulting to the
        present tick
        :arg start_tick: tick at which to start the delta, default 0

        """
        branch = branch or self.branch
        turn = turn or self.turn
        tick = tick or self.tick
        delta = super().get_turn_delta(branch, turn, start_tick, tick)
        if branch in self._avatarness_cache.settings \
                and turn in self._avatarness_cache.settings[branch]:
            for chara, graph, node, is_av in self._avatarness_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(chara, {}).setdefault(
                    'avatars', {}).setdefault(graph, {})[node] = is_av
        if branch in self._things_cache.settings \
                and turn in self._things_cache.settings[branch]:
            for chara, thing, location in self._things_cache.settings[
                                              branch][turn][start_tick:tick]:
                thingd = delta.setdefault(chara, {}).setdefault(
                    'node_val', {}).setdefault(thing, {})
                thingd['location'] = location
        delta['rulebooks'] = rbdif = {}
        if branch in self._rulebooks_cache.settings \
                and turn in self._rulebooks_cache.settings[branch]:
            for _, rulebook, rules in self._rulebooks_cache.settings[
                                          branch][turn][start_tick:tick]:
                rbdif[rulebook] = rules
        delta['rules'] = rdif = {}
        if branch in self._triggers_cache.settings \
                and turn in self._triggers_cache.settings[branch]:
            for _, rule, funs in self._triggers_cache.settings[
                                     branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['triggers'] = funs
        if branch in self._prereqs_cache.settings \
                and turn in self._prereqs_cache.settings[branch]:
            for _, rule, funs in self._prereqs_cache.settings[
                                     branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['prereqs'] = funs
        if branch in self._actions_cache.settings \
                and turn in self._triggers_cache.settings[branch]:
            for _, rule, funs in self._triggers_cache.settings[
                                     branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['actions'] = funs

        if branch in self._characters_rulebooks_cache.settings \
                and turn in self._characters_rulebooks_cache.settings[branch]:
            for _, character, rulebook in \
                    self._characters_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(
                    character, {})['character_rulebook'] = rulebook
        if branch in self._avatars_rulebooks_cache.settings \
                and turn in self._avatars_rulebooks_cache.settings[branch]:
            for _, character, rulebook in \
                    self._avatars_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['avatar_rulebook'] = rulebook
        if branch in self._characters_things_rulebooks_cache.settings \
                and turn in self._characters_things_rulebooks_cache.settings[
                    branch]:
            for _, character, rulebook in \
                    self._characters_things_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(
                    character, {})['character_thing_rulebook'] = rulebook
        if branch in self._characters_places_rulebooks_cache.settings \
                and turn in self._characters_places_rulebooks_cache.settings[
                    branch]:
            for _, character, rulebook in \
                    self._characters_places_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(
                    character, {})['character_place_rulebook'] = rulebook
        if branch in self._characters_portals_rulebooks_cache.settings \
                and turn in self._characters_portals_rulebooks_cache.settings[
                    branch]:
            for _, character, rulebook in \
                    self._characters_portals_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_portal_rulebook'] = rulebook

        if branch in self._nodes_rulebooks_cache.settings \
                and turn in self._nodes_rulebooks_cache.settings[branch]:
            for character, node, rulebook in \
                    self._nodes_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(character, {}).setdefault(
                    'node_val', {}).setdefault(node, {})['rulebook'] = rulebook
        if branch in self._portals_rulebooks_cache.settings \
                and turn in self._portals_rulebooks_cache.settings[branch]:
            for character, orig, dest, rulebook in \
                    self._portals_rulebooks_cache.settings[
                        branch][turn][start_tick:tick]:
                delta.setdefault(character, {}).setdefault('edge_val', {}) \
                    .setdefault(orig, {}).setdefault(dest, {})[
                        'rulebook'] = rulebook
        return delta

    def _del_rulebook(self, rulebook):  # TODO: fix this for new cache style
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
            is_avatar
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
        from .xcollections import (
            FunctionStore,
            CharacterMapping,
            UniversalMapping
        )
        from .cache import (
            NodeContentsCache,
            InitializedCache,
            EntitylessCache,
            InitializedEntitylessCache,
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
        from .rule import AllRuleBooks, AllRules

        super()._init_caches()
        self._things_cache = ThingsCache(self)
        self._node_contents_cache = NodeContentsCache(self)
        self.character = self.graph = CharacterMapping(self)
        self._universal_cache = EntitylessCache(self)
        self._universal_cache.name = 'universal_cache'
        self._rulebooks_cache = InitializedEntitylessCache(self)
        self._rulebooks_cache.name = 'rulebooks_cache'
        self._characters_rulebooks_cache = InitializedEntitylessCache(self)
        self._characters_rulebooks_cache.name = 'characters_rulebooks_cache'
        self._avatars_rulebooks_cache = InitializedEntitylessCache(self)
        self._avatars_rulebooks_cache.name = 'avatars_rulebooks_cache'
        self._characters_things_rulebooks_cache = \
            InitializedEntitylessCache(self)
        self._characters_things_rulebooks_cache.name = \
            'characters_things_rulebooks_cache'
        self._characters_places_rulebooks_cache = \
            InitializedEntitylessCache(self)
        self._characters_places_rulebooks_cache.name = \
            'characters_places_rulebooks_cache'
        self._characters_portals_rulebooks_cache = \
            InitializedEntitylessCache(self)
        self._characters_portals_rulebooks_cache.name = \
            'characters_portals_rulebooks_cache'
        self._nodes_rulebooks_cache = InitializedCache(self)
        self._nodes_rulebooks_cache.name = 'nodes_rulebooks_cache'
        self._portals_rulebooks_cache = InitializedCache(self)
        self._portals_rulebooks_cache.name = 'portals_rulebooks_cache'
        self._triggers_cache = InitializedEntitylessCache(self)
        self._triggers_cache.name = 'triggers_cache'
        self._prereqs_cache = InitializedEntitylessCache(self)
        self._prereqs_cache.name = 'prereqs_cache'
        self._actions_cache = InitializedEntitylessCache(self)
        self._actions_cache.name = 'actions_cache'
        self._node_rules_handled_cache = NodeRulesHandledCache(self)
        self._node_rules_handled_cache.name = 'node_rules_handled_cache'
        self._portal_rules_handled_cache = PortalRulesHandledCache(self)
        self._portal_rules_handled_cache.name = 'portal_rules_handled_cache'
        self._character_rules_handled_cache = CharacterRulesHandledCache(self)
        self._character_rules_handled_cache.name = \
            'character_rules_handled_cache'
        self._avatar_rules_handled_cache = AvatarRulesHandledCache(self)
        self._avatar_rules_handled_cache.name = 'avatar_rules_handled_cache'
        self._character_thing_rules_handled_cache \
            = CharacterThingRulesHandledCache(self)
        self._character_thing_rules_handled_cache.name = \
            'character_thing_rules_handled_cache'
        self._character_place_rules_handled_cache \
            = CharacterPlaceRulesHandledCache(self)
        self._character_place_rules_handled_cache.name = \
            'character_place_rules_handled_cache'
        self._character_portal_rules_handled_cache \
            = CharacterPortalRulesHandledCache(self)
        self._character_portal_rules_handled_cache.name = \
            'character_portal_rules_handled_cache'
        self._avatarness_cache = AvatarnessCache(self)
        self._avatarness_cache.name = 'avatarness_cache'
        self._turns_completed = defaultdict(lambda: max((0, self.turn - 1)))
        """The last turn when the rules engine ran in each branch"""
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

    def _load_graphs(self):
        for charn in self.query.characters():
            self._graph_objs[charn] = self.char_cls(
                self, charn, init_rulebooks=False)

    def __init__(
            self,
            prefix='.',
            *,
            string=None,
            trigger=None,
            prereq=None,
            action=None,
            function=None,
            method=None,
            connect_string=None,
            connect_args={},
            schema_cls=NullSchema,
            alchemy=False,
            commit_modulus=None,
            random_seed=None,
            logfun=None,
            validate=False,
            clear=False
    ):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        :arg prefix: directory containing the simulation and its code;
        defaults to the working directory
        :arg string: module storing strings to be used in the game
        :arg function: module containing utility functions
        :arg method: module containing functions taking this engine as
        first arg
        :arg trigger: module containing trigger functions, taking a LiSE
        entity and returning a boolean for whether to run a rule
        :arg prereq: module containing prereq functions, taking a LiSE entity and
        returning a boolean for whether to permit a rule to run
        :arg action: module containing action functions, taking a LiSE entity and
        mutating it (and possibly the rest of the world)
        :arg connect_string: a rfc1738 URI for a database to connect to;
        if absent, we'll use a SQLite database in the prefix directory.
        With ``alchemy=False`` we can only open SQLite databases,
        in which case ``connect_string`` is just a path to the database--
        unless it's ``":memory:"``, which is an in-memory database that
        won't be saved
        :arg connect_args: dictionary of keyword arguments for the
        database connection
        :arg schema: a Schema class that determines which changes to allow to
        the world; used when a player should not be able to change just anything.
        Defaults to `NullSchema`
        :arg alchemy: whether to use SQLAlchemy to connect to the
        database. If False, LiSE can only use SQLite
        :arg commit_modulus: LiSE will commit changes to disk every
        ``commit_modulus`` turns
        :arg random_seed: a number to initialize the randomizer
        :arg logfun: an optional function taking arguments
        ``level, message`` and
        :arg validate: whether to perform integrity tests while
        loading the game
        :arg clear: whether to delete *any and all* existing data
        and code in ``prefix``. Use with caution!

        """
        import os
        from .xcollections import StringStore
        self.exist_node_time = 0
        self.exist_edge_time = 0
        if string:
            self.string = string
        else:
            self._string_file = os.path.join(prefix, 'strings.json')
            if clear and os.path.exists(self._string_file):
                os.remove(self._string_file)
        if function:
            self.function = function
        else:
            self._function_file = os.path.join(prefix, 'function.py')
            if clear and os.path.exists(self._function_file):
                os.remove(self._function_file)
        if method:
            self.method = method
        else:
            self._method_file = os.path.join(prefix, 'method.py')
            if clear and os.path.exists(self._method_file):
                os.remove(self._method_file)
        if trigger:
            self.trigger = trigger
        else:
            self._trigger_file = os.path.join(prefix, 'trigger.py')
            if clear and os.path.exists(self._trigger_file):
                os.remove(self._trigger_file)
        if prereq:
            self.prereq = prereq
        else:
            self._prereq_file = os.path.join(prefix, 'prereq.py')
            if clear and os.path.exists(self._prereq_file):
                os.remove(self._prereq_file)
        if action:
            self.action = action
        else:
            self._action_file = os.path.join(prefix, 'action.py')
            if clear and os.path.exists(self._action_file):
                os.remove(self._action_file)
        self.schema = schema_cls(self)
        if connect_string and not alchemy:
            connect_string = connect_string.split('sqlite:///')[-1]
        super().__init__(
            connect_string or os.path.join(prefix, 'world.db'),
            connect_args=connect_args,
            alchemy=alchemy,
            validate=validate
        )
        self._things_cache.setdb = self.query.set_thing_loc
        self._universal_cache.setdb = self.query.universal_set
        self._rulebooks_cache.setdb = self.query.rulebook_set
        self.eternal = self.query.globl
        if hasattr(self, '_string_file'):
            self.string = StringStore(
                self.query,
                self._string_file,
                self.eternal.setdefault('language', 'eng')
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
        from random import Random
        self._rando = Random()
        if 'rando_state' in self.universal:
            self._rando.setstate(self.universal['rando_state'])
        else:
            self._rando.seed(self.random_seed)
            self.universal['rando_state'] = self._rando.getstate()
        if hasattr(self.method, 'init'):
            self.method.init(self)

    def _init_load(self, validate=False):
        from .rule import Rule
        q = self.query
        self._things_cache.load(q.things_dump())
        super()._init_load(validate=validate)
        things_kf = self._things_cache.keyframe
        nv_kf = self._node_val_cache.keyframe
        for node, branches in nv_kf.items():
            for branch, turns in branches.items():
                for turn, ticks in turns.items():
                    for tick, vals in ticks.items():
                        if 'location' in vals:
                            if node not in things_kf or branch not in \
                                    things_kf[node]:
                                things_kf[node][branch] = SettingsTurnDict({
                                    turn: WindowDict({tick: vals})
                                })
                            elif turn not in things_kf[node][branch]:
                                things_kf[node][branch][turn] = WindowDict({
                                    tick: vals
                                })
                            else:
                                things_kf[node][branch][turn][tick] = vals
        self._avatarness_cache.load(q.avatars_dump())
        self._universal_cache.load(q.universals_dump())
        self._rulebooks_cache.load(q.rulebooks_dump())
        self._characters_rulebooks_cache.load(
            q.character_rulebook_dump())
        self._avatars_rulebooks_cache.load(q.avatar_rulebook_dump())
        self._characters_things_rulebooks_cache.load(
            q.character_thing_rulebook_dump())
        self._characters_places_rulebooks_cache.load(
            q.character_place_rulebook_dump())
        self._characters_portals_rulebooks_cache.load(
            q.character_portal_rulebook_dump())
        self._nodes_rulebooks_cache.load(q.node_rulebook_dump())
        self._portals_rulebooks_cache.load(q.portal_rulebook_dump())
        self._triggers_cache.load(q.rule_triggers_dump())
        self._prereqs_cache.load(q.rule_prereqs_dump())
        self._actions_cache.load(q.rule_actions_dump())
        store_crh = self._character_rules_handled_cache.store
        for row in q.character_rules_handled_dump():
            store_crh(*row, loading=True)
        store_arh = self._avatar_rules_handled_cache.store
        for row in q.avatar_rules_handled_dump():
            store_arh(*row, loading=True)
        store_ctrh = self._character_thing_rules_handled_cache.store
        for row in q.character_thing_rules_handled_dump():
            store_ctrh(*row, loading=True)
        store_cprh = self._character_place_rules_handled_cache.store
        for row in q.character_place_rules_handled_dump():
            store_cprh(*row, loading=True)
        store_cporh = self._character_portal_rules_handled_cache.store
        for row in q.character_portal_rules_handled_dump():
            store_cporh(*row, loading=True)
        store_cnrh = self._node_rules_handled_cache.store
        for row in q.node_rules_handled_dump():
            store_cnrh(*row, loading=True)
        store_porh = self._portal_rules_handled_cache.store
        for row in q.portal_rules_handled_dump():
            store_porh(*row, loading=True)
        self._turns_completed.update(q.turns_completed_dump())
        self._rules_cache = {
            name: Rule(self, name, create=False) for name in q.rules_dump()}

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

    def commit(self):
        try:
            self.universal['rando_state'] = self._rando.getstate()
        except HistoryError:
            branch, turn, tick = self.branch, self.turn, self.tick
            self.turn = self._branches[branch][3]
            self.universal['rando_state'] = self._rando.getstate()
            self.turn = turn
            self.tick = tick
        super().commit()

    def close(self):
        """Commit changes and close the database."""
        import sys, os
        for store in self.stores:
            if hasattr(store, 'save'):
                store.save(reimport=False)
            path, filename = os.path.split(store._filename)
            modname = filename[:-3]
            if modname in sys.modules:
                del sys.modules[modname]
        super().close()

    def __enter__(self):
        """Return myself. For compatibility with ``with`` semantics."""
        return self

    def __exit__(self, *args):
        """Close on exit."""
        self.close()

    def _set_branch(self, v):
        oldrando = self.universal.get('rando_state')
        super()._set_branch(v)
        newrando = self.universal.get('rando_state')
        if newrando and newrando != oldrando:
            self._rando.setstate(newrando)
        self.time.send(self.time, branch=self._obranch, turn=self._oturn)

    def _set_turn(self, v):
        oldrando = self.universal.get('rando_state')
        oldturn = self._oturn
        super()._set_turn(v)
        newrando = self.universal.get('rando_state')
        if v > oldturn and newrando and newrando != oldrando:
            self._rando.setstate(newrando)
        self.time.send(self.time, branch=self._obranch, turn=self._oturn)

    def _set_tick(self, v):
        oldrando = self.universal.get('rando_state')
        oldtick = self._otick
        super()._set_tick(v)
        newrando = self.universal.get('rando_state')
        if v > oldtick and newrando and newrando != oldrando:
            self._rando.setstate(newrando)

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

    def _handled_av(
            self, character, graph, avatar, rulebook, rule,
            branch, turn, tick):
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

    def _handled_char_thing(
            self, character, thing, rulebook, rule, branch, turn, tick):
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

    def _handled_char_place(
            self, character, place, rulebook, rule, branch, turn, tick):
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

    def _handled_char_port(
            self, character, orig, dest, rulebook, rule, branch, turn, tick):
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

    def _handled_node(
            self, character, node, rulebook, rule, branch, turn, tick):
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

    def _handled_portal(
            self, character, orig, dest, rulebook, rule, branch, turn, tick):
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

    def _follow_rule(self, rule, handled_fun, *args):
        self.debug("following rule: " + repr(rule))

    def _follow_rules(self):
        # TODO: roll back changes done by rules that raise an exception
        # TODO: if there's a paradox while following some rule,
        #  start a new branch, copying handled rules
        from collections import defaultdict
        branch, turn, tick = self._btt()
        charmap = self.character
        rulemap = self.rule
        todo = defaultdict(list)

        def check_triggers(rule, handled_fun, entity):
            for trigger in rule.triggers:
                res = trigger(entity)
                if res:
                    return True
            else:
                handled_fun()
                return False

        def check_prereqs(rule, handled_fun, entity):
            for prereq in rule.prereqs:
                res = prereq(entity)
                if not res:
                    handled_fun()
                    return False
            return True

        def do_actions(rule, handled_fun, entity):
            actres = []
            for action in rule.actions:
                res = action(entity)
                if res:
                    actres.append(res)
            handled_fun()
            return actres

        # TODO: triggers that don't mutate anything should be
        #  evaluated in parallel
        #  Ideally this would be implemented with a pool of
        #  "engines" that serve Facades
        #  mirroring the state of the world on turn start,
        #  kept in sync with deltas.
        #  I think I could do it with regular concurrent.futures pools though.
        for (
            charactername, rulebook, rulename
        ) in self._character_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if charactername not in charmap:
                continue
            rule = rulemap[rulename]
            handled = partial(
                self._handled_char, charactername, rulebook, rulename,
                branch, turn, tick)
            entity = charmap[charactername]
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        avcache_retr = self._avatarness_cache._base_retrieve
        node_exists = self._node_exists
        get_node = self._get_node
        for (
            charn, graphn, avn, rulebook, rulen
        ) in self._avatar_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if not node_exists(graphn, avn) or avcache_retr(
                    (charn, graphn, avn, branch, turn, tick)
            ) in (KeyError, None):
                continue
            rule = rulemap[rulen]
            handled = partial(
                self._handled_av, charn, graphn, avn, rulebook, rulen,
                branch, turn, tick)
            entity = get_node(graphn, avn)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        is_thing = self._is_thing
        handled_char_thing = self._handled_char_thing
        for (
            charn, thingn, rulebook, rulen
        ) in self._character_thing_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick):
            if not node_exists(charn, thingn) or not is_thing(charn, thingn):
                continue
            rule = rulemap[rulen]
            handled = partial(
                handled_char_thing, charn, thingn, rulebook, rulen,
                branch, turn, tick)
            entity = get_node(charn, thingn)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        handled_char_place = self._handled_char_place
        for (
            charn, placen, rulebook, rulen
        ) in self._character_place_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if not node_exists(charn, placen) or is_thing(charn, placen):
                continue
            rule = rulemap[rulen]
            handled = partial(
                handled_char_place, charn, placen, rulebook, rulen,
                branch, turn, tick)
            entity = get_node(charn, placen)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        edge_exists = self._edge_exists
        get_edge = self._get_edge
        handled_char_port = self._handled_char_port
        for (
            charn, orign, destn, rulebook, rulen
        ) in self._character_portal_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if not edge_exists(charn, orign, destn):
                continue
            rule = rulemap[rulen]
            handled = partial(
                handled_char_port, charn, orign, destn, rulebook, rulen,
                branch, turn, tick)
            entity = get_edge(charn, orign, destn)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        handled_node = self._handled_node
        for (
                charn, noden, rulebook, rulen
        ) in self._node_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if not node_exists(charn, noden):
                continue
            rule = rulemap[rulen]
            handled = partial(
                handled_node, charn, noden, rulebook, rulen,
                branch, turn, tick)
            entity = get_node(charn, noden)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))
        handled_portal = self._handled_portal
        for (
                charn, orign, destn, rulebook, rulen
        ) in self._portal_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if not edge_exists(charn, orign, destn):
                continue
            rule = rulemap[rulen]
            handled = partial(
                handled_portal, charn, orign, destn, rulebook, rulen,
                branch, turn, tick)
            entity = get_edge(charn, orign, destn)
            if check_triggers(rule, handled, entity):
                todo[rulebook].append((rule, handled, entity))

        # TODO: rulebook priorities (not individual rule priorities, just follow the order of the rulebook)
        for rulebook in sort_set(todo.keys()):
            for rule, handled, entity in todo[rulebook]:
                if check_prereqs(rule, handled, entity):
                    try:
                        yield do_actions(rule, handled, entity)
                    except StopIteration:
                        raise InnerStopIteration

    def advance(self):
        """Follow the next rule if available.

        If we've run out of rules, reset the rules iterator.

        """
        assert self.turn > self._turns_completed[self.branch]
        try:
            return next(self._rules_iter)
        except InnerStopIteration:
            self._rules_iter = self._follow_rules()
            return StopIteration()
        except StopIteration:
            self._rules_iter = self._follow_rules()
            return final_rule
        # except Exception as ex:
        #     self._rules_iter = self._follow_rules()
        #     return ex

    def new_character(self, name, data=None, **kwargs):
        """Create and return a new :class:`Character`."""
        self.add_character(name, data, **kwargs)
        return self.character[name]

    def add_character(self, name, data=None, **kwargs):
        """Create a new character.

        You'll be able to access it as a :class:`Character` object by
        looking up ``name`` in my ``character`` property.

        ``data``, if provided, should be a networkx-compatible graph
        object. Your new character will be a copy of it.

        Any keyword arguments will be set as stats of the new character.

        """
        self._init_graph(name, 'DiGraph', data)
        self._graph_objs[name] = graph_obj = self.char_cls(self, name)
        if kwargs:
            graph_obj.stat.update(kwargs)

    def del_character(self, name):
        """Remove the Character from the database entirely.

        This also deletes all its history. You'd better be sure.

        """
        self.query.del_character(name)
        self.del_graph(name)
        del self.character[name]

    def _is_thing(self, character, node):
        return self._things_cache.contains_entity(
            character, node, *self._btt())

    def _set_thing_loc(
            self, character, node, loc
    ):
        branch, turn, tick = self._nbtt()
        self._things_cache.store(character, node, branch, turn, tick, loc)
        self.query.set_thing_loc(
            character,
            node,
            branch,
            turn,
            tick,
            loc
        )

    def alias(self, v, stat='dummy'):
        """Return a pointer to a value for use in historical queries.

        It will behave much as if you assigned the value to some entity
        and then used its ``historical`` method to get a reference to
        the set of its past values, which happens to contain only the
        value you've provided here, ``v``.

        :arg v: the value to represent
        :arg stat: what name to pretend its stat has; usually irrelevant

        """
        from .util import EntityStatAccessor
        r = DummyEntity(self)
        r[stat] = v
        return EntityStatAccessor(r, stat, engine=self)

    def _entityfy(self, v, stat='dummy'):
        from .query import Query
        from .util import EntityStatAccessor
        if (
                isinstance(v, self.thing_cls) or
                isinstance(v, self.place_cls) or
                isinstance(v, self.portal_cls) or
                isinstance(v, Query) or
                isinstance(v, EntityStatAccessor)
        ):
            return v
        return self.alias(v, stat)

    def turns_when(self, qry):
        """Yield the turns in this branch when the query held true

        :arg qry: a Query, likely constructed by comparing the result
        of a call to an entity's ``historical`` method with the output
        of ``self.alias(..)`` or another ``historical(..)``

        """
        # yeah, it's just a loop over the query's method...I'm planning
        # on moving some iter_turns logic in here when I figure out what
        # of it is truly independent of any given type of query
        for branch, turn in qry.iter_turns():
            yield turn

    def _node_contents(self, character, node):
        return self._node_contents_cache.retrieve(
                character, node, *self._btt()
        )

    def apply_choice(self, entity, key, value, dry_run=False):
        schema = self.schema
        assert schema.entity_permitted(entity)
        val = schema.validate(self.turn, entity, key, value)
        if not val:
            return val
        if type(val) is tuple:
            res, msg = val
            if res and not dry_run:
                entity[key] = value
            return msg
        if not dry_run:
            entity[key] = value

    def apply_choices(self, choices, dry_run=False, perfectionist=False):
        """Validate changes a player wants to make, and apply if acceptable.

        Returns a pair of lists containing acceptance and rejection messages,
        which the UI may present as it sees fit. They are always in a pair with
        the change request as the zeroth item. The message may be None or a string.

        Validator functions may return only a boolean indicating acceptance.
        If they instead return a pair, the initial boolean indicates acceptance
        and the following item is the message.

        This function will not actually result in any simulation happening.
        It creates a plan. See my ``plan`` context manager for the precise
        meaning of this.

        With ``dry_run=True`` just return the acceptances and rejections without
        really planning anything. With ``perfectionist=True`` apply changes if
        and only if all of them are accepted.

        """
        schema = self.schema
        todo = defaultdict(list)
        acceptances = []
        rejections = []
        for track in choices:
            entity = track['entity']
            permissible = schema.entity_permitted(entity)
            if not permissible:
                msg = schema.get_not_permitted_entity_message(entity)
                for turn, changes in enumerate(track['changes'], start=self.turn):
                    rejections.extend(
                        ((turn, entity, k, v), msg) for (k, v) in changes
                    )
                continue
            for turn, changes in enumerate(track['changes'], start=self.turn):
                for k, v in changes:
                    ekv = (entity, k, v)
                    parcel = (turn, entity, k, v)
                    val = schema.validate(*parcel)
                    if type(val) is tuple:
                        accept, message = val
                        if accept:
                            todo[turn].append(ekv)
                            l = acceptances
                        else:
                            l = rejections
                        l.append((parcel, message))
                    elif val:
                        todo[turn].append(ekv)
                        acceptances.append((parcel, None))
                    else:
                        rejections.append((parcel, None))
        if dry_run or (perfectionist and rejections):
            return acceptances, rejections
        now = self.turn
        with self.plan():
            for turn in sorted(todo):
                self.turn = turn
                for entity, key, value in todo[turn]:
                    if isinstance(entity, self.char_cls):
                        entity.stat[key] = value
                    else:
                        entity[key] = value
        self.turn = now
        return acceptances, rejections
