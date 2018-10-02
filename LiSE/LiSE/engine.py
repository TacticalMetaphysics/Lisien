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

import umsgpack
from blinker import Signal
from allegedb import ORM as gORM
from .util import reify, sort_set

from . import exc


class NoPlanningAttrGetter:
    __slots__ = ('_real',)

    def __init__(self, attr, *attrs):
        self._real = attrgetter(attr, *attrs)

    def __call__(self, obj):
        if obj._planning:
            raise exc.PlanError("Don't use randomization in a plan")
        return self._real(obj)


def getnoplan(attribute_name):
    return property(NoPlanningAttrGetter(attribute_name))


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
        start_branch, start_turn, start_tick = engine.btt()
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
            raise exc.RulesEngineError("Can't run the rules engine on any turn but the latest")
        if start_turn == latest_turn:
            # As a side effect, the following assignment sets the tick to
            # the latest in the new turn, which will be 0 if that turn has not
            # yet been simulated.
            engine.turn += 1
            if engine.tick == 0:
                engine.universal['rando_state'] = engine._rando.getstate()
            else:
                engine._rando.setstate(engine.universal['rando_state'])
        with engine.advancing():
            for res in iter(engine.advance, final_rule):
                if res:
                    engine.universal['last_result'] = res
                    engine.universal['last_result_idx'] = 0
                    engine.universal['rando_state'] = engine._rando.getstate()
                    branch, turn, tick = engine.btt()
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

    Implements serialization methods and the __getattr__ for stored methods.

    By default, the deserializers will refuse to create LiSE entities. If
    you want them to, use my ``loading`` property to open a ``with`` block,
    in which deserialized entities will be created as needed.

    """
    from contextlib import contextmanager

    @contextmanager
    def loading(self):
        """Context manager for when you need to instantiate entities upon unpacking"""
        if getattr(self, '_initialized', False):
            raise ValueError("Already loading")
        self._initialized = False
        yield
        self._initialized = True

    def __getattr__(self, item):
        meth = super().__getattribute__('method').__getattr__(item)
        return MethodType(meth, self)

    def _pack_character(self, char):
        return umsgpack.Ext(MSGPACK_CHARACTER, umsgpack.packb(char.name, ext_handlers=self._pack_handlers))

    def _pack_place(self, place):
        return umsgpack.Ext(MSGPACK_PLACE, umsgpack.packb(
            (place.character.name, place.name), ext_handlers=self._pack_handlers
        ))

    def _pack_thing(self, thing):
        return umsgpack.Ext(MSGPACK_THING, umsgpack.packb(
            (thing.character.name, thing.name), ext_handlers=self._pack_handlers
        ))

    def _pack_portal(self, port):
        return umsgpack.Ext(MSGPACK_PORTAL, umsgpack.packb(
            (port.character.name, port.orig, port.dest), ext_handlers=self._pack_handlers
        ))

    def _pack_tuple(self, tup):
        return umsgpack.Ext(MSGPACK_TUPLE, umsgpack.packb(list(tup), ext_handlers=self._pack_handlers))

    def _pack_frozenset(self, frozs):
        return umsgpack.Ext(MSGPACK_FROZENSET, umsgpack.packb(list(frozs), ext_handlers=self._pack_handlers))

    def _pack_set(self, s):
        return umsgpack.Ext(MSGPACK_SET, umsgpack.packb(list(s), ext_handlers = self._pack_handlers))

    def _pack_exception(self, exc):
        return umsgpack.Ext(MSGPACK_EXCEPTION, umsgpack.packb(
            [exc.__class__.__name__] + list(exc.args), ext_handlers=self._pack_handlers
        ))

    def _pack_func(self, func):
        return umsgpack.Ext({
            'method': MSGPACK_METHOD,
            'function': MSGPACK_FUNCTION,
            'trigger': MSGPACK_TRIGGER,
            'prereq': MSGPACK_PREREQ,
            'action': MSGPACK_ACTION
        }[func.__module__], umsgpack.packb(func.__name__))

    def _pack_meth(self, func):
        return umsgpack.Ext(MSGPACK_METHOD, umsgpack.packb(func.__name__))

    def _unpack_char(self, ext):
        charn = umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers)
        try:
            return self.character[charn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.char_cls(self, charn)

    def _unpack_place(self, ext):
        charn, placen = umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers)
        try:
            char = self.character[charn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.place_cls(self.char_cls(self, charn), placen)
        try:
            return char.place[placen]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.place_cls(char, placen)

    def _unpack_thing(self, ext):
        charn, thingn = umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers)
        try:
            char = self.character[charn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.thing_cls(self.char_cls(self, charn), thingn)
        try:
            return char.thing[thingn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.thing_cls(char, thingn)

    def _unpack_portal(self, ext):
        charn, orign, destn = umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers)
        try:
            char = self.character[charn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            char = self.char_cls(self, charn)
        try:
            return char.portal[orign][destn]
        except KeyError:
            if getattr(self, '_initialized', True):
                raise
            return self.portal_cls(char, orign, destn)

    def _unpack_trigger(self, ext):
        return getattr(self.trigger, umsgpack.unpackb(ext.data))

    def _unpack_prereq(self, ext):
        return getattr(self.prereq, umsgpack.unpackb(ext.data))

    def _unpack_action(self, ext):
        return getattr(self.action, umsgpack.unpackb(ext.data))

    def _unpack_function(self, ext):
        return getattr(self.function, umsgpack.unpackb(ext.data))

    def _unpack_method(self, ext):
        return getattr(self.method, umsgpack.unpackb(ext.data))

    def _unpack_tuple(self, ext):
        return tuple(umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers))

    def _unpack_frozenset(self, ext):
        return frozenset(umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers))

    def _unpack_set(self, ext):
        return set(umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers))

    def _unpack_exception(self, ext):
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
        data = umsgpack.unpackb(ext.data, ext_handlers=self._unpack_handlers)
        if data[0] not in excs:
            return Exception(*data)
        return excs[data[0]](*data[1:])


    @reify
    def _unpack_handlers(self):
        return {
            MSGPACK_CHARACTER: self._unpack_char,
            MSGPACK_PLACE: self._unpack_place,
            MSGPACK_THING: self._unpack_thing,
            MSGPACK_PORTAL: self._unpack_portal,
            MSGPACK_FINAL_RULE: lambda obj: final_rule,
            MSGPACK_TUPLE: self._unpack_tuple,
            MSGPACK_FROZENSET: self._unpack_frozenset,
            MSGPACK_SET: self._unpack_set,
            MSGPACK_TRIGGER: self._unpack_trigger,
            MSGPACK_PREREQ: self._unpack_prereq,
            MSGPACK_ACTION: self._unpack_action,
            MSGPACK_FUNCTION: self._unpack_function,
            MSGPACK_METHOD: self._unpack_method,
            MSGPACK_EXCEPTION: self._unpack_exception
        }

    @reify
    def _pack_handlers(self):
        return {
            self.char_cls: self._pack_character,
            self.place_cls: self._pack_place,
            self.thing_cls: self._pack_thing,
            self.portal_cls: self._pack_portal,
            tuple: self._pack_tuple,
            frozenset: self._pack_frozenset,
            set: self._pack_set,
            FinalRule: lambda obj: umsgpack.Ext(MSGPACK_FINAL_RULE, b""),
            FunctionType: self._pack_func,
            MethodType: self._pack_meth,
            Exception: self._pack_exception
        }

    def pack(self, obj):
        return umsgpack.packb(obj, ext_handlers=self._pack_handlers)

    def unpack(self, bs):
        return umsgpack.unpackb(bs, ext_handlers=self._unpack_handlers)

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
        """Roll ``n`` dice with ``d`` sides, sum them, and return whether they
        are <= ``target``.

        If ``comparator`` is provided, use it instead of <=. You may
        use a string like '<' or '>='.

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
    illegal_graph_names = ['global', 'eternal', 'universal', 'rulebooks', 'rules']
    illegal_node_names = ['nodes', 'node_val', 'edges', 'edge_val', 'things']

    def _make_node(self, graph, node):
        if self._is_thing(graph.name, node):
            return self.thing_cls(graph, node)
        else:
            return self.place_cls(graph, node)

    def _make_edge(self, graph, orig, dest, idx=0):
        return self.portal_cls(graph, orig, dest)

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
        it gets a 'location'; when the 'location' is deleted,
        that means it's back to being a place.

        Keys at the top level that are not character names:

        * 'rulebooks', a dictionary keyed by the name of each changed rulebook, the value
        being a list of rule names
        * 'rules', a dictionary keyed by the name of each changed rule, containing any
        of the lists 'triggers', 'prereqs', and 'actions'

        """
        from allegedb.window import update_window, update_backward_window
        if turn_from == turn_to:
            return self.get_turn_delta(branch, turn_to, tick_to, start_tick=tick_from)
        delta = super().get_delta(branch, turn_from, tick_from, turn_to, tick_to)
        if turn_from < turn_to:
            updater = partial(update_window, turn_from, tick_from, turn_to, tick_to)
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
            charporbbranches = self._characters_portals_rulebooks_cache.settings
            noderbbranches = self._nodes_rulebooks_cache.settings
            edgerbbranches = self._portals_rulebooks_cache.settings
        else:
            updater = partial(update_backward_window, turn_from, tick_from, turn_to, tick_to)
            univbranches = self._universal_cache.presettings
            avbranches = self._avatarness_cache.presettings
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

        def upduniv(_, key, val):
            delta.setdefault('universal', {})[key] = val
        if branch in univbranches:
            updater(upduniv, univbranches[branch])

        def updav(char, graph, node, av):
            delta.setdefault(char, {}).setdefault('avatars', {}).setdefault(graph, {})[node] = bool(av)
        if branch in avbranches:
            updater(updav, avbranches[branch])

        def updthing(char, thing, loc):
            if (
                char in delta and 'nodes' in delta[char]
                and thing in delta[char]['nodes'] and not
                delta[char]['nodes'][thing]
            ):
                return
            thingd = delta.setdefault(char, {}).setdefault('node_val', {}).setdefault(thing, {})
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
            if (
                character in delta and 'nodes' in delta[character]
                and node in delta[character]['nodes'] and not delta[character]['nodes'][node]
            ):
                return
            delta.setdefault(character, {}).setdefault('node_val', {}).setdefault(node, {})['rulebook'] = rulebook

        if branch in noderbbranches:
            updater(updnoderb, noderbbranches[branch])

        def updedgerb(character, orig, dest, rulebook):
            if (
                character in delta and 'edges' in delta[character]
                and orig in delta[character]['edges'] and dest in delta[character]['edges'][orig]
                and not delta[character]['edges'][orig][dest]
            ):
                return
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
        if branch in self._avatarness_cache.settings and turn in self._avatarness_cache.settings[branch]:
            for chara, graph, node, is_av in self._avatarness_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(chara, {}).setdefault('avatars', {}).setdefault(graph, {})[node] = is_av
        if branch in self._things_cache.settings and turn in self._things_cache.settings[branch]:
            for chara, thing, location in self._things_cache.settings[branch][turn][start_tick:tick]:
                thingd = delta.setdefault(chara, {}).setdefault('node_val', {}).setdefault(thing, {})
                thingd['location'] = location
        delta['rulebooks'] = rbdif = {}
        if branch in self._rulebooks_cache.settings and turn in self._rulebooks_cache.settings[branch]:
            for _, rulebook, rules in self._rulebooks_cache.settings[branch][turn][start_tick:tick]:
                rbdif[rulebook] = rules
        delta['rules'] = rdif = {}
        if branch in self._triggers_cache.settings and turn in self._triggers_cache.settings[branch]:
            for _, rule, funs in self._triggers_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['triggers'] = funs
        if branch in self._prereqs_cache.settings and turn in self._prereqs_cache.settings[branch]:
            for _, rule, funs in self._prereqs_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['prereqs'] = funs
        if branch in self._actions_cache.settings and turn in self._triggers_cache.settings[branch]:
            for _, rule, funs in self._triggers_cache.settings[branch][turn][start_tick:tick]:
                rdif.setdefault(rule, {})['actions'] = funs

        if branch in self._characters_rulebooks_cache.settings and turn in self._characters_rulebooks_cache.settings[branch]:
            for _, character, rulebook in self._characters_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_rulebook'] = rulebook
        if branch in self._avatars_rulebooks_cache.settings and turn in self._avatars_rulebooks_cache.settings[branch]:
            for _, character, rulebook in self._avatars_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['avatar_rulebook'] = rulebook
        if branch in self._characters_things_rulebooks_cache.settings and turn in self._characters_things_rulebooks_cache.settings[branch]:
            for _, character, rulebook in self._characters_things_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_thing_rulebook'] = rulebook
        if branch in self._characters_places_rulebooks_cache.settings and turn in self._characters_places_rulebooks_cache.settings[branch]:
            for _, character, rulebook in self._characters_places_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_place_rulebook'] = rulebook
        if branch in self._characters_portals_rulebooks_cache.settings and turn in self._characters_portals_rulebooks_cache.settings[branch]:
            for _, character, rulebook in self._characters_portals_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {})['character_portal_rulebook'] = rulebook

        if branch in self._nodes_rulebooks_cache.settings and turn in self._nodes_rulebooks_cache.settings[branch]:
            for character, node, rulebook in self._nodes_rulebooks_cache.settings[branch][turn][start_tick:tick]:
                delta.setdefault(character, {}).setdefault('node_val', {}).setdefault(node, {})['rulebook'] = rulebook
        if branch in self._portals_rulebooks_cache.settings and turn in self._portals_rulebooks_cache.settings[branch]:
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
            StringStore,
            FunctionStore,
            CharacterMapping,
            UniversalMapping
        )
        from .cache import (
            Cache,
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
        self._node_contents_cache = Cache(self)
        self.character = self.graph = CharacterMapping(self)
        self._universal_cache = EntitylessCache(self)
        self._rulebooks_cache = InitializedEntitylessCache(self)
        self._characters_rulebooks_cache = InitializedEntitylessCache(self)
        self._avatars_rulebooks_cache = InitializedEntitylessCache(self)
        self._characters_things_rulebooks_cache = InitializedEntitylessCache(self)
        self._characters_places_rulebooks_cache = InitializedEntitylessCache(self)
        self._characters_portals_rulebooks_cache = InitializedEntitylessCache(self)
        self._nodes_rulebooks_cache = InitializedCache(self)
        self._portals_rulebooks_cache = InitializedCache(self)
        self._triggers_cache = InitializedEntitylessCache(self)
        self._prereqs_cache = InitializedEntitylessCache(self)
        self._actions_cache = InitializedEntitylessCache(self)
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
        self._turns_completed = defaultdict(lambda: max((0, self.turn - 1)))
        """The last turn when the rules engine ran in each branch"""
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
            self._graph_objs[charn] = self.char_cls(self, charn, init_rulebooks=False)

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
            validate=False,
            clear_code=False,
            clear_world=False
    ):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        import os
        worlddbpath = worlddb.replace('sqlite:///', '')
        if clear_world and os.path.exists(worlddbpath):
            os.remove(worlddbpath)
        if isinstance(string, str):
            self._string_file = string
            if clear_code and os.path.exists(string):
                os.remove(string)
        else:
            self.string = string
        if isinstance(function, str):
            self._function_file = function
            if clear_code and os.path.exists(function):
                os.remove(function)
        else:
            self.function = function
        if isinstance(method, str):
            self._method_file = method
            if clear_code and os.path.exists(method):
                os.remove(method)
        else:
            self.method = method
        if isinstance(trigger, str):
            self._trigger_file = trigger
            if clear_code and os.path.exists(trigger):
                os.remove(trigger)
        else:
            self.trigger = trigger
        if isinstance(prereq, str):
            self._prereq_file = prereq
            if clear_code and os.path.exists(prereq):
                os.remove(prereq)
        else:
            self.prereq = prereq
        if isinstance(action, str):
            self._action_file = action
            if clear_code and os.path.exists(action):
                os.remove(action)
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
        self._things_cache.load(q.things_dump(), validate)
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
        for row in q.character_rules_handled_dump():
            self._character_rules_handled_cache.store(*row, loading=True)
        for row in q.avatar_rules_handled_dump():
            self._avatar_rules_handled_cache.store(*row, loading=True)
        for row in q.character_thing_rules_handled_dump():
            self._character_thing_rules_handled_cache.store(*row, loading=True)
        for row in q.character_place_rules_handled_dump():
            self._character_place_rules_handled_cache.store(*row, loading=True)
        for row in q.character_portal_rules_handled_dump():
            self._character_portal_rules_handled_cache.store(*row, loading=True)
        for row in q.node_rules_handled_dump():
            self._node_rules_handled_cache.store(*row, loading=True)
        for row in q.portal_rules_handled_dump():
            self._portal_rules_handled_cache.store(*row, loading=True)
        self._turns_completed.update(q.turns_completed_dump())
        self._rules_cache = {name: Rule(self, name, create=False) for name in q.rules_dump()}

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
        self.debug("following rule: " + repr(rule))
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
        # TODO: roll back changes done by rules that raise an exception
        # TODO: if there's a paradox while following some rule, start a new branch, copying handled rules
        from collections import defaultdict
        branch, turn, tick = self.btt()
        charmap = self.character
        rulemap = self.rule
        todo = defaultdict(list)

        def do_rule(tup):
            # Returns None if the entity following the rule no longer exists.
            # Better way to handle this?
            return {
                'character': lambda charactername, rulebook, rulename: self._follow_rule(
                    rulemap[rulename],
                    partial(self._handled_char, charactername, rulebook, rulename, branch, turn, tick),
                    branch, turn,
                    charmap[charactername]
                ) if charactername in charmap else None,
                'avatar': lambda charn, rulebook, graphn, avn, rulen: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_av, charn, graphn, avn, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[graphn].node[avn]
                ) if self._node_exists(graphn, avn) else None,
                'character_thing': lambda charn, rulebook, rulen, thingn: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_char_thing, charn, thingn, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[charn].thing[thingn]
                ) if charn in charmap and thingn in charmap[charn].thing else None,
                'character_place': lambda charn, rulebook, rulen, placen: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_char_place, charn, placen, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[charn].place[placen]
                ) if charn in charmap and placen in charmap[charn].place else None,
                'character_portal': lambda charn, rulebook, rulen, orign, destn: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_char_port, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[charn].portal[orign][destn]
                ) if self._edge_exists(charn, orign, destn) else None,
                'node': lambda charn, noden, rulebook, rulen: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_node, charn, noden, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[charn].node[noden]
                ) if self._node_exists(charn, noden) else None,
                'portal': lambda charn, orign, destn, rulebook, rulen: self._follow_rule(
                    rulemap[rulen],
                    partial(self._handled_portal, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                    branch, turn,
                    charmap[charn].portal[orign][destn]
                ) if self._edge_exists(charn, orign, destn) else None
            }[tup[0]](*tup[1:])
        for (
            charactername, rulebook, rulename
        ) in self._character_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if charactername not in charmap:
                continue
            todo[rulebook].append(('character', charactername, rulebook, rulename))
        for (
            charn, graphn, avn, rulebook, rulen
        ) in self._avatar_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if charn not in charmap:
                continue
            char = charmap[charn]
            if graphn not in char.avatar or avn not in char.avatar[graphn]:
                continue
            todo[rulebook].append(('avatar', charn, rulebook, graphn, avn, rulen))
        for (
            charn, thingn, rulebook, rulen
        ) in self._character_thing_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            if charn not in charmap or thingn not in charmap[charn].thing:
                continue
            todo[rulebook].append(('character_thing', charn, rulebook, rulen, thingn))
        for (
            charn, placen, rulebook, rulen
        ) in self._character_place_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if charn not in charmap or placen not in charmap[charn].place:
                continue
            todo[rulebook].append(('character_place', charn, rulebook, rulen, placen))
        for (
            charn, orign, destn, rulebook, rulen
        ) in self._character_portal_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if charn not in charmap:
                continue
            char = charmap[charn]
            if orign not in char.portal or destn not in char.portal[orign]:
                continue
            todo[rulebook].append(('character_portal', charn, rulebook, rulen, orign, destn))
        for (
                charn, noden, rulebook, rulen
        ) in self._node_rules_handled_cache.iter_unhandled_rules(
            branch, turn, tick
        ):
            if charn not in charmap or noden not in charmap[charn]:
                continue
            todo[rulebook].append(('node', charn, noden, rulebook, rulen))
        for (
                charn, orign, destn, rulebook, rulen
        ) in self._portal_rules_handled_cache.iter_unhandled_rules(
                branch, turn, tick
        ):
            if charn not in charmap:
                continue
            char = charmap[charn]
            if orign not in char.portal or destn not in char.portal[orign]:
                continue
            todo[rulebook].append(('portal', charn, orign, destn, rulebook, rulen))

        # TODO: rulebook priorities (not individual rule priorities, just follow the order of the rulebook)
        for rulebook in sort_set(todo.keys()):
            for rule in todo[rulebook]:
                try:
                    yield do_rule(rule)
                except StopIteration:
                    raise InnerStopIteration

    def advance(self):
        """Follow the next rule if available.

        If we've run out of rules, reset the rules iterator.

        """
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
        self._init_graph(name, 'DiGraph')
        self._graph_objs[name] = self.char_cls(self, name, data, **kwargs)

    def del_character(self, name):
        """Remove the Character from the database entirely.

        This also deletes all its history. You'd better be sure.

        """
        self.query.del_character(name)
        self.del_graph(name)
        del self.character[name]

    def _is_thing(self, character, node):
        return self._things_cache.contains_entity(character, node, *self.btt())

    def _set_thing_loc(
            self, character, node, loc
    ):
        branch, turn, tick = self.nbtt()
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
        from .util import EntityStatAccessor
        r = DummyEntity(self)
        r[stat] = v
        return EntityStatAccessor(r, stat, engine=self)

    def entityfy(self, v, stat='dummy'):
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
        for branch, turn in qry.iter_turns():
            yield turn