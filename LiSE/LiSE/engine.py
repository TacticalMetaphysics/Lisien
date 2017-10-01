# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The "engine" of LiSE is an object relational mapper with special
stores for game data and entities, as well as properties for manipulating the
flow of time.

"""
from random import Random
from functools import partial
from json import dumps, loads, JSONEncoder
from operator import gt, lt, ge, le, eq, ne
from blinker import Signal
from allegedb import ORM as gORM
from allegedb.cache import HistoryError
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


class TimeSignal(Signal):
    # TODO: always time travel to the last tick in the turn
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __iter__(self):
        yield self.engine.branch
        yield self.engine.turn

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i in ('branch', 0):
            return self.engine.branch
        if i in ('turn', 1):
            return self.engine.tick

    def __setitem__(self, i, v):
        if i in ('branch', 0):
            self.engine.branch = v
        if i in ('tick', 'rev', 1):
            self.engine.tick = v


class TimeSignalDescriptor(object):
    signals = {}

    def __get__(self, inst, cls):
        if id(inst) not in self.signals:
            self.signals[id(inst)] = TimeSignal(inst)
        return self.signals[id(inst)]

    def __set__(self, inst, val):
        if id(inst) not in self.signals:
            self.signals[id(inst)] = TimeSignal(inst)
        real = self.signals[id(inst)]
        branch_then, turn_then, tick_then = real.engine.btt()
        branch_now, turn_now = val
        # make sure I'll end up within the revision range of the
        # destination branch
        e = real.engine
        parbtt = real.engine._parent_btt
        tick_now = None
        if branch_now != 'trunk':
            if branch_now in parbtt:
                parturn = parbtt[branch_now][1]
                if turn_now < parturn:
                    raise ValueError(
                        "Tried to jump to branch {br}, "
                        "which starts at turn {t}. "
                        "Go to turn {t} or later to use branch {br}.".format(
                            br=branch_now,
                            t=parturn
                        )
                    )
            else:
                tick_now = real.engine._turn_end.get((branch_now, turn_now), 0)
                parbtt[branch_now] = (
                    branch_then, turn_now, tick_now
                )
                e.query.new_branch(branch_now, branch_then, turn_now, tick_now)
        e._obranch, e._oturn = branch, turn = val
        e._branch_end[branch] = max((e._branch_end[branch], turn))
        e._otick = tick_now or real.engine._turn_end.get((branch_now, turn_now), 0)
        real.send(
            e,
            branch_then=branch_then,
            turn_then=turn_then,
            tick_then=tick_then,
            branch_now=branch_now,
            turn_now=turn_now,
            tick_now=tick_now
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
        for res in iter(engine.advance, final_rule):
            if res:
                branch, turn, tick = engine.btt()
                engine.universal['rando_state'] = engine.rando.getstate()
                self.send(
                    engine,
                    branch=branch,
                    turn=turn,
                    tick=tick
                )
                return res
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


class DummyEntity(dict):
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
    def encode(self, o):
        return super().encode(self.listify(o))

    def default(self, o):
        try:
            from numpy import sctypes
        except ImportError:
            return super().default(o)
        t = type(o)
        if t in sctypes['int']:
            return int(o)
        elif t in sctypes['float']:
            return float(o)
        else:
            return super().default(o)


class AbstractEngine(object):
    def __getattr__(self, att):
        if hasattr(super(), 'method') and hasattr(self.method, att):
            return partial(getattr(self.method, att), self)
        raise AttributeError('No attribute or stored method: {}'.format(att))

    @reify
    def _listify_dispatch(self):
        return {
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
                "portal", obj.character.name, obj.orig, obj.dest]
        }

    def listify(self, obj):
        try:
            return self._listify_dispatch[type(obj)](obj)
        except KeyError:
            return obj

    @reify
    def _delistify_dispatch(self):
        def nodeget(obj):
            return self._node_objs[(
                self.delistify(obj[1]), self.delistify(obj[2])
            )]
        return {
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
            )]
        }

    def delistify(self, obj):
        if isinstance(obj, list) or isinstance(obj, tuple):
            return self._delistify_dispatch[obj[0]](obj)
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
      objects, but this one *is* sensitive to sim-time. Each tick, the
      state of the randomizer is saved here under the key
      ``'rando_state'``.
    - ``rando``: The randomizer used by all of the rules.

    """
    char_cls = Character
    thing_cls = Thing
    place_cls = node_cls = Place
    portal_cls = edge_cls = _make_edge = Portal
    query_engine_cls = QueryEngine
    time = TimeSignalDescriptor()

    def _make_node(self, char, node):
        if self._is_thing(char.name, node):
            return Thing(char, node)
        else:
            return Place(char, node)

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

    def _set_character_rulebook(self, character, which, rulebook):
        if which not in (
                'character',
                'avatar',
                'character_thing',
                'character_place',
                'character_node',
                'character_portal'
        ):
            raise ValueError("Not a character rulebook: {}".format(which))
        self.query.upd_rulebook_char(which, rulebook, character)
        self._characters_rulebooks_cache.store(character, **{which: rulebook})

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
                self._string_file,
                self.eternal.setdefault('language', 'eng')
            )

    def load_graphs(self):
        for charn in self.query.characters():
            self._graph_objs[charn] = Character(self, charn)

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
            self._character_rules_handled_cache.store(character, rulebook, rule, branch, turn)
        for character, rulebook, rule, graph, avatar, branch, turn, tick in \
                q.avatar_rules_handled_dump():
            self._avatar_rules_handled_cache.store(character, rulebook, rule, graph, avatar, branch, turn)
        for character, rulebook, rule, thing, branch, turn, tick in \
                q.character_thing_rules_handled_dump():
            self._character_thing_rules_handled_cache.store(character, rulebook, rule, thing, branch, turn)
        for character, rulebook, rule, place, branch, turn, tick in \
                q.character_place_rules_handled_dump():
            self._character_place_rules_handled_cache.store(character, rulebook, rule, place, branch, turn)
        for character, rulebook, rule, orig, dest, branch, turn, tick in \
                q.character_portal_rules_handled_dump():
            self._character_portal_rules_handled_cache.store(character, rulebook, rule, orig, dest, branch, turn)
        for character, node, rulebook, rule, branch, turn, tick in q.node_rules_handled_dump():
            self._node_rules_handled_cache.store(character, node, rulebook, rule, branch, turn, tick)
        for character, orig, dest, rulebook, rule, branch, turn, tick in q.portal_rules_handled_dump():
            self._portal_rules_handled_cache.store(character, orig, dest, rulebook, rule, branch, turn, tick)
        self._rules_cache = {name: Rule(self, name, typ, create=False) for name, typ in q.rules_dump()}

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

    def commit(self):
        super().commit()

    def close(self):
        """Commit changes and close the database."""
        self.commit()
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

    @property
    def branch(self):
        if self._obranch is not None:
            return self._obranch
        return self.query.globl['branch']

    @branch.setter
    def branch(self, v):
        """Set my allegedb's branch and call listeners"""
        (b, t) = self.time
        if v == b:
            return
        self.time = (v, t)

    @property
    def turn(self):
        return self._oturn

    @turn.setter
    def turn(self, v):
        if not isinstance(v, int):
            raise TypeError("turn must be integer")
        if v == self.turn:
            return
        self.time = (self.branch, v)

    def _handled_char(self, charn, rulebook, rulen, branch, turn, tick):
        try:
            self._character_rules_handled_cache.store(
                charn, rulebook, rulen, branch, turn, tick
            )
        except ValueError:
            assert rulen in self._character_rules_handled_cache.shallow[
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
            assert rule in self._avatar_rules_handled_cache.shallow[
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
            assert rule in self._character_thing_rules_handled_cache.shallow[
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
            assert rule in self._character_place_rules_handled_cache.shallow[
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
            assert rule in self._character_portal_rules_handled_cache.shallow[
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
            assert rule in self._node_rules_handled_cache.shallow[
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
            assert rule in self._portal_rules_handled_cache.shallow[
                character, orig, dest, rulebook, branch, turn
            ]
            return
        self.query.handled_portal_rule(
            character, orig, dest, rulebook, rule, branch, turn, tick
        )

    def _follow_rule(self, rule, handled_fun, branch, turn, *args):
        for trigger in rule.triggers:
            res = trigger(*args)
            self.time = branch, turn
            if res:
                break
        else:
            return handled_fun()
        satisfied = True
        for prereq in rule.prereqs:
            res = prereq(*args)
            self.time = branch, turn
            if not res:
                satisfied = False
                break
        if not satisfied:
            return handled_fun()
        actres = []
        for action in rule.actions:
            res = action(*args)
            if res:
                actres.append(res)
            self.time = branch, turn
        handled_fun()
        return actres

    def _follow_rules(self):
        # Currently the user doesn't have a lot of control over the order that
        # rulebooks get run in. I should implement that.
        branch, turn, tick = self.btt()
        charmap = self.character
        rulemap = self.rule

        for (
            charactername, rulebook, rulename
        ) in self._character_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulename],
                partial(self._handled_char, charactername, rulebook, rulename, branch, turn, tick),
                branch, turn,
                charmap[charactername]
            )
        for (
            charn, rulebook, graphn, avn, rulen
        ) in self._avatar_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_av, charn, graphn, avn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn],
                charmap[graphn].node[avn]
            )
        for (
            charn, rulebook, rulen, thingn
        ) in self._character_thing_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_thing, charn, thingn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].thing[thingn]
            )
        for (
            charn, rulebook, rulen, placen
        ) in self._character_place_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_place, charn, placen, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].place[placen]
            )
        for (
            charn, rulebook, rulen, orign, destn
        ) in self._character_portal_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_char_port, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].portal[orign][destn]
            )
        for (
                charn, noden, rulebook, rulen
        ) in self._node_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_node, charn, noden, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].node[noden]
            )
        for (
                charn, orign, destn, rulebook, rulen
        ) in self._portal_rules_handled_cache.iter_unhandled_rules(branch, turn, tick):
            yield self._follow_rule(
                rulemap[rulen],
                partial(self._handled_port, charn, orign, destn, rulebook, rulen, branch, turn, tick),
                branch, turn,
                charmap[charn].portal[orign][destn]
            )

    def advance(self):
        """Follow the next rule if available, or advance to the next tick."""
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
        self.query.init_character(
            name,
            *self.time,
            **kwargs
        )
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

    def _exist_node(self, character, node, exist=True):
        branch, turn, tick = self.nbtt()
        self.query.exist_node(
            character,
            node,
            branch,
            turn,
            tick,
            exist
        )
        self._nodes_cache.store(character, node, branch, turn, tick, exist)
        self._nodes_rulebooks_cache.store(character, node, branch, turn, tick, (character, node))

    def _exist_edge(
            self, character, orig, dest, exist=True
    ):
        branch, turn, tick = self.nbtt()
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
            character, orig, dest, 0, branch, turn, tick, exist
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
