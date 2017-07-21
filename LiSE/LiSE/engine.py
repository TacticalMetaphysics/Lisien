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
from .rule import AllRuleBooks, AllRules
from .query import Query, QueryEngine
from .util import getatt, reify, EntityStatAccessor
from .cache import (
    Cache,
    EntitylessCache,
    AvatarnessCache,
    CharacterRulebooksCache,
    ActiveRulesCache,
    NodeRulesHandledCache,
    PortalRulesHandledCache,
    CharacterRulesHandledCache,
    ThingsCache
)


class TimeSignal(Signal):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __iter__(self):
        yield self.engine.branch
        yield self.engine.tick

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i in ('branch', 0):
            return self.engine.branch
        if i in ('tick', 'rev', 1):
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
        (branch_then, tick_then) = real.engine.time
        (branch_now, tick_now) = val
        # make sure I'll end up within the revision range of the
        # destination branch
        if branch_now != 'trunk':
            if branch_now in real.engine._parentbranch_rev:
                parrev = real.engine._parentbranch_rev[branch_now][1]
                if tick_now < parrev:
                    raise ValueError(
                        "Tried to jump to branch {br}, "
                        "which starts at tick {t}. "
                        "Go to tick {t} or later to use branch {br}.".format(
                            br=branch_now,
                            t=parrev
                        )
                    )
            else:
                real.engine._parentbranch_rev[branch_now] = (
                    branch_then, tick_now
                )
                real.engine.query.new_branch(branch_now, branch_then, tick_now)
        (real.engine._obranch, real.engine._orev) = val
        real.send(
            real,
            engine=real.engine,
            branch_then=branch_then,
            tick_then=tick_then,
            branch_now=branch_now,
            tick_now=tick_now
        )


class NextTick(Signal):
    """Make time move forward in the simulation.

    Calls ``advance`` repeatedly, appending its results to a list until
    the tick has ended.  Returns the list.

    I am also a ``Signal``, so you can register functions to be
    called when the simulation runs. Pass them to my ``connect``
    method.

    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __call__(self):
        curtick = self.engine.tick
        r = []
        while self.engine.tick == curtick:
            r.append(self.engine.advance())
        # The last element is always None, but is not a sentinel; any
        # rule may return None.
        self.send(
            self.engine,
            branch=self.engine.branch,
            tick=self.engine.tick,
            result=r
        )
        return r[:-1]


class DummyEntity(dict):
    __slots__ = ['engine']

    def __init__(self, engine):
        self.engine = engine


json_dump_hints = {}
json_load_hints = {}


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
                "portal", obj.character.name, obj._origin, obj._destination]
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

    LiSE tracks history as a series of ticks. In each tick, each
    simulation rule is evaluated once for each of the simulated
    entities it's been applied to. World changes in a given tick are
    remembered together, such that the whole world state can be
    rewound: simply set the properties ``branch`` and ``tick`` back to
    what they were just before the change you want to undo.

    Properties:

    - ``branch``: The fork of the timestream that we're on.
    - ``tick``: Units of time that have passed since the sim started.
    - ``time``: ``(branch, tick)``
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
        self._nodes_rulebooks_cache.store(character, node, rulebook)
        self.engine.query.set_node_rulebook(character, node, rulebook)

    def _set_portal_rulebook(self, character, orig, dest, rulebook):
        self._portals_rulebooks_cache.store(character, orig, dest, rulebook)
        self.query.set_portal_rulebook(character, orig, dest, rulebook)

    def _remember_avatarness(
            self, character, graph, node,
            is_avatar=True, branch=None, tick=None
    ):
        """Use this to record a change in avatarness.

        Should be called whenever a node that wasn't an avatar of a
        character now is, and whenever a node that was an avatar of a
        character now isn't.

        ``character`` is the one using the node as an avatar,
        ``graph`` is the character the node is in.

        """
        branch = self.branch if branch is None else branch
        tick = self.tick if tick is None else tick
        self._avatarness_cache.store(
            character,
            graph,
            node,
            branch,
            tick,
            is_avatar
        )
        self.query.avatar_set(
            character,
            graph,
            node,
            branch,
            tick,
            is_avatar
        )

    def _set_rule_activeness(
            self, rulebook, rule, active, branch=None, tick=None
    ):
        branch = branch or self.branch
        tick = tick if tick is not None else self.tick
        self._active_rules_cache.store(rulebook, rule, branch, tick, active)
        # note the use of the world DB, not the code DB
        self.query.set_rule_activeness(rulebook, rule, branch, tick, active)

    def _init_caches(self):
        super()._init_caches()
        self._portal_objs = {}
        self._things_cache = ThingsCache(self)
        self.character = self.graph = CharacterMapping(self)
        self._universal_cache = EntitylessCache(self)
        self._rulebooks_cache = Cache(self)
        self._characters_rulebooks_cache = CharacterRulebooksCache(self)
        self._nodes_rulebooks_cache = Cache(self)
        self._portals_rulebooks_cache = Cache(self)
        self._active_rules_cache = ActiveRulesCache(self)
        self._node_rules_handled_cache = NodeRulesHandledCache(self)
        self._portal_rules_handled_cache = PortalRulesHandledCache(self)
        self._character_rules_handled_cache = CharacterRulesHandledCache(self)
        self._avatar_rules_handled_cache = CharacterRulesHandledCache(self)
        self._character_thing_rules_handled_cache \
            = CharacterRulesHandledCache(self)
        self._character_place_rules_handled_cache \
            = CharacterRulesHandledCache(self)
        self._character_node_rules_handled_cache \
            = CharacterRulesHandledCache(self)
        self._character_portal_rules_handled_cache \
            = CharacterRulesHandledCache(self)
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

    def _init_load(self):
        # I have to load thingness first, because it affects my _make_node method
        for row in self.query.things_dump():
            self._things_cache.store(*row)
        super()._init_load()
        for row in self.query.universals_dump():
            self._universal_cache.store(*row)
        for row in self.query.rulebooks_dump():
            self._rulebooks_cache.store(*row)
        for row in self.query.characters_dump():
            self._characters_rulebooks_cache.store(*row)
        for row in self.query.node_rulebook_dump():
            self._nodes_rulebooks_cache.store(*row)
        for row in self.query.portal_rulebook_dump():
            self._portals_rulebooks_cache.store(*row)
        for row in self.query.node_rules_handled_dump():
            self._node_rules_handled_cache.store(*row)
        for row in self.query.portal_rules_handled_dump():
            self._portal_rules_handled_cache.store(*row)
        for row in self.query.character_rules_handled_dump():
            self._character_rules_handled_cache.store(*row)
        for row in self.query.avatar_rules_handled_dump():
            self._avatar_rules_handled_cache.store(*row)
        for row in self.query.character_thing_rules_handled_dump():
            self._character_thing_rules_handled_cache.store(*row)
        for row in self.query.character_place_rules_handled_dump():
            self._character_place_rules_handled_cache.store(*row)
        for row in self.query.character_portal_rules_handled_dump():
            self._character_portal_rules_handled_cache.store(*row)
        for row in self.query.avatars_dump():
            self._avatarness_cache.store(*row)

    def _load_graphs(self):
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
            logfun=None
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
            query_engine_class=QueryEngine,
            connect_args=connect_args,
            alchemy=alchemy,
            json_dump=self.json_dump,
            json_load=self.json_load,
        )
        self.next_tick = NextTick(self)
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
        if v != 'trunk':
            if v in self._parentbranch_rev:
                partick = self._parentbranch_rev[v][1]
                if self.tick < partick:
                    raise ValueError(
                        "Tried to jump to branch {br}, "
                        "which starts at tick {rv}. "
                        "Go to tick {rv} or later to use this branch.".format(
                            br=v,
                            rv=partick
                        )
                    )
            else:
                parent = b
                child = v
                self._parentbranch_rev[child] = parent, t
                self._childbranch[parent].add(child)
                self.query.new_branch(child, parent, t)
        self._obranch = v
        if not hasattr(self, 'locktime'):
            self.time.send(
                self,
                branch_then=b,
                tick_then=t,
                branch_now=v,
                tick_now=t
            )

    @property
    def tick(self):
        return self.rev

    @tick.setter
    def tick(self, v):
        """Update allegedb's ``rev``, and call listeners"""
        if not isinstance(v, int):
            raise TypeError("tick must be integer")
        (branch_then, tick_then) = self.time
        if v == self.tick:
            return
        self.rev = v
        if not hasattr(self, 'locktime'):
            self.time.send(
                self,
                branch_then=branch_then,
                tick_then=tick_then,
                branch_now=branch_then,
                tick_now=v
            )

    def _rule_active(self, rulebook, rule):
        if hasattr(rulebook, 'name'):
            rulebook = rulebook.name
        if hasattr(rule, 'name'):
            rule = rule.name
        return self._active_rules_cache.retrieve(
            rulebook, rule, *self.time
        ) is True

    def _poll_char_rules(self):
        unhandled_iter = self._character_rules_handled_cache.\
                         iter_unhandled_rules
        for char in self.character:
            for (
                    rulemap,
                    rulebook
            ) in self._characters_rulebooks_cache.retrieve(char).items():
                for rule in unhandled_iter(
                        char, rulemap, rulebook, *self.time
                ):
                    yield (rulemap, char, rulebook, rule)

    def _poll_node_rules(self):
        unhandled_iter = self._node_rules_handled_cache.iter_unhandled_rules
        for chara in self.character.values():
            char = chara.name
            for node in chara.node:
                try:
                    rulebook = self._nodes_rulebooks_cache.retrieve(char, node)
                except KeyError:
                    rulebook = (char, node)
                for rule in unhandled_iter(
                    char, node, rulebook, *self.time
                ):
                    yield (char, node, rulebook, rule)

    def _poll_portal_rules(self):
        cache = self._portals_rulebooks_cache
        for chara in self.character.values():
            for orig in chara.portal:
                for dest in chara.portal[orig]:
                    try:
                        rulebook = cache.retrieve(chara.name, orig, dest)
                    except KeyError:
                        rulebook = (chara.name, orig, dest)
                    unhanditer = self._portal_rules_handled_cache.\
                                 iter_unhandled_rules
                    for rule in unhanditer(
                            chara.name, orig, dest, rulebook, *self.time
                    ):
                        yield (
                            chara,
                            orig,
                            dest,
                            rulebook,
                            rule
                        )

    def _poll_rules(self):
        """Iterate over tuples containing rules yet unresolved in the current tick.

        The tuples are of the form: ``(ruletype, character, entity,
        rulebook, rule)`` where ``ruletype`` is what kind of entity
        the rule is about (character', 'thing', 'place', or
        'portal'), and ``entity`` is the :class:`Place`,
        :class:`Thing`, or :class:`Portal` that the rule is attached
        to. For character-wide rules it is ``None``.

        """
        for (
                rulemap, character, rulebook, rule
        ) in self._poll_char_rules():
            try:
                yield (
                    rulemap,
                    self.character[character],
                    None,
                    rulebook,
                    self.rule[rule]
                )
            except KeyError:
                continue
        for (
                character, node, rulebook, rule
        ) in self._poll_node_rules():
            try:
                c = self.character[character]
                n = c.node[node]
            except KeyError:
                continue
            typ = 'thing' if hasattr(n, 'location') else 'place'
            yield typ, c, n, rulebook, self.rule[rule]
        for (
                character, a, b, i, rulebook, rule
        ) in self._poll_portal_rules():
            try:
                c = self.character[character]
                yield 'portal', c, c.portal[a][b], rulebook, self.rule[rule]
            except KeyError:
                continue

    def _handled_thing_rule(self, char, thing, rulebook, rule, branch, tick):
        self._node_rules_handled_cache.store(
            char, thing, rulebook, rule, branch, tick
        )
        self.query.handled_thing_rule(
            char, thing, rulebook, rule, branch, tick
        )

    def _handled_place_rule(self, char, place, rulebook, rule, branch, tick):
        self._node_rules_handled_cache.store(
            char, place, rulebook, rule, branch, tick
        )
        self.query.handled_place_rule(
            char, place, rulebook, rule, branch, tick
        )

    def _handled_portal_rule(
            self, char, orig, dest, rulebook, rule, branch, tick
    ):
        self._portal_rules_handled_cache.store(
            char, orig, dest, rulebook, rule, branch, tick
        )
        self.query.handled_portal_rule(
            char, orig, dest, rulebook, rule, branch, tick
        )

    def _handled_character_rule(
            self, typ, char, rulebook, rule, branch, tick
    ):
        self._character_rules_handled_cache.store(
            char, typ, rulebook, rule, branch, tick
        )
        self.query.handled_character_rule(
            typ, char, rulebook, rule, branch, tick
        )

    def _follow_rules(self):
        """For each rule in play at the present tick, call it and yield a
        tuple describing the results.

        Tuples are of the form: ``(returned, rulename, ruletype,
        rulebook)`` where ``returned`` is whatever the rule itself
        returned upon being called, and ``ruletype`` is what sort of
        entity the rule applies to.

        """
        (branch, tick) = self.time
        rules = list(self._poll_rules())
        for (typ, character, entity, rulebook, rule) in rules:
            def follow(*args):
                return (rule(self, *args), rule.name, typ, rulebook)

            if typ in ('thing', 'place', 'portal'):
                yield follow(character, entity)
                if typ == 'thing':
                    self._handled_thing_rule(
                        character.name,
                        entity.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
                elif typ == 'place':
                    self._handled_place_rule(
                        character.name,
                        entity.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
                else:
                    self._handled_portal_rule(
                        character.name,
                        entity.origin.name,
                        entity.destination.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
            else:
                if typ == 'character':
                    yield follow(character)
                elif typ == 'avatar':
                    for avatar in character.avatars():
                        yield follow(character, avatar)
                elif typ == 'character_thing':
                    for thing in character.thing.values():
                        yield follow(character, thing)
                elif typ == 'character_place':
                    for place in character.place.values():
                        yield follow(character, place)
                elif typ == 'character_node':
                    for node in character.node.values():
                        yield follow(character, node)
                elif typ == 'character_portal':
                    for portal in character.portal.values():
                        yield follow(character, portal)
                else:
                    raise ValueError('Unknown type of rule')
                self._handled_character_rule(
                    typ, character.name, rulebook, rule.name, branch, tick
                )

    def advance(self):
        """Follow the next rule if available, or advance to the next tick."""
        try:
            r = next(self._rules_iter)
        except StopIteration:
            self.tick += 1
            self._rules_iter = self._follow_rules()
            self.universal['rando_state'] = self.rando.getstate()
            if self.commit_modulus and self.tick % self.commit_modulus == 0:
                self.commit()
            r = None
        return r

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
        return self._things_cache.contains_entity(character, node, *self.time)

    def _set_thing_loc_and_next(
            self, character, node, loc, nextloc=None, branch=None, tick=None
    ):
        branch = branch or self.branch
        tick = tick or self.tick
        self.query.thing_loc_and_next_set(
            character,
            node,
            branch,
            tick,
            loc,
            nextloc
        )
        self._things_cache.store(character, node, branch, tick, loc, nextloc)

    def _node_exists(self, character, node):
        return self._nodes_cache.contains_entity(character, node, *self.time)

    def _exist_node(self, character, node, exist=True, branch=None, tick=None):
        branch = branch or self.branch
        tick = tick or self.tick
        self.query.exist_node(
            character,
            node,
            branch,
            tick,
            exist
        )
        self._nodes_cache.store(character, node, branch, tick, exist)
        self._nodes_rulebooks_cache.store(character, node, branch, tick, (character, node))

    def _exist_edge(
            self, character, orig, dest, exist=True, branch=None, tick=None
    ):
        branch = branch or self.branch
        tick = tick or self.tick
        self.query.exist_edge(
            character,
            orig,
            dest,
            0,
            branch,
            tick,
            exist
        )
        self._edges_cache.store(
            character, orig, dest, 0, branch, tick, exist
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
