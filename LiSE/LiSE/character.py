# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The top level of the LiSE world model, the Character.

Based on NetworkX DiGraph objects with various additions and
conveniences.

A Character is a graph that follows rules. Its rules may be assigned
to run on only some portion of it: just edges (called Portals), just
nodes, or just nodes of the kind that have a location in another node
(called Places and Things, respectively). Each Character has a
``stat`` property that acts very much like a dictionary, in which you
can store game-relevant data for the rules to use.

You can designate some nodes in one Character as avatars of another,
and then assign a rule to run on all of a Character's avatars. This is
useful for the common case where someone in your game has a location
in the physical world (here, a Character, called 'physical') but also
has a behavior flowchart, or a skill tree, that isn't part of the
physical world. In that case the flowchart is the person's Character,
and their node in the physical world is an avatar of it.

"""

from abc import abstractmethod
from collections import (
    Mapping,
    MutableMapping,
    Callable
)
from operator import ge, gt, le, lt, eq
from blinker import Signal

import networkx as nx
from allegedb.graph import (
    DiGraph,
    GraphNodeMapping,
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping
)
from allegedb.wrap import MutableMappingUnwrapper

from .xcollections import CompositeDict
from .rule import RuleMapping
from .rule import RuleFollower as BaseRuleFollower
from .node import Node
from .thing import Thing
from .place import Place
from .portal import Portal
from .util import getatt, singleton_get
from .exc import AmbiguousAvatarError, WorldIntegrityError


class SpecialMappingDescriptor:
    def __init__(self, mapclsname):
        self.mapps = {}
        self.mapclsname = mapclsname

    def __get__(self, instance, owner):
        if id(instance) in self.mapps:
            return self.mapps[id(instance)]
        mappcls = getattr(instance, self.mapclsname)
        ret = self.mapps[id(instance)] = mappcls(instance)
        return ret

    def __set__(self, instance, value):
        if id(instance) not in self.mapps:
            self.mapps[id(instance)] = getattr(instance, self.mapclsname)(instance)
        it = self.mapps[id(instance)]
        it.clear()
        it.update(value)


class AbstractCharacter(MutableMapping):

    """The Character API, with all requisite mappings and graph generators.

    Mappings resemble those of a NetworkX digraph:

    * ``thing`` and ``place`` are subsets of ``node``
    * ``edge``, ``adj``, and ``succ`` are aliases of ``portal``
    * ``pred`` is an alias to ``preportal``
    * ``stat`` is a dict-like mapping of data that changes over game-time,
    to be used in place of graph attributes

    """
    engine = getatt('db')
    no_unwrap = True

    @abstractmethod
    def add_place(self, name, **kwargs): pass

    def add_node(self, name, **kwargs):
        self.add_place(name, **kwargs)

    @abstractmethod
    def add_places_from(self, seq, **attrs): pass

    def add_nodes_from(self, seq, **attrs):
        self.add_places_from(seq, **attrs)

    def new_place(self, name, **kwargs):
        if name not in self.node:
            self.add_place(name, **kwargs)
            return self.place[name]
        if isinstance(name, str):
            n = 0
            while name + str(n) in self.node:
                n += 1
            self.add_place(name + str(n), **kwargs)
            return self.place[name]
        raise KeyError("Already have a node named {}".format(name))

    def new_node(self, name, **kwargs):
        return self.new_place(name, **kwargs)

    @abstractmethod
    def add_thing(self, name, location, **kwargs): pass

    @abstractmethod
    def add_things_from(self, seq, **attrs): pass

    def new_thing(
            self, name, location, **kwargs
    ):
        if name not in self.node:
            self.add_thing(name, location, **kwargs)
            return self.thing[name]
        if isinstance(name, str):
            n = 0
            while name + str(n) in self.node:
                n += 1
            self.add_thing(name + str(n), location, **kwargs)
            return self.thing[name]
        raise KeyError("Already have a thing named {}".format(name))

    @abstractmethod
    def thing2place(self, name): pass

    @abstractmethod
    def place2thing(self, name, location): pass

    @abstractmethod
    def add_portal(self, orig, dest, symmetrical=False, **kwargs): pass

    def add_edge(self, orig, dest, **kwargs):
        self.add_portal(orig, dest, **kwargs)

    def new_portal(self, orig, dest, symmetrical=False, **kwargs):
        self.add_portal(orig, dest, symmetrical, **kwargs)
        return self.portal[orig][dest]

    @abstractmethod
    def add_portals_from(self, seq, **attrs): pass

    def add_edges_from(self, seq, **attrs):
        self.add_portals_from(seq, **attrs)

    def remove_portals_from(self, seq):
        for orig, dest in seq:
            del self.portal[orig][dest]

    def remove_edges_from(self, seq):
        self.remove_portals_from(seq)

    @abstractmethod
    def add_avatar(self, a, b=None): pass

    @abstractmethod
    def del_avatar(self, a, b=None): pass

    def __eq__(self, other):
        return isinstance(other, AbstractCharacter) and self.name == other.name

    def __iter__(self):
        return iter(self.node)

    def __len__(self):
        return len(self.node)

    def __contains__(self, k):
        return k in self.node

    def __getitem__(self, k):
        return self.node[k]

    def __setitem__(self, k, v):
        self.node[k] = v

    def __delitem__(self, k):
        del self.node[k]

    thing = SpecialMappingDescriptor('ThingMapping')
    place = SpecialMappingDescriptor('PlaceMapping')
    node = _node = SpecialMappingDescriptor('ThingPlaceMapping')
    portal = adj = succ = edge = _adj = _succ = SpecialMappingDescriptor('PortalSuccessorsMapping')
    preportal = pred = _pred = SpecialMappingDescriptor('PortalPredecessorsMapping')
    avatar = SpecialMappingDescriptor('AvatarGraphMapping')
    stat = getatt('graph')

    def historical(self, stat):
        from .query import StatusAlias
        return StatusAlias(
            entity=self.stat,
            stat=stat
        )

    def do(self, func, *args, **kwargs):
        """Apply the function to myself, and return myself.

        Look up the function in the database if needed. Pass it any
        arguments given, keyword or positional.

        Useful chiefly when chaining.

        """
        if not callable(func):
            func = getattr(self.engine.function, func)
        func(self, *args, **kwargs)
        return self

    def perlin(self, stat='perlin'):
        """Apply Perlin noise to my nodes, and return myself.

        I'll try to use the name of the node as its spatial position
        for this purpose, or use its stats 'x', 'y', and 'z', or skip
        the node if neither are available. z is assumed 0 if not
        provided for a node.

        Result will be stored in a node stat named 'perlin' by default.
        Supply the name of another stat to use it instead.

        """
        from math import floor
        p = self.engine.shuffle([
            151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7,
            225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190,
            6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117,
            35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136,
            171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146,
            158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41,
            55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80,
            73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116,
            188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226,
            250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207,
            206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
            119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43,
            172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178,
            185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144,
            12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
            192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50,
            45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72,
            243, 141, 128, 195, 78, 66, 215, 61, 156, 180
        ]) * 2

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(hsh, x, y, z):
            """CONVERT LO 4 BITS OF HASH CODE INTO 12 GRADIENT DIRECTIONS."""
            h = hsh & 15
            u = x if h < 8 else y
            v = y if h < 4 else x if h == 12 or h == 14 else z
            return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

        def noise(x, y, z):
            # FIND UNIT CUBE THAT CONTAINS POINT.
            X = int(x) & 255
            Y = int(y) & 255
            Z = int(z) & 255
            # FIND RELATIVE X, Y, Z OF POINT IN CUBE.
            x -= floor(x)
            y -= floor(y)
            z -= floor(z)
            # COMPUTE FADE CURVES FOR EACH OF X, Y, Z.
            u = fade(x)
            v = fade(y)
            w = fade(z)
            # HASH COORDINATES OF THE 8 CUBE CORNERS,
            A = p[X] + Y
            AA = p[A] + Z
            AB = p[A+1] + Z
            B = p[X+1] + y
            BA = p[B] + Z
            BB = p[B+1] + Z
            # AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
            return lerp(
                w,
                lerp(
                    v,
                    lerp(
                        u,
                        grad(p[AA], x, y, z),
                        grad(p[BA], x-1, y, z)
                    ),
                    lerp(
                        u,
                        grad(p[AB], x, y-1, z),
                        grad(p[BB], x-1, y-1, z)
                    )
                ),
                lerp(
                    v,
                    lerp(
                        u,
                        grad(p[AA+1], x, y, z-1),
                        grad(p[BA+1], x-1, y, z-1)
                    ),
                    lerp(
                        u,
                        grad(p[AB+1], x, y-1, z-1),
                        grad(p[BB+1], x-1, y-1, z-1)
                    )
                )
            )

        for node in self.node.values():
            try:
                (x, y, z) = node.name
            except ValueError:
                try:
                    (x, y) = node.name
                    z = 0.0
                except ValueError:
                    try:
                        x = node['x']
                        y = node['y']
                        z = node.get('z', 0.0)
                    except KeyError:
                        continue
            x, y, z = map(float, (x, y, z))
            node[stat] = noise(x, y, z)

        return self

    def copy_from(self, g):
        """Copy all nodes and edges from the given graph into this.

        Return myself.

        """
        renamed = {}
        for k, v in g.node.items():
            ok = k
            if k in self.place:
                n = 0
                while k in self.place:
                    k = ok + (n,) if isinstance(ok, tuple) else (ok, n)
                    n += 1
            renamed[ok] = k
            self.place[k] = v
        if type(g) is nx.MultiDiGraph:
            g = nx.DiGraph(g)
        elif type(g) is nx.MultiGraph:
            g = nx.Graph(g)
        if type(g) is nx.DiGraph:
            for u, v in g.edges:
                self.edge[renamed[u]][renamed[v]] = g.adj[u][v]
        else:
            assert type(g) is nx.Graph
            for u, v, d in g.edges.data():
                self.add_portal(renamed[u], renamed[v], symmetrical=True, **d)
        return self

    def become(self, g):
        """Erase all my nodes and edges. Replace them with a copy of the graph
        provided.

        Return myself.

        """
        self.clear()
        self.copy_from(g)
        return self

    def balanced_tree(self, r, h):
        return self.copy_from(nx.balanced_tree(r, h))

    def barbell_graph(self, m1, m2):
        return self.copy_from(nx.barbell_graph(m1, m2))

    def complete_graph(self, n):
        return self.copy_from(nx.complete_graph(n))

    def circular_ladder_graph(self, n):
        return self.copy_from(nx.circular_ladder_graph(n))

    def cycle_graph(self, n):
        return self.copy_from(nx.cycle_graph(n))

    def empty_graph(self, n):
        return self.copy_from(nx.empty_graph(n))

    def grid_2d_graph(self, m, n, periodic=False):
        return self.copy_from(nx.grid_2d_graph(m, n, periodic))

    def grid_2d_8graph(self, m, n):
        """Make a 2d graph that's connected 8 ways, enabling diagonal movement"""
        me = nx.Graph()
        node = me.node
        add_node = me.add_node
        add_edge = me.add_edge
        for i in range(m):
            for j in range(n):
                add_node((i, j))
                if i > 0:
                    add_edge((i, j), (i-1, j))
                    if j > 0:
                        add_edge((i, j), (i-1, j-1))
                if j > 0:
                    add_edge((i, j), (i, j-1))
                if (i - 1, j + 1) in node:
                    add_edge((i, j), (i-1, j+1))
        return self.copy_from(me)

    def grid_graph(self, dim, periodic=False):
        return self.copy_from(nx.grid_graph(dim, periodic))

    def ladder_graph(self, n):
        return self.copy_from(nx.ladder_graph(n))

    def lollipop_graph(self, m, n):
        return self.copy_from(nx.lollipop_graph(m, n))

    def path_graph(self, n):
        return self.copy_from(nx.path_graph(n))

    def star_graph(self, n):
        return self.copy_from(nx.star_graph(n))

    def wheel_graph(self, n):
        return self.copy_from(nx.wheel_graph(n))

    def fast_gnp_random_graph(self, n, p, seed=None):
        return self.copy_from(nx.fast_gnp_random_graph(
            n, p, seed, directed=True
        ))

    def gnp_random_graph(self, n, p, seed=None):
        return self.copy_from(nx.gnp_random_graph(n, p, seed, directed=True))

    def gnm_random_graph(self, n, m, seed=None):
        return self.copy_from(nx.gnm_random_graph(n, m, seed, directed=True))

    def erdos_renyi_graph(self, n, p, seed=None):
        return self.copy_from(nx.erdos_renyi_graph(n, p, seed, directed=True))

    def binomial_graph(self, n, p, seed=None):
        return self.erdos_renyi_graph(n, p, seed)

    def newman_watts_strogatz_graph(self, n, k, p, seed=None):
        return self.copy_from(nx.newman_watts_strogatz_graph(n, k, p, seed))

    def watts_strogatz_graph(self, n, k, p, seed=None):
        return self.copy_from(nx.watts_strogatz_graph(n, k, p, seed))

    def connected_watts_strogatz_graph(self, n, k, p, tries=100, seed=None):
        return self.copy_from(nx.connected_watts_strogatz_graph(
            n, k, p, tries, seed
        ))

    def random_regular_graph(self, d, n, seed=None):
        return self.copy_from(nx.random_regular_graph(d, n, seed))

    def barabasi_albert_graph(self, n, m, seed=None):
        return self.copy_from(nx.barabasi_albert_graph(n, m, seed))

    def powerlaw_cluster_graph(self, n, m, p, seed=None):
        return self.copy_from(nx.powerlaw_cluster_graph(n, m, p, seed))

    def duplication_divergence_graph(self, n, p, seed=None):
        return self.copy_from(nx.duplication_divergence_graph(n, p, seed))

    def random_lobster(self, n, p1, p2, seed=None):
        return self.copy_from(nx.random_lobster(n, p1, p2, seed))

    def random_shell_graph(self, constructor, seed=None):
        return self.copy_from(nx.random_shell_graph(constructor, seed))

    def random_powerlaw_tree(self, n, gamma=3, seed=None, tries=100):
        return self.copy_from(nx.random_powerlaw_tree(n, gamma, seed, tries))

    def configuration_model(self, deg_sequence, seed=None):
        return self.copy_from(nx.configuration_model(deg_sequence, seed=seed))

    def directed_configuration_model(
            self,
            in_degree_sequence,
            out_degree_sequence,
            seed=None
    ):
        return self.copy_from(nx.directed_configuration_model(
            in_degree_sequence,
            out_degree_sequence,
            seed=seed
        ))

    def expected_degree_graph(self, w, seed=None, selfloops=True):
        return self.copy_from(nx.expected_degree_graph(w, seed, selfloops))

    def havel_hakmi_graph(self, deg_sequence):
        return self.copy_from(nx.havel_hakimi_graph(deg_sequence))

    def directed_havel_hakmi_graph(
            self,
            in_degree_sequence,
            out_degree_sequence
    ):
        return self.copy_from(nx.directed_havel_hakmi_graph(
            in_degree_sequence,
            out_degree_sequence
        ))

    def degree_sequence_tree(self, deg_sequence):
        return self.copy_from(nx.degree_sequence_tree(deg_sequence))

    def random_degree_sequence_graph(self, sequence, seed=None, tries=10):
        return self.copy_from(nx.random_degree_sequence_graph(
            sequence, seed, tries
        ))

    def random_clustered_graph(self, joint_degree_sequence, seed=None):
        return self.copy_from(nx.random_clustered_graph(
            joint_degree_sequence, seed=seed
        ))

    def gn_graph(self, n, kernel=None, seed=None):
        return self.copy_from(nx.gn_graph(n, kernel, seed=seed))

    def gnr_graph(self, n, p, seed=None):
        return self.copy_from(nx.gnr_graph(n, p, seed=seed))

    def gnc_graph(self, n, seed=None):
        return self.copy_from(nx.gnc_graph(n, seed=seed))

    def scale_free_graph(
            self, n,
            alpha=0.41,
            beta=0.54,
            gamma=0.05,
            delta_in=0.2,
            delta_out=0,
            seed=None
    ):
        return self.copy_from(nx.scale_free_graph(
            n,
            alpha,
            beta,
            gamma,
            delta_in,
            delta_out,
            seed=seed
        ))

    def random_geometric_graph(self, n, radius, dim=2, pos=None):
        return self.copy_from(nx.random_geometric_graph(n, radius, dim, pos))

    def geographical_threshold_graph(
            self, n, theta,
            alpha=2,
            dim=2,
            pos=None,
            weight=None
    ):
        return self.copy_from(nx.geographical_threshold_graph(
            n, theta, alpha, dim, pos, weight
        ))

    def waxman_graph(
            self, n,
            alpha=0.4,
            beta=0.1,
            L=None,
            domain=(0, 0, 1, 1)
    ):
        return self.copy_from(nx.waxman_graph(
            n, alpha, beta, L, domain
        ))

    def navigable_small_world_graph(self, n, p=1, q=1, r=2, dim=2, seed=None):
        return self.copy_from(nx.navigable_small_world_graph(
            n, p, q, r, dim, seed
        ))

    def line_graph(self):
        lg = nx.line_graph(self)
        self.clear()
        return self.copy_from(lg)

    def ego_graph(
            self, n,
            radius=1,
            center=True,
            distance=None
    ):
        return self.become(nx.ego_graph(
            self, n,
            radius,
            center,
            False,
            distance
        ))

    def stochastic_graph(self, weight='weight'):
        nx.stochastic_graph(self, copy=False, weight=weight)
        return self

    def uniform_random_intersection_graph(self, n, m, p, seed=None):
        return self.copy_from(nx.uniform_random_intersection_graph(
            n, m, p, seed=seed
        ))

    def k_random_intersection_graph(self, n, m, k, seed=None):
        return self.copy_from(nx.k_random_intersection_graph(
            n, m, k, seed=seed
        ))

    def general_random_intersection_graph(self, n, m, p, seed=None):
        return self.copy_from(
            nx.general_random_intersection_graph(n, m, p, seed=seed)
        )

    def caveman_graph(self, l, k):
        return self.copy_from(nx.caveman_graph(l, k))

    def connected_caveman_graph(self, l, k):
        return self.copy_from(nx.connected_caveman_graph(l, k))

    def relaxed_caveman_graph(self, l, k, p, seed=None):
        return self.copy_from(nx.relaxed_caveman_graph(l, k, p, seed=seed))

    def random_partition_graph(self, sizes, p_in, p_out, seed=None):
        return self.copy_from(nx.random_partition_graph(
            sizes, p_in, p_out, seed=seed, directed=True
        ))

    def planted_partition_graph(self, l, k, p_in, p_out, seed=None):
        return self.copy_from(nx.planted_partition_graph(
            l, k, p_in, p_out, seed=seed, directed=True
        ))

    def gaussian_random_partition_graph(self, n, s, v, p_in, p_out, seed=None):
        return self.copy_from(nx.gaussian_random_partition_graph(
            n, s, v, p_in, p_out, seed=seed
        ))

    def _lookup_comparator(self, comparator):
        if callable(comparator):
            return comparator
        ops = {
            'ge': ge,
            'gt': gt,
            'le': le,
            'lt': lt,
            'eq': eq
        }
        if comparator in ops:
            return ops[comparator]
        return getattr(self.engine.function, comparator)

    def cull_nodes(self, stat, threshold=0.5, comparator=ge):
        """Delete nodes whose stat >= ``threshold`` (default 0.5).

        Optional argument ``comparator`` will replace >= as the test
        for whether to cull. You can use the name of a stored function.

        """
        comparator = self._lookup_comparator(comparator)
        dead = [
            name for name, node in self.node.items()
            if stat in node and comparator(node[stat], threshold)
        ]
        self.remove_nodes_from(dead)
        return self

    def cull_portals(self, stat, threshold=0.5, comparator=ge):
        """Delete portals whose stat >= ``threshold`` (default 0.5).

        Optional argument ``comparator`` will replace >= as the test
        for whether to cull. You can use the name of a stored function.

        """
        comparator = self._lookup_comparator(comparator)
        dead = []
        for u in self.portal:
            for v in self.portal[u]:
                if stat in self.portal[u][v] and comparator(
                        self.portal[u][v][stat], threshold
                ):
                    dead.append((u, v))
        self.remove_edges_from(dead)
        return self

    def cull_edges(self, stat, threshold=0.5, comparator=ge):
        """Delete edges whose stat >= ``threshold`` (default 0.5).

        Optional argument ``comparator`` will replace >= as the test
        for whether to cull. You can use the name of a stored function.

        """
        return self.cull_portals(stat, threshold, comparator)


class CharRuleMapping(RuleMapping):

    """Wraps one of a character's rulebooks so you can get its rules by name.

    You can access the rules in this either dictionary-style or as
    attributes. This is for convenience if you want to get at a rule's
    decorators, eg. to add an Action to the rule.

    Using this as a decorator will create a new rule, named for the
    decorated function, and using the decorated function as the
    initial Action.

    Using this like a dictionary will let you create new rules,
    appending them onto the underlying :class:`RuleBook`; replace one
    rule with another, where the new one will have the same index in
    the :class:`RuleBook` as the old one; and activate or deactivate
    rules. The name of a rule may be used in place of the actual rule,
    so long as the rule already exists.

    You can also set a rule active or inactive by setting it to
    ``True`` or ``False``, respectively. Inactive rules are still in
    the rulebook, but won't be followed.

    """

    def __init__(self, character, rulebook, booktyp):
        """Initialize as usual for the ``rulebook``, mostly.

        My ``character`` property will be the one passed in, and my
        ``_table`` will be the ``booktyp`` with ``"_rules"`` appended.

        """
        super().__init__(rulebook.engine, rulebook)
        self.character = character
        self._table = booktyp + "_rules"


class RuleFollower(BaseRuleFollower):

    """Mixin class. Has a rulebook, which you can get a RuleMapping into."""

    def _get_rule_mapping(self):
        return CharRuleMapping(
            self.character,
            self.rulebook,
            self._book
        )

    @abstractmethod
    def _get_rulebook_cache(self):
        pass

    def _get_rulebook_name(self):
        try:
            return self._get_rulebook_cache().retrieve(
                self.character.name, *self.engine.btt()
            )
        except KeyError:
            return self.character.name, self._book

    def _set_rulebook_name(self, n):
        branch, turn, tick = self.engine.nbtt()
        self.engine.query._set_rulebook_on_character(self._book, self.character.name, branch, turn, tick, n)
        self._get_rulebook_cache().store(self.character.name, branch, turn, tick, n)

    def __contains__(self, k):
        return self.engine._active_rules_cache.contains_key(
            self._get_rulebook_name(), *self.engine.btt()
        )


class SenseFuncWrap(object):

    """Wrapper for a sense function that looks it up in the code store if
    provided with its name, and prefills the first two arguments.

    """

    engine = getatt('character.engine')

    def __init__(self, character, fun):
        """Store the character and the function.

        Look up the function in the engine's ``sense`` function store,
        if needed.

        """
        if not callable(fun):
            raise TypeError("function is not callable")
        self.character = character
        if isinstance(fun, str):
            self.fun = self.engine.sense[fun]
        else:
            self.fun = fun

    def __call__(self, observed):
        """Call the function, prefilling the engine and observer arguments."""
        if isinstance(observed, str):
            observed = self.engine.character[observed]
        return self.fun(self.engine, self.character, Facade(observed))


class CharacterSense(object):

    """Mapping for when you've selected a sense for a character to use
    but haven't yet specified what character to look at

    """

    engine = getatt('container.engine')
    observer = getatt('container.character')

    def __init__(self, container, sensename):
        """Store the container and the name of the sense."""
        self.container = container
        self.sensename = sensename

    @property
    def func(self):
        """Return the function most recently associated with this sense."""
        fn = self.engine.query.sense_func_get(
            self.observer.name,
            self.sensename,
            *self.engine.btt()
        )
        if fn is not None:
            return SenseFuncWrap(self.observer, fn)

    def __call__(self, observed):
        """Call my sense function and make sure it returns the right type,
        then return that.

        """
        r = self.func(observed)
        if not (
                isinstance(r, Character) or
                isinstance(r, Facade)
        ):
            raise TypeError(
                "Sense function did not return a character-like object"
            )
        return r


class CharacterSenseMapping(MutableMappingUnwrapper, Signal):

    """Used to view other Characters as seen by one, via a particular sense."""

    # TODO: cache senses properly
    _book = "character"

    engine = getatt('character.engine')

    def __init__(self, character):
        """Store the character."""
        super().__init__()
        self.character = character

    def __iter__(self):
        """Iterate over active sense names."""
        yield from self.engine.query.sense_active_items(
            self.character.name, *self.engine.btt()
        )

    def __len__(self):
        """Count active senses."""
        n = 0
        for sense in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Get a :class:`CharacterSense` named ``k`` if it exists."""
        if not self.engine.query.sense_is_active(
                self.character.name,
                k,
                *self.engine.btt()
        ):
            raise KeyError("Sense isn't active or doesn't exist")
        return CharacterSense(self.character, k)

    def __setitem__(self, k, v):
        """Use the function for the sense from here on out."""
        if isinstance(v, str):
            funn = v
        else:
            funn = v.__name__
        if funn not in self.engine.sense:
            if not isinstance(v, Callable):
                raise TypeError("Not a function")
            self.engine.sense[funn] = v
        branch, turn, tick = self.engine.btt()
        # TODO: cache
        self.engine.query.sense_fun_set(
            self.character.name,
            k,
            branch,
            turn,
            tick,
            funn,
            True
        )
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        """Stop having the given sense."""
        branch, turn, tick = self.engine.btt()
        # TODO: cache
        self.engine.query.sense_set(
            self.character.name,
            k,
            branch,
            turn,
            tick,
            False
        )
        self.send(self, key=k, val=None)

    def __call__(self, fun, name=None):
        """Decorate the function so it's mine now."""
        if not isinstance(fun, Callable):
            raise TypeError(
                "I need a function here"
            )
        if name is None:
            name = fun.__name__
        self[name] = fun


class FacadeEntity(MutableMapping, Signal):
    def __init__(self, mapping, **kwargs):
        super().__init__()
        self.facade = mapping.facade
        self._real = mapping
        self._patch = kwargs
        self._masked = set()

    def __contains__(self, item):
        if item in self._masked:
            return False
        return item in self._patch or item in self._real

    def __iter__(self):
        seen = set()
        for k in self._real:
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if (
                    k not in self._masked and
                    k not in seen
            ):
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("{} has been masked.".format(k))
        if k in self._patch:
            return self._patch[k]
        ret = self._real[k]
        if hasattr(ret, 'unwrap'):  # a wrapped mutable object from the allegedb.wrap module
            ret = ret.unwrap()
            self._patch[k] = ret  # changes will be reflected in the facade but not the original
        return ret

    def __setitem__(self, k, v):
        if k == 'name':
            raise TypeError("Can't change names")
        self._masked.discard(k)
        self._patch[k] = v
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        self._masked.add(k)
        self.send(self, key=k, val=None)


class FacadePlace(FacadeEntity):

    """Lightweight analogue of Place for Facade use."""

    @property
    def name(self):
        return self['name']

    def contents(self):
        for thing in self.facade.thing.values():
            if thing.container is self:
                yield thing

    def __init__(self, mapping, real_or_name, **kwargs):
        super().__init__(mapping, **kwargs)
        if isinstance(real_or_name, Place) or \
           isinstance(real_or_name, FacadePlace):
            self._real = real_or_name
        else:
            self._real = {'name': real_or_name}


class FacadeThing(FacadeEntity):
    @property
    def name(self):
        return self._real['name']

    def __init__(self, mapping, real_or_name, **kwargs):
        location = kwargs.pop('location', None)
        super().__init__(mapping, **kwargs)
        if location is None and not (
                isinstance(real_or_name, Thing) or
                isinstance(real_or_name, FacadeThing)
        ):
            raise TypeError(
                "FacadeThing needs to wrap a real Thing or another "
                "FacadeThing, or have a location of its own."
            )
        self._real = {
            'name': real_or_name.name if hasattr(real_or_name, 'name')
            else real_or_name,
            'location': location
        }

    @property
    def location(self):
        try:
            return self.facade.node[self['location']]
        except KeyError:
            return None


class FacadePortal(FacadeEntity):
    """Lightweight analogue of Portal for Facade use."""

    def __init__(self, mapping, other, **kwargs):
        super().__init__(mapping, **kwargs)
        if hasattr(mapping, 'orig'):
            self.orig = mapping.orig
            self.dest = other
        else:
            self.dest = mapping.dest
            self.orig = other
        try:
            self._real = self.facade.character.portal[self.orig][self.dest]
        except (KeyError, AttributeError):
            self._real = {}

    def __setitem__(self, k, v):
        if k in ('origin', 'destination'):
            raise TypeError("Portals have fixed origin and destination")
        super().__setitem__(k, v)

    @property
    def origin(self):
        return self.facade.node[self.orig]

    @property
    def destination(self):
        return self.facade.node[self.dest]


class FacadeEntityMapping(MutableMappingUnwrapper, Signal):

    """Mapping that contains entities in a Facade.

    All the entities are of the same type, ``facadecls``, possibly
    being distorted views of entities of the type ``innercls``.

    """
    def _make(self, k, v):
        return self.facadecls(self, k, **v)

    engine = getatt('facade.engine')

    def __init__(self, facade):
        """Store the facade."""
        super().__init__()
        self.facade = facade
        self._patch = {}
        self._masked = set()

    def __contains__(self, k):
        return (
            k not in self._masked and (
                k in self._patch or
                k in self._get_inner_map()
            )
        )

    def __iter__(self):
        seen = set()
        for k in self._get_inner_map():
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if k not in seen:
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k not in self:
            raise KeyError
        if k not in self._patch:
            self._patch[k] = self.facadecls(self, k, **self._get_inner_map()[k])
        return self._patch[k]

    def __setitem__(self, k, v):
        if not isinstance(v, self.facadecls):
            v = self._make(k, v)
        self._masked.discard(k)
        self._patch[k] = v
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        self._masked.add(k)
        self.send(self, key=k, val=None)


class FacadePortalSuccessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, origname):
        super().__init__(facade)
        self.orig = origname

    def _make(self, k, v):
        return self.facadecls(self, k, **v)

    def _get_inner_map(self):
        try:
            return self.facade.character.portal[self.orig]
        except AttributeError:
            return {}


class FacadePortalPredecessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, destname):
        super().__init__(facade)
        self.dest = destname

    def _make(self, k, v):
        return self.facadecls(self.facade.portal[k], self.dest, v)

    def _get_inner_map(self):
        try:
            return self.facade.character.preportal[self._destname]
        except AttributeError:
            return {}


class FacadePortalMapping(FacadeEntityMapping):
    def __getitem__(self, node):
        if node in self._masked:
            raise KeyError("Node {} is in the inner Character, but has been masked".format(node))
        if node not in self:
            raise KeyError("No such node: {}".format(node))
        if node not in self._patch:
            self._patch[node] = self.cls(self.facade, node)
        return self._patch[node]

    def __setitem__(self, node, value):
        self._masked.discard(node)
        v = self.cls(self.facade, node)
        v.update(value)
        self._patch[node] = v

    def __delitem__(self, node):
        self._masked.add(node)


class Facade(AbstractCharacter, nx.DiGraph):
    engine = getatt('character.engine')

    def __getstate__(self):
        ports = {}
        for o in self.portal:
            if o not in ports:
                ports[o] = {}
            for d in self.portal[o]:
                ports[o][d] = dict(self.portal[o][d])
        things = {k: dict(v) for (k, v) in self.thing.items()}
        places = {k: dict(v) for (k, v) in self.place.items()}
        return things, places, ports

    def __setstate__(self, state):
        self.character = None
        self.thing, self.place, self.portal = state

    def add_places_from(self, seq, **attrs):
        for place in seq:
            self.add_place(place, **attrs)

    def add_things_from(self, seq, **attrs):
        for thing in seq:
            self.add_thing(thing, **attrs)

    def thing2place(self, name):
        self.place[name] = self.thing.pop(name)

    def place2thing(self, name, location):
        it = self.place.pop(name)
        it['location'] = location
        self.thing[name] = it

    def add_portals_from(self, seq, **attrs):
        for it in seq:
            self.add_portal(*it, **attrs)

    def del_avatar(self, a, b=None):
        raise NotImplementedError("Facades don't have avatars")

    def add_place(self, name, **kwargs):
        self.place[name] = kwargs

    def add_node(self, name, **kwargs):
        self.place[name] = kwargs

    def add_thing(self, name, **kwargs):
        self.thing[name] = kwargs

    def add_portal(self, orig, dest, symmetrical=False, **kwargs):
        self.portal[orig][dest] = kwargs
        if symmetrical:
            mirror = dict(kwargs)
            mirror['is_mirror'] = True
            self.portal[dest][orig] = mirror

    def add_edge(self, orig, dest, **kwargs):
        self.add_portal(orig, dest, **kwargs)

    def add_avatar(self, a, b=None):
        raise NotImplementedError("Facades don't have avatars")

    def __init__(self, character=None):
        """Store the character."""
        self.character = character
        self.graph = self.StatMapping(self)

    class ThingMapping(FacadeEntityMapping):
        facadecls = FacadeThing
        innercls = Thing

        def _get_inner_map(self):
            try:
                return self.facade.character.thing
            except AttributeError:
                return {}

    class PlaceMapping(FacadeEntityMapping):
        facadecls = FacadePlace
        innercls = Place

        def _get_inner_map(self):
            try:
                return self.facade.character.place
            except AttributeError:
                return {}

    def ThingPlaceMapping(self, *args):
        return CompositeDict(self.thing, self.place)

    class PortalSuccessorsMapping(FacadePortalMapping):
        cls = FacadePortalSuccessors

        def __contains__(self, item):
            return item in self.facade.node

        def _get_inner_map(self):
            try:
                return self.facade.character.portal
            except AttributeError:
                return {}

    class PortalPredecessorsMapping(FacadePortalMapping):
        cls = FacadePortalPredecessors

        def __contains__(self, item):
            return item in self.facade.node

        def _get_inner_map(self):
            try:
                return self.facade.character.preportal
            except AttributeError:
                return {}

    class StatMapping(MutableMappingUnwrapper, Signal):
        def __init__(self, facade):
            super().__init__()
            self.facade = facade
            self._patch = {}
            self._masked = set()

        def __iter__(self):
            seen = set()
            if hasattr(self.facade.character, 'graph'):
                for k in self.facade.character.graph:
                    if k not in self._masked:
                        yield k
                    seen.add(k)
            for k in self._patch:
                if k not in seen:
                    yield k

        def __len__(self):
            n = 0
            for k in self:
                n += 1
            return n

        def __contains__(self, k):
            if k in self._masked:
                return False
            if hasattr(self.facade.character, 'graph') and k in self.facade.character.graph:
                return True
            return k in self._patch

        def __getitem__(self, k):
            if k in self._masked:
                raise KeyError("masked")
            if k not in self._patch and hasattr(self.facade.character, 'graph'):
                ret = self.facade.character.graph[k]
                if not hasattr(ret, 'unwrap'):
                    return ret
                self._patch[k] = ret.unwrap()
            return self._patch[k]

        def __setitem__(self, k, v):
            self._masked.discard(k)
            self._patch[k] = v
            self.send(self, key=k, val=v)

        def __delitem__(self, k):
            self._masked.add(k)
            self.send(self, key=k, val=None)


class Character(DiGraph, AbstractCharacter, RuleFollower):
    """A graph that follows game rules and has a containment hierarchy.

    Nodes in a Character are subcategorized into Things and
    Places. Things have locations, and those locations may be Places
    or other Things.

    Characters may have avatars in other Characters. These are just
    nodes. You can apply rules to a Character's avatars, and thus to
    any collection of nodes you want, perhaps in many different
    Characters. But you may want a Character to have exactly one
    avatar, representing their location in physical space -- the
    Character named 'physical'. So when a Character has only one
    avatar, you can treat the ``avatar`` property as an alias of the
    avatar.


    """
    _book = "character"

    @property
    def character(self):
        return self

    def _get_rulebook_cache(self):
        return self.engine._characters_rulebooks_cache

    def __repr__(self):
        return "{}.character[{}]".format(repr(self.engine), repr(self.name))

    def __init__(self, engine, name, data=None, *, init_rulebooks=True, **attr):
        """Store engine and name, and set up mappings for Thing, Place, and
        Portal

        """
        from allegedb.cache import FuturistWindowDict, PickyDefaultDict
        super().__init__(engine, name, data, **attr)
        self._avatars_cache = PickyDefaultDict(FuturistWindowDict)
        if not init_rulebooks:
            return
        cachemap = {
            'character': engine._characters_rulebooks_cache,
            'avatar': engine._avatars_rulebooks_cache,
            'character_thing': engine._characters_things_rulebooks_cache,
            'character_place': engine._characters_places_rulebooks_cache,
            'character_portal': engine._characters_portals_rulebooks_cache
        }
        for rulebook, cache in cachemap.items():
            branch, turn, tick = engine.nbtt()
            rulebook_or_name = attr.get(rulebook, (name, rulebook))
            rulebook_name = getattr(rulebook_or_name, 'name', rulebook_or_name)
            engine.query._set_rulebook_on_character(rulebook, name, branch, turn, tick, rulebook_name)
            cache.store((name, rulebook), branch, turn, tick, rulebook_name)

    class ThingMapping(MutableMappingUnwrapper, RuleFollower, Signal):
        """:class:`Thing` objects that are in a :class:`Character`"""
        _book = "character_thing"

        engine = getatt('character.engine')
        name = getatt('character.name')

        def _get_rulebook_cache(self):
            return self.engine._characters_things_rulebooks_cache

        def __init__(self, character):
            """Store the character and initialize cache."""
            super().__init__()
            self.character = character

        def __iter__(self):
            cache = self.engine._things_cache
            char = self.name
            branch, turn, tick = self.engine.btt()
            for key in cache.iter_keys(char, branch, turn, tick):
                try:
                    if cache.retrieve(char, key, branch, turn, tick) is not None:
                        yield key
                except KeyError:
                    continue

        def __contains__(self, thing):
            args = self.character.name, thing, *self.engine.btt()
            cache = self.engine._things_cache
            return cache.contains_key(*args) and cache.retrieve(*args) is not None

        def __len__(self):
            return self.engine._things_cache.count_keys(
                self.character.name, *self.engine.btt()
            )

        def __getitem__(self, thing):
            if thing not in self:
                raise KeyError("No such thing: {}".format(thing))
            return self._make_thing(thing)

        def _make_thing(self, thing, val=None):
            cache = self.engine._node_objs
            if isinstance(val, Thing):
                th = cache[self.name, thing] = val
            elif (self.name, thing) in cache:
                th = cache[(self.name, thing)]
                if type(th) is not Thing:
                    th = cache[self.name, thing] = Thing(self.character, thing)
            else:
                th = cache[(self.name, thing)] = Thing(self.character, thing)
            return th

        def __setitem__(self, thing, val):
            if not isinstance(val, Mapping):
                raise TypeError('Things are made from Mappings')
            if 'location' not in val:
                raise ValueError('Thing needs location')
            created = thing not in self
            self.engine._exist_node(self.character.name, thing)
            self.engine._set_thing_loc(
                self.character.name,
                thing,
                val['location']
            )
            th = self._make_thing(thing, val)
            th.clear()
            th.update(val)
            if created:
                self.send(self, thing_name=thing, exists=True)

        def __delitem__(self, thing):
            self[thing].delete()
            self.send(self, thing_name=thing, exists=False)

        def __repr__(self):
            return "{}.character[{}].thing".format(repr(self.engine), repr(self.name))

    class PlaceMapping(MutableMappingUnwrapper, RuleFollower, Signal):
        """:class:`Place` objects that are in a :class:`Character`"""
        _book = "character_place"

        engine = getatt('character.engine')
        name = getatt('character.name')

        def _get_rulebook_cache(self):
            return self.engine._characters_places_rulebooks_cache

        def __init__(self, character):
            """Store the character."""
            super().__init__()
            self.character = character

        def __iter__(self):
            for node in self.engine._nodes_cache.iter_entities(
                    self.character.name, *self.engine.btt()
            ):
                if not self.engine._things_cache.contains_entity(
                        self.character.name, node, *self.engine.btt()
                ):
                    yield node

        def __len__(self):
            return self.engine._nodes_cache.count_entities(
                self.character.name, *self.engine.btt()
            ) - self.engine._things_cache.count_entities(
                self.character.name, *self.engine.btt()
            )

        def __contains__(self, place):
            # TODO: maybe a special cache just for places and not just
            # nodes in general
            return (
                self.engine._nodes_cache.contains_entity(
                    self.character.name, place, *self.engine.btt()
                ) and not self.engine._things_cache.contains_entity(
                    self.character.name, place, *self.engine.btt()
                )
            )

        def __getitem__(self, place):
            if place not in self:
                raise KeyError("No such place: {}".format(place))
            cache = self.engine._node_objs
            if (self.name, place) not in cache or not isinstance(
                    cache[(self.name, place)], Place
            ):
                ret = cache[(self.name, place)] = Place(self.character, place)
                return ret
            return cache[(self.name, place)]

        def __setitem__(self, place, v):
            pl = self.engine._get_node(self.character, place)
            if not isinstance(pl, Place):
                raise KeyError("{} is not a place".format(place))
            pl.clear()
            pl.update(v)
            self.send(self, key=place, val=v)

        def __delitem__(self, place):
            self[place].delete()

        def __repr__(self):
            return "{}.character[{}].place".format(repr(self.engine), repr(self.name))

    class ThingPlaceMapping(GraphNodeMapping, Signal):
        """GraphNodeMapping but for Place and Thing"""
        _book = "character_node"

        character = getatt('graph')
        engine =  getatt('db')
        name = getatt('character.name')

        def __init__(self, character):
            """Store the character."""
            super().__init__(character)
            Signal.__init__(self)

        def __contains__(self, k):
            return self.engine._node_exists(self.character.name, k)

        def __getitem__(self, k):
            if k not in self:
                raise KeyError
            return self.engine._get_node(self.character, k)

        def __setitem__(self, k, v):
            self.character.place[k] = v

        def __delitem__(self, k):
            if k not in self:
                raise KeyError
            if self.engine._is_thing(
                self.character.name, k
            ):
                del self.character.thing[k]
            else:
                del self.character.place[k]
    node_map_cls = ThingPlaceMapping

    class PortalSuccessorsMapping(GraphSuccessorsMapping, RuleFollower):
        """Mapping of nodes that have at least one outgoing edge.

        Maps them to another mapping, keyed by the destination nodes,
        which maps to Portal objects.

        """
        _book = "character_portal"

        character = getatt('graph')
        engine = getatt('graph.engine')

        def _get_rulebook_cache(self):
            return self.engine._characters_portals_rulebooks_cache

        def __getitem__(self, orig):
            if self.engine._node_exists(
                    self.graph.name,
                    orig
            ):
                if orig not in self._cache:
                    self._cache[orig] = self.Successors(self, orig)
                return self._cache[orig]
            raise KeyError("No such node")

        def __setitem__(self, orig, val):
            if orig not in self._cache:
                self._cache[orig] = self.Successors(self, orig)
            sucs = self._cache[orig]
            sucs.clear()
            sucs.update(val)
            self.send(self, key=orig, val=sucs)

        def __delitem__(self, orig):
            super().__delitem__(orig)
            self.send(self, key=orig, val=None)

        class Successors(GraphSuccessorsMapping.Successors):
            """Mapping for possible destinations from some node."""

            engine = getatt('graph.engine')

            @staticmethod
            def send(self, **kwargs):
                """Call all listeners to ``dest`` and to my ``orig``."""
                super().send(self, **kwargs)
                self.container.send(self, **kwargs)

            def __getitem__(self, dest):
                if dest in self:
                    return self.engine._get_edge(self.graph, self.orig, dest, 0)
                raise KeyError("No such portal: {}->{}".format(
                    self.orig, dest
                ))

            def __setitem__(self, dest, value):
                self.engine._exist_edge(
                    self.graph.name,
                    self.orig,
                    dest
                )
                p = self.engine._get_edge(self.graph, self.orig, dest, 0)
                p.clear()
                p.update(value)
                self.send(self, key=dest, val=p)

            def __delitem__(self, dest):
                if dest not in self:
                    raise KeyError("No portal to {}".format(dest))
                self[dest].delete()
    adj_cls = PortalSuccessorsMapping

    class PortalPredecessorsMapping(
            DiGraphPredecessorsMapping,
            RuleFollower
    ):
        """Mapping of nodes that have at least one incoming edge.

        Maps to another mapping keyed by the origin nodes, which maps to
        Portal objects.

        """
        _book = "character_portal"

        def _get_rulebook_cache(self):
            return self.engine._characters_portals_rulebooks_cache

        class Predecessors(DiGraphPredecessorsMapping.Predecessors):
            """Mapping of possible origins from some destination."""
            def __setitem__(self, orig, value):
                key = (self.graph.name, orig, self.dest)
                if key not in self.db._portal_objs:
                    self.db._portal_objs[key] = Portal(
                        self.graph,
                        orig,
                        self.dest
                    )
                p = self.db._portal_objs[key]
                p.clear()
                p.update(value)
                p.engine._exist_edge(self.graph.name, self.dest, orig)
    pred_cls = PortalPredecessorsMapping

    class AvatarGraphMapping(Mapping, RuleFollower):
        """A mapping of other characters in which one has an avatar.

        Maps to a mapping of the avatars themselves, unless there's
        only one other character you have avatars in, in which case
        this maps to those.

        If you have only one avatar anywhere, you can pretend this
        is that entity.

        """
        _book = "avatar"

        engine = getatt('character.engine')
        name = getatt('character.name')

        def _get_rulebook_cache(self):
            return self.engine._avatars_rulebooks_cache

        def __init__(self, char):
            """Remember my character."""
            self.character = char
            self._char_av_cache = {}

        def __call__(self, av):
            """Add the avatar. It must be an instance of Place or Thing."""
            if av.__class__ not in (Place, Thing):
                raise TypeError("Only Things and Places may be avatars")
            self.character.add_avatar(av.name, av.character.name)

        def __iter__(self):
            """Iterate over every avatar graph that has at least one avatar node
            in it presently

            """
            return iter(self.engine._avatarness_cache.get_char_graphs(
                self.character.name, *self.engine.btt()
            ))

        def __contains__(self, k):
            return k in self.engine._avatarness_cache.get_char_graphs(
                self.character.name, *self.engine.btt()
            )

        def __len__(self):
            """Number of graphs in which I have an avatar."""
            return len(self.engine._avatarness_cache.get_char_graphs(
                self.character.name, *self.engine.btt()
            ))

        def _get_char_av_cache(self, g):
            if g not in self:
                raise KeyError
            if g not in self._char_av_cache:
                self._char_av_cache[g] = self.CharacterAvatarMapping(self, g)
            return self._char_av_cache[g]

        def __getitem__(self, g):
            return self._get_char_av_cache(g)

        @property
        def node(self):
            """If I have avatars in only one graph, return a map of them.

            Otherwise, raise AttributeError.

            """
            try:
                return self._get_char_av_cache(
                    self.engine._avatarness_cache.get_char_only_graph(
                        self.character.name, *self.engine.btt()
                    )
                )
            except KeyError:
                raise AttributeError(
                    "I have no avatar, or I have avatars in many graphs"
                )

        @property
        def only(self):
            """If I have only one avatar, return it.

            Otherwise, raise AttributeError.

            """
            try:
                charn, noden = self.engine._avatarness_cache.get_char_only_av(
                    self.character.name, *self.engine.btt()
                )
                return self.engine._get_node(self.engine.character[charn], noden)
            except KeyError:
                raise AttributeError(
                    "I have no avatar, or more than one avatar"
                )

        class CharacterAvatarMapping(Mapping):
            """Mapping of avatars of one Character in another Character."""
            def __init__(self, outer, graphn):
                """Store the character and the name of the "graph", ie. the other
                character.

                """
                self.character = outer.character
                self.engine = outer.engine
                self.name = outer.name
                self.graph = graphn

            def __iter__(self):
                """Iterate over the names of all the presently existing nodes in the
                graph that are avatars of the character

                """
                return iter(self.engine._avatarness_cache.get_char_graph_avs(
                    self.name, self.graph, *self.engine.btt()
                ))

            def __contains__(self, av):
                return av in self.engine._avatarness_cache.get_char_graph_avs(
                    self.name, self.graph, *self.engine.btt()
                )

            def __len__(self):
                """Number of presently existing nodes in the graph that are avatars of
                the character"""
                return len(self.engine._avatarness_cache.get_char_graph_avs(
                    self.name, self.graph, *self.engine.btt()
                ))

            def __getitem__(self, av):
                if av in self:
                    return self.engine._get_node(self.engine.character[self.graph], av)
                raise KeyError("No avatar: {}".format(av))

            @property
            def only(self):
                mykey = singleton_get(self.keys())
                if mykey is None:
                    raise AttributeError("No avatar, or more than one")
                return self.engine._get_node(self.engine.character[self.graph], mykey)

            def __setitem__(self, k, v):
                mykey = singleton_get(self.keys())
                if mykey is None:
                    raise AmbiguousAvatarError(
                        "More than one avatar in {}; "
                        "be more specific to set the stats of one.".format(
                            self.graph
                        )
                    )
                self.engine._get_node(self.graph, mykey)[k] = v

            def __repr__(self):
                return "{}.character[{}].avatar".format(repr(self.engine), repr(self.name))

    def facade(self):
        return Facade(self)

    def add_place(self, n, **kwargs):
        super().add_node(n, **kwargs)

    def add_places_from(self, seq, **attrs):
        """Take a series of place names and add the lot."""
        super().add_nodes_from(seq, **attrs)

    def add_thing(self, name, location, **kwargs):
        """Create a Thing, set its location,
        and set its initial attributes from the keyword arguments (if
        any).

        """
        if name in self.thing:
            raise WorldIntegrityError(
                "Already have a Thing named {}".format(name)
            )
        self.add_node(name, **kwargs)
        if isinstance(location, Node):
            location = location.name
        self.place2thing(name, location,)

    def add_things_from(self, seq, **attrs):
        for tup in seq:
            name = tup[0]
            location = tup[1]
            kwargs = tup[2] if len(tup) > 2 else attrs
            self.add_thing(name, location, **kwargs)

    def place2thing(self, name, location):
        """Turn a Place into a Thing with the given location.
        
        It will keep all its attached Portals.

        """
        self.engine._set_thing_loc(
            self.name, name, location
        )
        if (self.name, name) in self.engine._node_objs:
            del self.engine._node_objs[self.name, name]

    def thing2place(self, name):
        """Unset a Thing's location, and thus turn it into a Place."""
        self.engine._set_thing_loc(
            self.name, name, None
        )
        if (self.name, name) in self.engine._node_objs:
            del self.engine._node_objs[self.name, name]

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        """Connect the origin to the destination with a :class:`Portal`.

        Keyword arguments are the :class:`Portal`'s
        attributes. Exception: if keyword ``symmetrical`` == ``True``,
        a mirror-:class:`Portal` will be placed in the opposite
        direction between the same nodes. It will always appear to
        have the placed :class:`Portal`'s stats, and any change to the
        mirror :class:`Portal`'s stats will affect the placed
        :class:`Portal`.

        """
        if isinstance(origin, Node):
            origin = origin.name
        if isinstance(destination, Node):
            destination = destination.name
        super().add_edge(origin, destination, **kwargs)
        if symmetrical:
            self.add_portal(destination, origin, is_mirror=True)

    def new_portal(self, origin, destination, symmetrical=False, **kwargs):
        if isinstance(origin, Node):
            origin = origin.name
        if isinstance(destination, Node):
            destination = destination.name
        self.add_portal(origin, destination, symmetrical, **kwargs)
        return self.engine._get_edge(self, origin, destination, 0)

    def add_portals_from(self, seq, symmetrical=False):
        """Take a sequence of (origin, destination) pairs and make a
        :class:`Portal` for each.

        Actually, triples are acceptable too, in which case the third
        item is a dictionary of stats for the new :class:`Portal`.

        If optional argument ``symmetrical`` is set to ``True``, all
        the :class:`Portal` instances will have a mirror portal going
        in the opposite direction, which will always have the same
        stats.

        """
        for tup in seq:
            orig = tup[0]
            dest = tup[1]
            kwargs = tup[2] if len(tup) > 2 else {}
            if symmetrical:
                kwargs['symmetrical'] = True
            self.add_portal(orig, dest, **kwargs)

    def add_avatar(self, a, b=None):
        """Start keeping track of a :class:`Thing` or :class:`Place` in a
        different :class:`Character`.

        """
        if b is None:
            if not (
                    isinstance(a, Place) or
                    isinstance(a, Thing)
            ):
                raise TypeError(
                    'when called with one argument, '
                    'it must be a place or thing'
                )
            g = a.character.name
            n = a.name
        else:
            if isinstance(a, Character):
                g = a.name
            elif not isinstance(a, str):
                raise TypeError(
                    'when called with two arguments, '
                    'the first is a character or its name'
                )
            else:
                g = a
            if isinstance(b, Place) or isinstance(b, Thing):
                n = b.name
            elif not isinstance(b, str):
                raise TypeError(
                    'when called with two arguments, '
                    'the second is a thing/place or its name'
                )
            else:
                n = b
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        self.engine._exist_node(g, n)
        # Declare that the node is my avatar
        branch, turn, tick = self.engine.nbtt()
        self.engine._remember_avatarness(self.name, g, n, branch=branch, turn=turn, tick=tick)

    def del_avatar(self, a, b=None):
        """This is no longer my avatar, though it still exists on its own."""
        if b is None:
            if not isinstance(a, Node):
                raise TypeError(
                    "In single argument form, "
                    "del_avatar requires a Node object "
                    "(Thing or Place)."
                )
            g = a.character.name
            n = a.name
        else:
            g = a.name if isinstance(a, Character) else a
            n = b.name if isinstance(b, Node) else b
        self.engine._remember_avatarness(
            self.character.name, g, n, False
        )

    def portals(self):
        """Iterate over all portals."""
        char = self.character
        make_edge = self.engine._get_edge
        for (o, d) in self.engine._edges_cache.iter_keys(
                self.character.name, *self.engine.btt()
        ):
            yield make_edge(char, o, d)

    def avatars(self):
        """Iterate over all my avatars, regardless of what character they are
        in.

        """
        charname = self.character.name
        branch, turn, tick = self.engine.btt()
        charmap = self.engine.character
        avit = self.engine._avatarness_cache.iter_entities
        makenode = self.engine._get_node
        for graph in avit(
                charname, branch, turn, tick
        ):
            for node in avit(
                    charname, graph, branch, turn, tick
            ):
                try:
                    yield makenode(charmap[graph], node)
                except KeyError:
                    continue
