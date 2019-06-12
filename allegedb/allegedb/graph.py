# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
"""allegedb's special implementations of the NetworkX graph objects"""
import networkx
from networkx.exception import NetworkXError
from collections import defaultdict, MutableMapping
from .wrap import MutableMappingUnwrapper

class EntityCollisionError(ValueError):
    """For when there's a discrepancy between the kind of entity you're creating and the one by the same name"""


def getatt(attribute_name):
    """An easy way to make an alias"""
    from operator import attrgetter
    return property(attrgetter(attribute_name))


def convert_to_networkx_graph(data, create_using=None, multigraph_input=False):
    """Convert an AllegedGraph to the corresponding NetworkX graph type."""
    if isinstance(data, AllegedGraph):
        result = networkx.convert.from_dict_of_dicts(
            data.adj,
            create_using=create_using,
            multigraph_input=data.is_multigraph()
        )
        result.graph = dict(data.graph)
        result.node = {k: dict(v) for k, v in data.node.items()}
        return result
    return networkx.convert.to_networkx_graph(
        data, create_using, multigraph_input
    )


_alleged_receivers = defaultdict(list)


class AllegedMapping(MutableMappingUnwrapper):
    """Common amenities for mappings"""
    __slots__ = ()

    def connect(self, func):
        """Arrange to call this function whenever something changes here.

        The arguments will be this object, the key changed, and the value set.

        """
        l = _alleged_receivers[id(self)]
        if func not in l:
            l.append(func)

    def disconnect(self, func):
        """No longer call the function when something changes here."""
        if id(self) not in _alleged_receivers:
            return
        l = _alleged_receivers[id(self)]
        try:
            l.remove(func)
        except ValueError:
            return
        if not l:
            del _alleged_receivers[id(self)]

    def send(self, sender, **kwargs):
        """Internal. Call connected functions."""
        if id(self) not in _alleged_receivers:
            return
        for func in _alleged_receivers[id(self)]:
            func(sender, **kwargs)

    def clear(self):
        """Delete everything"""
        for k in list(self.keys()):
            del self[k]

    def update(self, other, **kwargs):
        """Version of ``update`` that doesn't clobber the database so much"""
        from itertools import chain
        if hasattr(other, 'items'):
            other = other.items()
        for (k, v) in chain(other, kwargs.items()):
            if (
                    k not in self or
                    self[k] != v
            ):
                self[k] = v


class AbstractEntityMapping(AllegedMapping):
    __slots__ = ()

    def _get_cache(self, key, branch, turn, tick):
        raise NotImplementedError

    def _get_cache_now(self, key):
        return self._get_cache(key, *self.db._btt())

    def _cache_contains(self, key, branch, turn, tick):
        raise NotImplementedError

    def _set_db(self, key, branch, turn, tick, value):
        """Set a value for a key in the database (not the cache)."""
        raise NotImplementedError

    def _set_cache(self, key, branch, turn, tick, value):
        raise NotImplementedError

    def _set_cache_now(self, key, value):
        branch, turn, tick = self.db._nbtt()
        self._set_cache(key, branch, turn, tick, value)

    def _del_db(self, key, branch, turn, tick):
        """Delete a key from the database (not the cache)."""
        self._set_db(key, branch, turn, tick, None)

    def _del_cache(self, key, branch, turn, tick):
        self._set_cache(key, branch, turn, tick, None)

    def __getitem__(self, key):
        """If key is 'graph', return myself as a dict, else get the present
        value of the key and return that

        """
        def wrapval(v):
            from functools import partial
            from .wrap import DictWrapper, ListWrapper, SetWrapper
            if isinstance(v, list):
                return ListWrapper(partial(self._get_cache_now, key), partial(self._set_cache_now, key), self, key)
            elif isinstance(v, dict):
                return DictWrapper(partial(self._get_cache_now, key), partial(self._set_cache_now, key), self, key)
            elif isinstance(v, set):
                return SetWrapper(partial(self._get_cache_now, key), partial(self._set_cache_now, key), self, key)
            else:
                return v

        return wrapval(self._get_cache_now(key))

    def __contains__(self, item):
        return self._cache_contains(item, *self.db._btt())

    def __setitem__(self, key, value):
        """Set key=value at the present branch and revision"""
        if value is None:
            raise ValueError(
                "allegedb uses None to indicate that a key's been deleted"
            )
        branch, turn, tick = self.db._nbtt()
        try:
            if self._get_cache(key, branch, turn, tick) != value:
                self._set_cache(key, branch, turn, tick, value)
        except KeyError:
            self._set_cache(key, branch, turn, tick, value)
        self._set_db(key, branch, turn, tick, value)
        self.send(self, key=key, value=value)

    def __delitem__(self, key):
        branch, turn, tick = self.db._nbtt()
        self._del_cache(key, branch, turn, tick)
        self._del_db(key, branch, turn, tick)
        self.send(self, key=key, value=None)


class GraphMapping(AbstractEntityMapping):
    """Mapping for graph attributes"""
    __slots__ = ('graph', 'db', '_iter_stuff', '_cache_contains_stuff',
                 '_len_stuff', '_get_stuff', '_set_db_stuff',
                 '_set_cache_stuff', '_del_db_stuff', '_get_cache_stuff')

    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.db = db = graph.db
        btt = db._btt
        graph_val_cache = db._graph_val_cache
        graphn = graph.name
        self._iter_stuff = (graph_val_cache.iter_entity_keys, graphn, btt)
        self._cache_contains_stuff = (graph_val_cache.contains_key, graphn)
        self._len_stuff = (graph_val_cache.count_entities, graphn, btt)
        self._get_stuff = (self._get_cache, btt)
        graph_val_set = db.query.graph_val_set
        self._set_db_stuff = (graph_val_set, graphn)
        self._set_cache_stuff = (graph_val_cache.store, graphn)
        self._del_db_stuff = (graph_val_set, graphn)
        self._get_cache_stuff = (graph_val_cache.retrieve, graphn)

    def __iter__(self):
        iter_entity_keys, graphn, btt = self._iter_stuff
        yield 'name'
        yield from iter_entity_keys(graphn, *btt())

    def _cache_contains(self, key, branch, turn, tick):
        contains_key, graphn = self._cache_contains_stuff
        return contains_key(
            graphn, key, branch, turn, tick
        )

    def __len__(self):
        count_entities, graphn, btt = self._len_stuff
        return 1 + count_entities(graphn, *btt())

    def __getitem__(self, item):
        if item == 'name':
            return self.graph.name
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        if key == 'name':
            raise KeyError("name cannot be changed after creation")
        super().__setitem__(key, value)

    def _get_cache(self, key, branch, turn, tick):
        retrieve, graphn = self._get_cache_stuff
        return retrieve(
            graphn, key, branch, turn, tick
        )

    def _get(self, key):
        get_cache, btt = self._get_stuff
        return get_cache(key, *btt())

    def _set_db(self, key, branch, turn, tick, value):
        graph_val_set, graphn = self._set_db_stuff
        graph_val_set(
            graphn,
            key,
            branch, turn, tick,
            value
        )

    def _set_cache(self, key, branch, turn, tick, value):
        store, graphn = self._set_cache_stuff
        store(
            graphn, key, branch, turn, tick, value
        )

    def _del_db(self, key, branch, turn, tick):
        graph_val_set, graphn = self._del_db_stuff
        graph_val_set(
            graphn,
            key,
            branch, turn, tick, None
        )

    def clear(self):
        keys = set(self.keys())
        keys.remove('name')
        for k in keys:
            del self[k]

    def unwrap(self):
        return {k: v.unwrap() if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v for (k, v) in self.items()}


class Node(AbstractEntityMapping):
    """Mapping for node attributes"""
    __slots__ = ('graph', 'node', 'db', '__weakref__', '_iter_stuff',
                 '_cache_contains_stuff', '_len_stuff', '_get_cache_stuff',
                 '_set_db_stuff', '_set_cache_stuff')

    def __new__(cls, graph, node):
        if (graph.name, node) in graph.db._node_objs:
            ret = graph.db._node_objs[graph.name, node]
            if not isinstance(ret, cls):
                raise EntityCollisionError("Already have node {} in graph {}, but it's of class {}".format(
                    node, graph.name, type(ret)))
            return ret
        return super(Node, cls).__new__(cls)

    def __init__(self, graph, node):
        """Store name and graph"""
        super().__init__()
        self.graph = graph
        self.node = node
        self.db = db = graph.db
        node_val_cache = db._node_val_cache
        graphn = graph.name
        btt = db._btt
        self._iter_stuff = (node_val_cache.iter_entity_keys, graphn, node, btt)
        self._cache_contains_stuff = (node_val_cache.contains_key, graphn, node)
        self._len_stuff = (node_val_cache.count_entity_keys, graphn, node, btt)
        self._get_cache_stuff = (node_val_cache.retrieve, graphn, node)
        self._set_db_stuff = (db.query.node_val_set, graphn, node)
        self._set_cache_stuff = (db._node_val_cache.store, graphn, node)

    def __repr__(self):
        return "{}(graph={}, node={})".format(self.__class__.__name__, self.graph, self.node)

    def __str__(self):
        return "{}(graph={}, node={}, data={})".format(self.__class__.__name__, self.graph, self.node, repr(dict(self)))

    def __iter__(self):
        iter_entity_keys, graphn, node, btt = self._iter_stuff
        return iter_entity_keys(graphn, node, *btt())

    def _cache_contains(self, key, branch, turn, tick):
        contains_key, graphn, node = self._cache_contains_stuff
        return contains_key(
            graphn, node, key, branch, turn, tick
        )

    def __len__(self):
        count_entity_keys, graphn, node, btt = self._len_stuff
        return count_entity_keys(graphn, node, *btt())

    def _get_cache(self, key, branch, turn, tick):
        retrieve, graphn, node = self._get_cache_stuff
        return retrieve(
            graphn, node, key, branch, turn, tick
        )

    def _set_db(self, key, branch, turn, tick, value):
        node_val_set, graphn, node = self._set_db_stuff
        node_val_set(
            graphn,
            node,
            key,
            branch, turn, tick,
            value
        )

    def _set_cache(self, key, branch, turn, tick, value):
        store, graphn, node = self._set_cache_stuff
        store(
            graphn,
            node,
            key,
            branch, turn, tick,
            value
        )


class Edge(AbstractEntityMapping):
    """Mapping for edge attributes"""
    __slots__ = ('graph', 'orig', 'dest', 'idx', 'db', '__weakref__',
                 '_iter_stuff', '_cache_contains_stuff', '_len_stuff',
                 '_get_cache_stuff', '_set_db_stuff', '_set_cache_stuff')

    def __new__(cls, graph, orig, dest, idx=0):
        if (graph.name, orig, dest, idx) in graph.db._edge_objs:
            ret = graph.db._edge_objs[graph.name, orig, dest, idx]
            if not isinstance(ret, cls):
                raise EntityCollisionError(
                    "Already have an edge {}->{}[{}] in graph {}, but of class {}".format(
                        orig, dest, idx, graph.name, type(ret)
                    )
                )
            return ret
        return super(Edge, cls).__new__(cls)

    def __init__(self, graph, orig, dest, idx=0):
        """Store the graph, the names of the nodes, and the index.

        For non-multigraphs the index is always 0.

        """
        super().__init__()
        self.graph = graph
        self.db = db = graph.db
        self.orig = orig
        self.dest = dest
        self.idx = idx
        edge_val_cache = db._edge_val_cache
        graphn = graph.name
        btt = db._btt
        self._iter_stuff = (edge_val_cache.iter_entity_keys, graphn, orig, dest, idx, btt)
        self._cache_contains_stuff = (edge_val_cache.contains_key, graphn, orig, dest, idx)
        self._len_stuff = (edge_val_cache.count_entity_keys, graphn, orig, dest, idx, btt)
        self._get_cache_stuff = (edge_val_cache.retrieve, graphn, orig, dest, idx)
        self._set_db_stuff = (db.query.edge_val_set, graphn, orig, dest, idx)
        self._set_cache_stuff = (edge_val_cache.store, graphn, orig, dest, idx)

    def __repr__(self):
        if self.idx == 0:
            return "{}(graph={}, orig={}, dest={})".format(
                self.__class__.__name__, self.graph, self.orig, self.dest
            )
        return "{}(graph={}, orig={}, dest={}, idx={})".format(
            self.__class__.__name__, self.graph, self.orig, self.dest, self.idx
        )

    def __str__(self):
        if self.idx == 0:
            return "{}(graph={}, orig={}, dest={}, data={})".format(
                self.__class__.__name__, self.graph, self.orig, self.dest, dict(self)
            )
        else:
            return "{}(graph={}, orig={}, dest={}, idx={}, data={})".format(
            self.__class__.__name__, self.graph, self.orig, self.dest, self.idx, dict(self)
        )

    def __iter__(self):
        iter_entity_keys, graphn, orig, dest, idx, btt = self._iter_stuff
        return iter_entity_keys(graphn, orig, dest, idx, *btt())

    def _cache_contains(self, key, branch, turn, tick):
        contains_key, graphn, orig, dest, idx = self._cache_contains_stuff
        return contains_key(graphn, orig, dest, idx, key, branch, turn, tick)

    def __len__(self):
        count_entity_keys, graphn, orig, dest, idx, btt = self._len_stuff
        return count_entity_keys(graphn, orig, dest, idx, *btt())

    def _get_cache(self, key, branch, turn, tick):
        retrieve, graphn, orig, dest, idx = self._get_cache_stuff
        return retrieve(graphn, orig, dest, idx, key, branch, turn, tick)

    def _set_db(self, key, branch, turn, tick, value):
        edge_val_set, graphn, orig, dest, idx = self._set_db_stuff
        edge_val_set(
            graphn,
            orig,
            dest,
            idx,
            key,
            branch, turn, tick,
            value
        )

    def _set_cache(self, key, branch, turn, tick, value):
        store, graphn, orig, dest, idx = self._set_cache_stuff
        store(
            graphn,
            orig,
            dest,
            idx,
            key,
            branch, turn, tick,
            value
        )


class GraphNodeMapping(AllegedMapping):
    """Mapping for nodes in a graph"""
    __slots__ = ('graph',)

    db = getatt('graph.db')

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def __iter__(self):
        """Iterate over the names of the nodes"""
        return self.db._nodes_cache.iter_entities(
            self.graph.name, *self.db._btt()
        )

    def __eq__(self, other):
        from collections import Mapping
        if not isinstance(other, Mapping):
            return NotImplemented
        if self.keys() != other.keys():
            return False
        for k in self.keys():
            me = self[k]
            you = other[k]
            if hasattr(me, 'unwrap') and not hasattr(me, 'no_unwrap'):
                me = me.unwrap()
            if hasattr(you, 'unwrap') and not hasattr(you, 'no_unwrap'):
                you = you.unwrap()
            if me != you:
                return False
        else:
            return True

    def __contains__(self, node):
        """Return whether the node exists presently"""
        return self.db._nodes_cache.contains_entity(
            self.graph.name, node, *self.db._btt()
        )

    def __len__(self):
        """How many nodes exist right now?"""
        return self.db._nodes_cache.count_entities(
            self.graph.name, *self.db._btt()
        )

    def __getitem__(self, node):
        """If the node exists at present, return it, else throw KeyError"""
        if node not in self:
            raise KeyError
        return self.db._get_node(self.graph, node)

    def __setitem__(self, node, dikt):
        """Only accept dict-like values for assignment. These are taken to be
        dicts of node attributes, and so, a new GraphNodeMapping.Node
        is made with them, perhaps clearing out the one already there.

        """
        created = False
        db = self.db
        graph = self.graph
        gname = graph.name
        if not db._node_exists(gname, node):
            created = True
            db._exist_node(gname, node, True)
        n = db._get_node(graph, node)
        n.clear()
        n.update(dikt)
        if created:
            self.send(self, node_name=node, exists=True)

    def __delitem__(self, node):
        """Indicate that the given node no longer exists"""
        if node not in self:
            raise KeyError("No such node")
        branch, turn, tick = self.db._nbtt()
        self.db.query.exist_node(
            self.graph.name,
            node,
            branch, turn, tick,
            False
        )
        self.db._nodes_cache.store(
            self.graph.name,
            node,
            branch, turn, tick,
            False
        )
        key = (self.graph.name, node)
        if node in self.db._node_objs:
            del self.db._node_objs[key]
        self.send(self, node_name=node, exists=False)

    def __eq__(self, other):
        """Compare values cast into dicts.

        As I serve the custom Node class, rather than dicts like
        networkx normally would, the normal comparison operation would
        not let you compare my nodes with regular networkx
        nodes-that-are-dicts. So I cast my nodes into dicts for this
        purpose, and cast the other argument's nodes the same way, in
        case it is a db graph.

        """
        if not hasattr(other, 'keys'):
            return False
        if self.keys() != other.keys():
            return False
        for k in self.keys():
            if dict(self[k]) != dict(other[k]):
                return False
        return True


class GraphEdgeMapping(AllegedMapping):
    """Provides an adjacency mapping and possibly a predecessor mapping
    for a graph.

    """
    __slots__ = ('graph',)

    _metacache = defaultdict(dict)

    @property
    def _cache(self):
        return self._metacache[id(self)]

    db = getatt('graph.db')

    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def __eq__(self, other):
        """Compare dictified versions of the edge mappings within me.

        As I serve custom Predecessor or Successor classes, which
        themselves serve the custom Edge class, I wouldn't normally be
        comparable to a networkx adjacency dictionary. Converting
        myself and the other argument to dicts allows the comparison
        to work anyway.

        """
        if not hasattr(other, 'keys'):
            return False
        if self.keys() != other.keys():
            return False
        for k in self.keys():
            if dict(self[k]) != dict(other[k]):
                return False
        return True

    def __iter__(self):
        return iter(self.graph.node)


class AbstractSuccessors(GraphEdgeMapping):
    __slots__ = ('graph', 'container', 'orig')

    db = getatt('graph.db')
    _metacache = defaultdict(dict)

    def _order_nodes(self, node):
        raise NotImplemented

    @property
    def _cache(self):
        return self._metacache[id(self)]

    def __init__(self, container, orig):
        """Store container and node"""
        super().__init__(container.graph)
        self.container = container
        self.orig = orig

    def __iter__(self):
        """Iterate over node IDs that have an edge with my orig"""
        return self.db._edges_cache.iter_successors(
            self.graph.name,
            self.orig,
            *self.db._btt()
        )

    def __contains__(self, dest):
        """Is there an edge leading to ``dest`` at the moment?"""
        orig, dest = self._order_nodes(dest)
        return self.db._edges_cache.has_successor(
            self.graph.name,
            orig,
            dest,
            *self.db._btt()
        )

    def __len__(self):
        """How many nodes touch an edge shared with my orig?"""
        return self.db._edges_cache.count_successors(
            self.graph.name,
            self.orig,
            *self.db._btt()
        )

    def _make_edge(self, dest):
        return Edge(self.graph, *self._order_nodes(dest))

    def __getitem__(self, dest):
        """Get the edge between my orig and the given node"""
        if dest not in self:
            raise KeyError("No edge {}->{}".format(self.orig, dest))
        orig, dest = self._order_nodes(dest)
        return self.db._get_edge(self.graph, orig, dest, 0)

    def __setitem__(self, dest, value):
        """Set the edge between my orig and the given dest to the given
        value, a mapping.

        """
        real_dest = dest
        orig, dest = self._order_nodes(dest)
        created = dest not in self
        if orig not in self.graph.node:
            self.graph.add_node(orig)
        if dest not in self.graph.node:
            self.graph.add_node(dest)
        branch, turn, tick = self.db._nbtt()
        self.db.query.exist_edge(
            self.graph.name,
            orig,
            dest,
            0,
            branch, turn, tick,
            True
        )
        self.db._edges_cache.store(
            self.graph.name,
            orig,
            dest,
            0,
            branch, turn, tick,
            True
        )
        e = self[real_dest]
        e.clear()
        e.update(value)
        if created:
            self.send(self, orig=orig, dest=dest, idx=0, exists=True)

    def __delitem__(self, dest):
        """Remove the edge between my orig and the given dest"""
        branch, turn, tick = self.db._nbtt()
        orig, dest = self._order_nodes(dest)
        self.db.query.exist_edge(
            self.graph.name,
            orig,
            dest,
            0,
            branch, turn, tick,
            False
        )
        self.db._edges_cache.store(
            self.graph.name,
            orig,
            dest,
            0,
            branch, turn, tick,
            None
        )
        self.send(self, orig=orig, dest=dest, idx=0, exists=False)

    def __repr__(self):
        cls = self.__class__
        return "<{}.{} object containing {}>".format(cls.__module__, cls.__name__, dict(self))

    def clear(self):
        """Delete every edge with origin at my orig"""
        for dest in list(self):
            del self[dest]


class GraphSuccessorsMapping(GraphEdgeMapping):
    """Mapping for Successors (itself a MutableMapping)"""
    __slots__ = ('graph',)

    class Successors(AbstractSuccessors):
        __slots__ = ('graph', 'container', 'orig')

        def _order_nodes(self, dest):
            if dest < self.orig:
                return (dest, self.orig)
            else:
                return (self.orig, dest)

    def __getitem__(self, orig):
        if orig not in self:
            raise KeyError("No edges from {}".format(orig))
        if orig not in self._cache:
            self._cache[orig] = self.Successors(self, orig)
        return self._cache[orig]

    def __setitem__(self, key, val):
        """Wipe out any edges presently emanating from orig and replace them
        with those described by val

        """
        if key in self:
            sucs = self[key]
            created = False
        else:
            sucs = self._cache[key] = self.Successors(self, key)
            created = True
        sucs.clear()
        sucs.update(val)
        if created:
            self.send(self, key=key, val=val)

    def __delitem__(self, key):
        """Wipe out edges emanating from orig"""
        self[key].clear()
        del self._cache[key]
        self.send(self, key=key, val=None)

    def __iter__(self):
        return iter(self.graph.node)

    def __len__(self):
        return len(self.graph.node)

    def __contains__(self, key):
        return key in self.graph.node

    def __repr__(self):
        cls = self.__class__
        return "<{}.{} object containing {}>".format(
            cls.__module__, cls.__name__, {
                k: dict(v) for (k, v) in self.items()
        })


class DiGraphSuccessorsMapping(GraphSuccessorsMapping):
    __slots__ = ('graph',)

    class Successors(AbstractSuccessors):
        __slots__ = ('graph', 'container', 'orig')

        def _order_nodes(self, dest):
            return (self.orig, dest)


class DiGraphPredecessorsMapping(GraphEdgeMapping):
    """Mapping for Predecessors instances, which map to Edges that end at
    the dest provided to this

    """
    __slots__ = ('graph',)

    def __contains__(self, dest):
        return dest in self.graph.node

    def __getitem__(self, dest):
        """Return a Predecessors instance for edges ending at the given
        node

        """
        if dest not in self:
            raise KeyError("No edges available")
        if dest not in self._cache:
            self._cache[dest] = self.Predecessors(self, dest)
        return self._cache[dest]

    def __setitem__(self, key, val):
        """Interpret ``val`` as a mapping of edges that end at ``dest``"""
        created = key not in self
        if key not in self._cache:
            self._cache[key] = self.Predecessors(self, key)
        preds = self._cache[key]
        preds.clear()
        preds.update(val)
        if created:
            self.send(self, key=key, val=val)

    def __delitem__(self, key):
        """Delete all edges ending at ``dest``"""
        it = self[key]
        it.clear()
        del self._cache[key]
        self.send(self, key=key, val=None)

    def __iter__(self):
        return iter(self.graph.node)

    def __len__(self):
        return len(self.graph.node)

    class Predecessors(GraphEdgeMapping):
        """Mapping of Edges that end at a particular node"""
        __slots__ = ('graph', 'container', 'dest')

        def __init__(self, container, dest):
            """Store container and node ID"""
            super().__init__(container.graph)
            self.container = container
            self.dest = dest

        def __iter__(self):
            """Iterate over the edges that exist at the present (branch, rev)

            """
            return self.db._edges_cache.iter_predecessors(
                self.graph.name,
                self.dest,
                *self.db._btt()
            )

        def __contains__(self, orig):
            """Is there an edge from ``orig`` at the moment?"""
            return self.db._edges_cache.has_predecessor(
                self.graph.name,
                self.dest,
                orig,
                *self.db._btt()
            )

        def __len__(self):
            """How many edges exist at this rev of this branch?"""
            return self.db._edges_cache.count_predecessors(
                self.graph.name,
                self.dest,
                *self.db._btt()
            )

        def _make_edge(self, orig):
            return Edge(self.graph, orig, self.dest)

        def __getitem__(self, orig):
            """Get the edge from the given node to mine"""
            return self.graph.adj[orig][self.dest]

        def __setitem__(self, orig, value):
            """Use ``value`` as a mapping of edge attributes, set an edge from the
            given node to mine.

            """
            branch, turn, tick = self.db._nbtt()
            try:
                e = self[orig]
                e.clear()
                created = False
            except KeyError:
                self.db.query.exist_edge(
                    self.graph.name,
                    orig,
                    self.dest,
                    0,
                    branch, turn, tick,
                    True
                )
                e = self._make_edge(orig)
            e.update(value)
            self.db._edges_cache.store(
                self.graph.name,
                orig,
                self.dest,
                0,
                branch, turn, tick,
                True
            )
            self.send(self, key=orig, val=value)

        def __delitem__(self, orig):
            """Unset the existence of the edge from the given node to mine"""
            branch, turn, tick = self.db._nbtt()
            if 'Multi' in self.graph.__class__.__name__:
                for idx in self[orig]:
                    self.db.query.exist_edge(
                        self.graph.name,
                        orig,
                        self.dest,
                        idx,
                        branch, turn, tick,
                        False
                    )
                    self.db._edges_cache.store(
                        self.graph.name,
                        orig,
                        self.dest,
                        idx,
                        branch, turn, tick,
                        False
                    )
                    self.send(self, key=orig, val=None)
                    return
                else:
                    raise KeyError("No edges from {}".format(orig))
            self.db.query.exist_edge(
                self.graph.name,
                orig,
                self.dest,
                0,
                branch, turn, tick,
                False
            )
            self.db._edges_cache.store(
                self.graph.name,
                orig,
                self.dest,
                0,
                branch, turn, tick,
                None
            )
            self.send(self, key=orig, val=None)


class AbstractMultiEdges(GraphEdgeMapping):
    """Mapping of Edges between two nodes"""
    __slots__ = ('graph', 'orig', 'dest')

    db = getatt('graph.db')

    def _order_nodes(self):
        """Swap my orig and dest if desired"""
        raise NotImplemented

    def __init__(self, graph, orig, dest):
        super().__init__(graph)
        self.orig = orig
        self.dest = dest

    def __iter__(self):
        orig, dest = self._order_nodes()
        return self.db._edges_cache.iter_keys(
            self.graph.name, orig, dest,
            *self.db._btt()
        )

    def __len__(self):
        """How many edges currently connect my two nodes?"""
        n = 0
        for idx in iter(self):
            n += 1
        return n

    def __contains__(self, i):
        orig, dest = self._order_nodes()
        return self.db._edges_cache.contains_key(
            self.graph.name, orig, dest, i,
            *self.db._btt()
        )

    def _getedge(self, idx):
        orig, dest = self._order_nodes()
        return self.db._get_edge(self.graph, orig, dest, idx)

    def __getitem__(self, idx):
        """Get an Edge with a particular index, if it exists at the present
        (branch, rev)

        """
        if idx not in self:
            raise KeyError("No edge at that index")
        return self._getedge(idx)

    def __setitem__(self, idx, val):
        """Create an Edge at a given index from a mapping. Delete the existing
        Edge first, if necessary.

        """
        orig, dest = self._order_nodes()
        created = idx not in self
        if orig not in self.graph.node:
            self.graph.add_node(orig)
        if dest not in self.graph.node:
            self.graph.add_node(dest)
        branch, turn, tick = self.db._nbtt()
        self.db.query.exist_edge(
            self.graph.name,
            orig,
            dest,
            idx,
            branch, turn, tick,
            True
        )
        self.db._edges_cache.store(
            self.graph.name, orig, dest, idx,
            branch, turn, tick, True
        )
        e = self._getedge(idx)
        e.clear()
        e.update(val)
        if created:
            self.send(self, orig=orig, dest=dest, idx=idx, exists=True)

    def __delitem__(self, idx):
        """Delete the edge at a particular index"""
        branch, turn, tick = self.db._btt()
        orig, dest = self._order_nodes()
        tick += 1
        e = self._getedge(idx)
        e.clear()
        if idx in self._cache:
            del self._cache[idx]
        self.db._edges_cache.store(
            self.graph.name, orig, dest, idx,
            branch, turn, tick, None
        )
        self.db.tick = tick
        self.send(self, orig=orig, dest=dest, idx=idx, exists=False)

    def clear(self):
        """Delete all edges between these nodes"""
        for idx in self:
            del self[idx]


class MultiGraphSuccessorsMapping(GraphSuccessorsMapping):
    """Mapping of Successors that map to MultiEdges"""
    __slots__ = ('graph',)

    def __getitem__(self, orig):
        """If the node exists, return its Successors"""
        if orig not in self.graph.node:
            raise KeyError("No such node")
        return self._getsucc(orig)

    def _getsucc(self, orig):
        if orig not in self._cache:
            self._cache[orig] = self.Successors(self, orig)
        return self._cache[orig]

    def __setitem__(self, orig, val):
        """Interpret ``val`` as a mapping of successors, and turn it into a
        proper Successors object for storage

        """
        r = self._getsucc(orig)
        r.clear()
        r.update(val)
        self.send(self, key=orig, val=val)

    def __delitem__(self, orig):
        """Disconnect this node from everything"""
        succs = self._getsucc(orig)
        succs.clear()
        del self._cache[orig]
        self.send(self, key=orig, val=None)

    class Successors(AbstractSuccessors):
        """Edges succeeding a given node in a multigraph"""
        __slots__ = ('graph', 'container', 'orig', '_multedge')

        class MultiEdges(AbstractMultiEdges):
            def _order_nodes(self):
                if self.dest < self.orig:
                    return (self.dest, self.orig)
                else:
                    return (self.orig, self.dest)

        def __init__(self, container, orig):
            super().__init__(container, orig)
            self._multedge = {}

        def __contains__(self, item):
            orig, dest = self._order_nodes(item)
            return self.db._edges_cache.has_successor(
                self.graph.name, orig, dest, *self.db._btt()
            )

        def _order_nodes(self, dest):
            if dest < self.orig:
                return(dest, self.orig)
            else:
                return (self.orig, dest)

        def _get_multedge(self, dest):
            if dest not in self._multedge:
                self._multedge[dest] = self.MultiEdges(
                    self.graph, *self._order_nodes(dest)
                )
            return self._multedge[dest]

        def __getitem__(self, dest):
            """Return MultiEdges to ``dest`` if it exists"""
            if dest in self.graph.node:
                return self._get_multedge(dest)
            raise KeyError("No such node")

        def __setitem__(self, dest, val):
            """Interpret ``val`` as a dictionary of edge attributes for edges
            between my ``orig`` and the given ``dest``

            """
            it = self[dest]
            it.clear()
            it.update(val)
            assert dest in self
            self.send(self, key=dest, val=val)

        def __delitem__(self, dest):
            """Delete all edges between my ``orig`` and the given ``dest``"""
            self[dest].clear()
            del self._multedge[dest]
            self.send(self, key=dest, val=None)


class MultiDiGraphPredecessorsMapping(DiGraphPredecessorsMapping):
    """Version of DiGraphPredecessorsMapping for multigraphs"""
    __slots__ = ('graph',)

    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        """Predecessor edges from a given node"""
        __slots__ = ('graph', 'container', 'dest')

        class MultiEdges(AbstractMultiEdges):
            def _order_nodes(self):
                return self.orig, self.dest

        def __getitem__(self, orig):
            """Get MultiEdges"""
            return self.MultiEdges(self.graph, orig, self.dest)

        def __setitem__(self, orig, val):
            self[orig].update(val)
            self.send(self, key=orig, val=val)

        def __delitem__(self, orig):
            self[orig].clear()
            self.send(self, key=orig, val=None)


class MultiDiGraphSuccessorsMapping(MultiGraphSuccessorsMapping):
    __slots__ = ('graph',)

    class Successors(MultiGraphSuccessorsMapping.Successors):
        __slots__ = ('graph', 'container', 'orig', '_multedge')

        class MultiEdges(AbstractMultiEdges):
            def _order_nodes(self):
                return self.orig, self.dest

        def _order_nodes(self, dest):
            return self.orig, dest


class AllegedGraph(object):
    """Class giving the graphs those methods they share in
    common.

    """
    _succs = {}
    _statmaps = {}
    graph_map_cls = GraphMapping
    node_map_cls = GraphNodeMapping

    def __new__(cls, db, name, data=None, **attr):
        if name in db._graph_objs:
            ret = db._graph_objs[name]
            if not isinstance(ret, cls):
                raise EntityCollisionError("Already have a graph named {}, but it's of class {}".format(
                    name, type(ret)
                ))
            return ret
        return super(AllegedGraph, cls).__new__(cls)

    def __init__(self, db, name, data=None, **attr):
        self._name = name
        self.db = db
        if name not in self.db._graph_objs:
            self.db._graph_objs[name] = self
        if data is not None:
            data = data.copy()
            if isinstance(data, dict) and 'name' in data:
                del data['name']
            if hasattr(data, 'graph') and 'name' in data.graph:
                del data.graph['name']
            convert_to_networkx_graph(data, create_using=self)
        self.graph.update(attr)

    @property
    def graph(self):
        if id(self) not in self._statmaps:
            self._statmaps[id(self)] = self.graph_map_cls(self)
        return self._statmaps[id(self)]

    @graph.setter
    def graph(self, v):
        self.graph.clear()
        self.graph.update(v)

    _nodemaps = {}

    @property
    def node(self):
        if id(self) not in self._nodemaps:
            self._nodemaps[id(self)] = self.node_map_cls(self)
        return self._nodemaps[id(self)]

    @node.setter
    def node(self, v):
        self.node.clear()
        self.node.update(v)
    _node = node

    _succmaps = {}

    @property
    def adj(self):
        if self._name not in self._succmaps:
            self._succmaps[id(self)] = self.adj_cls(self)
        return self._succmaps[id(self)]

    @adj.setter
    def adj(self, v):
        self.adj.clear()
        self.adj.update(v)
    edge = succ = _succ = _adj = adj

    _predmaps = {}

    @property
    def pred(self):
        if not hasattr(self, 'pred_cls'):
            raise TypeError("Undirected graph")
        if self._name not in self._predmaps:
            self._predmaps[id(self)] = self.pred_cls(self)
        return self._predmaps[id(self)]

    @pred.setter
    def pred(self, v):
        self.pred.clear()
        self.pred.update(v)
    _pred = pred

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, v):
        raise TypeError("graphs can't be renamed")

    def _and_previous(self):
        """Return a 4-tuple that will usually be (current branch, current
        revision - 1, current branch, current revision), unless
        current revision - 1 is before the start of the current
        branch, in which case the first element will be the parent
        branch.

        """
        branch = self.db.branch
        rev = self.db.rev
        (parent, parent_rev) = self.db.sql('parparrev', branch).fetchone()
        before_branch = parent if parent_rev == rev else branch
        return (before_branch, rev-1, branch, rev)

    def clear(self):
        """Remove all nodes and edges from the graph.

        Unlike the regular networkx implementation, this does *not*
        remove the graph's name. But all the other graph, node, and
        edge attributes go away.

        """
        self.adj.clear()
        self.node.clear()
        self.graph.clear()


class Graph(AllegedGraph, networkx.Graph):
    """A version of the networkx.Graph class that stores its state in a
    database.

    """
    adj_cls = GraphSuccessorsMapping


class DiGraph(AllegedGraph, networkx.DiGraph):
    """A version of the networkx.DiGraph class that stores its state in a
    database.

    """
    adj_cls = DiGraphSuccessorsMapping
    pred_cls = DiGraphPredecessorsMapping

    def remove_edge(self, u, v):
        """Version of remove_edge that's much like normal networkx but only
        deletes once, since the database doesn't keep separate adj and
        succ mappings

        """
        try:
            del self.succ[u][v]
        except KeyError:
            raise NetworkXError(
                "The edge {}-{} is not in the graph.".format(u, v)
            )

    def remove_edges_from(self, ebunch):
        """Version of remove_edges_from that's much like normal networkx but only
        deletes once, since the database doesn't keep separate adj and
        succ mappings

        """
        for e in ebunch:
            (u, v) = e[:2]
            if u in self.succ and v in self.succ[u]:
                del self.succ[u][v]

    def add_edge(self, u, v, attr_dict=None, **attr):
        """Version of add_edge that only writes to the database once"""
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(
                    "The attr_dict argument must be a dictionary."
                )
        if u not in self.node:
            self.node[u] = {}
        if v not in self.node:
            self.node[v] = {}
        if u in self.adj:
            datadict = self.adj[u].get(v, {})
        else:
            self.adj[u] = {v: {}}
            datadict = self.adj[u][v]
        datadict.update(attr_dict)
        self.succ[u][v] = datadict
        assert u in self.succ, "Failed to add edge {u}->{v} ({u} not in successors)".format(u=u, v=v)
        assert v in self.succ[u], "Failed to add edge {u}->{v} ({v} not in succ[{u}])".format(u=u, v=v)

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        """Version of add_edges_from that only writes to the database once"""
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(
                    "The attr_dict argument must be a dict."
                )
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
                assert hasattr(dd, "update")
            elif ne == 2:
                u, v = e
                dd = {}
            else:
                raise NetworkXError(
                    "Edge tupse {} must be a 2-tuple or 3-tuple.".format(e)
                )
            if u not in self.node:
                self.node[u] = {}
            if v not in self.node:
                self.node[v] = {}
            datadict = self.adj.get(u, {}).get(v, {})
            datadict.update(attr_dict)
            datadict.update(dd)
            self.succ[u][v] = datadict
            assert(u in self.succ)
            assert(v in self.succ[u])


class MultiGraph(AllegedGraph, networkx.MultiGraph):
    """A version of the networkx.MultiGraph class that stores its state in a
    database.

    """
    adj_cls = MultiGraphSuccessorsMapping


class MultiDiGraph(AllegedGraph, networkx.MultiDiGraph):
    """A version of the networkx.MultiDiGraph class that stores its state in a
    database.

    """
    adj_cls = MultiDiGraphSuccessorsMapping
    pred_cls = MultiDiGraphPredecessorsMapping

    def remove_edge(self, u, v, key=None):
        """Version of remove_edge that's much like normal networkx but only
        deletes once, since the database doesn't keep separate adj and
        succ mappings

        """
        try:
            d = self.adj[u][v]
        except KeyError:
            raise NetworkXError(
                "The edge {}-{} is not in the graph.".format(u, v)
            )
        if key is None:
            d.popitem()
        else:
            try:
                del d[key]
            except KeyError:
                raise NetworkXError(
                    "The edge {}-{} with key {} is not in the graph.".format
                    (u, v, key)
                )
        if len(d) == 0:
            del self.succ[u][v]

    def remove_edges_from(self, ebunch):
        """Version of remove_edges_from that's much like normal networkx but only
        deletes once, since the database doesn't keep separate adj and
        succ mappings

        """
        for e in ebunch:
            (u, v) = e[:2]
            if u in self.succ and v in self.succ[u]:
                del self.succ[u][v]

    def add_edge(self, u, v, key=None, attr_dict=None, **attr):
        """Version of add_edge that only writes to the database once."""
        if attr_dict is None:
            attr_dict = attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise NetworkXError(
                    "The attr_dict argument must be a dictionary."
                )
        if u not in self.node:
            self.node[u] = {}
        if v not in self.node:
            self.node[v] = {}
        if v in self.succ[u]:
            keydict = self.adj[u][v]
            if key is None:
                key = len(keydict)
                while key in keydict:
                    key += 1
            datadict = keydict.get(key, {})
            datadict.update(attr_dict)
            keydict[key] = datadict
        else:
            if key is None:
                key = 0
            datadict = {}
            datadict.update(attr_dict)
            keydict = {key: datadict}
            self.succ[u][v] = keydict
        return key


class GraphsMapping(MutableMapping):
    def __init__(self, orm):
        self.orm = orm

    def __iter__(self):
        return iter(self.orm._graph_objs)

    def __len__(self):
        return len(self.orm._graph_objs)

    def __getitem__(self, item):
        return self.orm.get_graph(item)

    def __setitem__(self, key, value):
        if isinstance(value, networkx.MultiDiGraph):
            self.orm.new_multidigraph(key, data=value)
        elif isinstance(value, networkx.DiGraph):
            self.orm.new_digraph(key, data=value)
        elif isinstance(value, networkx.MultiGraph):
            self.orm.new_multigraph(key, data=value)
        else:
            self.orm.new_graph(key, data=value)

    def __delitem__(self, key):
        self.orm.del_graph(key)
