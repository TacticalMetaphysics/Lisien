# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector.
import networkx
from networkx.exception import NetworkXError
from blinker import Signal
from collections import MutableMapping, defaultdict
from operator import attrgetter
from .xjson import (
    JSONWrapper,
    JSONListWrapper,
    JSONReWrapper,
    JSONListReWrapper
)


def getatt(attribute_name):
    """An easy way to make an alias"""
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


class NeatMapping(MutableMapping):
    """Common amenities for mappings"""
    def clear(self):
        """Delete everything"""
        for k in list(self.keys()):
            del self[k]

    def __repr__(self):
        return "{}(graph={}, data={})".format(
            self.__class__.__name__, self.graph.name, repr(dict(self))
        )

    def update(self, other):
        """Version of ``update`` that doesn't clobber the database so much"""
        iteratr = (
            other.iteritems
            if hasattr(other, 'iteritems')
            else other.items
        )
        for (k, v) in iteratr():
            if (
                    k not in self or
                    self[k] != v
            ):
                self[k] = v


class AbstractEntityMapping(NeatMapping, Signal):

    def _iter_keys_db(self):
        """Return a list of keys from the database (not the cache)."""
        raise NotImplementedError

    def _iter_keys_cache(self):

        raise NotImplementedError

    def _get_db(self, key):
        """Return a value of a key from the database (not the cache)."""
        raise NotImplementedError

    def _get_cache(self, key):
        raise NotImplementedError

    def _set_db(self, key, value):
        """Set a value for a key in the database (not the cache)."""
        raise NotImplementedError

    def _set_cache(self, key, value):
        raise NotImplementedError

    def _del_db(self, key):
        """Delete a key from the database (not the cache)."""
        raise NotImplementedError

    def _del_cache(self, key):
        self._set_cache(key, None)

    def __iter__(self):
        if self.db.caching:
            return self._iter_keys_cache()
        return self._iter_keys_db()

    def __len__(self):
        """Number of set keys"""
        n = 0
        for k in iter(self):
            n += 1
        return n

    def __getitem__(self, key):
        """If key is 'graph', return myself as a dict, else get the present
        value of the key and return that

        """
        def wrapval(v):
            if isinstance(v, list):
                if self.db.caching:
                    return JSONListReWrapper(self, key, v)
                return JSONListWrapper(self, key)
            elif isinstance(v, dict):
                if self.db.caching:
                    return JSONReWrapper(self, key, v)
                return JSONWrapper(self, key)
            else:
                return v

        if self.db.caching:
            return wrapval(self._get_cache(key))
        return wrapval(self._get(key))

    def __setitem__(self, key, value):
        """Set key=value at the present branch and revision"""
        if value is None:
            raise ValueError(
                "allegedb uses None to indicate that a key's been deleted"
            )
        if self.db.caching:
            try:
                if self._get_cache(key) != value:
                    self._set_cache(key, value)
            except KeyError:
                self._set_cache(key, value)
        self._set_db(key, value)
        self.send(self, key=key, value=value)

    def __delitem__(self, key):
        """Indicate that the key has no value at this time"""
        if self.db.caching:
            self._set_cache(key, None)
        self._del_db(key)
        self.send(self, key=key, value=None)


class GraphMapping(AbstractEntityMapping):
    """Mapping for graph attributes"""
    db = getatt('graph.db')

    def __init__(self, graph):
        """Initialize private dict and store pointers to the graph and ORM"""
        super().__init__()
        self.graph = graph

    def _iter_keys_db(self):
        """Return keys from the database"""
        return iter(self.db.query.graph_val_keys(
                self.graph.name,
                self.db.branch,
                self.db.rev
        ))

    def _iter_keys_cache(self):
        return self.db._graph_val_cache.iter_entity_keys(
            self.graph.name, self.db.branch, self.db.rev
        )

    def _get_db(self, key):
        """Just load value from database and return"""
        return self.db.query.graph_val_get(
            self.graph.name,
            key,
            self.db.branch,
            self.db.rev
        )

    def _get_cache(self, key):
        return self.db._graph_val_cache.retrieve(
            self.graph.name, key, self.db.branch, self.db.rev
        )
    _get = _get_cache

    def _set_db(self, key, value):
        """Set key=value in the database (not the cache)"""
        self.db.query.graph_val_set(
            self.graph.name,
            key,
            self.db.branch,
            self.db.rev,
            value
        )

    def _set_cache(self, key, value):
        """Set key=value in db's _graph_val_cache"""
        self.db._graph_val_cache.store(
            self.graph.name, key, self.db.branch, self.db.rev, value
        )

    def _del_db(self, key):
        """Delete the value from the database (not the cache)"""
        self.db.query.graph_val_del(
            self.graph.name,
            key,
            self.db.branch,
            self.db.rev
        )


class Node(AbstractEntityMapping):
    """Mapping for node attributes"""
    db = getatt('graph.db')

    def __init__(self, graph, node):
        """Store name and graph"""
        super().__init__()
        self.graph = graph
        self.node = node

    def _iter_keys_db(self):
        return self.db.query.node_val_keys(
            self.graph.name,
            self.node,
            self.db.branch,
            self.db.rev
        )

    def _iter_keys_cache(self):
        return self.db._node_val_cache.iter_entity_keys(
            self.graph.name, self.node, self.db.branch, self.db.rev
        )

    def _get_db(self, key):
        return self.db.query.node_val_get(
            self.graph.name,
            self.node,
            key,
            self.db.branch,
            self.db.rev
        )

    def _get_cache(self, key):
        return self.db._node_val_cache.retrieve(
            self.graph.name, self.node, key, self.db.branch, self.db.rev
        )

    def _set_db(self, key, value):
        self.db.query.node_val_set(
            self.graph.name,
            self.node,
            key,
            self.db.branch,
            self.db.rev,
            value
        )

    def _set_cache(self, key, value):
        self.db._node_val_cache.store(
            self.graph.name,
            self.node,
            key,
            self.db.branch,
            self.db.rev,
            value
        )

    def _del_db(self, key):
        self.db.query.node_val_del(
            self.graph.name,
            self.node,
            key,
            self.db.branch,
            self.db.rev
        )


class Edge(AbstractEntityMapping):
    """Mapping for edge attributes"""
    db = getatt('graph.db')

    def __init__(self, graph, nodeA, nodeB, idx=0):
        """Store the graph, the names of the nodes, and the index.

        For non-multigraphs the index is always 0.

        """
        super().__init__()
        self.graph = graph
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.idx = idx

    def _iter_keys_db(self):
        return self.db.query.edge_val_keys(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            self.db.branch,
            self.db.rev
        )

    def _iter_keys_cache(self):
        return self.db._edge_val_cache.iter_entity_keys(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            self.db.branch,
            self.db.rev
        )

    def _get_db(self, key):
        return self.db.query.edge_val_get(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            key,
            self.db.branch,
            self.db.rev
        )

    def _get_cache(self, key):
        return self.db._edge_val_cache.retrieve(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            key,
            self.db.branch,
            self.db.rev
        )

    def _set_db(self, key, value):
        self.db.query.edge_val_set(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            key,
            self.db.branch,
            self.db.rev,
            value
        )

    def _set_cache(self, key, value):
        self.db._edge_val_cache.store(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            key,
            self.db.branch,
            self.db.rev,
            value
        )

    def _del_db(self, key):
        self.db.query.edge_val_del(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.idx,
            key,
            self.db.branch,
            self.db.rev
        )


class GraphNodeMapping(NeatMapping):
    """Mapping for nodes in a graph"""
    created = Signal()
    deleted = Signal()

    def __init__(self, graph):
        self.graph = graph
        self.db = graph.db

    def __iter__(self):
        """Iterate over the names of the nodes"""
        for node in self.graph.nodes():
            yield node

    def __contains__(self, node):
        """Return whether the node exists presently"""
        if self.db.caching:
            return self.db._nodes_cache.contains_entity(
                self.graph.name, node, self.db.branch, self.db.rev
            )
        return self.db.query.node_exists(
            self.graph.name,
            node,
            self.db.branch,
            self.db.rev
        )

    def __len__(self):
        """How many nodes exist right now?"""
        n = 0
        for node in iter(self):
            n += 1
        return n

    def __getitem__(self, node):
        """If the node exists at present, return it, else throw KeyError"""
        if node not in self:
            raise KeyError
        return self.db._node_objs[(self.graph.name, node)]

    def __setitem__(self, node, dikt):
        """Only accept dict-like values for assignment. These are taken to be
        dicts of node attributes, and so, a new GraphNodeMapping.Node
        is made with them, perhaps clearing out the one already there.

        """
        created = node not in self
        self.db.query.exist_node(
            self.graph.name,
            node,
            self.db.branch,
            self.db.rev,
            True
        )
        if (self.graph.name, node) in self.db._node_objs:
            n = self.db._node_objs[(self.graph.name, node)]
            n.clear()
        else:
            n = self.db._node_objs[(self.graph.name, node)] = Node(
                self.graph, node
            )
        n.update(dikt)
        if self.db.caching:
            self.db._nodes_cache.store(
                self.graph.name,
                node,
                self.db.branch,
                self.db.rev,
                True
            )
        if created:
            self.created.send(self, node=n)

    def __delitem__(self, node):
        """Indicate that the given node no longer exists"""
        if node not in self:
            raise KeyError("No such node")
        self.db.query.exist_node(
            self.graph.name,
            node,
            self.db.branch,
            self.db.rev,
            False
        )
        if self.db.caching:
            self.db._nodes_cache.store(
                self.graph.name,
                node,
                self.db.branch,
                self.db.rev,
                False
            )
        self.deleted.send(self, name=node)

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


class GraphEdgeMapping(NeatMapping, Signal):
    """Provides an adjacency mapping and possibly a predecessor mapping
    for a graph.

    """
    _metacache = defaultdict(dict)

    @property
    def _cache(self):
        return self._metacache[id(self)]

    @property
    def db(self):
        return self.graph.db

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
    graph = getatt('container.graph')
    db = getatt('container.graph.db')
    _metacache = defaultdict(dict)
    created = Signal()
    deleted = Signal()

    @property
    def _cache(self):
        return self._metacache[id(self)]

    def __init__(self, container, nodeA):
        """Store container and node"""
        self.container = container
        self.nodeA = nodeA

    def __iter__(self):
        """Iterate over node IDs that have an edge with my nodeA"""
        if self.db.caching:
            return self.db._edges_cache.iter_entities(
                self.graph.name,
                self.nodeA,
                self.db.branch,
                self.db.rev
            )
        return self.db.query.nodeBs(
            self.graph.name,
            self.nodeA,
            self.db.branch,
            self.db.rev
        )

    def __contains__(self, nodeB):
        """Is there an edge leading to ``nodeB`` at the moment?"""
        if self.db.caching:
            return self.db._edges_cache.contains_entity(
                self.graph.name,
                self.nodeA,
                nodeB,
                0,
                self.db.branch,
                self.db.rev
            )
        for i in self.db.query.multi_edges(
                self.graph.name,
                self.nodeA,
                nodeB,
                self.db.branch,
                self.db.rev
        ):
            return True
        return False

    def __len__(self):
        """How many nodes touch an edge shared with my nodeA?"""
        n = 0
        for nodeB in iter(self):
            n += 1
        return n

    def _make_edge(self, nodeB):
        return Edge(self.graph, self.nodeA, nodeB)

    def __getitem__(self, nodeB):
        """Get the edge between my nodeA and the given node"""
        if nodeB not in self:
            raise KeyError("No edge {}->{}".format(self.nodeA, nodeB))
        if self.db.caching:
            if nodeB not in self._cache:
                self._cache[nodeB] = self._make_edge(nodeB)
            return self._cache[nodeB]
        return self._make_edge(nodeB)

    def __setitem__(self, nodeB, value):
        """Set the edge between my nodeA and the given nodeB to the given
        value, a mapping.

        """
        created = nodeB not in self
        self.db.query.exist_edge(
            self.graph.name,
            self.nodeA,
            nodeB,
            0,
            self.db.branch,
            self.db.rev,
            True
        )
        if self.db.caching:
            self.db._edges_cache.store(
                self.graph.name,
                self.nodeA,
                nodeB,
                0,
                self.db.branch,
                self.db.rev,
                True
            )
        e = self[nodeB]
        e.clear()
        e.update(value)
        if created:
            self.created.send(self, edge=e)

    def __delitem__(self, nodeB):
        """Remove the edge between my nodeA and the given nodeB"""
        self.db.query.exist_edge(
            self.graph.name,
            self.nodeA,
            nodeB,
            0,
            self.db.branch,
            self.db.rev,
            False
        )
        if self.db.caching:
            self.db._edges_cache.store(
                self.graph.name,
                self.nodeA,
                nodeB,
                0,
                self.db.branch,
                self.db.rev,
                False
            )
        self.deleted.send(self, from_node=self.nodeA, to_node=nodeB)

    def clear(self):
        """Delete every edge with origin at my nodeA"""
        for nodeB in self:
            del self[nodeB]


class GraphSuccessorsMapping(GraphEdgeMapping):
    """Mapping for Successors (itself a MutableMapping)"""
    class Successors(AbstractSuccessors):
        def _order_nodes(self, nodeB):
            if nodeB < self.nodeA:
                return (nodeB, self.nodeA)
            else:
                return (self.nodeA, nodeB)

    def __getitem__(self, nodeA):
        if nodeA not in self:
            raise KeyError("No edges from {}".format(nodeA))
        if nodeA not in self._cache:
            self._cache[nodeA] = self.Successors(self, nodeA)
        return self._cache[nodeA]

    def __setitem__(self, key, val):
        """Wipe out any edges presently emanating from nodeA and replace them
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
        """Wipe out edges emanating from nodeA"""
        self[key].clear()
        del self._cache[key]
        self.send(self, key=key, val=None)

    def __iter__(self):
        return iter(self.graph.node)

    def __len__(self):
        return len(self.graph.node)

    def __contains__(self, key):
        return key in self.graph.node


class DiGraphSuccessorsMapping(GraphSuccessorsMapping):
    class Successors(AbstractSuccessors):
        def _order_nodes(self, nodeB):
            return (self.nodeA, nodeB)


class DiGraphPredecessorsMapping(GraphEdgeMapping):
    """Mapping for Predecessors instances, which map to Edges that end at
    the nodeB provided to this

    """
    _predcache = defaultdict(dict)

    def __contains__(self, nodeB):
        return nodeB in self.graph.node

    def __getitem__(self, nodeB):
        """Return a Predecessors instance for edges ending at the given
        node

        """
        if nodeB not in self:
            raise KeyError("No edges available")
        if nodeB not in self._cache:
            self._cache[nodeB] = self.Predecessors(self, nodeB)
        return self._cache[nodeB]

    def _getpreds(self, nodeB):
        cache = self._predcache[id(self)]
        if nodeB not in cache:
            cache[nodeB] = self.Predecessors(self, nodeB)
        return cache[nodeB]

    def __setitem__(self, key, val):
        """Interpret ``val`` as a mapping of edges that end at ``nodeB``"""
        created = key not in self
        preds = self._getpreds(key)
        preds.clear()
        preds.update(val)
        if created:
            self.send(self, key=key, val=val)

    def __delitem__(self, key):
        """Delete all edges ending at ``nodeB``"""
        self._getpreds(key).clear()
        self.send(self, key=key, val=None)

    def __iter__(self):
        return iter(self.graph.node)

    def __len__(self):
        return len(self.graph.node)

    class Predecessors(GraphEdgeMapping):
        """Mapping of Edges that end at a particular node"""
        @property
        def graph(self):
            return self.container.graph

        def __init__(self, container, nodeB):
            """Store container and node ID"""
            self.container = container
            self.nodeB = nodeB

        def __iter__(self):
            """Iterate over the edges that exist at the present (branch, rev)

            """
            if self.db.caching:
                cache = self.db._edges_cache.predecessors[
                    (self.graph.name, self.nodeB)]
                for nodeA in cache:
                    seen = False
                    for idx in cache[nodeA]:
                        if seen:
                            break
                        for (branch, rev) in self.db._active_branches():
                            if branch in cache[nodeA][idx]:
                                v = cache[nodeA][idx][branch][rev]
                                self.db._edges_cache.store(
                                    self.graph.name,
                                    nodeA,
                                    self.nodeB,
                                    idx,
                                    branch,
                                    rev,
                                    v
                                )
                                if v:
                                    yield nodeA
                                seen = True
                                break
                return
            return self.db.query.nodeAs(
                self.graph.name,
                self.nodeB,
                self.db.branch,
                self.db.rev
            )

        def __contains__(self, nodeA):
            """Is there an edge from ``nodeA`` at the moment?"""
            if self.db.caching:
                cache = self.db._edges_cache.predecessors[
                    (self.graph.name, self.nodeB)][nodeA]
                for (branch, rev) in self.db._active_branches():
                    for idx in cache:
                        if branch in cache[idx]:
                            v = cache[idx][branch][rev]
                            self.db._edges_cache.store(
                                self.graph.name,
                                nodeA,
                                self.nodeB,
                                idx,
                                branch,
                                rev,
                                v
                            )
                            return v
                return False
            for i in self.db.query.multi_edges(
                    self.graph.name,
                    self.nodeA,
                    self.nodeB,
                    self.db.branch,
                    self.db.rev
            ):
                return True
            return False

        def __len__(self):
            """How many edges exist at this rev of this branch?"""
            n = 0
            for nodeA in iter(self):
                n += 1
            return n

        def _make_edge(self, nodeA):
            return Edge(self.graph, nodeA, self.nodeB)

        def __getitem__(self, nodeA):
            """Get the edge from the given node to mine"""
            return self.graph.adj[nodeA][self.nodeB]

        def __setitem__(self, nodeA, value):
            """Use ``value`` as a mapping of edge attributes, set an edge from the
            given node to mine.

            """
            try:
                e = self[nodeA]
                e.clear()
                created = False
            except KeyError:
                self.db.query.exist_edge(
                    self.graph.name,
                    nodeA,
                    self.nodeB,
                    0,
                    self.db.branch,
                    self.db.rev,
                    True
                )
                e = self._make_edge(nodeA)
                created = True
            e.update(value)
            if self.db.caching:
                self.db._edges_cache.store(
                    self.graph.name,
                    nodeA,
                    self.nodeB,
                    0,
                    self.db.branch,
                    self.db.rev,
                    True
                )
            if created:
                self.created.send(self, key=nodeA, val=value)

        def __delitem__(self, nodeA):
            """Unset the existence of the edge from the given node to mine"""
            if 'Multi' in self.graph.__class__.__name__:
                for idx in self[nodeA]:
                    self.db.query.exist_edge(
                        self.graph.name,
                        nodeA,
                        self.nodeB,
                        idx,
                        self.db.branch,
                        self.db.rev,
                        False
                    )
                    if self.db.caching:
                        self.db._edges_cache.store(
                            self.graph.name,
                            nodeA,
                            self.nodeB,
                            idx,
                            self.db.branch,
                            self.db.rev,
                            False
                        )
                    self.deleted.send(self, key=nodeA)
                    return
            self.db.query.exist_edge(
                self.graph.name,
                nodeA,
                self.nodeB,
                0,
                self.db.branch,
                self.db.rev,
                False
            )
            if self.db.caching:
                self.db._edges_cache.store(
                    self.graph.name,
                    nodeA,
                    self.nodeB,
                    0,
                    self.db.branch,
                    self.db.rev,
                    False
                )
            self.deleted.send(self, key=nodeA)


class MultiEdges(GraphEdgeMapping):
    """Mapping of Edges between two nodes"""
    def __init__(self, graph, nodeA, nodeB):
        """Store graph and node IDs"""
        self.graph = graph
        self.db = graph.db
        self.nodeA = nodeA
        self.nodeB = nodeB
        self._cache = {}

    def __iter__(self):
        if self.db.caching:
            return self.db._edges_cache.iter_keys(
                self.graph.name, self.nodeA, self.nodeB,
                self.db.brach, self.db.rev
            )
        return self.db.query.multi_edges(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            self.db.branch,
            self.db.rev
        )

    def __len__(self):
        """How many edges currently connect my two nodes?"""
        n = 0
        for idx in iter(self):
            n += 1
        return n

    def __contains__(self, i):
        if self.db.caching:
            return self.db._edges_cache.contains_key(
                self.graph.name, self.nodeA, self.nodeB, i,
                self.db.branch, self.db.rev
            )
        return self.db.query.edge_exists(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            i,
            self.db.branch,
            self.db.rev
        )

    def _getedge(self, idx):
        if idx not in self._cache:
            self._cache[idx] = Edge(self.graph, self.nodeA, self.nodeB, idx)
        return self._cache[idx]

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
        created = idx not in self
        self.db.query.exist_edge(
            self.graph.name,
            self.nodeA,
            self.nodeB,
            idx,
            self.db.branch,
            self.db.rev,
            True
        )
        e = self._getedge(idx)
        e.clear()
        e.update(val)
        if self.db.caching:
            self.db._edges_cache.store(
                self.graph.name, self.nodeA, self.nodeB, idx,
                self.db.branch, self.db.rev, True
            )
        if created:
            self.created.send(self, key=idx, val=val)

    def __delitem__(self, idx):
        """Delete the edge at a particular index"""
        e = self._getedge(idx)
        if not e.exists:
            raise KeyError("No edge at that index")
        e.clear()
        del self._cache[idx]
        if self.db.caching:
            self.db._edges_cache.remember(
                self.graph.name, self.nodeA, self.nodeB, idx,
                self.db.branch, self.db.rev
            )
        self.deleted.send(self, key=idx)

    def clear(self):
        """Delete all edges between these nodes"""
        for idx in self:
            del self[idx]


class MultiGraphSuccessorsMapping(GraphSuccessorsMapping):
    """Mapping of Successors that map to MultiEdges"""
    def __getitem__(self, nodeA):
        """If the node exists, return its Successors"""
        if nodeA not in self.graph.node:
            raise KeyError("No such node")
        return self.Successors(self, nodeA)

    def _getsucc(self, nodeA):
        if nodeA not in self._cache:
            self._cache[nodeA] = self.Successors(self, nodeA)
        return self._cache[nodeA]

    def __setitem__(self, nodeA, val):
        """Interpret ``val`` as a mapping of successors, and turn it into a
        proper Successors object for storage

        """
        created = nodeA in self
        r = self._getsucc(nodeA)
        r.clear()
        r.update(val)
        if created:
            self.created.send(self, key=nodeA, val=val)

    def __delitem__(self, nodeA):
        """Disconnect this node from everything"""
        succs = self._getsucc(nodeA)
        succs.clear()
        del self._cache[nodeA]
        self.deleted.send(self, key=nodeA)

    class Successors(AbstractSuccessors):
        """Edges succeeding a given node in a multigraph"""
        def _order_nodes(self, nodeB):
            if nodeB < self.nodeA:
                return(nodeB, self.nodeA)
            else:
                return (self.nodeA, nodeB)

        _multedge = {}

        def _get_multedge(self, nodeB):
            if nodeB not in self._multedge:
                self._multedge[nodeB] = MultiEdges(
                    self.graph, *self._order_nodes(nodeB)
                )
            return self._multedge[nodeB]

        def __getitem__(self, nodeB):
            """Return MultiEdges to ``nodeB`` if it exists"""
            if nodeB in self.graph.node:
                return self._get_multedge(nodeB)
            raise KeyError("No such node")

        def __setitem__(self, nodeB, val):
            """Interpret ``val`` as a dictionary of edge attributes for edges
            between my ``nodeA`` and the given ``nodeB``

            """
            created = nodeB not in self
            self[nodeB].update(val)
            if created:
                self.created.send(self, key=nodeB, val=val)

        def __delitem__(self, nodeB):
            """Delete all edges between my ``nodeA`` and the given ``nodeB``"""
            self[nodeB].clear()
            del self._multedge[nodeB]
            self.deleted.send(self, key=nodeB)


class MultiDiGraphPredecessorsMapping(DiGraphPredecessorsMapping):
    """Version of DiGraphPredecessorsMapping for multigraphs"""
    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        """Predecessor edges from a given node"""
        def __getitem__(self, nodeA):
            """Get MultiEdges"""
            return MultiEdges(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, val):
            created = nodeA not in self
            self[nodeA].update(val)
            if created:
                self.created.send(self, key=nodeA, val=val)

        def __delitem__(self, nodeA):
            self[nodeA].clear()
            self.deleted.send(self, key=nodeA)


class AllegedGraph(object):
    """Class giving the graphs those methods they share in
    common.

    """
    _succs = {}
    _statmaps = {}

    def __init__(self, db, name, data=None, **attr):
        self._name = name
        self.db = db
        if self.db.caching:
            self.db.graph[name] = self
        if data is not None:
            convert_to_networkx_graph(data, create_using=self)
        self.graph.update(attr)

    @property
    def graph(self):
        if self._name not in self._statmaps:
            self._statmaps[self._name] = GraphMapping(self)
        return self._statmaps[self._name]

    @graph.setter
    def graph(self, v):
        self.graph.clear()
        self.graph.update(v)

    _nodemaps = {}

    @property
    def node(self):
        if self._name not in self._nodemaps:
            self._nodemaps[self._name] = GraphNodeMapping(self)
        return self._nodemaps[self._name]

    @node.setter
    def node(self, v):
        self.node.clear()
        self.node.update(v)

    _succmaps = {}

    @property
    def adj(self):
        if self._name not in self._succmaps:
            self._succmaps[self._name] = self.adj_cls(self)
        return self._succmaps[self._name]

    @adj.setter
    def adj(self, v):
        self.adj.clear()
        self.adj.update(v)
    edge = succ = adj

    _predmaps = {}

    @property
    def pred(self):
        if not hasattr(self, 'pred_cls'):
            raise TypeError("Undirected graph")
        if self._name not in self._predmaps:
            self._predmaps[self._name] = self.pred_cls(self)
        return self._predmaps[self._name]

    @pred.setter
    def pred(self, v):
        self.pred.clear()
        self.pred.update(v)

    def nodes(self):
        if self.db.caching:
            for n in self.db._nodes_cache.iter_entities(
                    self._name, self.db.branch, self.db.rev
            ):
                yield n
            return
        else:
            for node in self.db.query.nodes_extant(
                self._name, self.db.branch, self.db.rev
            ):
                yield node

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
        datadict = self.adj[u].get(v, {})
        datadict.update(attr_dict)
        if u not in self.node:
            self.node[u] = {}
        if v not in self.node:
            self.node[v] = {}
        self.succ[u][v] = datadict
        assert(
            u in self.succ and
            v in self.succ[u]
        )

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
    adj_cls = MultiGraphSuccessorsMapping
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
