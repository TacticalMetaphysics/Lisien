from igraph import Graph, Vertex, Edge
from util import (
    SaveableMetaclass,
    dictify_row,
    TabdictIterator)
from collections import defaultdict


__metaclass__ = SaveableMetaclass


class BranchDictIter:
    def __init__(self, branchdict):
        self.realiter = branchdict.iteritems()

    def __iter__(self):
        return self

    def next(self):
        (branch, (parent, tick_from, tick_to)) = self.realiter.next()
        return (branch, parent, tick_from, tick_to)


class Timestream:
    """A graph of many timelines, some of which share some of their time.

    The first argument is a dictionary keyed with branch indices, with
    values composed of tuples like:

    (parent_branch, start, end)

    parent_branch must be another key in the branchdict. start and end
    are the tick when a branch begins (which cannot change) and ends
    (which can, and probably will, perhaps as often as once per
    update).

    Call the update method to rearrange the contents of this board to
    reflect the state of the branches.

    """
    tables = [
        ("timestream",
         {"branch": "integer not null",
          "parent": "integer not null",
          "tick_from": "integer not null",
          "tick_to": "integer not null"},
         ("branch",),
         {"parent": ("timestream", "branch")},
         ["branch>=0", "tick_from>=0",
          "tick_to>=tick_from", "parent=0 or parent<>branch"])]
    def __init__(self, rumor):
        self.rumor = rumor
        td = self.rumor.tabdict
        self.branch_edges = defaultdict(set)
        self.branch_done_to = defaultdict(lambda: -1)
        self.branchdict = {}
        for rd in TabdictIterator(td["timestream"]):
            self.branchdict[rd["branch"]] = (
                rd["parent"], rd["tick_from"], rd["tick_to"])
        self.graph = Graph(directed=True)
        self.graph.add_vertices(2)
        self.graph.vs["tick"] = [0, 0]
        self.graph.add_edge(0, 1, branch=0)
        self.branch_edges[0].add(0)
        self.branch_head = {0: self.graph.vs[0]}
        # When the player travels to the past and then branches the
        # timeline, it may result in a new vertex in the middle of
        # what once was an unbroken edge. The edge succeeding the new
        # vertex, representing how things went *originally*, is still
        # representative of the old branch, even though it is now a
        # successor of the vertex for a different branch
        # altogether. That original branch now has another edge
        # representing it.
        self.update_handlers = set()
        self.update(0)

    def __hash__(self):
        b = []
        for t in self.branchdict.itervalues():
            b.extend(t)
        return hash(tuple(b))

    def update(self, ts=0):
        """Update the tree to reflect the current state of branchdict.

For every branch in branchdict, there should be one vertex at the
start and one at the end. If the branch has grown, but has as many
child branches as previously, change the tick of the end vertex to
reflect the growth.

If there are more child branches than before, split an edge to place a
vertex at the start point of each child branch. Then extend a new edge
out to the end of the child branch. The start and end vertices may
be on the same tick, in which case they are connected by an edge of
length zero.

        """
        for (branch, (parent, tick_from, tick_to)) in self.branchdict.iteritems():
            done_to = self.branch_done_to[branch]
            if tick_to > done_to:
                # I am now looking at a tick-window that has not been
                # put into the graph yet.
                #
                # Where does it belong?
                #
                # Is its branch, at least, already in the graph somewhere?
                if branch in self.branch_edges:
                    # I may have to extend an edge to make it fit the
                    # whole tick-window.
                    e_to = self.get_edge(branch, tick_to)
                    if e_to is None:
                        e_to = self.latest_edge(branch)
                        v = self.graph.vs[e_to.target]
                        growth = tick_to - v["tick"]
                        v["tick"] += growth
                    # Otherwise there's not really much to do here.
                else:
                    # I assume that this dict reflects the genealogy
                    # of the branches accurately
                    try:
                        self.split_branch(
                            parent, branch, tick_from, tick_to - tick_from)
                    except KeyError:
                        assert(branch == 0)
                        self.graph.add_vertices(2)
                        self.graph.vs["tick"] = [0, tick_to]
                        self.graph.add_edge(0, 1, branch=0)
                        eid =self.graph.get_eid(0, 1)
                        self.branch_edges[0].add(eid)
            self.branch_done_to[branch] = tick_to
            for handler in self.update_handlers:
                handler.on_timestream_update()

    def get_edge_len(self, e):
        if isinstance(e, int):
            e = self.graph.es[e]
        vo = self.graph.vs[e.source]
        vd = self.graph.vs[e.target]
        return vd["tick"] - vo["tick"]

    def sanitize_vert(self, v):
        if isinstance(v, Vertex):
            vert = v
            vid = vert.index
        else:
            vid = v
            vert = self.graph.vs[vid]
        return (vert, vid)

    def sanitize_edge(self, e):
        if isinstance(e, Edge):
            edge = e
            eid = edge.index
        else:
            eid = e
            edge = self.graph.es[eid]
        return (edge, eid)

    def vertex_in_branch(self, v, branch):
        v = self.sanitize_vert(v)[1]
        for eid in self.graph.incident(v):
            if eid in self.branch_edges[branch]:
                return True
        return False

    def add_edge(self, branch, vert_from, vert_to):
        assert(branch in self.branchdict)
        (vert_from, vi1) = self.sanitize_vert(vert_from)
        (vert_to, vi2) = self.sanitize_vert(vert_to)
        self.graph.add_edge(vi1, vi2, branch=branch)
        eid = self.graph.get_eid(vi1, vi2)
        self.branch_edges[branch].add(eid)
        (p, a, z) = self.branchdict[branch]
        if vert_from["tick"] < self.branchdict[branch][1]:
            a = vert_from["tick"]
        if vert_to["tick"] > self.branchdict[branch][2]:
            z = vert_to["tick"]
        self.branchdict[branch] = (p, a, z)
        return self.graph.es[eid]

    def delete_edge(self, e):
        (e, eid) = self.sanitize_edge(e)
        old_branch = e["branch"]
        self.branch_edges[old_branch].discard(eid)
        self.graph.delete_edges(eid)

    def add_vert(self, tick):
        i = len(self.graph.vs)
        self.graph.add_vertex(tick=tick)
        v = self.graph.vs[i]
        return v

    def vert_branch(self, vert):
        if isinstance(vert, int):
            vert = self.graph.vs[vert]
        try:
            eid = self.graph.incident(vert)[0]
            return self.graph.es[eid]["branch"]
        except:
            return -1

    def add_vert_on(self, e, tick):
        (e, eid) = self.sanitize_edge(e)
        former = self.graph.vs[e.source]
        latter = self.graph.vs[e.target]
        old_branch = e["branch"]
        v = self.add_vert(tick)
        i = v.index
        e1 = self.add_edge(old_branch, former, i)
        e2 = self.add_edge(old_branch, i, latter)
        return (e1, v, e2)

    def get_edge(
            self,
            vert_from_or_branch,
            vert_to_or_tick,
            mode="branch_tick"):
        if mode == "branch_tick":
            return self.get_edge_from_branch_tick(
                vert_from_or_branch,
                vert_to_or_tick)
        elif mode == "verts":
            return self.get_edge_from_verts(
                vert_from_or_branch,
                vert_to_or_tick)
        else:
            raise Exception("Invalid mode")

    def get_edge_from_verts(self, vert_from, vert_to):
        if isinstance(vert_from, Vertex):
            vert_from = vert_from.index
        if isinstance(vert_to, Vertex):
            vert_to = vert_to.index
        eid = self.graph.get_eid(vert_from, vert_to)
        return self.graph.es[eid]

    def get_edge_from_branch_tick(self, branch, tick):
        """Return the edge that contains the given tick in the given branch,
or None if no such edge exists."""
        v = self.branch_head[branch]
        if tick < v["tick"]:
            raise Exception("This branch started after that tick.")
        return self.successor_on_branch_tick(v, branch, tick)

    def successor_on_branch_tick(self, v, branch, tick):
        """Traverse the graph starting from the given vertex. Return the edge
containing the given tick in the given branch. If it doesn't exist,
return None."""
        if tick == v["tick"]:
            # I'll consider ticks coinciding exactly with a vertex to
            # be in the descendant edge in that branch.
            if self.vertex_in_branch(v, branch):
                for e in self.graph.incident(v):
                    if e in self.branch_edges[branch]:
                        return e
            return None
        for eid in self.graph.incident(v):
            e = self.graph.es[eid]
            v_to = self.graph.vs[e.target]
            if eid in self.branch_edges[branch]:
                if v_to["tick"] > tick:
                    return e
                else:
                    # None of these edges are right! You want the ones
                    # after this vertex here.
                    return self.successor_on_branch_tick(v_to, branch, tick)
            elif v_to["tick"] >= tick:
                return e
        return None

    def split_branch(self, old_branch, new_branch, tick_from, tick_to):
        """Find the edge in old_branch in the given tick, split it, and start
a new edge off the split. The new edge will be a member of
new_branch.

        """
        assert(new_branch not in self.branchdict)
        self.branchdict[new_branch] = (old_branch, tick_from, tick_to)
        e = self.get_edge_from_branch_tick(old_branch, tick_from)
        if e is None:
            vseq = self.graph.vs(tick_eq=tick_from)
            v1 = vseq[0]
            v2 = self.add_vert(tick=tick_to)
        else:
            (e1, v1, e2) = self.add_vert_on(e, tick_from)
            v2 = self.add_vert(tick=tick_to)
        return self.add_edge(new_branch, v1, v2)

    def latest_edge(self, branch):
        """Return the edge in the given branch that ends on the highest
tick."""
        edges = set(self.branch_edges[branch])
        late = self.graph.es[edges.pop()]
        v_late = self.graph.vs[late.target]
        while len(edges) > 0:
            e = self.graph.es[edges.pop()]
            v_e = self.graph.vs[e.target]
            if v_e["tick"] > v_late["tick"]:
                late = e
                v_late = v_e
        return late

    def extend_branch(self, branch, n):
        """Make the branch so many ticks longer."""
        edge = self.latest_edge(branch)
        vert = self.graph.vs[edge.target]
        vert["tick"] += n
        self.branchdict[branch] = (
            self.branchdict[branch][0], self.branchdict.branch[1] + n)

    def extend_branch_to(self, branch, tick_to):
        """Make the branch end on the given tick, but only if it is later than
the branch's current end."""
        edge = self.latest_edge(branch)
        vert = self.graph.vs[edge.target]
        if tick_to > vert["tick"]:
            vert["tick"] = tick_to
        if tick_to > self.branchdict[branch][1]:
            self.branchdict[branch] = (
                self.branchdict[branch][0], tick_to)

    def get_tabdict(self):
        return {"timestream": BranchDictIter(self.branchdict)}

class TimestreamException(Exception):
    pass
