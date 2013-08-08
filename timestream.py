from igraph import Graph, Vertex, Edge, IN
from spot import AbstractSpot
from arrow import Arrow
from collections import defaultdict
import pyglet


class Spot(AbstractSpot):
    def __init__(self, board, v):
        super(Spot, self).__init__(board, v, saveable=False)

    def __getattr__(self, attrn):
        # horrible hack
        if attrn == "x":
            return self.v["branch"] * 20
        elif attrn == "y":
            return self.v["tick"] * 20
        else:
            return super(Spot, self).__getattr__(attrn)


class SpotIter:
    def __init__(self, timestream):
        self.realiter = iter(self.timestream.graph.vs)

    def __iter__(self):
        return self

    def next(self):
        return self.realiter.next()["spot"]


class ArrowIter:
    def __init__(self, timestream):
        self.realiter = iter(self.timestream.graph.es)

    def __iter__(self):
        return self

    def next(self):
        return self.realiter.next()["arrow"]


class Timestream(Board):
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
    def __init__(self, branchdict, parentdict, timescale=1.0):
        # How am I going to populate the branchdict?
        self.branchdict = branchdict
        self.branch_parent = parentdict
        self.branch_done_to = defaultdict(lambda: -1)
        self.graph = Graph(directed=True)
        self.branch_head = {}
        # When the player travels to the past and then branches the
        # timeline, it may result in a new vertex in the middle of
        # what once was an unbroken edge. The edge succeeding the new
        # vertex, representing how things went *originally*, is still
        # representative of the old branch, even though it is now a
        # successor of the vertex for a different branch
        # altogether. That original branch now has another edge
        # representing it.
        self.branch_edges = defaultdict(set)
        self.branch_edges[0] = self.graph.es[0]
        self.update()

    def update(self):
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
        for branch in self.branchdict:
            done_to = self.branch_done_to[branch]
            (tick_from, tick_to) = self.branchdict[branch]
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
                    e_from = self.get_edge(branch, tick_from)
                    e_to = self.get_edge(branch, tick_to)
                    if e_to is None:
                        e_to = self.latest_edge(branch)
                        v = self.graph.vs[e.target]
                        growth = v["tick"] - tick_to
                        v["tick"] += growth
                        e_to["length"] += growth
                    # Otherwise there's not really much to do here.
                else:
                    # I assume that this dict reflects the genealogy
                    # of the branches accurately
                    try:
                        parent = self.branch_parent[branch]
                        self.split_branch(
                            parent, branch, tick_from, tick_to - tick_from)
                    except KeyError:
                        assert(branch == 0)
                        self.graph.add_vertices(2)
                        self.graph.vs["tick"] = [0, tick_to]
                        self.graph.add_edge(0, 1, branch=0)
                        e = self.graph.es[self.graph.get_eid(0, 1)]
                        self.branch_edges[0].add(e)

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
        for e in self.graph.incident(v, mode=IN):
            if e in self.branch_edges[branch]:
                return True
        return False

    def add_edge(self, vert_from, vert_to, branch, length=0):
        (vert_from, vi1) = self.sanitize_vert(vert_from)
        (vert_to, vi2) = self.sanitize_vert(vert_to)
        self.graph.add_edge(vi1, vi2, branch=branch, length=length)
        e = self.graph.es[self.graph.get_eid(vi1, vi2)]
        self.branch_edges[branch].add(e)
        e["arrow"] = Arrow(self, e)
        return e

    def delete_edge(self, e):
        (e, eid) = self.sanitize_edge(e)
        old_branch = e["branch"]
        self.branch_edges[old_branch].discard(e)
        e["arrow"].delete()
        self.graph.delete_edges(eid)

    def add_vert(self, tick):
        i = len(self.graph.vs)
        self.graph.add_vertex(tick=tick)
        v = self.graph.vs[i]
        v["spot"] = Spot(self, v)
        return v

    def add_vert_on(self, e, tick):
        (e, eid) = self.sanitize_edge(e)
        former = self.graph.vs[e.source]
        latter = self.graph.vs[e.target]
        old_branch = e["branch"]
        e1 = self.add_edge(former, i, old_branch)
        e2 = self.add_edge(i, latter, old_branch)
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
        if isinstance(vert_from, Vertex)
            vert_from = vert_from.index
        if isinstance(vert_to, Vertex)
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
            if self.vertex_in_branch(v):
                return e
            return None
        for e in v.incident(mode=IN):
            v_to = self.graph.vs[e.target]
            if e in self.branch_members[branch]:
                if v_to["tick"] > tick:
                    return e
                else:
                    # None of these edges are right! You want the ones
                    # after this vertex here.
                    return self.successor_on(v_to, branch, tick)
            elif v_to["tick"] >= tick:
                return e
        return None
        
    def split_branch(self, old_branch, new_branch, tick, length=0):
        """Find the edge in old_branch in the given tick, split it, and start
a new edge off the split. The new edge will be a member of
new_branch.

        """
        e = self.get_edge(old_branch, tick)
        (e1, v1, e2) = self.add_vert_on(e, tick)
        v2 = self.add_vert(tick=tick+length)
        return self.add_edge(v1, v2, new_branch, length)

    def latest_edge(self, branch):
        """Return the edge in the given branch that ends on the highest
tick."""
        edges = set(self.branch_edges[branch])
        late = edges.pop()
        v_late = self.graph.vs[late.target]
        while len(edges) > 0:
            e = edges.pop()
            v_e = self.graph.vs[e.target]
            if v_e["tick"] > v_late["tick"]:
                late = e
                v_late = v_e
        return late
