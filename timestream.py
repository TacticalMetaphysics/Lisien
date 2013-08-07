from igraph import Graph, Vertex, Edge, IN
from board import Board
from spot import AbstractSpot
from arrow import Arrow
from collections import defaultdict
import pyglet

class Spot(AbstractSpot):
    def __init__(self, board, v):
        super(Spot, self).__init__(board, v, saveable=False)

class Timestream(Board):
    """The extra, special board that shows the various ways a timeline can
    branch and lets you browse it and stuff.

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
    def __init__(self, branchdict, width, height, wallpaper, timescale=1.0):
        self.branchdict = branchdict
        self.width = width
        self.height = height
        self.wallpaper = wallpaper
        self.graph = Graph(directed=True)
        self.graph.add_vertices(2)
        self.graph.vs["tick"] = [0, 1]
        self.graph.vs["branch"] = [0, 1]
        self.graph.add_edge(0, 1)
        self.branch_heads = {0: self.graph.vs[0]}
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
        branches_updated = set()
        for (
                branch, (parent_branch, tick_from, tick_to)
                ) in self.branchdict.iteritems():
            if branch in self.branch_heads and branch not in branches_updated:
                self.update_branch(branch, parent_branch, tick_from, tick_to)
                branches_updated.add(branch)
            else:
                e = self.split_branch(parent_branch, tick_from, tick_to)
                self.branch_edges[branch].add(e)

    def update_branch(self, branch, parent_branch, tick_from, tick_to):
        branch_edges = self.branch_edges[branch]
        branch_vertis = set()
        for edge in iter(branch_edges):
            branch_vertis.add(edge.source)
            branch_vertis.add(edge.target)
        branch_verts = [self.graph.vs[i] for i in iter(branch_vertis)]
        branch_ticks = [v["tick"] for v in branch_verts]
        earliest_tick = min(branch_ticks)
        latest_tick = max(branch_ticks)
        

    def vertex_in_branch(self, v, branch):
        for e in self.graph.incident(v, mode=IN):
            if e in self.branch_edges[branch]:
                return True
        return False

    def add_edge(self, v, branch, tick_to):
        if isinstance(v, Vertex):
            vid = v.index
        else:
            vid = v
            v = self.graph.vs[v]
        if branch not in self.branch_heads:
            self.branch_heads[branch] = v
        i = len(self.graph.vs)
        self.graph.add_vertex(tick=tick_to)
        v2 = self.graph.vs[i]
        v2["spot"] = Spot(self, v2)
        self.graph.add_edge(vid, i)
        eid = self.graph.get_eid(vid, i)
        e = self.graph.es[eid]
        self.branch_edges[branch].add(e)
        return e

    def get_edge(self, vert_from, vert_to):
        if isinstance(vert_from, Vertex)
            vert_from = vert_from.index
        if isinstance(vert_to, Vertex)
            vert_to = vert_to.index
        eid = self.graph.get_eid(vert_from, vert_to)
        return self.graph.es[eid]

    def add_vert_on(self, eid, tick):
        if isinstance(eid, Edge):
            e = eid
        else:
            e = self.graph.es[eid]
        former = self.graph.vs[e.source]
        latter = self.graph.vs[e.target]
        if (
                former.tick > tick or
                latter.tick < tick):
            raise Exception("Impossible node")
        branch = former["branch"]
        self.branch_edges[branch].discard(e)
        e["arrow"].delete()
        self.graph.delete_edges(eid)
        i = len(self.graph.vs)
        self.graph.add_vertex(tick=tick)
        v = self.graph.vs[i]
        v["spot"] = Spot(self, v)
        new_edges = [(former, i), (i, latter)]
        self.graph.add_edges(new_edges)
        for (ifrom, ito) in new_edges:
            self.branch_edges[branch].add(self.get_edge(ifrom, ito))
        return v

    def edge_at(self, branch, tick):
        """Return the edge that contains the given tick in the given branch,
or None if no such edge exists."""
        v = self.branch_heads[branch]
        if tick < v["tick"]:
            raise Exception("This branch started after that tick.")
        return self.successor_on(v_to, branch, tick)

    def successor_on(self, v, branch, tick):
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
                    # None of these edges are right! You want the ones after this vertex here.
                    return self.successor_on(v_to, branch, tick)
            elif v_to["tick"] >= tick:
                return e
        return None
        
    def extend_branch(self, branch, n):
        v1 = self.branch_heads[branch]
        e = None
        for edge in v1.incident(mode=IN):
            if edge in self.branch_edges[branch]:
                e = edge
                break
        if e is None:
            raise Exception("There's no edge in that branch yet")
        v2 = self.graph.vs[e.target]
        v2["tick"] += n

    def split_branch(self, old_branch, new_branch, tick_from, tick_to):
        """Find the edge in old_branch in the given tick, split it,
and start a new edge off the split. The new edge will extend to tick_to, and will be a member of new_branch

        """
        e = self.edge_at(old_branch, tick_from)
        v1 = self.add_vert_on(e, tick_from)
        i = 
