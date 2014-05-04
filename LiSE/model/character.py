# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from networkx import MultiDiGraph
from networkx import shortest_path as sp
from collections import defaultdict
from LiSE.orm import SaveableMetaclass
from thing import Thing
from place import Place
from portal import Portal


class AbstractCharacter(object):
    __metaclass__ = SaveableMetaclass

    def __init__(self, closet, name):
        self.closet = closet
        self.name = name
        self.thing_d = {}
        self.thing_loc_d = {}
        self.thing_contents_d = {}
        self.thing_stat_d = {}
        self.place_d = {}
        self.portal_d = {}

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            hasattr(other, 'closet') and
            self.closet is other.closet and
            self.name == other.name)

    def __hash__(self):
        return hash(self.name)

    def __getitem__(self, key):
        return self.graph.graph[key]

    @property
    def graph(self):
        return self.get_graph()

    def get_graph(self):
        # TODO avoid regenerating the graph unnecessarily
        return self.make_graph()

    def make_graph(self):
        """Generate and return a MultiDiGraph representing all the Places,
        Portals, and Things in me.

        """
        # Initialize the dicts that hold state.  thing_d, place_d, and
        # portal_d only hold stateless objects that are views onto me.
        self.thing_loc_d = {}
        self.thing_contents_d = {}
        self.thing_stat_d = defaultdict(dict)
        r = MultiDiGraph()
        portals_done = set()
        for portalbone in self.iter_portal_loc_bones():
            if portalbone.name in portals_done:
                continue
            for placename in (portalbone.origin, portalbone.destination):
                if placename not in self.place_d:
                    self.place_d[placename] = Place(self, placename)
                place = self.place_d[placename]
                if placename not in r.node:
                    r.add_node(placename, {
                        'place': place,
                        'contents': set()})
            if portalbone.name not in self.portal_d:
                self.portal_d[portalbone.name] = Portal(self, portalbone.name)
            portal = self.portal_d[portalbone.name]
            r.add_edge(portalbone.origin, portalbone.destination, {
                'name': portalbone.name,
                'portal': portal,
                'contents': set()})
            portals_done.add(portalbone.name)
        create_places_after_thing = {}
        for thingbone in self.iter_thing_loc_bones():
            if thingbone.name not in self.thing_d:
                self.thing_d[thingbone.name] = Thing(self, thingbone.name)
            thing = self.thing_d[thingbone.name]
            if thingbone.location in r.node:
                r.node[thingbone.location]['contents'].add(thing)
            elif thingbone.location in self.portal_d:
                portal = self.portal_d[thingbone.location]
                edge = r.node[portal.origin][portal.destination]
                edge["contents"].add(thing)
            elif thingbone.location in self.thing_d:
                self.thing_contents_d[thingbone.location].add(thing)
                if thingbone.name in create_places_after_thing:
                    del create_places_after_thing[thingbone.name]
            else:
                create_places_after_thing[thingbone.name] = thingbone.location
            self.thing_loc_d[thingbone.name] = thingbone.location
        for (thingn, locn) in create_places_after_thing.iteritems():
            thing = self.thing_d[thingn]
            if locn in self.thing_d:
                if locn in self.thing_contents_d:
                    self.thing_contents_d[locn].add(thing)
                else:
                    self.thing_contents_d[locn] = set([thing])
            elif locn in self.portal_d:
                pbone = self.portal_d[locn].get_loc_bone()
                r[pbone.origin][pbone.destination]['contents'].add(thing)
            elif locn not in r.node:
                r.add_node(locn,
                           {'contents': set([thing])})
            else:
                r.node[locn]['contents'].add(thing)
        for bone in self.iter_place_stat_bones():
            r.node[bone.name][bone.key] = bone.value
        for bone in self.iter_portal_stat_bones():
            r[bone.origin][bone.destination][bone.key] = bone.value
        for bone in self.iter_thing_stat_bones():
            self.thing_stat_d[bone.name][bone.key] = bone.value
        for bone in self.iter_char_stat_bones():
            r.graph[bone.key] = bone.value

        def shortest_path(o, d, weight=''):
            return sp(r, o, d, weight)
        shortest_path.__doc__ = sp.__doc__
        r.shortest_path = shortest_path
        return r

    def iter_portal_names(self):
        already = set()
        for bone in self.iter_portal_loc_bones():
            if bone.name not in already:
                yield bone.name
                already.add(bone.name)

    def iter_thing_names(self):
        already = set()
        for bone in self.iter_thing_loc_bones():
            if bone.name not in already:
                yield bone.name
                already.add(bone.name)

    def transport_thing_to(self, thingn, destn, graph_for_pathfinding=None):
        """Set the thing to travel along the shortest path to the destination.

        It will spend however long it must in each of the places and
        portals along the path.

        With optional argument graph_for_pathfinding, it will try to
        follow the shortest path it can find in the given graph--which
        might cause it to stop moving earlier than expected, if the
        path it finds doesn't exist within me.

        """
        movegraph = self.graph
        pathgraph = graph_for_pathfinding\
            if graph_for_pathfinding else movegraph
        o = self.thing_loc_d[thingn]
        path_sl = pathgraph.shortest_path(o, destn)
        placens = iter(path_sl)
        placen = next(placens)
        path_ports = []
        for placenxt in placens:
            try:
                path_ports.append(movegraph[placen][placenxt]['portal'])
                placen = placenxt
            except KeyError:
                break
        self.thing_d[thingn].follow_path(path_ports)


class Character(AbstractCharacter):
    provides = ["character"]
    tables = [
        ("character_stat", {
            "columns": {
                "character": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "key", "branch", "tick")})]

    def __init__(self, closet, name):
        self.closet = closet
        self.name = name

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def _iter_bones(self, tabn, xkeys=[]):

        """Iterate over the bones that are hosted in a tab, at a branch and
        tick.

        """
        skel = self.closet.skeleton[tabn][unicode(self)]
        (branch, tick) = self.closet.sanetime()
        for name in skel:
            subskel = skel[name]
            for key in xkeys:
                subskel = subskel[key]
            if branch in subskel:
                yield subskel[branch].value_during(tick)

    def iter_thing_loc_bones(self):
        """Iterate over all the thing_loc bones that are active at the
        moment.

        """
        for b in self._iter_bones(u"thing_loc"):
            yield b

    def iter_portal_loc_bones(self):
        """Iterate over all the portal_loc bones that are active at the
        moment.

        """
        for b in self._iter_bones(u"portal_loc"):
            yield b


class AbstractFacade(AbstractCharacter):
    """View onto one Character as seen by another."""
    def __init__(self, observer, observed):
        self.observer = observer
        self.observed = observed


class PlugboardFacade(AbstractFacade):
    """A Facade whose behavior--what to include, what to exclude, and what
    to include even though it isn't in the underlying Character--is
    defined by non-method functions that can be swapped out at
    runtime."""
    def iter_thing_loc_bones(self):
        for bone in self.iter_extra_thing_loc_bones():
            yield bone
        for bone in self.observed.iter_thing_loc_bones():
            if not self.omit_thing_loc_bone(bone):
                yield self.distort_thing_loc_bone(bone)

    def iter_portal_loc_bones(self):
        for bone in self.iter_extra_portal_loc_bones():
            yield bone
        for bone in self.observed.iter_portal_loc_bones():
            if not self.omit_portal_loc_bone(bone):
                yield self.distort_portal_loc_bone(bone)


class NullFacade(AbstractFacade):
    """Facade that's a simple passthrough to the observed Character."""
    def __init__(self, observer, observed):
        super(NullFacade, self).__init__(observer, observed)
        for replicant in (
                "iter_thing_loc_bones",
                "iter_portal_loc_bones",
                "iter_place_stat_bones",
                "iter_portal_stat_bones",
                "iter_thing_stat_bones"):
            setattr(self, replicant, getattr(observed, replicant))


class TransformerFacade(AbstractFacade):
    """Facade class that turns itself into some other Facade class, based
    on what characters it's about."""
    def __new__(cls, observer, observed):
        return cls.facade_cls_map[observer][observed](observer, observed)


class Facade(TransformerFacade):
    """Default Facade implementation. Turns into whatever kind of Facade
    is specified in the database-global variable
    default_facade_cls."""
    default_facade_cls_map = {
        "PlugboardFacade": PlugboardFacade,
        "NullFacade": NullFacade}

    def __new__(cls, observer, observed):
        return cls.default_facade_cls_map[
            observed.closet.get_global("default_facade_cls")](
            observer, observed)
