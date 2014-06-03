# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from networkx import DiGraph
from networkx import shortest_path as sp
from LiSE.orm import SaveableMetaclass
from thing import Thing
from place import Place
from portal import Portal


class AbstractCharacter(object):
    """Basis for classes implementing the Character API.

    The Character API is a graph structure built to reflect the state
    of its corner of the game world at the present time. Places and
    Portals are nodes and edges in this graph. Their stats and
    contents are accessible here as well.

    """
    __metaclass__ = SaveableMetaclass

    def __init__(self, closet, name):
        """Store the closet and name, and initialize dictionaries in which to
        store accessor objects.

        """
        self.closet = closet
        self.name = name
        self.thing_d = {}
        self.thing_loc_d = {}
        self.thing_contents_d = {}
        self.thing_stat_d = {}
        self.place_d = {}
        self.portal_d = {}

    def __eq__(self, other):
        """Compare based on the name and the closet"""
        return (
            hasattr(other, 'name') and
            hasattr(other, 'closet') and
            self.closet is other.closet and
            self.name == other.name
        )

    def __hash__(self):
        """Hash of the name"""
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
        """Generate and return a DiGraph representing all the Places,
        Portals, and Things in me.

        To find what's in me, I iterate over
        ``self._current_bones()``. Anything not yielded by that method
        doesn't go in the graph. Things yielded by that method may be
        prevented from going into the graph if the relevant validation
        method returns ``False``.

        Validation methods:
        * _has_place_named
        * _has_portal_between
        * _has_thing_named
        * _place_has_stat
        * _portal_has_stat
        * _thing_has_stat
        * _character_has_stat

        These should not be used for any other purpose! If you
        actually want to know whether a Place exists at the moment,
        look for its node in the graph.

        """
        self.thing_contents_d = {}
        self.thing_stat_d = {}
        r = DiGraph()

        def cast(b):
            if b.type == 'text':
                return b.value
            elif b.type == 'real':
                return float(b.value)
            elif b.type == 'boolean':
                return bool(b.value)
            elif b.type == 'integer':
                return int(b.value)
            else:
                raise TypeError("Unsupported stat type: {}".format(b.type))

        def add_place(name):
            if name not in self.place_d:
                self.place_d[name] = Place(self, name)
            place = self.place_d[name]
            if name not in r.node:
                r.add_node(
                    name,
                    {
                        "place": place,
                        "contents": set()
                    }
                )
            return place

        def add_portal(origin, destination):
            add_place(origin)
            add_place(destination)
            if origin not in self.portal_d:
                self.portal_d = {}
            if destination not in self.portal_d[origin]:
                self.portal_d[origin][destination] = Portal(
                    self, origin, destination)
            portal = self.portal_d[origin][destination]
            if destination not in r.edge[origin]:
                r.add_edge(
                    origin,
                    destination,
                    {
                        "portal": portal,
                        "contents": set()
                    }
                )
            return portal

        def add_thing(name):
            if name not in self.thing_d:
                self.thing_d[name] = Thing(self, name)
            return self.thing_d[name]

        def process_portal_stat_bone(b):
            if not (
                    self._has_portal_between(b.origin, b.destination) and
                    self._portal_has_stat(b.origin, b.destination, b.key)
            ):
                return
            add_portal(b.origin, b.destination)
            r.edge[b.origin][b.destination][b.key] = cast(b)

        things2b = {}

        def process_thing_stat_bone(b):
            if not (
                    self._has_thing_named(b.name) and
                    self._thing_has_stat(b.name, b.key)
            ):
                return
            thing = add_thing(b.name)
            if b.key == 'location':
                if '->' in b.value:
                    (origin, destination) = b.value.split('->')
                    r.edge[origin][destination]["contents"].add(thing)
                else:
                    if b.value in r.node:
                        r.node[b.value]["contents"].add(thing)
                    else:
                        things2b[b.name] = b.value
            else:
                if b.name not in self.thing_stat_d:
                    self.thing_stat_d[b.name] = {}
                self.thing_stat_d[b.name][b.key] = cast(b)

        def postprocess_things():
            for thingn in things2b:
                thing = self.thing_d[thingn]
                if things2b[thingn] in things2b:
                    locn = things2b[thingn]
                    if locn not in self.thing_contents_d:
                        self.thing_contents_d[locn] = set()
                    self.thing_contents_d[locn].add(thing)
                elif thingn in r.node:
                    r.node[locn].contents.add(thing)
                else:
                    r.add_node(
                        locn,
                        {
                            "place": add_place(locn),
                            "contents": set([thing])
                        }
                    )

        def process_place_stat_bone(b):
            if not (
                    self.has_place_named(b.name) and
                    self.place_has_stat(b.name, b.key)
            ):
                return
            add_place(b.name)
            r.node[b.name][b.key] = cast(b)

        def process_character_stat_bone(b):
            if not self.character_has_stat(b.key):
                return
            r.graph[b.key] = cast(b)

        def shortest_path(o, d, weight=''):
            return sp(r, o, d, weight)
        shortest_path.__doc__ = sp.__doc__
        r.shortest_path = shortest_path

        for bone in self._current_bones():
            if isinstance(bone, Place.bonetype):
                process_place_stat_bone(bone)
            elif isinstance(bone, Portal.bonetype):
                process_portal_stat_bone(bone)
            elif isinstance(bone, Thing.bonetype):
                process_thing_stat_bone(bone)
            elif isinstance(bone, Character.bonetypes["character_stat"]):
                process_character_stat_bone(bone)
            else:
                raise TypeError("Unknown bonetype")
        postprocess_things()
        return r

    def _has_thing_named(self, name):
        raise NotImplementedError("Abstract class")

    def _has_place_named(self, name):
        raise NotImplementedError("Abstract class")

    def _has_portal_between(self, origin, destination):
        raise NotImplementedError("Abstract class")

    def _thing_has_stat(self, name, key):
        raise NotImplementedError("Abstract class")

    def _place_has_stat(self, name, key):
        raise NotImplementedError("Abstract class")

    def _portal_has_stat(self, origin, destination, key):
        raise NotImplementedError("Abstract class")

    def _character_has_stat(self, key):
        raise NotImplementedError("Abstract class")

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
        pathgraph = (
            graph_for_pathfinding
            if graph_for_pathfinding else movegraph
        )
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
    """A part of the world model, comprising a graph, things in the graph,
    and stats of items in the graph.

    A Character instance's ``graph`` attribute will always give a
    representation of the character's state at the present
    sim-time.

    """
    provides = ["character"]
    tables = [
        (
            "character_stat",
            {
                "columns":
                {
                    "character": "text not null",
                    "key": "text not null",
                    "branch": "integer not null default 0",
                    "tick": "integer not null default 0",
                    "value": "text"
                },
                "primary_key":
                ("character", "key", "branch", "tick")
            }
        ),
        (
            "character_avatar",
            {
                "columns":
                {
                    "character": "text not null",
                    "host": "text not null default 'Physical'",
                    "type": "text not null default 'thing'",
                    "name": "text not null"
                },
                "primary_key":
                ("character", "host", "type", "name"),
                "checks":
                ["type in ('thing', 'place', 'portal')"]
            }
        )
    ]

    def __init__(self, closet, name):
        self.closet = closet
        self.name = name

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def _has_thing_named(self, name):
        return True

    def _has_place_named(self, name):
        return True

    def _has_portal_between(self, origin, destination):
        return True

    def _thing_has_stat(self, name, key):
        return True

    def _place_has_stat(self, name, key):
        return True

    def _portal_has_stat(self, name, key):
        return True

    def _character_has_stat(self, key):
        return True

    def _current_bones(self):
        (branch, tick) = self.closet.timestream.time
        skel = self.closet.skeleton
        thingskel = skel[u'thing_stat'][self.name]
        placeskel = skel[u'place_stat'][self.name]
        portalskel = skel[u'portal_stat'][self.name]
        charskel = skel[u'character_stat'][self.name]
        # TODO: more efficient iteration?
        for thing in thingskel:
            for key in thingskel[thing]:
                if branch in thingskel[thing][key]:
                    yield thingskel[thing][key][branch].value_during(tick)
        for place in placeskel:
            for key in placeskel[place]:
                if branch in placeskel[place][key]:
                    yield placeskel[place][key][branch].value_during(tick)
        for origin in portalskel:
            for destination in portalskel[origin]:
                for key in portalskel[origin][destination]:
                    if branch in portalskel[origin][destination][key]:
                        yield portalskel[origin][destination][
                            key][branch].value_during(tick)
        for key in charskel:
            if branch in charskel[key]:
                yield charskel[key][branch].value_during(tick)


class AbstractFacade(AbstractCharacter):
    """View onto one Character as seen by another."""
    def __init__(self, observer, observed):
        self.observer = observer
        self.observed = observed
        self.closet = self.observed.closet
        self.thing_d = {}
        self.thing_loc_d = {}
        self.thing_contents_d = {}
        self.thing_stat_d = {}
        self.place_d = {}
        self.portal_d = {}

    def __eq__(self, other):
        return (
            hasattr(other, 'observer') and
            hasattr(other, 'observed') and
            self.observer == other.observer and
            self.observed == other.observed)

    def __hash__(self):
        return hash((self.observer, self.observed))

    def __unicode__(self):
        return u"Facade({},{})".format(self.observer, self.observed)

    def _current_bones(self):
        """Based on the bones in the observed character, yield the bones that
        should be presented to the observer.

        The default implementation defers to the methods
        ``gen_thing_stat_bones``, ``gen_place_stat_bones``,
        ``gen_portal_stat_bones``, and ``gen_character_stat_bones``.

        """
        raise NotImplementedError("Abstract class")


class NullFacade(AbstractFacade):
    """Facade that's a simple passthrough to the observed Character."""

    def _has_thing_named(self, name):
        return self.observed._has_thing_named(name)

    def _has_place_named(self, name):
        return self.observed._has_place_named(name)

    def _has_portal_between(self, origin, destination):
        return self.observed._has_portal_between(origin, destination)

    def _thing_has_stat(self, name, key):
        return self.observed._thing_has_stat(name, key)

    def _place_has_stat(self, name, key):
        return self.observed._place_has_stat(name, key)

    def _portal_has_stat(self, observer, observed, key):
        return self.observed._portal_has_stat(observer, observed, key)

    def _character_has_stat(self, key):
        return self.observed._character_has_stat(key)

    def _current_bones(self):
        for bone in self.observed._current_bones():
            yield bone


def truth(*args):
    return True


def falsity(*args):
    return False


class DecoratorFacade(NullFacade):
    """A Facade whose behavior is defined by non-method functions that
    can be swapped out at runtime.

    """
    tables = [
        (
            "decorator_facade_config",
            {
                "columns":
                {
                    "observer": "text not null",
                    "observed": "text not null",
                    "decorator": "text not null",
                    "branch": "integer not null default 0",
                    "tick": "integer not null default 0",
                    "function": "text"
                },
                "primary_key":
                ("observer", "observed", "decorator", "branch", "tick"),
                "foreign_keys":
                {
                    "observer, observed, 'decorator'":
                    ("facade", "observer, observed, type")
                },
                "checks":
                [
                    "decorator in ('@hasthing', '@hasplace', '@hasportal', "
                    "'@thingstat', '@placestat', '@portalstat', '@charstat', "
                    "'@curbones')"
                ]
            }
        )
    ]
    funcs = {
        "true": truth,
        "false": falsity
    }

    def __init__(self, observer, observed):
        super(DecoratorFacade, self).__init__(observer, observed)
        (branch, tick) = observed.closet.timestream.time
        skel = observed.closet.skeleton[u'decorator_facade_config']
        obsrvr = unicode(observer)
        obsrvd = unicode(observed)
        if obsrvr not in skel:
            skel[obsrvr] = {}
        if obsrvd not in skel[obsrvr]:
            skel[obsrvr][obsrvd] = {}
        skel = skel[obsrvr][obsrvd]
        for decorator in skel:
            if decorator not in skel:
                skel[decorator] = {}
            if branch not in skel[decorator]:
                continue
            bone = skel[branch].value_during(tick)
            fun = self.funcs[bone.function]
            {
                '@hasthing': self.hasthing,
                '@hasplace': self.hasplace,
                '@hasportal': self.hasportal,
                '@thingstat': self.thingstat,
                '@placestat': self.placestat,
                '@portalstat': self.portalstat,
                '@charstat': self.charstat,
                '@curbones': self.curbones
            }[decorator](fun)

    def hasthing(self, fun):
        self._has_thing_named = fun

    def hasplace(self, fun):
        self._has_place_named = fun

    def hasportal(self, fun):
        self._has_portal_between = fun

    def thingstat(self, fun):
        self._thing_has_stat = fun

    def placestat(self, fun):
        self._place_has_stat = fun

    def portalstat(self, fun):
        self._portal_has_stat = fun

    def charstat(self, fun):
        self._character_has_stat = fun

    def curbones(self, fun):
        self._current_bones = fun


class Facade(AbstractFacade):
    """A class that, instead of instantiating itself, turns into some
    other subclass of AbstractFacade.

    """
    tables = [
        (
            "facade",
            {
                "columns":
                {
                    "observer": "text not null",
                    "observed": "text not null",
                    "type": "text not null"
                },
                "primary_key":
                ("observer", "observed")
            }
        )
    ]
    classes = [
        NullFacade,
        DecoratorFacade
    ]
    default_cls = DecoratorFacade

    def __new__(cls, observer, observed):
        obsrvr = unicode(observer)
        obsrvd = unicode(observed)
        skel = observed.closet.skeleton[u'facade']
        if obsrvr not in skel or obsrvd not in skel[obsrvr]:
            return cls.mk_default(observer, observed)
        b = skel[obsrvr][obsrvd]
        if (
                b.type in SaveableMetaclass.clsmap and
                SaveableMetaclass.clsmap[b.type] in Facade.classes
        ):
            return SaveableMetaclass.clsmap[b.type](
                observer,
                observed
            )
        return cls.mk_default(observer, observed)

    @classmethod
    def mk_default(cls, observer, observed):
        return cls.default_cls(observer, observed)

    @classmethod
    def defaultmaker(cls, fun):
        cls.mkdefault = fun

    @classmethod
    def default(cls, fun):
        cls.default_cls = fun
