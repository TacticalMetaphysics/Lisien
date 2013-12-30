# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from igraph import Graph, InternalError

from LiSE.util import (
    selectif,
    upbranch,
    KnowledgeException,
    SaveableMetaclass,
    Skeleton)
from thing import Thing
from place import Place
from portal import Portal


"""Things that should have character sheets."""


def skelget(tl, skel, branch, tick):
    while branch != 0:
        try:
            return skel[branch].value_during(tick)
        except KeyError:
            branch = tl.parent(branch)
    # may throw KeyError
    return skel[0].value_during(tick)


def skel_filter_iter(skel, keys=[]):
    if len(keys) == 0:
        for v in skel.itervalues():
            yield v
    else:
        for k in keys:
            yield skel[k]


def iter_skel_keys(key_skel, branches=[], ticks=[]):
    accounted = set()
    for key in key_skel:
        if key in accounted:
            break
            for branch_skel in skel_filter_iter(key_skel[key], branches):
                if key in accounted:
                    break
                    for tick_skel in skel_filter_iter(branch_skel, ticks):
                        if key in accounted:
                            break
                            accounted.add(key)
                            yield key


class Character(object):
    __metaclass__ = SaveableMetaclass
    """A collection of :class:`Thing`, :class:`Place`, :class:`Portal`,
    and possibly even :class:`Dimension` instances that are regarded
    as one entity for some purpose.

    :class:`Character` is chiefly inspired by the concept from
    tabletop roleplaying game systems. These are normally assumed to
    represent the sort of entity that has a body, a mind, and some
    ability to affect the game world, but none of those are strictly
    needed for a character. In the roleplaying system _GURPS_, it is
    not unusual to make characters that represent vegetables, mindless
    robots, swarms of insects, vehicles, or spirits. LiSE characters
    are meant to be at least that flexible.

    LiSE assumes only that characters are diegetic items. That means
    they exist in the simulated universe, and that they are composed
    of some collection of simulation elements, including physical
    objects as well as abstractions. A quality of a physical object
    may be part of a character, even when the object itself is not,
    and likewise, characters may have qualities that no particular
    object in the simulation possesses; but even in those cases,
    characters "exist," and therefore, interact with other things that
    exist in the simulation. They do not interact with anything
    outside the simulation. It is senseless for :class:`Character` to
    perform file operations or make network connections, unless your
    game *simulates* those things, and the character is a *simulation*
    of software on a *simulated* computer. This is the meaning of the
    term "diegetic" in LiSE.

    An instance of :class:`Character` is a collection of elements in
    the simulation, of classes :class:`Thing`, :class:`Place`,
    :class:`Portal`. To deal with elements too abstract or otherwise
    inconvenient to represent with those classes, you may assign
    "stats" to a character. All of these parts are subject to change
    as time passes in the simulation. Those changes will be recorded,
    such that if you rewind time in the simulation, you get the old
    values.

    Stats are strings. There are no special restrictions on their
    values. Assigning a stat to a character results in the creation of
    a new :class:`Cause` to indicate that the stat is present. If the
    value of the stat can make a difference to whether an event
    triggers, you need to create your own :class:`Cause` for that
    case.

    """
    demands = ["thing"]
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

    def __init__(self, closet, name, knows_self=True,
                 self_omitters=[], self_liars=[]):
        """Initialize a character from the data in the closet.

        A character is a collection of items in the game world that
        are somehow related. The significance of that relation is
        arbitrary--sometimes it determines what laws of physics apply
        to the items, but it might just be a set of collectible cards
        scattered about for somebody to put together. In most games,
        there will be one character that represents the physical
        reality that all of the other characters inhabit. That
        character is conventionally named 'Physical'. Characters
        representing people will have items in 'Physical' to represent
        the characters' bodies, and items elsewhere to represent their
        minds, social relations, and whatever spiritual notions you
        choose to model in your game.

        LiSE divides the world of items into :class:`Thing`,
        :class:`Place`, and :class:`Portal`. Things are located in
        other items. Portals connect other items together. Places do
        nothing but provide points of reference for other items. If
        you need a place that moves, you can use a thing
        instead--things can be located inside other things if you
        like.

        The places and portals within a given character are connected
        together in its graph. Things refer to that graph to say where
        they are. Places can only exist in the character upon which
        they are defined, but that restriction does not apply to
        portals or things, both of which have a 'host' character that
        may or may not be the character on which they are
        defined. Thus, parts of some characters can occupy other
        characters. That makes it possible for the physical body of a
        person to occupy someplace in the character called 'Physical',
        and thereby exist in the physical world without necessarily
        being a part of it.

        Note that it is possible for something to be part of more than
        one character at a time. This is desirable for when, eg., six
        adventurers band together to form a party, which looks a lot
        more intimidating as a group than any of its individual
        members. The party in that instance would be a new chanacter
        containing all the same things as the six who make it up, and
        it would have its own intimidation stat, among others.

        """
        self.closet = closet
        self.name = name
        if unicode(self) not in self.closet.skeleton[u"thing"]:
            self.closet.skeleton[u"thing"][unicode(self)] = {}
        if unicode(self) not in self.closet.skeleton[u"portal"]:
            self.closet.skeleton[u"portal"][unicode(self)] = {}
        self.thing_d = dict([
            (thingn, Thing(self, thingn)) for thingn in
            self.closet.skeleton[u"thing"][unicode(self)]])
        self.facade_d = {}
        if knows_self:
            self.facade_d[unicode(self)] = Facade(
                observer=self, observed=self, omitters=self_omitters,
                liars=self_liars)
        self.graph = Graph(directed=True)
        self.closet.character_d[unicode(self)] = self
        self.update()

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def _get_skel(self):
        """Get the skeleton from the closet. Debugging method."""
        return self.closet.skeleton

    def update(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        for placebone in self.iter_place_bones(
                None, branch, tick):
            if (
                    "name" not in self.graph.vs.attributes() or
                    placebone.place not in self.graph.vs["name"]):
                print("Creating place {} in a graph".format(placebone.place))
                self.make_place(placebone.place)
        for bone in self.iter_hosted_portal_loc_bones(branch, tick):
            try:
                oi = self.graph.vs["name"].index(bone.origin)
            except ValueError:
                oi = len(self.graph.vs)
            try:
                di = self.graph.vs["name"].index(bone.destination)
            except ValueError:
                di = len(self.graph.vs)
            for placen in (bone.origin, bone.destination):
                if placen not in self.graph.vs["name"]:
                    self.make_place(placen)
            try:
                self.graph.get_eid(oi, di)
            except (ValueError, InternalError):
                print("adding edge of {}".format(bone.name))
                i = len(self.graph.es)
                self.graph.add_edge(
                    oi, di, name=bone.name)
                # the portal is merely *hosted* here.
                # it should know what it's a *part* of, too
                character = self.closet.get_character(bone.character)
                self.graph.es[i]["portal"] = Portal(character, bone.name)
        for v in self.graph.vs:
            try:
                self.get_place_bone(v["name"], branch, tick)
            except (KeyError, KnowledgeException):
                self.graph.delete_vertices(v)
        for e in self.graph.es:
            try:
                self.get_portal_loc_bone(e["name"], branch, tick)
            except (KeyError, KnowledgeException):
                self.graph.delete_edges(e)

    def get_facade(self, observer):
        if observer is None:
            raise ValueError("Every facade must have an observer.")
        if unicode(observer) not in self.facade_d:
            self.facade_d[unicode(observer)] = Facade(
                observer, observed=self)
        return self.facade_d[unicode(observer)]

    def sanetime(self, branch=None, tick=None):
        """If branch or tick are None, replace them with the current value
        from the closet.

        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        return (branch, tick)

    def get_bone(self, name):
        """Try to get the bone for the named item without knowing what type it
        is.

        First try Place, then Portal, then Thing.

        """
        try:
            return self.get_place(name)
        except KeyError:
            try:
                return self.get_portal(name)
            except KeyError:
                return self.get_thing(name)

    def get_whatever(self, bone):
        """Get the item of the appropriate type, based on the type of the bone
        supplied."""
        return {
            Place.bonetype: self.get_place,
            Portal.bonetypes.portal: self.get_portal,
            Thing.bonetypes.thing: self.get_thing}[type(bone)](bone.name)

    ### Thing

    def get_thing(self, name):
        """Return a thing already created."""
        return self.thing_d[name]

    def make_thing(self, name):
        """Make a thing, remember it, and return it."""
        self.thing_d[name] = Thing(self, name)
        return self.thing_d[name]

    def _get_thing_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timestream, skel, branch, tick)

    def get_thing_bone(self, name):
        """Return a bone describing a thing's location at some particular
        time."""
        return self.closet.skeleton[u"thing"][unicode(self)][name]

    def get_thing_stat_skel(self, name, stat, branch=None):
        r = self.closet.skeleton[u"thing_stat"][unicode(self)][name][stat]
        if branch:
            return r[branch]
        return r

    def iter_thing_stat_bones(self, name, stat, branch=None):
        skel = self.closet.skeleton[u"thing_stat"][unicode(self)][name][stat]
        if branch is not None:
            skel = skel[branch]
        for bone in skel.iterbones():
            yield bone

    def get_thing_stat_bone(self, thing, stat, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.get_thing_stat_skel(thing, stat, branch).value_during(tick)

    def get_thing_stat(self, thing, stat, branch=None, tick=None):
        return self.get_thing_stat_bone(thing, stat, branch, tick).value

    def _iter_thing_skel_bones(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        for name in skel:
            yield skelget(self.closet.tl, skel, branch, tick)

    def iter_things(self, branch=None, tick=None):
        """Iterate over things that have been defined for this character,
        possibly creating them. If a time is given, the iteration will
        be restricted to things that exist then.

        """
        try:
            skel = self.closet.skeleton[u"thing_loc"][unicode(self)]
            leks = self.closet.skeleton[u"thing"][unicode(self)]
        except KeyError:
            return
        if branch is None and tick is None:
            boneiter = leks.iterbones()
        elif branch is None:
            def bitter():
                for thingn in skel:
                    for branch in skel[thingn]:
                        if skel[thingn][branch].key_or_key_before(
                                tick) is None:
                            continue
                        yield leks[thingn]
            boneiter = bitter()
        elif tick is None:
            def bitter():
                for thingn in skel:
                    if branch in skel[thingn]:
                        yield leks[thingn]
            boneiter = bitter()
        else:
            def bitter():
                for thingn in skel:
                    if branch in skel[thingn]:
                        if skel[thingn][branch].key_or_key_before(
                                tick) is None:
                            continue
                        yield leks[thingn]
            boneiter = bitter()
        for bone in boneiter:
            try:
                r = self.get_thing(bone.name)
            except KeyError:
                r = self.make_thing(bone.name)
            assert(r is not None)
            yield r

    def iter_thing_bones(self, branch=None, tick=None):
        """Iterate over all things present in this character at the time
        specified, or the present time if not specified."""
        skel = self.closet.skeleton[u"thing"][unicode(self)]
        for bone in self._iter_thing_skel_bones(skel, branch, tick):
            yield bone

    def iter_thing_loc_bones(self, thing=None, branch=None):
        skel = self.closet.skeleton[u"thing_loc"][unicode(self)]
        for thing_skel in selectif(skel, unicode(thing)):
            for branch_skel in selectif(thing_skel, branch):
                for bone in branch_skel.iterbones():
                    yield bone

    def get_thing_locations(self, name, branch=None):
        r = self.closet.skeleton[u"thing_loc"][unicode(self)][name]
        if branch is None:
            return r
        else:
            return r[branch]

    def del_thing_locations(self, name, branch=None):
        if branch is None:
            del self.closet.skeleton[u"thing_loc"][unicode(self)][name]
        else:
            del self.closet.skeleton[u"thing_loc"][unicode(self)][name][
                branch]

    def get_thing_location(self, name, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.get_thing_locations(name, branch).value_during(tick)
        if bone is None:
            return None
        corebone = self.get_thing_bone(name)
        host = self.closet.get_character(corebone.host)
        # suppose the location is a portal, and therefore has a
        # "real" bone
        for portal_bone in host.iter_hosted_portal_loc_bones(branch, tick):
            if portal_bone.name == bone.location:
                char = self.closet.get_character(portal_bone.character)
                return char.get_portal(bone.location)
        # I guess not. Maybe it's a thing?
        if bone.location in host.thing_d:
            return host.thing_d[name]
        # Nope, must be a place
        # Ensure it has a vertex
        try:
            v = self.graph.vs.find(name=bone.location)
            return v["place"]
        except (KeyError, ValueError):
            place = Place(self, bone.location)
            self.graph.add_vertex(name=bone.location, place=place)
            return place

    def iter_hosted_thing_bones(self):
        skel = self.closet.skeleton[u"thing"]
        for bone in skel.iterbones():
            if bone.host == self.name:
                yield bone

    def iter_hosted_thing_loc_bones(self, branch=None, tick=None):
        for thib in self.iter_hosted_thing_bones():
            skel = self.closet.skeleton[u"thing_loc"][thib.character]
            for branchskel in selectif(skel, branch):
                if tick is None:
                    for bone in branchskel.iterbones():
                        yield bone
                else:
                    yield branchskel.value_during(tick)

    ### Place

    def get_place(self, name):
        v = self.graph.vs.find(name=name)
        return v["place"]

    def make_place(self, name):
        place = Place(self, name)
        self.graph.add_vertex(
            name=name,
            place=place)
        return place

    def iter_places(self):
        for bone in self.closet.skeleton[u"place"].iterbones():
            try:
                yield self.get_place(bone.place)
            except KeyError:  # attribute does not exist
                return

    def get_place_bone(self, name, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(
            self.closet.timestream,
            self.closet.skeleton[u"place"][unicode(self)][name],
            branch,
            tick)

    def iter_place_bones(self, name=None, branch=None, tick=None):
        self.closet.query_place()
        try:
            skel = self.closet.skeleton[u"place"][unicode(self)]
        except KeyError:
            return
        for nameskel in selectif(skel, name):
            for branchskel in selectif(nameskel, branch):
                if tick is None:
                    for bone in branchskel.itervalues():
                        yield bone
                else:
                    r = branchskel.value_during(tick)
                    if r is not None:
                        yield r

    def iter_place_contents_bones(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_location"]

        def yield_if_here(bone):
            if bone.host == unicode(self) and bone.location == name:
                yield bone

        for charskel in skel.itervalues():
            for thingskel in charskel.itervalues():
                for branchskel in selectif(thingskel, branch):
                    if tick is None:
                        for bone in branchskel.iterbones():
                            yield_if_here(bone)
                    else:
                        yield_if_here(branchskel.value_during(tick))

    def get_place_contents(self, name, branch=None, tick=None):
        return set([bone for bone in self.iter_place_contents(
            name, branch, tick)])

    ### Portal

    def get_portal(self, name):
        return Portal(self, name)

    def make_portal(self, name=None, origin=None, destination=None,
                    host=None, branch=None, tick=None):
        if name is None:
            if None in (origin, destination):
                raise ValueError(
                    "Need either a name referring to a portal record, "
                    "or an origin and a destination")
            name = "{}->{}".format(origin, destination)
        if origin is None and destination is None and host is None:
            return self.make_portal_simple(name, branch, tick)
        if host is None:
            host = self
        (branch, tick) = self.sanetime(branch, tick)
        littlebone = Portal.bonetypes["portal"](
            name=unicode(name),
            character=unicode(self),
            host=unicode(host))
        bigbone = Portal.bonetypes["portal_loc"](
            name=unicode(name),
            character=unicode(self),
            origin=unicode(origin),
            destination=unicode(destination),
            branch=int(branch),
            tick=int(tick))
        self.closet.set_bone(littlebone)
        self.closet.set_bone(bigbone)
        return self.make_portal_simple(name, branch, tick)

    def make_portal_simple(self, name, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.closet.skeleton[u"portal"][unicode(self)][name]
        host = self.closet.get_character(bone.host)
        host.update()
        port = Portal(self, name)
        return port

    def iter_portal_loc_bones(self, name=None, branch=None, tick=None):
        try:
            skel = self.closet.skeleton[u"portal_loc"][unicode(self)]
        except KeyError:
            return

        for nameskel in selectif(skel, name):
            for branchskel in selectif(nameskel, branch):
                if tick is None:
                    for bone in branchskel.iterbones():
                        yield bone
                else:
                    yield branchskel.value_during(tick)

    def iter_hosted_portal_loc_bones(self, branch=None, tick=None):
        for portalbone in self.closet.skeleton[u"portal"].iterbones():
            if portalbone.host == self.name:
                skel = self.closet.skeleton[u"portal_loc"][
                    portalbone.character][portalbone.name]
                for branchskel in selectif(skel, branch):
                    if tick is None:
                        for bone in branchskel.iterbones():
                            yield bone
                    else:
                        yield branchskel.value_during(tick)

    def get_portal_bone(self, name):
        return self.closet.skeleton[u"portal"][unicode(self)][name]

    def get_portal_locations(self, name, branch=None):
        skel = self.closet.skeleton[u"portal_loc"][unicode(self)][name]
        if branch is not None:
            skel = skel[branch]
        return skel

    def get_portal_loc_bone(self, name, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.closet.skeleton[u"portal_loc"][unicode(self)][
            name][branch].value_during(tick)

    def iter_portal_bones(self):
        skel = self.closet.skeleton[u"portal"][unicode(self)]
        for bone in skel.iterbones():
            yield bone

    def iter_portals(self, branch=None, tick=None):
        accounted = set()
        for bone in self.iter_portal_loc_bones(
                name=None, branch=branch, tick=tick):
            if bone.name not in accounted:
                yield self.get_portal(bone.name)
                accounted.add(bone.name)

    ### Stat

    def iter_stat_bones(self, keys=[], branches=[], ticks=[]):
        skel = self.closet.skeleton[u"character_stat"]
        if unicode(self) not in skel:
            return
        key_skel = skel[unicode(self)]
        for branch_skel in skel_filter_iter(key_skel, keys):
            for tick_skel in skel_filter_iter(branch_skel, branches):
                for bone in skel_filter_iter(tick_skel, ticks):
                    yield bone

    def iter_stat_keys(self, branches=[], ticks=[]):
        skel = self.closet.skeleton[u"character_stat"]
        if unicode(self) not in skel:
            return
        for key in iter_skel_keys(skel[unicode(self)], branches, ticks):
            yield key

    def _iter_noun_stat_bones(self, nountab, nounnames=[],
                              stats=[], branches=[], ticks=[]):
        skel = self.closet.skeleton[nountab]
        if unicode(self) not in skel:
            return
        skel = skel[unicode(self)]
        for nounname in nounnames:
            if nounname not in skel:
                continue
            for stat_skel in skel_filter_iter(skel[nounname], stats):
                for branch_skel in skel_filter_iter(stat_skel, branches):
                    for bone in skel_filter_iter(branch_skel, ticks):
                        yield bone

    def _iter_noun_stat_keys(self, nountab, nounname, branches=[], ticks=[]):
        skel = self.closet.skeleton[nountab]
        if unicode(self) not in skel:
            return
        if nounname not in skel[unicode(self)]:
            return
        for key in iter_skel_keys(
                skel[unicode(self)][nounname], branches, ticks):
            yield key

    def iter_thing_stat_keys(self, thingn, branches=[], ticks=[]):
        for key in self._iter_noun_stat_keys(
                u"thing_stat", thingn, branches, ticks):
            yield key

    def iter_place_stat_keys(self, placen, branches=[], ticks=[]):
        for key in self._iter_noun_stat_keys(
                u"place_stat", placen, branches, ticks):
            yield key

    def iter_portal_stat_keys(self, portn, branches=[], ticks=[]):
        for key in self._iter_noun_stat_keys(
                u"portal_stat", portn, branches, ticks):
            yield key

    def new_branch(self, parent, branch, tick):
        for portal in self.iter_portals(branch=parent, tick=tick):
            for bone in portal.new_branch(parent, branch, tick):
                yield bone
        for thing in self.iter_things(branch=parent, tick=tick):
            for bone in thing.new_branch(parent, branch, tick):
                yield bone
        try:
            skel = self.closet.skeleton[u"stat"][unicode(self)]
        except KeyError:
            return

        def iterstatbones():
            for stat in skel:
                for bone in skel[stat][parent].iterbones():
                    yield bone
        for bone in upbranch(self.closet, iterstatbones(), branch, tick):
            yield bone


class Omitter(object):
    __metaclass__ = SaveableMetaclass
    demands = ["facade"]
    tables = [
        ("omitter", {
            "columns": {
                "list": "text not null",
                "i": "integer not null",
                "name": "text not null"},
            "primary_key": ("list", "i"),
            "foreign_keys": {
                "list": ("facade", "omitter_list")},
            "checks": ["i>=0"]})]
    functions = {
        "true": lambda bone: True,
        "false": lambda bone: False}

    def __init__(self, name):
        self.name = name

    def __call__(self, bone):
        r = self.functions[self.name](bone)
        if not isinstance(r, bool):
            raise TypeError(
                "Omitters must return ``bool`` values.")
        return r

    @classmethod
    def register_function(cls, name, fun):
        cls.functions[name] = fun


class Liar(object):
    __metaclass__ = SaveableMetaclass
    demands = ["facade"]
    tables = [
        ("liar", {
            "columns": {
                "list": "text not null",
                "i": "integer not null",
                "name": "text not null"},
            "primary_key": ("list", "i"),
            "foreign_keys": {
                "list": ("facade", "liar_list")},
            "checks": ["i>=0"]})]
    functions = {
        "noop": lambda bone: bone}

    def __init__(self, name):
        self.name = name

    def __call__(self, bone):
        typ = type(bone)
        r = self.functions[self.name](bone)
        if not isinstance(r, typ):
            raise TypeError(
                "Liars must return the same bone type they take.")
        return r

    @classmethod
    def register_function(cls, name, fun):
        cls.functions[name] = fun


class Facade(Character):
    """A view onto a :class:`Character`.

    A character's facade is the way it *seems* when observed by some
    other character--though, actually, characters that observe
    themselves look at facades of themselves.

    The :class:`Facade` API is mostly similar to that of
    :class:`Character`, but it doesn't always return the same
    information. To choose what it returns, supply the constructor
    with omitters and liars.

    """
    __metaclass__ = SaveableMetaclass
    demands = ["character"]
    tables = [
        ("facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "omitter_list": "text default null",
                "liar_list": "text default null"},
            "primary_key": ("observer", "observed")}),
        ("character_stat_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "observer", "observed", "key", "branch", "tick")})]

    def __init__(self, observer, observed, omitters=[], liars=[]):
        """Construct a facade for how the observer sees the observed.

        When a facade is about to return an instance of some subclass
        of :class:`LiSE.util.Bone`, it first calls all of its omitters
        with the bone as the argument. If even a single omitter
        returns ``True``, the facade will pretend it didn't find
        anything. Depending on the context, this may mean returning
        ``None``, raising an exception, or skipping an element in a
        generator. Supposing none of the omitters speak up, the facade
        decides what exactly to return by passing the bone through the
        liars. Every liar must output a bone, which will be passed to
        the next liar in the list. This could result in the truthful
        bone coming out the end and getting returned--liars don't need
        to lie *all the time*--but the intended use is to permit
        systematic deception, both against the player, and against
        other characters.

        """
        if not (
                isinstance(observer, Character) and
                isinstance(observed, Character)):
            raise TypeError("Facades are between two Characters.")
        self.observer = observer
        self.observed = observed
        self.omitters = omitters
        self.liars = liars
        self.closet = self.observed.closet
        self.graph = Graph(directed=True)
        if unicode(self.observer) not in self.closet.facade_d:
            self.closet.facade_d[unicode(self.observer)] = {}
        self.closet.facade_d[unicode(self.observer)][
            unicode(self.observed)] = self

    def __str__(self):
        return str(self.observed)

    def __unicode__(self):
        return unicode(self.observed)

    @property
    def name(self):
        return self.observed.name

    def evade(self, bone):
        """Raise KnowledgeException if the bone triggers an omitter. Otherwise
        return it.

        """
        for omitter in self.omitters:
            if omitter(bone):
                raise KnowledgeException(
                    "Found bone {}, but omitted due to {}".format(
                        bone, omitter))
        return bone

    def deceive(self, bone):
        """Allow my liars to mutilate the bone however they please before
        returning it."""
        for liar in self.liars:
            bone = liar(bone)
        return bone

    def distort(self, bone):
        """Let my omitters and liars at the bone, and return it if it
        survives. May raise KnowledgeException."""
        return self.deceive(self.evade(bone))

    # override
    def get_whatever(self, bone):
        """Return a thing or a portal, depending on the type of the bone."""
        return self.distort({
            Thing.bonetypes.thing_facade: self.get_thing,
            Portal.bonetypes.portal_facade: self.get_portal}[
            type(bone)](bone.name))

    ### Thing

    # override
    def get_thing_bone(self, name, branch=None, tick=None):
        bone = self.observed.get_thing_bone(name)
        return self.distort(bone)

    # override
    def iter_thing_stat_bones(self, thing, stat, branch=None):
        for bone in self.observed.iter_thing_stat_bones(thing, stat, branch):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                continue

    # override
    def get_thing_stat_skel(self, thing, stat, branch=None):
        r = Skeleton()
        if branch is None:
            for bone in self.iter_thing_stat_bones(thing, stat, branch):
                if bone.branch not in r:
                    r[bone.branch] = []
                r[bone.branch][bone.tick] = bone
        else:
            for bone in self.iter_thing_stat_bones(thing, stat, branch):
                r[bone.tick] = bone
        return r

    # override
    def get_thing_stat_bone(self, name, stat, branch=None, tick=None):
        return self.distort(self.observed.get_thing_stat_bone(
            name, stat, branch, tick))

    # override
    def iter_thing_bones(self, branch=None, tick=None):
        """Iterate over all bones for all things at the given time.

        """
        for bone in super(Facade, self).iter_thing_bones(branch, tick):
            yield self.distort(bone)

    # override
    def iter_thing_loc_bones(self, thing=None, branch=None):
        for bone in super(Facade, self).iter_thing_loc_bones(
                thing, branch):
            yield self.distort(bone)

    # override
    def iter_hosted_thing_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_facade"]
        for bone in self._iter_hosted_thing_bones_skel(skel, branch, tick):
            yield self.distort(bone)

    # override
    def get_thing_locations(self, thing, branch=None):
        """Get the part of the skeleton that shows the history of where the
        thing has been in the given branch, if specified; otherwise
        the current branch.

        For the facade, this is constructed off-the-cuff out of bones
        iterated over and run through ``distort``.

        """
        r = Skeleton([])
        if branch is None:
            for bone in self.iter_thing_loc_bones(thing):
                if bone.branch not in r:
                    r[bone.branch] = []
                r[bone.branch][bone.tick] = bone
        else:
            for bone in self.iter_thing_loc_bones(thing, branch):
                r[bone.tick] = bone
        return r

    def iter_place_bones(self, name=None, branch=None, tick=None):
        for bone in self.observed.iter_place_bones(name, branch, tick):
            try:
                yield self.deceive(bone)
            except KnowledgeException:
                pass

    # override
    def iter_place_contents(self, name, branch=None, tick=None):
        """Iterate over the bones for all the things that are located in the
        given place."""
        # incase of passing a Place object and not a string
        label = unicode(name)
        for bone in super(Facade, self).iter_place_contents(
                label, branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass

    # override
    def iter_place_stat_bones(self, place, stat, branch=None):
        for bone in self.observed.iter_place_stat_bones(place, stat, branch):
            try:
                yield self.deceive(bone)
            except KnowledgeException:
                continue

    ### Portal

    # override
    def get_portal_loc_bone(self, name, branch=None, tick=None):
        """Return a portal bone, possibly distorted.

        If the portal exists in the ``observed`` character, but is not
        apparent to the ``observer``, raise KnowledgeException.

        """
        bone = self.observed.get_portal_loc_bone(name, branch, tick)
        return self.distort(bone)

    # override
    def iter_portal_loc_bones(self, name=None, branch=None, tick=None):
        for bone in super(Facade, self).iter_portal_loc_bones(
                name, branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass

    # override
    def iter_hosted_portal_bones(self, branch=None, tick=None):
        for bone in super(Facade, self).iter_hosted_portal_bones(branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass

    # override
    def get_stat_bone(self, name, branch=None, tick=None):
        bone = super(Facade, self).get_stat_bone(name, branch, tick)
        return self.distort(bone)

    # override
    def iter_stat_bones(self, name=None, branch=None, tick=None):
        for bone in super(Facade, self).iter_stat_bones(name, branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass

    def new_branch(self, parent, branch, tick):
        skel = self.closet.skeleton[u"character_stat_facade"][
            unicode(self.observer)][unicode(self.observed)]

        def bonito():
            for stat in skel:
                if parent in skel[stat]:
                    for bone in skel[stat][parent].iterbones():
                        yield bone

        for bone in upbranch(self.closet, bonito(), branch, tick):
            yield bone
