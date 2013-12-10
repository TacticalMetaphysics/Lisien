# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from igraph import Graph
from igraph._igraph import InternalError as IgraphError

from re import match

from LiSE.util import (
    SaveableMetaclass,
    portex)
from LiSE.model import (
    Thing,
    Place,
    Portal)


"""Things that should have character sheets."""


def skelget(tl, skel, branch, tick):
    while branch != 0:
        try:
            return skel[branch].value_during(tick)
        except KeyError:
            branch = tl.parent(branch)
    # may throw KeyError
    return skel[0].value_during(tick)


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

    provides = ["character"]
    tables = [
        ("stat",
         {"character": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "stat": "text"},
         ("character", "name", "branch", "tick"),
         {},
         [])]

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

        """
        self.closet = closet
        self.name = name
        self.thing_d = {}
        self.facade_d = {}
        if knows_self:
            self.facade_d[unicode(self)] = Facade(
                observer=self, observed=self, omitters=self_omitters,
                liars=self_liars)
        self.graph = Graph(directed=True)
        self.closet.character_d[unicode(self)] = self

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def update(self, branch=None, tick=None):
        for v in self.graph.vs:
            try:
                self.get_place_bone(
                    v["name"], branch, tick)
            except KeyError:
                self.graph.delete_vertex(v.index)
        for e in self.graph.es:
            try:
                self.get_portal_bone(
                    e["name"], branch, tick)
            except KeyError:
                self.graph.delete_edge(e.index)
        for v in self.iter_place_bones(branch, tick):
            self.graph.add_vertex(name=v.name, place=Place(self, v.name))
        for e in self.iter_hosted_portal_bones(branch, tick):
            assert e.origin in self.graph.vs["name"], (
                "The portal {} originates from {}, "
                "which is not in {}.".format(
                    e, e.origin, self))
            oi = self.graph.vs["name"].index(e.origin)
            assert e.destination in self.graph.vs["name"], (
                "The portal {} leads to {}, "
                "which is not in {}.".format(
                    e, e.destination, self))
            di = self.graph.vs["name"].index(e.destination)
            self.graph.add_edge(
                oi, di, name=e.name, portal=Portal(self, e.name))

    def get_facade(self, observer):
        if observer is None:
            raise ValueError("Every facade must have an observer.")
        if unicode(observer) not in self.facade_d:
            self.facade_d[unicode(observer)] = Facade(
                observer, observed=self, honest=self.truthful)
        return self.facade_d[unicode(observer)]

    def sanetime(self, branch, tick):
        """If branch or tick are None, replace them with the current value
        from the closet.

        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        return (branch, tick)

    @staticmethod
    def skelset(skel, bone):
        """Put ``bone`` into ``skel``, nested according to the database
        schema."""
        if bone.character not in skel:
            skel[bone.character] = {}
        if bone.label not in skel[bone.character]:
            skel[bone.character][bone.label] = []
        if bone.branch not in skel[bone.character][bone.label]:
            skel[bone.character][bone.label][bone.branch] = []
        skel[bone.character][bone.label][bone.branch][bone.tick] = bone

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

    def _get_thing_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.tl, skel, branch, tick)

    def get_thing_bone(self, name, branch=None, tick=None):
        """Return a bone describing a thing's location at some particular
        time."""
        skel = self.closet.skeleton[u"thing"][unicode(self)][name]
        return self._get_thing_skel_bone(name, skel, branch, tick)

    def _iter_thing_skel_bones(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        for name in skel:
            yield skelget(self.closet.tl, skel, branch, tick)

    def iter_thing_bones(self, branch=None, tick=None):
        """Iterate over all things present in this character at the time
        specified, or the present time if not specified."""
        skel = self.closet.skeleton[u"thing"][unicode(self)]
        for bone in self._iter_thing_skel_bones(skel, branch, tick):
            yield bone

    def _get_thing_skel_locations(self, skel, branch):
        if branch is None:
            branch = self.closet.branch
        while branch != 0:
            try:
                return skel[branch]
            except KeyError:
                branch = self.closet.timeline.parent(branch)
        return skel[0]

    def get_thing_locations(self, name, branch=None):
        skel = self.closet.skeleton[u"thing"][unicode(self)][name]
        return self._get_thing_skel_locations(skel, branch)

    def get_thing_location(self, labl, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.get_thing_bone(labl, branch, tick)
        if bone is None:
            return None
        host = self.closet.get_character(bone.host)
        try:
            return host.graph.vs.find(name=bone.label)["place"]
        except IgraphError:
            try:
                m = match(portex, bone.label)
                (origin, destination) = m.groups()
                oi = host.graph.vs.find(name=origin)
                di = host.graph.vs.find(name=destination)
                ei = host.graph.get_eid(oi, di)
                return host.graph.es[ei]["portal"]
            except (IgraphError, IndexError):
                return host.get_thing(bone.label)

    def set_thing_bone(self, bone):
        skel = self.closet.skeleton[u"thing"]
        self.skelset(skel, bone)

    def set_thing_values(self, label, branch, tick, host, location):
        self.set_thing_bone(Thing.bonetypes.thing(
            character=unicode(self),
            label=label,
            branch=branch,
            tick=tick,
            host=host,
            location=location))

    ### Place

    def get_place(self, name):
        v = self.graph.vs.find(name=name)
        return v["place"]

    def _get_place_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timeline, skel, branch, tick)

    def get_place_bone(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"place"][unicode(self)][name]
        return self._get_place_skel_bone(skel, branch, tick)

    def _iter_place_skel_bones(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        for name in skel:
            yield skelget(self.closet.timeline, skel, branch, tick)

    def iter_place_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"place"][unicode(self)]
        for bone in self._iter_skel_place_bones(skel, branch, tick):
            yield bone

    def _iter_place_skel_contents(self, skel, name, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        for naem in skel:
            try:
                bone = skel[naem][branch].value_during(tick)
            except KeyError:
                continue
            if bone.location == name:
                yield bone

    def iter_place_contents(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing"]
        for bone in self._iter_place_skel_contents(skel, name, branch, tick):
            yield bone

    def get_place_contents(self, name, branch=None, tick=None):
        return set([bone for bone in self.iter_place_contents(
            name, branch, tick)])

    def _set_place_skel_bone(self, skel, bone):
        if bone.name not in self.graph.vs["name"]:
            self.graph.add_vertex(
                name=bone.name)
        self.skelset(skel, bone)

    def set_place_bone(self, bone):
        skel = self.closet.skeleton[u"place"]
        self._set_place_skel_bone(skel, bone)

    ### Portal

    def get_portal(self, name):
        e = self.graph.es.find(name=name)
        return e["portal"]

    def _get_portal_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timestream, skel, branch, tick)

    def get_portal_bone(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal"][unicode(self)][name]
        return self._get_portal_skel_bone(skel, branch, tick)

    def _iter_portal_skel_bones(self, skel, branch, tick):
        for label in skel:
            yield self._get_portal_skel_bone(skel, label, branch, tick)

    def iter_portal_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal"][unicode(self)]
        for bone in self._iter_portal_skel_bones(skel, branch, tick):
            yield bone

    def _iter_hosted_portal_bones_skel(self, skel, branch=None, tick=None):
        for character in skel:
            for name in skel[character]:
                skel = skel[character][name]
                bone = self._get_portal_skel_bone(skel, branch, tick)
                if bone.host == unicode(self):
                    yield bone

    def iter_hosted_portal_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal"]
        for bone in self._iter_hosted_portal_bones_skel(skel, branch, tick):
            yield bone

    def _portal_init_skel_branch_label(self, skel, name, parent, branch, tick):
        prev = None
        started = False
        for bone in skel[parent].iterbones():
            if bone.tick >= tick:
                bone2 = bone._replace(branch=branch)
                skel[branch][bone2.tick] = bone2
                if (
                        not started and prev is not None and
                        bone2.tick > tick and prev.tick < tick):
                    bone3 = prev._replace(branch=branch, tick=tick)
                    skel[branch][bone3.tick] = bone3
                started = True
            prev = bone

    def portal_init_branch_label(self, name, parent, branch, tick):
        skel = self.closet.skeleton[u"portal"][unicode(self)][name]
        self._portal_init_skel_branch_label(skel, name, parent, branch, tick)

    def _portal_init_skel_branch(self, skel, parent, branch, tick):
        for name in skel:
            self.portal_init_skel_label(skel, name, parent, branch, tick)

    def portal_init_branch(self, parent, branch, tick):
        skel = self.closet.skeleton[u"portal"][unicode(self)]
        self._portal_init_skel_branch(skel, parent, branch, tick)

    def _set_portal_skel_bone(self, skel, bone):
        self.skelset(skel, bone)
        if bone.origin not in self.graph.vs["name"]:
            self.set_place_idx(
                bone.origin, len(self.graph.vs),
                bone.branch, bone.tick)
        if bone.destination not in self.graph.vs["name"]:
            self.set_place_idx(
                bone.destination, len(self.graph.vs),
                bone.branch, bone.tick)
        if bone.label not in self.graph.es["name"]:
            oi = self.graph.vs["name"][bone.origin]
            di = self.graph.vs["name"][bone.destination]
            self.graph.add_edge(
                oi, di,
                name=bone.label,
                bone=lambda branch, tick: self.get_portal_bone(branch, tick))

    def set_portal_bone(self, bone):
        skel = self.closet.skeleton[u"portal"]
        self._set_portal_skel_bone(skel, bone)

    ### Stat

    def _get_stat_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timeline, skel, branch, tick)

    def get_stat_bone(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"stat"][unicode(self)][name]
        return self._get_stat_skel_bone(skel, branch, tick)

    def _iter_stat_skel_bones(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        for label in skel:
            yield self.get_stat_bone(label, branch, tick)

    def iter_stat_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"stat"][unicode(self)]
        for bone in self._iter_stat_skel_bones(skel, branch, tick):
            yield bone

    def set_stat_bone(self, bone):
        skel = self.closet.skeleton[u"stat"]
        self.skelset(skel, bone)


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
    provides = ["facade"]
    tables = [
        ("stat_facade",
         {"observer": "text not null",
          "observed": "text not null",
          "label": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "name": "text",
          "stat": "text"},
         ("character", "facade", "label", "branch", "tick"),
         {"character, label": ("character_label", "character, label")},
         [])]

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

    @staticmethod
    def skelset(skel, bone):
        """Put ``bone`` into ``skel``, nested according to the database
        schema."""
        if bone.observer not in skel:
            skel[bone.observer] = {}
        if bone.observed not in skel[bone.observer]:
            skel[bone.observer][bone.observed] = {}
        if bone.label not in skel[bone.observer][bone.observed]:
            skel[bone.observer][bone.observed][bone.label] = []
        if bone.branch not in skel[bone.observer][bone.observed][bone.label]:
            skel[bone.observer][bone.observed][bone.label][bone.branch] = []
        skel[bone.observer][bone.observed][bone.label][
            bone.branch][bone.tick] = bone

    def tact(self, bone):
        """Raise KeyError if the bone triggers an omitter. Otherwise return
        it."""
        for omitter in self.omitters:
            if omitter(bone):
                raise KeyError("Found bone {}, but omitted due to {}".format(
                    bone, omitter))
        return bone

    def spin(self, bone):
        """Allow my liars to mutilate the bone however they please before
        returning it."""
        for liar in self.liars:
            bone = liar(bone)
        return bone

    def deceive(self, bone):
        """Let my omitters and liars at the bone, and return it if it
        survives."""
        return self.spin(self.tact(bone))

    # override
    def get_whatever(self, bone):
        return self.deceive({
            Thing.bonetypes.thing_facade: self.get_thing,
            Portal.bonetypes.portal_facade: self.get_portal}[
            type(bone)](bone.name))

    ### Thing

    def iter_thing_facade_bones(self, bone):
        """Take a thing_bone and yield the corresponding thing_facade_bones in
        this facade right here."""
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)]
        for inbone in self._iter_thing_skel_bones(
                skel, bone.branch, bone.tick):
            try:
                inbone = self.deceive(inbone)
            except KeyError:
                continue
            if inbone.name == bone.name and inbone.host == bone.host:
                yield inbone

    # override
    def get_thing_bone(self, label, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)][
            label]
        return self.deceive(self._get_thing_skel_bone(skel, branch, tick))

    # override
    def iter_thing_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)]
        for bone in self._iter_thing_skel_bones(skel, branch, tick):
            try:
                yield self.deceive(bone)
            except KeyError:
                continue

    # override
    def get_thing_locations(self, thing, branch=None):
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)][thing]
        return self.deceive(self._get_thing_skel_locations(skel, branch))

    # override
    def set_thing_bone(self, bone):
        skel = self.closet.skeleton[u"thing_facade"]
        self.skelset(skel, bone)

    # override
    def get_place_bone(self, label, branch=None, tick=None):
        """Return a bone describing a place in the observer's opinion."""
        skel = self.closet.skeleton[u"place_facade"][
            unicode(self.observer)][unicode(self.observed)][label]
        return self.deceive(self._get_place_skel_bone(skel, branch, tick))

    # override
    def iter_place_contents(self, label, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)]
        for bone in self._iter_place_skel_contents(skel, label, branch, tick):
            try:
                yield self.deceive(bone)
            except KeyError:
                continue

    # override
    def set_place_bone(self, bone):
        skel = self.closet.skeleton[u"place_facade"]
        self._set_place_skel_bone(skel, bone)

    # override
    def get_portal_bone(self, label, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal_facade"][
            unicode(self.observer)][unicode(self.observed)][label]
        return self.deceive(self._get_portal_skel_bone(skel, branch, tick))

    # override
    def iter_hosted_portal_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal_facade"]
        for bone in self._iter_hosted_portal_bones_skel(skel, branch, tick):
            try:
                yield self.deceive(bone)
            except KeyError:
                continue

    # override
    def set_portal_bone(self, bone):
        skel = self.closet.skeleton[u"portal_facade"]
        self._set_portal_skel_bone(skel, bone)

    # override
    def get_stat_bone(self, label, branch=None, tick=None):
        skel = self.observed.closet.skeleton[u"stat_facade"][
            unicode(self.observer)][unicode(self.observed)][
            label]
        return self.deceive(self._get_stat_skel_bone(skel, branch, tick))

    # override
    def set_stat_bone(self, bone):
        skel = self.closet.skeleton[u"stat_facade"]
        self.skelset(skel, bone)
