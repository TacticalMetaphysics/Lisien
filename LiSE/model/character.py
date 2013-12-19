# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from igraph import Graph

from LiSE.util import (
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

    def update(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        for v in self.graph.vs:
            try:
                self.get_place_bone(
                    v["name"], branch, tick)
            except (KeyError, KnowledgeException):
                self.graph.delete_vertex(v.index)
        for e in self.graph.es:
            try:
                self.get_portal_bone(
                    e["name"], branch, tick)
            except (KeyError, KnowledgeException):
                self.graph.delete_edge(e.index)
        for v in self.iter_place_bones(branch, tick):
            self.graph.add_vertex(name=v.place, place=Place(self, v.place))
        for e in self.iter_hosted_portal_bones(branch, tick):
            bone = self.closet.skeleton[u"portal_loc"][
                unicode(self)][e.name][branch].value_during(tick)
            oi = self.graph.vs["name"].index(bone.origin)
            di = self.graph.vs["name"].index(bone.destination)
            self.graph.add_edge(
                oi, di, name=e.name, portal=Portal(self, e.name))

    def get_facade(self, observer):
        if observer is None:
            raise ValueError("Every facade must have an observer.")
        if unicode(observer) not in self.facade_d:
            self.facade_d[unicode(observer)] = Facade(
                observer, observed=self)
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

    def _iter_thing_loc_bones_skel(self, skel, thing=None, branch=None):
        """Iterate over location bones, perhaps for only one thing, perhaps
        for a particular branch."""
        if thing is not None:
            thingit = iter([unicode(thing)])
        else:
            thingit = skel.iterkeys()
        for thingn in thingit:
            if branch is None:
                branchit = skel[thingn].iterkeys()
            else:
                branchit = iter([branch])
            for branchi in branchit:
                for bone in skel[thingn][branchi].iterbones():
                    yield bone

    def iter_thing_loc_bones(self, thing=None, branch=None):
        skel = self.closet.skeleton[u"thing_loc"][unicode(self)]
        for bone in self._iter_thing_loc_bones_skel(skel, thing, branch):
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
        skel = self.closet.skeleton[u"thing_loc"][unicode(self)][name]
        return self._get_thing_skel_locations(skel, branch)

    def get_thing_location(self, name, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.get_thing_locations(name, branch).value_during(tick)
        if bone is None:
            return None
        corebone = self.get_thing_bone(name)
        host = self.closet.get_character(corebone.host)
        # suppose the location is a portal, and therefore has a
        # "real" bone
        for portal_bone in host.iter_hosted_portal_bones(branch, tick):
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
            place.upd_skel_from_bone(bone)
            self.graph.add_vertex(name=bone.location, place=place)
            return place

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

    def _get_place_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timeline, skel, branch, tick)

    def get_place_bone(self, name, branch=None, tick=None):
        skel = self.closet.skeleton[u"place"][unicode(self)][name]
        return self._get_place_skel_bone(skel, branch, tick)

    def iter_place_bones(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        self.closet.query_place()
        try:
            skel = self.closet.skeleton[u"place"][unicode(self)]
        except KeyError:
            return
        for place in skel.iterbones():
            yield place

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

    ### Portal

    def get_portal(self, name):
        e = self.graph.es.find(name=name)
        return e["portal"]

    def make_portal(self, origin, destination, host=None, name=None,
                    branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        origin = unicode(origin)
        destination = unicode(destination)
        if host is None:
            host = self
        host = unicode(host)
        if name is None:
            name = "{}->{}".format(origin, destination)
        bone = Portal.bonetypes["portal"](
            character=unicode(self),
            name=name,
            host=host)
        self.closet.set_bone(bone)
        bigbone = Portal.bonetypes["portal_loc"](
            character=unicode(self),
            name=name,
            branch=branch,
            tick=tick,
            origin=origin,
            destination=destination)
        self.closet.set_bone(bigbone)
        p = Portal(self, name)
        self.graph.add_vertex(name=name, portal=p)
        return p

    def _get_portal_skel_bone(self, skel, branch, tick):
        (branch, tick) = self.sanetime(branch, tick)
        return skelget(self.closet.timestream, skel, branch, tick)

    def get_portal_bone(self, name):
        return self.closet.skeleton[u"portal"][unicode(self)][name]

    def get_portal_loc_bone(self, name, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.closet.skeleton[u"portal_loc"][unicode(self)][
            name][branch].value_during(tick)

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
                bone = skel[character][name]
                if bone.host == unicode(self):
                    yield bone

    def iter_hosted_portal_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"portal"]
        for bone in self._iter_hosted_portal_bones_skel(skel, branch, tick):
            yield bone

    def _iter_hosted_thing_bones_skel(self, skel, branch=None, tick=None):
        for character in skel:
            for name in skel[character]:
                bone = skel[character][name]
                if bone.host == unicode(self):
                    yield bone

    def iter_hosted_thing_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing"]
        for bone in self._iter_hosted_thing_bones_skel(skel, branch, tick):
            yield bone

    def iter_hosted_thing_loc_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_loc"]
        for bone in self.iter_hosted_thing_bones(branch, tick):
            yield skel[bone.character][bone.name][branch].value_during(tick)

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
    def get_real_thing_bone(self, label, branch, tick):
        """Return a thing bone specific to this one facade.

        This method does not distort the bone. It never returns a bone
        from my ``observed`` character.

        """
        (branch, tick) = self.sanetime(branch, tick)
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)][
            label]
        return self._get_thing_skel_bone(skel, branch, tick)

    def get_thing_bone(self, name, branch, tick):
        """Return a thing bone, possibly from this facade, possibly from its
        ``observed`` character, and possibly distorted by ``liars``.

        In case the named thing is in the ``observed`` character, but
        is not apparent to the ``observer``, raise KnowledgeException.

        """
        try:
            bone = self.get_real_thing_bone(name, branch, tick)
        except KeyError:
            bone = self.observed.get_thing_bone(name, branch, tick)
        return self.distort(bone)

    # override
    def iter_thing_bones(self, branch=None, tick=None):
        """Iterate over all bones for all things at the given time.

        Beware, this yields bones with null values too. Just because
        you get a bone, doesn't mean there's a thing for it at the
        moment.

        """
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)][unicode(self.observed)]
        accounted = set()
        for bone in self._iter_thing_skel_bones(skel, branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass
            accounted.add(bone.name)
        for bone in super(Facade, self).iter_thing_bones(branch, tick):
            if bone.name not in accounted:
                yield bone

    # override
    def iter_thing_loc_bones(self, thing=None, branch=None):
        skel = self.closet.skeleton[u"thing_loc_facade"][
            unicode(self.observer)][unicode(self.observed)]
        for bone in self._iter_thing_loc_bones_skel(skel, thing, branch):
            yield bone

    # override
    def iter_hosted_thing_bones(self, branch=None, tick=None):
        skel = self.closet.skeleton[u"thing_facade"]
        for bone in self._iter_hosted_thing_bones_skel(skel, branch, tick):
            yield bone

    # override
    def get_thing_locations(self, thing, branch=None):
        """Get the part of the skeleton that shows the history of where the
        thing has been in the given branch, if specified; otherwise
        the current branch.

        For the facade, this is constructed off-the-cuff out of bones
        iterated over and run through ``distort``.

        """
        skel = Skeleton([])
        for bone in self.observed.iter_thing_loc_bones(thing, branch):
            if bone.branch not in skel:
                skel[bone.branch] = []
            skel[bone.branch][bone.tick] = self.distort(bone)
        return skel

    def iter_place_bones(self, branch=None, tick=None):
        for bone in self.observed.iter_place_bones(branch, tick):
            for omitter in self.omitters:
                if omitter(bone):
                    continue
            for liar in self.liars:
                bone = liar(bone)
            yield bone

    # override
    def iter_place_contents(self, name, branch=None, tick=None):
        """Iterate over the bones for all the things that are located in the
        given place."""
        # incase of passing a Place object and not a string
        label = unicode(name)
        skel = self.closet.skeleton[u"thing_facade"][
            unicode(self.observer)]
        accounted = set()
        for bone in self._iter_place_skel_contents(skel, label, branch, tick):
            try:
                yield self.distort(bone)
            except KnowledgeException:
                pass
            accounted.add(bone.name)
        for bone in super(Facade, self).iter_place_contents(
                label, branch, tick):
            if bone.name not in accounted:
                try:
                    yield self.distort(bone)
                except KnowledgeException:
                    pass

    ### Portal

    def get_real_portal_bone(self, label, branch=None, tick=None):
        """Return a portal bone specific to this facade.

        This method does not distort the bone. It never returns bones
        from my ``observed`` character.

        """
        skel = self.closet.skeleton[u"portal_loc_facade"][
            unicode(self.observer)][unicode(self.observed)][label]
        return self._get_portal_skel_bone(skel, branch, tick)

    # override
    def get_portal_loc_bone(self, name, branch=None, tick=None):
        """Return a portal bone, possibly specific to this facade, possibly
        from its ``observed`` character, and possibly distorted by its
        ``liars``.

        If the portal exists in the ``observed`` character, but is not
        apparent to the ``observer``, raise KnowledgeException.

        """
        try:
            bone = self.get_real_portal_bone(name, branch, tick)
        except KeyError:
            bone = self.observed.get_portal_loc_bone(name, branch, tick)
        return self.distort(bone)

    # override
    def iter_hosted_portal_bones(self, branch=None, tick=None):
        """Iterate first over the portal bones that this facade has its own
        records about, and then over those the underlying character
        knows about, but only if its name was not somehow accounted
        for by this facade.

        The ``distort`` method accounts for a name by
        - returning a bone for it
        - raising a KnowledgeException

        """
        accounted = set()
        try:
            skel = self.closet.skeleton[u"portal_facade"]
            # It's a bit dodgy to distort a bone from portal_facade,
            # because the contents of portal_facade are *already*
            # distortions. Even so, I will let it happen, because I cannot
            # rule out the usefulness.
            for bone in self._iter_hosted_portal_bones_skel(
                    skel, branch, tick):
                try:
                    yield self.distort(bone)
                except KnowledgeException:
                    pass
                accounted.add(bone.name)
        except KeyError:
            pass
        for bone in super(Facade, self).iter_hosted_portal_bones(branch, tick):
            if bone.name not in accounted:
                try:
                    yield self.distort(bone)
                except KnowledgeException:
                    pass

    # override
    def get_stat_bone(self, name, branch=None, tick=None):
        """Return a bone for the stat by the given name at the current time,
        or the given branch and tick if specified.

        """
        skel = self.observed.closet.skeleton[u"stat_facade"][
            unicode(self.observer)][unicode(self.observed)][
            name]
        return self._get_stat_skel_bone(skel, branch, tick)
