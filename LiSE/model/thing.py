# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.util import (
    TimeParadox,
    JourneyException
)

from container import Container
from kivy.logger import Logger


class Thing(Container):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    demands = ["portal"]
    tables = [
        ("thing", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "host": "text not null default 'Physical'"},
            "primary_key": ("character", "name")}),
        ("thing_loc", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "location": "text"},
            "primary_key": (
                "character", "name", "branch", "tick"),
            "foreign_keys": {
                "character, name": (
                    "thing", "character, name")}}),
        ("thing_stat", {
            "columns": {
                "character": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "name", "key", "branch", "tick"),
            "foreign_keys": {
                "character, name": (
                    "thing", "character, name")}}),
        ("thing_loc_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "location": "text"},
            "primary_key": (
                "observer", "observed", "name", "branch", "tick"),
            "foreign_keys": {
                "observed, name": (
                    "thing", "character, name")}}),
        ("thing_stat_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "observer", "observed", "name",
                "key", "branch", "tick"),
            "foreign_keys": {
                "observed, name": (
                    "thing", "character, name")}})]
    """Things exist in particular places--but what exactly it *means* for
    a thing to be in a place will vary by context. Each of those
    contexts gets a "host," which is another character. Things are
    *in* their host, but are *part of* their character.

    Each thing has only one record in the table ``thing``, but may
    have several records in the table ``thing_facade``. The record in
    ``thing`` is regarded as the true one for the purpose of resolving
    the outcomes of events, but when characters make decisions, they
    can only use the information from ``thing_facade``. This permits
    you to hide information and lie to the player.

    There are various methods here to get information on a thing's
    status. Called with no arguments, they return the truth at the
    present time. During simulation, they are always called with at
    least the argument ``observer``, and will therefore respond with
    the way the observer *understands* them.

    """

    @property
    def location(self):
        return self.get_location()

    @property
    def locations(self):
        return self.get_locations()

    @property
    def host(self):
        hostn = self.character.get_thing_bone(self.name).host
        return self.character.closet.get_character(hostn)

    def __init__(self, character, name):
        self.character = character
        self.name = name
        self.new_branch_blank = False

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        return "Thing({})".format(self)

    def __contains__(self, that):
        return that.location is self

    def get_bone(self, observer=None, branch=None, tick=None):
        """Return a bone describing my status.

        With optional argument ``observer``, the bone will be munged
        by a facade, possibly resulting in a deliberately misleading
        KeyError.

        """
        if observer is None:
            return self.character.get_thing_bone(self.name)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_bone(branch, tick)

    def get_location(self, observer=None, branch=None, tick=None):
        """Return the thing, place, or portal I am in.

        With optional argument ``observer``, return the thing, place,
        or portal I *seem* to be in.

        """
        if observer is None:
            return self.character.get_thing_location(self.name, branch, tick)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_location(self.name, branch, tick)

    def get_locations(self, observer=None, branch=None):
        if observer is None:
            return self.character.get_thing_locations(self.name, branch)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_locations(self.name, branch)

    def set_location(self, location, branch=None, tick=None):
        self.character.set_thing_location(self, location, branch, tick)

    def get_stats(self, observer=None, branch=None):
        if observer is None:
            return self.character.get_thing_stat_skel(self.name, branch)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_stat_skel(self.name, branch)

    def get_stat(self, stat, observer=None, branch=None, tick=None):
        if observer is None:
            return self.character.get_thing_stat(self.name, stat, branch, tick)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_stat(self.name, stat, branch, tick)

    def set_stat(self, key, value, branch=None, tick=None):
        self.character.set_thing_stat(self, key, value, branch, tick)

    def get_speed(self, observer=None, branch=None, tick=None):
        lo = self.get_location(observer, branch, tick)
        ticks = self.get_ticks_thru(lo, observer, branch, tick)
        return float(lo.weight) / float(ticks)

    def get_ticks_thru(
            self, portal, observer=None, branch=None, tick=None):
        """How many ticks would it take to get through that portal?"""
        if observer is None:
            observer = self.character
        # placeholder
        portal_weight = 1
        return int(portal_weight / 0.1)

    def get_progress(self, observer=None, branch=None, tick=None):
        """Return a float representing the proportion of the portal I have
        passed through.

        """
        (branch, tick) = self.character.sanetime(branch, tick)
        bone = self.get_locations(observer, branch).value_during(tick)
        # this is when I entered the portal
        t1 = bone.tick
        # this is when I will enter the destination
        t2 = self.get_locations(observer, branch).key_after(tick)
        if t2 is None:
            return None
        duration = float(t2 - t1)
        passed = float(tick - t1)
        return passed / duration

    def journey_to(self, destplace, branch=None, tick=None):
        """Schedule myself to travel somewhere.

        I'll attempt to find a path from wherever I am at the moment
        (or the time supplied) to the destination given. If I find
        one, I'll schedule myself to be in the places and portals in
        it at the appropriate times. Precisely what times are
        'appropriate' depends on the effective lengths of the portals.

        """
        if unicode(destplace) == unicode(self.get_location(branch, tick)):
            # Nothing to do
            return
        (branch, tick) = self.character.sanetime(branch, tick)
        oloc = self.get_location(None, branch, tick)
        otick = tick
        if "Portal" in oloc.__class__.__name__:
            loc = unicode(oloc.destination)
            tick = self.get_locations(None, branch).key_after(otick)
        else:
            loc = oloc
            tick = otick
        # Get a shortest path based on my understanding of the graph,
        # which may not match reality.  It would be weird to use some
        # other character's understanding.
        host = self.character.closet.get_character(self.get_bone().host)
        host.update(branch, tick)
        facade = host.get_facade(self.character)
        facade.update(branch, tick)
        ipath = facade.graph.get_shortest_paths(
            unicode(loc), to=unicode(destplace), output=str("epath"))
        path = None
        for p in ipath:
            if p == []:
                continue
            desti = facade.graph.es[p[-1]].target
            if desti == destplace.v.index:
                path = [facade.graph.es[i]["portals"][
                    unicode(facade.observer)] for i in p]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        locs = list(self.branch_loc_bones_gen(branch))
        # Attempt to follow the path based on how the graph is
        # actually laid out.
        try:
            Logger.debug("Thing: attempting to follow path {}".format(
                path))
            self.follow_path(path, branch, tick)
        except TimeParadox:
            tupme = (unicode(self.character), unicode(self))
            self.character.closet.new_branch_blank.add(tupme)
            self.restore_loc_bones(branch, locs)
            increment = 1
            while branch + increment in self.locations:
                increment += 1
            self.character.closet.new_branch(branch, branch+increment, tick)
            self.character.closet.time_travel(branch+increment, tick)
            self.character.closet.new_branch_blank.remove(tupme)
            if "Portal" in oloc.__class__.__name__:
                loc = oloc.origin
                tick = self.get_locations().key_after(otick)
            else:
                loc = oloc
                tick = otick
            self.follow_path(
                path, self.character.closet.branch, tick+1)

    def follow_path(self, path, branch, tick):
        """Presupposing I'm in the given host, follow the path by scheduling
        myself to be located in the appropriate place or portal at the
        appropriate time.

        Optional arguments ``branch`` and ``tick`` give the start
        time. If unspecified, the current diegetic time is used.

        Raises ``TimeParadox`` if I would contradict any locations
        already scheduled.

        """
        try:
            aft = self.get_locations(
                observer=None, branch=branch).key_after(tick)
            raise TimeParadox(
                "Tried to follow a path at tick {},"
                " but I was scheduled to be elsewhere "
                " at tick {}".format(tick, aft))
        except ValueError:
            pass
        host = self.character.closet.get_character(self.get_bone().host)
        bone = self.character.get_thing_locations(
            self.name, branch).value_during(tick)
        prevtick = tick + 1
        for port in path:
            bone = bone._replace(
                location=port.name,
                tick=prevtick)
            host.closet.set_bone(bone)
            prevtick += self.get_ticks_thru(
                port, observer=None, branch=branch, tick=prevtick)
            bone = bone._replace(
                location=unicode(port.destination),
                tick=prevtick)
            host.closet.set_bone(bone)
            prevtick += 1

    def new_branch(self, parent, branch, tick):
        """There's a new branch off of the parent branch, and it starts at the
        given tick, so I'll copy any locations I'm in *after* that, in
        the parent branch, to the child.

        """
        Logger.debug("Thing: new branch {} from parent {}".format(
            branch, parent))
        if (
                unicode(self.character), unicode(self)
        ) in self.character.closet.new_branch_blank:
            Logger.debug("Thing: new_branch_blank")
            start_loc = self.get_location(None, parent, tick)
            if hasattr(start_loc, 'destination'):
                tick = self.locations[parent].key_after(tick)
                start_loc = self.get_location(parent, tick)
            locb = self.bonetypes["thing_loc"](
                character=unicode(self.character),
                name=self.name,
                branch=branch,
                tick=tick,
                location=start_loc)
            yield locb
            return
        for bone in self.character.iter_thing_loc_bones(
                self, branch=parent):
            yield bone._replace(branch=branch)
        for bone in self.iter_stats_bones(branch=parent):
            yield bone._replace(branch=branch)
        for observer in self.character.facade_d.iterkeys():
            for bone in self.iter_loc_bones(observer, branch=parent):
                yield bone._replace(branch=branch)
            for bone in self.iter_stats_bones(
                    stats=[], observer=observer, branch=parent):
                yield bone._replace(branch=branch)

    def iter_loc_bones(self, observer=None, branch=None):
        if observer is None:
            for bone in self.character.iter_thing_loc_bones(self, branch):
                yield bone
        else:
            facade = self.character.get_facade(observer)
            for bone in facade.iter_thing_loc_bones(self, branch):
                yield bone

    def iter_stats_bones(self, stats=[], observer=None,
                         branch=None, tick=None):
        (branch, tick) = self.character.sanetime(branch, tick)
        if observer is None:
            try:
                for bone in self.character.iter_thing_stat_bones(
                        self.name, stats, [branch], [tick]):
                    yield bone
            except KeyError:
                return
        else:
            facade = self.character.get_facade(observer)
            for bone in facade.iter_thing_stat_bones(
                    self.name, stats, [branch], [tick]):
                yield bone

    def branch_loc_bones_gen(self, branch=None):
        """Iterate over all the location bones in the given branch, defaulting
        to the current branch.

        """
        for bone in self.get_locations(branch=branch).iterbones():
            yield bone

    def restore_loc_bones(self, branch, bones):
        """Delete all my location data in the given branch, and then set
        location data from the bones.

        """
        self.character.del_thing_locations(self.name, branch)
        for bone in bones:
            self.character.closet.set_bone(bone)

    def iter_stat_keys(self, observer=None, branch=None, tick=None):
        (branch, tick) = self.character.sanetime(branch, tick)
        for key in self.get_subjectively(
                'iter_thing_stat_keys', observer, [branch, tick]):
            yield key
