# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.util import (
    TimeParadox,
    JourneyException
)

from container import Container


class Thing(Container):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    demands = ["portal", "place_stat"]
    tables = [
        ("thing_loc", {
            "columns": {
                "host": "text not null default 'Physical'",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "location": "text"},
            "primary_key": (
                "host", "name", "branch", "tick")}),
        ("thing_stat", {
            "columns": {
                "character": "text not null",
                "host": "text not null",
                "name": "text not null",
                "key": "text",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "host", "name", "key", "branch", "tick"),
            "foreign_keys": {
                "character, name": (
                    "thing", "character, name")}})]

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

    def get_location(self, branch=None, tick=None):
        pass

    def get_progress(self, branch=None, tick=None):
        """Return a float representing the proportion of the portal I have
        passed through.

        """
        pass

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
        if self.new_branch_blank:
            start_loc = self.get_location(parent, tick)
            if hasattr(start_loc, 'destination'):
                tick = self.locations[parent].key_after(tick)
                start_loc = self.get_location(parent, tick)
            locb = self.bonetypes["thing_loc"](
                host=unicode(self.host),
                name=self.name,
                branch=branch,
                tick=tick,
                location=start_loc)
            yield locb
            return
        for bone in self.iter_loc_bones(branch=parent):
            yield bone._replace(branch=branch)
        for bone in self.iter_stats_bones(branch=parent):
            yield bone._replace(branch=branch)
        for observer in self.character.facade_d.iterkeys():
            for bone in self.iter_loc_bones(observer, branch=parent):
                yield bone._replace(branch=branch)
            for bone in self.iter_stats_bones(
                    stats=[], observer=observer, branch=parent):
                yield bone._replace(branch=branch)
