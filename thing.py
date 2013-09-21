# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    LocationException,
    TimeParadox,
    JourneyException)
from re import match, compile
from logging import getLogger
from collections import deque


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)

portex = compile("Portal\((.+)->(.+)\)")


class Thing:
    """The sort of item that has a particular location at any given time.

Every Thing has a Journey and a Schedule, but either may be empty.

Things can contain other Things and be contained by other Things. This
is independent of where they are "located," which is always a
Place. But any things that contain one another should share locations,
too.

    """
    tables = [
        ("thing_location",
         {"dimension": "text not null default 'Physical'",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "location": "text not null"},
         ("dimension", "thing", "branch", "tick_from"),
         {"dimension, thing": ("thing", "dimension, name"),
          "dimension, location": ("place", "dimension, name")},
         [])]
    basic_speed = 0.1

    def __init__(self, closet, dimension, name):
        self.closet = closet
        self.update_handlers = set()
        self.dimension = dimension
        self._name = str(name)
        self.new_branch_blank = False
        self.locations = self.closet.skeleton["thing_location"][
            str(self.dimension)][str(self)]
        self.dimension.thingdict[name] = self
        self.branches_in = set()

    def __getattr__(self, attrn):
        if attrn == "locations":
            return self.closet.skeleton["thing_location"][
                str(self.dimension)][str(self)]
        elif attrn == 'location':
            return self.get_location()
        elif attrn == 'speed':
            return self.get_speed()
        elif attrn == 'distance':
            return self.get_distance()
        elif attrn == 'progress':
            return self.get_progress()
        else:
            raise AttributeError(
                "Thing instance {0} has no attribute {1}".format(
                    str(self), attrn))

    def __setattr__(self, attrn, val):
        if attrn == 'location':
            self.set_location(val)
        else:
            super(Thing, self).__setattr__(attrn, val)

    def __str__(self):
        return self._name

    def __contains__(self, that):
        return that.location is self

    def register_update_handler(self, that):
        self.update_handlers.add(that)

    def update(self):
        self.closet.timestream.update()
        for handler in self.update_handlers:
            handler(self)

    def assert_can_enter(self, it):
        """If I can't enter the given location, raise a LocationException.

Assume that the given location has no objections."""
        pass

    def assert_can_leave_location(self):
        """If I can't leave the location I'm currently in, raise a
LocationException."""
        pass

    def assert_can_contain(self, it):
        """If I can't contain the given item, raise a ContainmentException."""
        pass

    def get_location(self, branch=None, tick=None):
        """Return my current location by default, or where I was at the given
tick in the given branch."""
        rd = self.get_location_rd(branch, tick)
        if rd is None or rd["location"] is None:
            return None
        m = match(portex, rd["location"])
        if m is not None:
            return self.dimension.get_portal(*m.groups())
        elif rd["location"] in self.dimension.thingdict:
            return self.dimension.thingdict[rd["location"]]
        else:
            return self.dimension.get_place(rd["location"])

    def get_location_rd(self, branch=None, tick=None):
        r = self.get_location_rd_triad(branch, tick)
        if len(r) < 2:
            return None
        else:
            return r[1]

    def get_location_rd_triad(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        r = deque([], 3)
        for rd in self.locations[branch].iterrows():
            r.append(rd)
            if rd["tick_from"] > tick:
                break
        return tuple(r)

    def exists(self, branch=None, tick=None):
        rd = self.get_location_rd(branch, tick)
        return None not in (rd, rd["location"])

    def set_location(self, loc, branch=None, tick=None):
        """Declare that I'm in the given Place, Portal, or Thing.

With no tick_to argument, I'll stay in this location
indefinitely--that is, until I have anything else to do. With tick_to,
I'll stay in this location until then, even if I DO have something
else to do.

Return an Effect representing the change.
        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.locations[branch][tick] = {
            "dimension": str(self.dimension),
            "thing": str(self),
            "branch": branch,
            "tick_from": tick,
            "location": str(loc)}

    def get_speed(self, branch=None, tick=None):
        lo = self.get_location(branch, tick)
        ticks = self.get_ticks_thru(lo)
        return float(len(lo)) / float(ticks)

    def get_ticks_thru(self, po):
        """How many ticks would it take to get through that portal?"""
        # basic_speed should really be looked up in a character, this
        # is really a placeholder
        return len(po) / self.basic_speed

    def determined_ticks_thru(self, po):
        """Can I assume that it will always take *the same* number of ticks to
get through that portal?"""
        return True

    def get_distance(self, branch=None, tick=None):
        """Return a float representing the number of spans across the portal
I've gone.

Presupposes that I'm in a portal.

        """
        return NotImplemented

    def get_progress(self, branch=None, tick=None):
        """Return a float representing the proportion of the portal I have
passed through.

If I'm not in a Portal, raise LocationException.

        """
        if branch is None:
            branch = self.closet.branch
        if branch not in self.locations:
            raise LocationException("I am nowhere in that branch")
        rds = self.get_location_rd_triad(branch, tick)
        if rds[1] is None or rds[1]["location"] is None:
            raise LocationException("I am nowhere at that tick")
        m = match(portex, rds[1]["location"])
        if m is None:
            raise LocationException("I am not in a portal at that tick")
        duration = float(rds[2]["tick_from"] - rds[1]["tick_from"])
        passed = float(tick - rds[2]["tick_from"])
        return passed / duration

    def journey_to(self, destplace, branch=None, tick=None):
        """Schedule myself to travel to the given place, interrupting whatever
other journey I may be on at the time."""
        # TODO if I overwrite my extant travel schedule, overwrite
        # *everything* after the start of this new stuff. Right now,
        # anywhere I'm scheduled to be in a tick after the end of the
        # new journey, I'll still be there. It makes no sense.
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        locrds = self.get_location_rd_triad(branch, tick)
        m = match(portex, locrds[1]["location"])
        if m is None:
            loc = locrds[1]["location"]
        else:
            # I assume I'm not staying indefinitely in the portal I am in
            loc = self.dimension.get_place(m.groups()[1])
            tick = locrds[2]["tick_from"]
        ipath = self.dimension.graph.get_shortest_paths(
            str(loc), to=str(destplace), output="epath")
        path = None
        for p in ipath:
            if p == []:
                continue
            desti = self.dimension.graph.es[p[-1]].target
            if desti == int(destplace):
                path = [e["portal"] for e in
                        [self.dimension.graph.es[i] for i in p]]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        locs = self.branch_loc_rds(branch)
        try:
            self.follow_path(path, branch, tick)
        except TimeParadox:
            del self.locations[branch]
            self.restore_loc_rds(locs)
            self.new_branch_blank = True
            increment = 1
            while branch + increment in self.locations:
                increment += 1
            self.closet.time_travel_inc_branch(branches=increment)
            self.dimension.check_thing_id(self)
            self.new_branch_blank = False
            branch = self.closet.branch
            self.follow_path(path, branch, tick)

    def follow_path(self, path, branch, tick):
        prevtick = tick
        for port in path:
            self.set_location(port, int(branch), int(prevtick))
            prevtick = self.get_ticks_thru(port) + prevtick
        destplace = path[-1].dest
        self.set_location(destplace, int(branch), int(prevtick))
        self.update()

    def new_branch(self, parent, branch, tick):
        if branch not in self.locations:
            self.locations[branch] = []
        if self.new_branch_blank:
            start_loc = self.get_location(parent, tick)
            self.set_location(start_loc, branch, tick)
            return
        prev = None
        started = False
        for rd in self.locations[parent].iterrows():
            if rd["tick_from"] >= tick:
                rd2 = dict(rd)
                rd2["branch"] = branch
                self.locations[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        rd["tick_from"] > tick and prev["tick_from"] < tick):
                    rd3 = dict(prev)
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.locations[branch][rd3["tick_from"]] = rd3
                started = True
            prev = rd

    def branch_loc_rds(self, branch=None):
        if branch is None:
            branch = self.closet.branch
        r = [rd.deepcopy() for rd in self.locations[branch].iterrows()]
        return r

    def restore_loc_rds(self, rds):
        logger.debug("Restoring locations")
        for rd in rds:
            if rd["branch"] not in self.locations:
                self.locations[rd["branch"]] = []
            self.locations[rd["branch"]][rd["tick_from"]] = rd
