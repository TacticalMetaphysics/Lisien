# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    LocationException,
    BranchTicksIter,
    SkeletonIterator,
    TimeParadox)
from logging import getLogger


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


class JourneyException(Exception):
    pass


class BranchError(Exception):
    pass


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
          "tick_to": "integer default null",
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
        self.indefinite_locations = {}
        self.new_branch_blank = False
        for rd in SkeletonIterator(self.locations):
            if rd["tick_to"] is None:
                self.indefinite_locations[rd["branch"]] = rd["tick_from"]
        self.dimension.thingdict[name] = self
        logger.debug(
            "Instantiated Thing %s. Its locations are:\n%s",
            str(self), repr(self.locations))

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
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if len(self.locations) < branch:
            return None
        if (
                branch in self.indefinite_locations and
                tick >= self.indefinite_locations[branch]):
            itf = self.indefinite_locations[branch]
            rd = self.locations[branch][itf]
            if rd["location"][:6] == "Portal":
                pstr = rd["location"][6:].strip("()")
                (orign, destn) = pstr.split("->")
                return self.dimension.get_portal(orign, destn)
            else:
                return self.dimension.get_place(rd["location"])
        for rd in SkeletonIterator(self.locations[branch]):
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                if rd["location"][:6] == "Portal":
                    pstr = rd["location"][6:].strip("()")
                    (orign, destn) = pstr.split("->")
                    return self.dimension.get_portal(orign, destn)
                else:
                    return self.dimension.get_place(rd["location"])
        return None

    def set_location(self, loc, branch=None, tick_from=None, tick_to=None):
        """Declare that I'm in the given Place, Portal, or Thing.

With no tick_to argument, I'll stay in this location
indefinitely--that is, until I have anything else to do. With tick_to,
I'll stay in this location until then, even if I DO have something
else to do.

Return an Effect representing the change.
        """
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        if branch in self.indefinite_locations:
            indef_start = self.indefinite_locations[branch]
            indef_rd = self.locations[indef_start]
            if tick_from > indef_start:
                indef_rd["tick_to"] = tick_from - 1
                del self.indefinite_locations[branch]
            elif tick_to > indef_start:
                del self.locations[branch][indef_start]
                del self.indefinite_locations[branch]
            elif (
                    tick_to == self.indef_start and
                    indef_rd["location"] == str(loc)):
                indef_rd["tick_from"] = tick_from
                self.indefinite_locations[branch] = tick_from
                return
        assert branch in self.locations, "Make a new branch first"
        self.locations[branch][tick_from] = {
            "dimension": str(self.dimension),
            "thing": str(self),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to,
            "location": str(loc)}
        print "remembered a location rowdict:"
        print self.locations[branch][tick_from]
        if tick_to is None:
            self.indefinite_locations[branch] = tick_from

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

Presupposes that I'm in a portal.

        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if len(self.locations) < branch:
            raise LocationException("I am nowhere in that branch")
        for rd in SkeletonIterator(self.locations[branch]):
            if rd["tick_to"] is None:
                continue
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return float(tick - rd["tick_from"]) / float(rd["tick_to"] - rd["tick_from"])
        raise LocationException("I am not in a portal at that time")

    def free_time(self, n, branch=None, tick=None):
        """Return the first tick after the one given, and after which there
are n ticks of free time."""
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if len(self.locations) < branch:
            return None
        laterthan = tick
        for (tick_from, (loc, tick_to)) in self.locations[branch].iteritems():
            # This is only a *travel* event if it puts me in a portal
            if not (hasattr(loc, 'orig') and hasattr(loc, 'dest')):
                continue
            if (tick_from - n <= laterthan and laterthan <= tick_to):
                laterthan = tick_to
        return laterthan + 1

    def end_location(self, branch=None, tick=None):
        """Find where I am at the given time. Arrange to stop being there
then."""
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if len(self.locations) < branch:
            raise BranchError("Branch not known")
        if branch in self.indefinite_locations:
            tick_from = self.indefinite_locations[branch]
            rd = self.locations[branch][tick_from]
            rd["tick_to"] = tick
            del self.indefinite_locations[branch]
        else:
            for rd in SkeletonIterator(self.locations[branch]):
                if rd["tick_from"] < tick and rd["tick_to"] > tick:
                    rd["tick_to"] = tick
                    return

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
        loc = self.get_location(branch, tick)
        if hasattr(loc, 'orig'):
            loc = loc.orig
        ipath = self.dimension.graph.get_shortest_paths(
            str(loc), to=str(destplace), output="epath")
        path = None
        for p in ipath:
            desti = self.dimension.graph.es[p[-1]].target
            if desti == int(destplace):
                path = [e["portal"] for e in [self.dimension.graph.es[i] for i in p]]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        prevtick = tick + 1
        try:
            self.follow_path(path, branch, tick)
        except TimeParadox:
            self.new_branch_blank = True
            self.closet.increment_branch()
            self.new_branch_blank = False
            self.follow_path(path, self.closet.branch, tick)
    
    def follow_path(self, path, branch, tick):
        self.end_location(branch, tick)
        prevtick = tick
        for port in path:
            tick_out = self.get_ticks_thru(port) + prevtick
            self.set_location(port, int(branch), int(prevtick), int(tick_out))
            prevtick = tick_out + 1
        destplace = path[-1].dest
        self.set_location(destplace, int(branch), int(prevtick))
        self.update()

    def new_branch(self, parent, branch, tick):
        if self.new_branch_blank:
            return
        for rd in SkeletonIterator(self.locations[parent]):
            if rd["tick_to"] is None or rd["tick_to"] >= tick:
                rd2 = dict(rd)
                rd2["branch"] = branch
                if rd2["tick_from"] < tick:
                    rd2["tick_from"] = tick
                    self.locations[branch][tick] = rd2
                    if rd2["tick_to"] is None:
                        self.indefinite_locations[branch] = tick
                else:
                    self.locations[branch][rd2["tick_from"]] = rd2
                    if rd2["tick_to"] is None:
                        self.indefinite_locations[branch] = rd2["tick_from"]
