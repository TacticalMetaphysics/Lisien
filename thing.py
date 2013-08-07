# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    LocationException,
    BranchTicksIter)
from portal import Portal
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

    def __init__(self, dimension, name, locations={}, indef_locs={}):
        self.name = name
        self.dimension = dimension
        self.rumor = self.dimension.rumor
        self.locations = locations
        self.indefinite_locations = indef_locs
        self.pawns = []

    def __getattr__(self, attrn):
        if attrn == 'location':
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
        return self.name

    def __int__(self):
        return self.dimension.things.index(self)

    def __contains__(self, that):
        return that.location is self

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
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.locations:
            return None
        if branch in self.indefinite_locations:
            istart = self.indefinite_locations[branch]
            if tick >= istart:
                return self.locations[branch][istart][0]
        for (tick_from, (loc, tick_to)) in self.locations[branch].iteritems():
            if tick_from <= tick and tick <= tick_to:
                return loc
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
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.locations:
            self.locations[branch] = {}
        if branch in self.indefinite_locations:
            ifrom = self.indefinite_locations[branch]
            (iloc, ito) = self.locations[branch][ifrom]
            if tick_from > ifrom:
                self.locations[branch][ifrom] = (iloc, tick_from - 1)
                del self.indefinite_locations[branch]
            elif tick_to > ifrom:
                del self.locations[branch][ifrom]
                del self.indefinite_locations[brarnch]
        self.locations[branch][tick_from] = (loc, tick_to)
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
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.locations:
            raise LocationException("I am nowhere in that branch")
        for (tick_from, (loc, tick_to)) in self.locations[branch].iteritems():
            if tick_to is None:
                continue
            if tick_from <= tick and tick <= tick_to:
                assert(hasattr(loc, 'orig') and hasattr(loc, 'dest'))
                return float(tick - tick_from) / float(tick_to - tick_from)
        raise LocationException("I am nowhere at that time")

    def free_time(self, n, branch=None, tick=None):
        """Return the first tick after the one given, and after which there
are n ticks of free time."""
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.locations:
            # Well, not existing is certainly ONE way not to have commitments
            return tick
        laterthan = tick
        for (tick_from, (loc, tick_to)) in self.locations[branch].iteritems():
            # This is only a *travel* event if it puts me in a portal
            if not (hasattr(loc, 'orig') and hasattr(loc, 'dest')):
                continue
            if (tick_from - n <= laterthan and laterthan <= tick_to):
                laterthan = tick_to
        return laterthan + 1


    def get_tabdict(self):
        return {
            "thing_location": [
                {
                    "dimension": str(self.dimension),
                    "thing": str(self),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "location": str(location)}
                for (branch, tick_from, tick_to, location) in
                BranchTicksIter(self.locations)]}

    def end_location(self, branch=None, tick=None):
        """Find where I am at the given time. Arrange to stop being there then."""
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.locations:
            raise BranchError("Branch not known")
        for (tick_from, (loc, tick_to)) in self.locations[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                self.locations[branch][tick_from] = (loc, tick)
                if tick_to is None:
                    del self.indefinite_locations[branch]
                return

    def journey_to(self, destplace, branch=None, tick=None):
        """Schedule myself to travel to the given place, interrupting whatever
other journey I may be on at the time."""
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        loc = self.get_location(branch, tick)
        print "{0} journeys from {1} to {2}".format(str(self), str(loc), str(destplace))
        ipath = self.dimension.graph.get_shortest_paths(
            str(loc), to=str(destplace), output="epath")
        path = None
        for p in ipath:
            desti = self.dimension.graph.es[p[-1]].target
            if desti == int(destplace):
                path = [
                    Portal(self.dimension, self.dimension.graph.es[step]) for step in p]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        prevtick = tick + 1
        self.end_location(branch, tick)
        for port in path:
            tick_out = self.get_ticks_thru(port) + prevtick
            print "At tick {0}, {1} will enter {2}. At tick {3}, it will leave.".format(
                int(prevtick), str(self), str(port), int(tick_out))
            self.set_location(port, int(branch), int(prevtick), int(tick_out))
            prevtick = tick_out + 1
        print "{0} will arrive at {1} at tick {2}.".format(
            str(self), str(destplace), int(prevtick))
        self.set_location(destplace, int(branch), int(prevtick))
