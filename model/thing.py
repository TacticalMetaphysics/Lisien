# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    LocationException,
    TimeParadox,
    JourneyException,
    thingex,
    placex,
    portex)
from re import match
from logging import getLogger

logger = getLogger(__name__)


class Thing(object):
    __metaclass__ = SaveableMetaclass
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
          "location": "text"},
         ("dimension", "thing", "branch", "tick_from"),
         {},
         [])]
    basic_speed = 0.1

    @property
    def locations(self):
        return self.closet.skeleton["thing_location"][
            unicode(self.dimension)][unicode(self)]

    @property
    def location(self):
        return self.get_location()

    @property
    def speed(self):
        return self.get_speed()

    @property
    def distance(self):
        return self.get_distance()

    @property
    def progress(self):
        return self.get_progress()

    def __init__(self, closet, dimension, name):
        self.closet = closet
        self.dimension = dimension
        self._name = unicode(name)
        self.new_branch_blank = False
        self.dimension.thingdict[name] = self

    def __str__(self):
        return str(self._name)

    def __unicode__(self):
        return unicode(self._name)

    def __repr__(self):
        return "Thing({})".format(self)

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
        try:
            bone = self.get_location_bone(branch, tick)
        except KeyError:
            return None
        if bone is None or bone.location is None:
            return None
        if bone.location in self.dimension.graph.vs["name"]:
            return self.dimension.get_place(bone.location)
        try:
            (orign, destn) = bone.location.split("->")
            oi = self.dimension.graph.vs.find(orign).index
            di = self.dimension.graph.vs.find(destn).index
            eid = self.dimension.graph.get_eid(oi, di)
            return self.dimension.graph.es[eid]["portal"]
        except Exception as e:
            return self.dimension.get_thing(bone.location)

    def get_location_bone(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if branch not in self.locations:
            return None
        return self.locations[branch].value_during(tick)

    def exists(self, branch=None, tick=None):
        """Have I got a location?

If not, I'm nowhere, and therefore don't exist."""
        try:
            rd = self.get_location_bone(branch, tick)
        except KeyError:
            return False
        return None not in (rd, rd["location"])

    def set_location(self, loc, branch=None, tick=None):
        """Declare that I'm in the given Place, Portal, or Thing.

        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if branch not in self.locations:
            self.locations[branch] = {}
        self.locations[branch][tick] = self.bonetypes.thing_location(
            dimension=unicode(self.dimension),
            thing=unicode(self),
            branch=branch,
            tick_from=tick,
            location=unicode(loc))
        self.closet.timestream.upbranch(branch)
        self.closet.timestream.uptick(tick)

    def get_speed(self, branch=None, tick=None):
        lo = self.get_location(branch, tick)
        ticks = self.get_ticks_thru(lo)
        return float(len(lo)) / float(ticks)

    def get_ticks_thru(self, po):
        """How many ticks would it take to get through that portal?"""
        # basic_speed should really be looked up in a character, this
        # is really a placeholder
        return int(len(po) / self.basic_speed)

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

        """
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if branch not in self.locations:
            raise LocationException("I am nowhere in that branch")
        # this is when I entered the portal
        t1 = self.get_location_bone(branch, tick)["tick_from"]
        # this is when I will enter the destination
        t2 = self.locations[branch].key_after(tick)
        if t2 is None:
            # I entered the portal without scheduling when to leave.
            # This should never happen *in play* but I guess a
            # developer might put me in the portal before scheduling
            # my time to leave it.  Return 0.5 so that I appear
            # halfway thru the portal, therefore, clearly "in" it.
            return 0.5
        duration = float(t2 - t1)
        passed = float(tick - t1)
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
        oloc = str(self.get_location(branch, tick))
        otick = tick
        m = match(portex, oloc)
        if m is not None:
            loc = m.groups()[0]
            tick = self.locations[branch].key_after(otick)
        else:
            loc = oloc
            tick = otick
        assert(tick is not None)
        ipath = self.dimension.graph.get_shortest_paths(
            loc, to=unicode(destplace), output=str("epath"))
        path = None
        for p in ipath:
            if p == []:
                continue
            desti = self.dimension.graph.es[p[-1]].target
            if desti == int(destplace):
                path = [self.dimension.graph.es[i]["portal"] for i in p]
#[e["portal"] for e in
#                        [self.dimension.graph.es[i] for i in p]]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        locs = list(self.branch_loc_bones_gen(branch))
        try:
            self.follow_path(path, branch, tick)
        except TimeParadox:
            del self.locations[branch]
            self.restore_loc_bones(locs)
            self.new_branch_blank = True
            increment = 1
            while branch + increment in self.locations:
                increment += 1
            self.closet.time_travel_inc_branch(branches=increment)
            self.new_branch_blank = False
            branch = self.closet.branch
            m = match(portex, oloc)
            if m is not None:
                loc = m.groups()[0]
                tick = self.locations[branch].key_after(otick)
            else:
                loc = oloc
                tick = otick
            self.follow_path(path, branch, tick)

    def follow_path(self, path, branch, tick):
        # only acceptable if I'm currently in the last place I'll be
        # in this branch
        try:
            if self.locations[branch].key_after(tick) is not None:
                raise TimeParadox
        except KeyError:
            # This just means the branch isn't there yet. Don't worry.
            pass
        prevtick = tick + 1
        for port in path:
            self.set_location(port, branch, prevtick)
            prevtick += self.get_ticks_thru(port)
            self.set_location(port.destination, branch, prevtick)
            prevtick += 1

    def new_branch(self, parent, branch, tick):
        def gethibranch():
            return self.dimension.closet.timestream.hi_branch
        if branch not in self.locations:
            self.locations[branch] = {}
        if self.new_branch_blank:
            start_loc = self.get_location(parent, tick)
            if hasattr(start_loc, 'destination'):
                tick = self.locations[parent].key_after(tick)
                start_loc = self.get_location(parent, tick)
            self.set_location(start_loc, branch, tick)
            return
        prev = None
        started = False
        i = 0
        for rd in self.locations[parent].iterbones():
            i += 1
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

    def branch_loc_bones_gen(self, branch=None):
        if branch is None:
            branch = self.closet.branch
        for bone in self.locations[branch].iterbones():
            yield bone

    def restore_loc_bones(self, bones):
        logger.debug("Restoring locations")
        for bone in bones:
            self.set_location(bone.location, bone.branch, bone.tick_from)
