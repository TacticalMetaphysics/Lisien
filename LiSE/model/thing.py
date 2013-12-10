# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from re import match

from LiSE.util import (
    TimeParadox,
    JourneyException,
    portex)

from container import Container


class Thing(Container):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    # dimension is now the name of the character that the *place* is in
    tables = [
        ("thing",
         {"character": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "host": "text",
          "location": "text"},
         ("character", "name", "branch", "tick"),
         {"host, location": ("place", "character, name")},
         []),
        ("thing_facade",
         {"observer": "text not null",
          "observed": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "host": "text",
          "location": "text"},
         ("observer", "observed", "name", "branch", "tick"),
         {"host, location": ("place", "character, name")},
         [])]
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
        if observer is None:
            return self.character.get_thing_bone(self.name, branch, tick)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_bone(branch, tick)

    def get_location(self, locn, observer=None, branch=None, tick=None):
        # Get the host character, to look up my location in.
        #
        # As the host is the character under observation, use its
        # facade, rather than my character's facade.
        my_bone = self.get_bone(None, branch, tick)
        host = self.character.closet.get_character(my_bone.host)
        if observer is None:
            getster = host
        else:
            getster = host.get_facade(observer)
        bone = getster.get_bone(locn)
        return getster.get_whatever(bone)

    def get_locations(self, observer=None, branch=None):
        if observer is None:
            return self.character.get_thing_locations(self.name, branch)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_thing_locations(self.name, branch)

    def get_speed(self, branch=None, tick=None):
        lo = self.get_bone(branch, tick).location
        ticks = self.get_ticks_thru(lo, branch, tick)
        return float(lo.weight) / float(ticks)

    def get_ticks_thru(
            self, portal, observer=None, branch=None, tick=None):
        """How many ticks would it take to get through that portal?"""
        if observer is None:
            observer = self.character
        portal_bone = portal.get_bone(observer, branch, tick)
        return int(portal_bone.weight / 0.1)

    def get_progress(self, observer=None, branch=None, tick=None):
        """Return a float representing the proportion of the portal I have
passed through.

        """
        bone = self.get_bone(observer, branch, tick)
        # this is when I entered the portal
        t1 = bone.tick_from
        # this is when I will enter the destination
        t2 = self.get_locations(branch).key_after(tick)
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

    def journey_to(self, destplace, host='Physical', branch=None, tick=None):
        """Schedule myself to travel to the given place, interrupting whatever
other journey I may be on at the time."""
        # TODO if I overwrite my extant travel schedule, overwrite
        # *everything* after the start of this new stuff. Right now,
        # anywhere I'm scheduled to be in a tick after the end of the
        # new journey, I'll still be there. It makes no sense.
        oloc = str(self.get_location(branch, tick))
        otick = tick
        m = match(portex, oloc)
        if m is not None:
            loc = m.groups()[0]
            tick = self.get_locations(branch).key_after(otick)
        else:
            loc = oloc
            tick = otick
        # It would be weird to use some other character's
        # understanding of the map.
        host = self.character.closet.get_character(host)
        facade = host.get_facade(self.character)
        ipath = facade.graph.get_shortest_paths(
            loc, to=unicode(destplace), output=str("epath"))
        path = None
        for p in ipath:
            if p == []:
                continue
            desti = facade.graph.es[p[-1]].target
            if desti == int(destplace):
                path = [facade.graph.es[i]["portal"] for i in p]
                break
        if path is None:
            raise JourneyException("Found no path to " + str(destplace))
        locs = list(self.branch_loc_bones_gen(branch))
        try:
            self.follow_path(path, branch, tick)
        except TimeParadox:
            self.restore_loc_bones(branch, locs)
            self.new_branch_blank = True
            increment = 1
            while branch + increment in self.locations:
                increment += 1
            self.character.closet.time_travel_inc_branch(branches=increment)
            self.new_branch_blank = False
            m = match(portex, oloc)
            if m is not None:
                loc = m.groups()[0]
                tick = self.get_locations().key_after(otick)
            else:
                loc = oloc
                tick = otick
            self.follow_path(path, None, tick)

    def follow_path(self, path, branch, tick):
        # only acceptable if I'm currently in the last place I'll be
        # in this branch
        try:
            if self.get_locations(observer=None, branch=branch).key_after(
                    tick) is not None:
                raise TimeParadox
        except KeyError:
            # This just means the branch isn't there yet. Don't worry.
            pass
        prevtick = tick + 1
        for port in path:
            self.set_location(port, observer=None,
                              branch=branch, tick=prevtick)
            prevtick += self.get_ticks_thru(
                port, observer=None, branch=branch, tick=prevtick)
            self.set_location(port.destination, observer=None,
                              branch=branch, tick=prevtick)
            prevtick += 1

    def new_branch(self, parent, branch, tick):
        def gethibranch():
            return self.dimension.closet.timestream.hi_branch
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
        for bone in self.get_locations(parent).iterbones():
            i += 1
            if bone.tick_from >= tick:
                bone2 = bone._replace(branch=branch)
                self.get_locations(branch)[bone2.tick_from] = bone2
                if (
                        not started and prev is not None and
                        bone.tick_from > tick and prev.tick_from < tick):
                    bone3 = prev._replace(branch=branch, tick_from=tick)
                    self.get_locations(branch)[bone3.tick_from] = bone3
                started = True
            prev = bone

    def branch_loc_bones_gen(self, branch=None):
        for bone in self.get_locations(branch).iterbones():
            yield bone

    def restore_loc_bones(self, branch, bones):
        self.character.del_thing_locations(branch)
        for bone in bones:
            self.set_location(bone.location, bone.branch, bone.tick_from)
