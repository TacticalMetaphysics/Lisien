# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.util import TimeParadox
from place import Place, Container


class Thing(Container):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    demands = ["portal_loc", "place_stat"]
    tables = [
        ("thing_loc", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "location": "text"},
            "primary_key": (
                "character", "name", "branch", "tick")}),
        ("thing_stat", {
            "columns": {
                "character": "text not null",
                "name": "text not null",
                "key": "text",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "name", "key", "branch", "tick"),
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

    def __getitem__(self, key):
        if key == 'contents':
            return self.character.thing_contents_d[self.name]
        elif key == 'location':
            return self.location
        else:
            return self.character.thing_stat_d[self.name][key]

    @property
    def loc_bone(self):
        (branch, tick) = self.character.closet.time
        return self.character.closet.skeleton[u"thing_loc"][
            self.character.name][self.name][branch].value_during(tick)

    @property
    def location(self):
        bone = self.loc_bone
        locn = bone.location
        if locn in self.character.thing_d:
            return self.character.thing_d[locn]
        elif locn in self.character.portal_d:
            return self.character.portal_d[locn]
        elif locn not in self.character.place_d:
            self.character.place_d[locn] = Place(self.character, locn)
        return self.character.place_d[locn]

    def follow_path(self, path):
        """Presupposing I'm in the given host, follow the path by scheduling
        myself to be located in the appropriate place or portal at the
        appropriate time.

        Optional arguments ``branch`` and ``tick`` give the start
        time. If unspecified, the current diegetic time is used.

        Raises ``TimeParadox`` if I would contradict any locations
        already scheduled.

        """
        (branch, tick) = self.character.closet.time
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
            # Currently, a Thing will stay in each Place on a path for
            # exactly one tick before leaving.  This may not always be
            # appropriate.
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
