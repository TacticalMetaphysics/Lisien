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
    tables = [
        (
            "thing_stat",
            {
                "columns":
                {
                    "character": "text not null",
                    "name": "text not null",
                    "key": "text not null",
                    "branch": "integer not null default 0",
                    "tick": "integer not null default 0",
                    # null value means this key doesn't apply to me
                    "value": "text",
                    "type": "text not null default 'text'"
                },
                "primary_key":
                ("character", "name", "key", "branch", "tick"),
                "checks":
                ["type in ('text', 'real', 'boolean', 'integer')"]
            }
        )
    ]

    def __init__(self, character, name):
        self.character = character
        self.name = name
        self.new_branch_blank = False

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        return "Thing({})".format(self.name)

    def __contains__(self, that):
        return that.location == self.name

    def __getitem__(self, key):
        if key == 'contents':
            return self.character.thing_contents_d[self.name]
        elif key == 'location':
            return self.location
        else:
            return self.character.thing_stat_d[self.name][key]

    @property
    def location(self):
        (branch, tick) = self.character.closet.timestream.time
        return (
            self.character.closet.skeleton
            [u'thing_stat']
            [self.character.name]
            [self.name]
            [u'location']
            [branch].value_during(tick)
        )

    def follow_path(self, path):
        """Presupposing I'm in the given host, follow the path by scheduling
        myself to be located in the appropriate place or portal at the
        appropriate time.

        Optional arguments ``branch`` and ``tick`` give the start
        time. If unspecified, the current diegetic time is used.

        Raises ``TimeParadox`` if I would contradict any locations
        already scheduled.

        """
        (branch, tick) = self.character.closet.timestream.time
        locs = (
            self.character.closet.skeleton
            [u'thing_stat']
            [self.character.name]
            [self.name]
            [self.location]
        )
        if branch in locs:
            try:
                aft = locs[branch].key_after(tick)
                raise TimeParadox(
                    "Tried to follow a path at tick {},"
                    " but I was scheduled to be elsewhere "
                    " at tick {}".format(tick, aft)
                )
            except ValueError:
                pass
        bone = locs[branch].value_during(tick)
        prevtick = tick + self.ticks_to_leave(bone.value)
        # TODO

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
                location=start_loc
            )
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
                    stats=[],
                    observer=observer,
                    branch=parent
            ):
                yield bone._replace(branch=branch)
