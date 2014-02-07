from LiSE.orm import SaveableMetaclass
from LiSE.util import HandleHandler


class Timestream(HandleHandler):
    """Tracks the genealogy of the various branches of time.


    Branches of time each have one parent; branch zero is its own parent."""
    __metaclass__ = SaveableMetaclass
    # I think updating the start and end ticks of a branch using
    # listeners might be a good idea
    tables = [
        ("timestream",
         {"columns":
          {"branch": "integer not null",
           "parent": "integer not null"},
          "primary_key": ("branch", "parent"),
          "foreign_keys":
          {"parent": ("timestream", "branch")},
          "checks":
          ["parent=0 or parent<>branch"]})
        ]

    def __getattr__(self, attrn):
        d = {'branch': self._branch,
             'tick': self._tick,
             'hi_branch': self._hi_branch,
             'hi_tick': self._hi_tick,
             'time': (self._branch, self._tick),
             'hi_time': (self._hi_branch, self._hi_tick)}
        if attrn in d:
            return d[attrn]
        else:
            raise AttributeError

    def __setattr__(self, attrn, value):
        """Fire listeners for branch, tick, hi_branch, hi_tick"""
        setter = super(Timestream, self).__setattr__
        fire = set()
        if attrn == 'time':
            (branch, tick) = value
            setter('_branch', branch)
            setter('_tick', tick)
            fire.update(['branch', 'tick'])
        elif attrn == 'hi_time':
            (hi_branch, hi_tick) = value
            setter('_hi_branch', hi_branch)
            setter('_hi_tick', hi_tick)
            fire.update(['hi_branch', 'hi_tick'])
        elif attrn == 'branch':
            setter('_branch', value)
            fire.add('branch')
            if self.branch > self.hi_branch:
                setter('_hi_branch', self.branch)
                fire.add('hi_branch')
        elif attrn == 'tick':
            setter('_tick', value)
            fire.add('tick')
            if self.tick > self.hi_tick:
                setter('_hi_tick', self.tick)
                fire.add('hi_tick')
        else:
            setter(attrn, value)
        for ln in fire:
            for f in getattr(self, ln + '_listeners'):
                f(getattr(self, ln))
        if 'branch' in fire or 'tick' in fire:
            for f in self.time_listeners:
                f(self.branch, self.tick)
        if 'hi_branch' in fire or 'hi_tick' in fire:
            for f in self.hi_time_listeners:
                f(self.hi_branch, self.hi_tick)

    def __init__(self, closet):
        """Initialize hi_branch and hi_tick to 0, and their listeners to
        empty.

        """
        self.closet = closet
        self._branch = 0
        self._tick = 0
        self._hi_branch = 0
        self._hi_tick = 0
        self.mk_handles('branch', 'tick', 'time',
                        'hi_branch', 'hi_tick', 'hi_time')

    def min_branch(self, table=None):
        """Return the lowest known branch.

        With optional argument ``table``, consider only records in
        that table.

        """
        lowest = None
        skel = self.closet.skeleton
        if table is not None:
            skel = skel[table]
        for bone in skel.iterbones():
            if hasattr(bone, 'branch') and (
                    lowest is None or bone.branch < lowest):
                lowest = bone.branch
        return lowest

    def max_branch(self, table=None):
        """Return the highest known branch.

        With optional argument ``table``, consider only records in
        that table.

        """
        highest = 0
        skel = self.closet.skeleton
        if table is not None:
            skel = skel[table]
        for bone in skel.iterbones():
            if hasattr(bone, 'branch') and bone.branch > highest:
                highest = bone.branch
        return highest

    def max_tick(self, branch=None, table=None):
        """Return the highest recorded tick in the given branch, or every
        branch if none given.

        With optional argument table, consider only records in that table.

        """
        highest = 0
        skel = self.closet.skeleton
        if table is not None:
            skel = skel[table]
        for bone in skel.iterbones():
            if branch is None or (
                    hasattr(bone, 'branch') and bone.branch == branch):
                for attrn in ('tick_from', 'tick_to', 'tick'):
                    if hasattr(bone, attrn):
                        tick = getattr(bone, attrn)
                        if tick is not None and tick > highest:
                            highest = tick
        return highest

    def min_tick(self, branch=None, table=None):
        """Return the lowest recorded tick in the given branch, or every
        branch if none given.

        With optional argument table, consider only records in that table.

        """
        lowest = None
        skel = self.closet.skeleton
        if table is not None:
            skel = skel[table]
        for bone in skel.iterbones():
            if branch is None or (
                    hasattr(bone, 'branch') and bone.branch == branch):
                for attrn in ('tick_from', 'tick'):
                    if hasattr(bone, attrn):
                        tick = getattr(bone, attrn)
                        if tick is not None and (
                                lowest is None or
                                tick < lowest):
                            lowest = tick
        return lowest

    def parent(self, branch):
        """Return the parent of the branch"""
        return self.closet.skeleton["timestream"][branch].parent

    def children(self, branch):
        """Generate all children of the branch"""
        for bone in self.closet.skeleton["timestream"].iterbones():
            if bone.parent == branch:
                yield bone.branch
