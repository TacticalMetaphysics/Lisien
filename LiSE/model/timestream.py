from LiSE.orm import SaveableMetaclass


class Timestream(object):
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

    @property
    def hi_time(self):
        return (self.hi_branch, self.hi_tick)

    def __init__(self, closet):
        """Initialize hi_branch and hi_tick to 0, and their listeners to
        empty.

        """
        self.closet = closet
        self.hi_branch_listeners = []
        self.hi_tick_listeners = []
        self.hi_time_listeners = []
        self.hi_branch = 0
        self.hi_tick = 0

    def __setattr__(self, attrn, val):
        """Trigger the listeners as needed"""
        if attrn == "hi_branch":
            for listener in self.hi_branch_listeners:
                listener(self, val)
            for listener in self.hi_time_listeners:
                listener(self, val, self.hi_tick)
        elif attrn == "hi_tick":
            for listener in self.hi_tick_listeners:
                listener(self, val)
            for listener in self.hi_time_listeners:
                listener(self, self.hi_branch, val)
        super(Timestream, self).__setattr__(attrn, val)

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

    def uptick(self, tick):
        """Set ``self.hi_tick`` to ``tick`` if the present value is lower."""
        self.hi_tick = max((tick, self.hi_tick))

    def upbranch(self, branch):
        """Set ``self.hi_branch`` to ``branch`` if the present value is
        lower."""
        self.hi_branch = max((branch, self.hi_branch))

    def parent(self, branch):
        """Return the parent of the branch"""
        return self.closet.skeleton["timestream"][branch].parent

    def children(self, branch):
        """Generate all children of the branch"""
        for bone in self.closet.skeleton["timestream"].iterbones():
            if bone.parent == branch:
                yield bone.branch
