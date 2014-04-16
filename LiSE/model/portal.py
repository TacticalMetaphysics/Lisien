# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container
from LiSE.util import upbranch


class Portal(Container):
    tables = [
        ("portal_loc", {
            "columns": {
                "host": "text not null",
                "name": "text not null",
                "branch": "integer not null",
                "tick": "integer not null",
                "origin": "text",
                "destination": "text"},
            "primary_key": (
                "host", "name", "branch", "tick")}),
        ("portal_stat", {
            "columns": {
                "host": "text not null",
                "name": "text not null",
                "key": "text",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "host", "name", "key", "branch", "tick")})]

    def __init__(self, host, name):
        self.host = host
        self.name = name

    def __eq__(self, other):
        return (
            isinstance(other, Portal) and
            self.host == other.host and
            self.name == other.name)

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        bone = self.get_loc_bone()
        return "{2}({0}->{1})".format(
            bone.origin, bone.destination, self.name)

    def get_loc_bone(self, branch=None, tick=None):
        pass

    def get_origin(self, branch=None, tick=None):
        pass

    def get_destination(self, branch=None, tick=None):
        pass

    def new_branch(self, parent, branch, tick):
        skel = self.character.closet.skeleton[u"portal_loc"][
            unicode(self.host)][unicode(self)][parent]
        for bone in upbranch(
                self.character.closet, skel.iterbones(), branch, tick):
            yield bone
