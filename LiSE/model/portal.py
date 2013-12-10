# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container


class Portal(Container):
    tables = [
        ("portal",
         {"character": "text not null",
          "host": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "origin": "text",
          "destination": "text",
          "weight": "integer default 1"},
         ("character", "name", "branch", "tick"),
         # portals that are directly in characters must refer to
         # places that "really exist" in their character.  whether
         # those places exist in any physical sense depends on the
         # character
         {"host, origin":
          ("place", "character, name"),
          "host, destination":
          ("place", "character, name")},
         []),
        ("portal_facade",
         {"observer": "text not null",
          "observed": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0",
          "host": "text",
          "origin": "text",
          "destination": "text",
          "weight": "integer default 1"},
         ("observer", "observed", "name", "branch", "tick"),
         {"host, origin": ("place", "character, name"),
          "host, destination": ("place", "character, name")},
         [])]

    def __init__(self, character, name):
        self.character = character
        self.name = name

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        bone = self.bone
        return "{2}({0}->{1})".format(
            bone.origin, bone.destination, self.name)

    def get_bone(self, observer=None, branch=None, tick=None):
        if observer is None:
            return self.character.get_portal_bone(self.name, branch, tick)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_portal_bone(self.name, branch, tick)
