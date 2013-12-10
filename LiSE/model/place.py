# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container


class Place(Container):
    """Where you go when you have to be someplace.

    Places are vertices in a character's graph where things can go,
    and where portals can lead. A place's name must be unique within
    its character.

    Places have no other distinguishing characteristics. They are made
    distinct by putting stuff in them. To find out what is here, use
    the methods ``get_contents`` and ``get_portals``.

    Places exist only in characters, and not in facades. But when
    somebody looks into a place, what they see there depends entirely
    on the facade.

    """
    tables = [
        ("place",
         {"character": "text not null",
          "name": "text not null",
          "branch": "integer not null default 0",
          "tick": "integer not null default 0"},
         ("character", "name", "branch", "tick"),
         {},
         [])]

    def __init__(self, character, name):
        self.character = character
        self.name = name

    def __contains__(self, that):
        return self.contains(that)

    def get_bone(self, branch=None, tick=None):
        return self.character.get_place_bone(branch, tick)

    def _iter_portals_bones(self, observer=None, branch=None, tick=None):
        if observer is None:
            for bone in self.character.iter_portal_bones(branch, tick):
                if (
                        bone.host == unicode(self.character) and
                        bone.location == self.name):
                    yield bone
            return
        facade = self.character.get_facade(observer)
        for bone in facade.iter_hosted_portal_bones(branch, tick):
            if bone.origin == self.name:
                yield bone

    def iter_portals(self, observer=None, branch=None, tick=None):
        if observer is None:
            getter = self.character.get_portal
        else:
            facade = self.character.get_facade(observer)
            getter = facade.get_portal
        for bone in self._iter_portals_bones(observer, branch, tick):
            yield getter(bone.name)

    def get_portals(self, observer=None, branch=None, tick=None):
        if observer is None:
            getter = self.character.get_portal
        else:
            facade = self.character.get_facade(observer)
            getter = facade.get_portal
        return set([
            getter(bone.name) for bone in
            self._iter_portals_bones(observer, branch, tick)])
