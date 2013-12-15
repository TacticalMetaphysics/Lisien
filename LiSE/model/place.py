# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container
from LiSE.util import PlaceBone


class Place(Container):
    """Where you go when you have to be someplace.

    Places are vertices in a character's graph where things can go,
    and where portals can lead. A place's name must be unique within
    its character.

    Unlike things and portals, places can't be hosted by characters
    other than the one they are part of. When something is "hosted" in
    a character, that means it's in a place that's in the
    character--always.

    You don't need to create a bone for each and every place you
    use--link to it with a portal, or put a thing there, and it will
    exist. Place bones are only for when a place needs stats.

    """
    tables = [
        ("place_stat", {
            "columns": {
                "character": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "name", "key", "branch", "tick")}),
        ("place_stat_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "observer", "observed", "name", "key", "branch", "tick")})]

    @property
    def v(self):
        return self.character.graph.vs.find(name=self.name)

    def __init__(self, character, name):
        """Initialize a place in a character by a name"""
        self.character = character
        self.name = name

    def __contains__(self, that):
        """Is that here?"""
        return self.contains(that)

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def _iter_portals_bones(self, observer=None, branch=None, tick=None):
        """Iterate over the bones of portals that lead out from me"""
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
        """Iterate over all those portals which lead from this place to
        elsewhere."""
        if observer is None:
            getter = self.character.get_portal
        else:
            facade = self.character.get_facade(observer)
            getter = facade.get_portal
        for bone in self._iter_portals_bones(observer, branch, tick):
            yield getter(bone.name)

    def get_portals(self, observer=None, branch=None, tick=None):
        """Get a set of names of portals leading out from here."""
        if observer is None:
            getter = self.character.get_portal
        else:
            facade = self.character.get_facade(observer)
            getter = facade.get_portal
        return set([
            getter(bone.name) for bone in
            self._iter_portals_bones(observer, branch, tick)])

    def upd_skel_from_bone(self, bone):
        """Update the 'place' skeleton so that it contains the place(s)
        referred to in the bone."""
        if hasattr(bone, 'origin'):
            pobone = self.character.closet.skeleton[u"portal"][
                bone.character][bone.name]
            host = pobone.host
            for name in (bone.origin, bone.destination):
                if not self.character.closet.place_exists(
                        host, name, bone.branch, bone.tick):
                    self.character.closet.set_bone(PlaceBone(
                        character=host, place=name,
                        branch=bone.branch, tick=bone.tick))
            return
        elif hasattr(bone, 'place'):
            if not self.character.closet.place_exists(
                    bone.character, bone.place, bone.branch, bone.tick):
                self.character.closet.set_bone(PlaceBone(
                    character=bone.character, place=bone.place,
                    branch=bone.branch, tick=bone.tick))
        elif hasattr(bone, 'location'):
            thbone = self.character.closet.skeleton[u"thing"][
                bone.character][bone.name]
            host = thbone.host
            if not self.character.closet.place_exists(
                    host, bone.location, bone.branch, bone.tick):
                self.character.closet.set_bone(PlaceBone(
                    character=host, place=bone.location,
                    branch=bone.branch, tick=bone.tick))
