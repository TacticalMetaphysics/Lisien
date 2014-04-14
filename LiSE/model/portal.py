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
                "character": "text not null default 'Physical'",
                "host": "text not null",
                "name": "text not null",
                "key": "text",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "host", "name", "key", "branch", "tick")})]

    @property
    def e(self):
        return self.character.graph.es.find(name=self.name)

    # @property
    # def loc_bone(self):
    #     return self.get_loc_bone()

    @property
    def host(self):
        return self.character.closet.get_character(self.bone.host)

    @property
    def origin(self):
        return self.get_origin()

    @property
    def destination(self):
        return self.get_destination()

    def __init__(self, character, name):
        self.character = character
        self.name = name
        self.closet = self.character.closet
        self.bone = self.closet.skeleton[u"portal"][
            self.character.name][self.name]

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        bone = self.get_loc_bone()
        return "{2}({0}->{1})".format(
            bone.origin, bone.destination, self.name)

    def get_bone(self, observer=None):
        if observer is None:
            return self.character.get_portal_bone(self.name)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_portal_bone(self.name)

    def get_loc_bone(self, observer=None, branch=None, tick=None):
        if observer is None:
            return self.character.get_portal_loc_bone(
                self.name, branch, tick)
        else:
            facade = self.character.get_facade(observer)
            return facade.get_portal_loc_bone(
                self.name, branch, tick)

    def get_origin(self, observer=None, branch=None, tick=None):
        return self._get_origdest('origin', observer, branch, tick)

    def get_destination(self, observer=None, branch=None, tick=None):
        return self._get_origdest('destination', observer, branch, tick)

    def _get_origdest(self, bone_att, observer, branch, tick):
        bone = self.get_loc_bone(observer, branch, tick)
        try:
            placebone = self.host.get_place_bone(getattr(bone, bone_att),
                                                 branch, tick)
            return self.host.get_place(placebone.place)
        except KeyError:
            skel = self.host.closet.skeleton[u"thing"]
            for thingbone in skel.iterbones():
                if (
                        thingbone.name == getattr(bone, bone_att) and
                        thingbone.host == unicode(self.host)):
                    char = self.host.closet.get_character(thingbone.character)
                    return char.get_thing(thingbone.name)
            raise KeyError("Noplace!")

    def new_branch(self, parent, branch, tick):
        skel = self.character.closet.skeleton[u"portal_loc"][
            unicode(self.character)][unicode(self)][parent]
        for bone in upbranch(
                self.character.closet, skel.iterbones(), branch, tick):
            yield bone

    def iter_stat_keys(self, observer=None, branch=None, tick=None):
        (branch, tick) = self.character.sanetime(branch, tick)
        if observer is None:
            for key in self.character.iter_portal_stat_keys(
                    self.name, [branch], [tick]):
                yield key
        else:
            facade = self.character.get_facade(observer)
            for key in facade.iter_portal_stat_keys(
                    self.name, [branch], [tick]):
                yield key
