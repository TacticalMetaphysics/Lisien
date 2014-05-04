# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from place import Place, Container
from LiSE.util import upbranch


class Portal(Container):
    tables = [
        ("portal_loc", {
            "columns": {
                "character": "text not null",
                "name": "text not null",
                "branch": "integer not null",
                "tick": "integer not null",
                "origin": "text",
                "destination": "text"},
            "primary_key": (
                "character", "name", "branch", "tick")}),
        ("portal_stat", {
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
                "character, name": ("portal_loc", "character, name")}})]

    def __init__(self, character, name):
        self.character = character
        self.name = name

    def __eq__(self, other):
        return (
            isinstance(other, Portal) and
            self.character == other.character and
            self.name == other.name)

    def __hash__(self):
        return hash((self.character, self.name))

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        bone = self.get_loc_bone()
        return "{2}({0}->{1})".format(
            bone.origin, bone.destination, self.name)

    def __getitem__(self, key):
        bone = self.loc_bone
        return self.character.graph[bone.origin][bone.destination][key]

    @property
    def contents(self):
        bone = self.loc_bone
        return self.character.graph[bone.origin][bone.destination]['contents']

    def _get_end(self, bprop):
        bone = self.loc_bone
        endn = getattr(bone, bprop)
        if endn not in self.character.place_d:
            self.character.place_d[endn] = Place(self.character, endn)
        return self.character.place_d[endn]

    @property
    def origin(self):
        return self._get_end('origin')

    @property
    def destination(self):
        return self._get_end('destination')

    @property
    def loc_bone(self):
        (branch, tick) = self.character.closet.time
        return self.character.closet.skeleton[u'portal_loc'][
            self.character.name][self.name][branch].value_during(tick)

    def new_branch(self, parent, branch, tick):
        skel = self.character.closet.skeleton[u"portal_loc"][
            unicode(self.character)][unicode(self)][parent]
        for bone in upbranch(
                self.character.closet, skel.iterbones(), branch, tick):
            yield bone
