# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container


class Portal(Container):
    tables = [
        ("portal", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "host": "text not null default 'Physical'"},
            "primary_key": (
                "character", "name")}),
        ("portal_loc", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "origin": "text",
                "destination": "text"},
            "primary_key": (
                "character", "name", "branch", "tick"),
            "foreign_keys": {
                "character, name": (
                    "portal", "character, name")}}),
        ("portal_stat", {
            "columns": {
                "character": "text not null default 'Physical'",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "name", "key", "branch", "tick"),
            "foreign_keys": {
                "character, name": (
                    "portal", "character, name")}}),
        ("portal_loc_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "origin": "text",
                "destination": "text"},
            "primary_key": (
                "observer", "observed", "name", "branch", "tick"),
            "foreign_keys": {
                "observed, name": (
                    "portal", "character, name")}}),
        ("portal_stat_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "observer", "observed", "name",
                "key", "branch", "tick"),
            "foreign_keys": {
                "observed, name": (
                    "portal", "character, name")}})]

    @property
    def bone(self):
        return self.get_bone()

    @property
    def loc_bone(self):
        return self.get_loc_bone()

    @property
    def origin(self):
        return self.get_origin()

    @property
    def destination(self):
        return self.get_destination()

    def __init__(self, character, name):
        self.character = character
        self.name = name

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        bone = self.loc_bone
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
        bone = self.get_loc_bone(observer, branch, tick)
        try:
            return self.character.get_place(bone.origin)
        except KeyError:
            return self.character.get_thing(bone.origin)

    def get_destination(self, observer=None, branch=None, tick=None):
        bone = self.get_loc_bone(observer, branch, tick)
        try:
            return self.character.get_place(bone.destination)
        except KeyError:
            return self.character.get_thing(bone.destination)
