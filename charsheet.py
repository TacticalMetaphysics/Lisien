# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from logging import getLogger
from util import SaveableMetaclass
from pyglet.text import Label
from pyglet.gl import GL_TRIANGLES, GL_LINES


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


def generate_items(skel, keykey, valkey):
    for rd in skel.iterrows():
        yield (rd[keykey], rd[valkey])


class CharSheetItem(object):
    charsheet_atts = set([
        "closet", "batch", "window_left", "window_right",
        "width", "window", "character", "visible",
        "interactive", "left", "right", "style",
        "character"])

    atrdic = {
        "window_top": lambda self: self.charsheet.item_window_top(self),
        "window_bot": lambda self: self.window_top - self.height,
        "height": lambda self: len(self) * self.rowheight}

    def __init__(self, charsheet):
        self.charsheet = charsheet

    def __len__(self):
        return len(self.skel)

    def __getattr__(self, attrn):
        if attrn in self.charsheet_atts:
            return getattr(self.charsheet, attrn)
        elif hasattr(self, 'atrdic') and attrn in self.atrdic:
            return self.atrdic[attrn](self)
        elif attrn in CharSheetItem.atrdic:
            return CharSheetItem.atrdic[attrn](self)
        raise AttributeError(
            "{1} instance does not have and "
            "cannot compute attribute {0}.".format(
                attrn, self.__class__.__name__))


class CharSheetTable(CharSheetItem):
    def __init__(self, charsheet):
        self.charsheet = charsheet

    def iter_skeleton(self, branch, tick):
        for rd in self.skeleton[branch].iterrows():
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                yield rd

    def iterrows(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        for rd in self.iter_skeleton(branch, tick):
            yield [rd[key] for key in self.colkeys]

    def row_labels(self, row, l, t, batch, group):
        step = self.width / len(row)
        for cell in row:
            yield Label(
                cell,
                self.style.fontface,
                self.style.fontsize,
                color=self.style.textcolor.tup,
                x=l,
                y=t,
                anchor_x='left',
                anchor_y='top',
                batch=batch,
                group=group)
            l += step

    def iterlabels(self, batch, group, branch=None, tick=None):
        t = self.window_top
        l = self.window_left
        for label in self.row_labels(self.colkeys, t, batch, group):
            yield label
        t -= self.rowheight
        for row in self.iterrows(branch, tick):
            for label in self.row_labels(row, l, t, batch, group):
                yield label
            t -= self.rowheight


class CharSheetThingTable(CharSheetTable):
    colkeys = ["dimension", "thing", "location"]

    def iter_skeleton(self, branch, tick):
        for rd in self.character.thingdict[branch]:
            if rd["tick_from"] >= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                yield {
                    "dimension": rd["dimension"],
                    "thing": rd["thing"],
                    "location": str(
                        self.closet.get_thing(
                            rd["dimension"],
                            rd["thing"]).location)}


class CharSheetPortalTable(CharSheetTable):
    atrdic = {
        "skeleton": lambda self: self.character.portaldict}
    colkeys = ["dimension", "origin", "destination"]


class CharSheetStatTable(CharSheetTable):
    atrdic = {
        "skeleton": lambda self: self.character.statdict}
    colkeys = ["stat", "value"]


class CharSheetSkillTable(CharSheetTable):
    atrdic = {
        "skeleton": lambda self: self.character.skilldict}
    colkeys = ["skill", "deck"]


class CharSheet(object):
    demands = ["character"]

    tables = [
        (
            "charsheet",
            {"window": "TEXT NOT NULL DEFAULT 'Main'",
             "character": "TEXT NOT NULL",
             "visible": "BOOLEAN NOT NULL DEFAULT 0",
             "interactive": "BOOLEAN NOT NULL DEFAULT 1",
             "left": "FLOAT NOT NULL DEFAULT 0.8",
             "right": "FLOAT NOT NULL DEFAULT 1.0",
             "bot": "FLOAT NOT NULL DEFAULT 0.0",
             "top": "FLOAT NOT NULL DEFAULT 1.0",
             "style": "TEXT NOT NULL DEFAULT 'default_style'"},
            ("window", "character"),
            {"window": ("window", "name"),
             "character": ("character", "name"),
             "style": ("style", "name")},
            [])
    ]

    rdfields = set(["visible", "interactive",
                    "left", "right", "bot", "top"])

    atrdic = {
        "batch": lambda self: self.window.batch,
        "group": lambda self: self.window.char_sheet_group,
        "window_left": lambda self: int(self.left * self.window.width),
        "window_bot": lambda self: int(self.bot * self.window.height),
        "window_top": lambda self: int(self.top * self.window.height),
        "window_right": lambda self: int(self.right * self.window.width)}

    def __init__(self, window, character):
        s = super(CharSheet, self)
        s.__setattr__("window", window)
        s.__setattr__("closet", window.closet)
        s.__setattr__("character", self.closet.get_character(character))
        s.__setattr__("items", [])

    def __getattr__(self, attrn):
        if attrn in self.rdfields:
            return self._rowdict[attrn]
        try:
            return CharSheet.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "CharSheet instance does not have and "
                "cannot compute attribute {0}".format(
                    attrn))

    def __setattr__(self, attrn, val):
        if attrn in self.rdfields:
            self._rowdict[attrn] = val
        else:
            super(CharSheet, self).__setattr__(attrn, val)

    def item_window_top(self, it):
        if isinstance(it, int):
            it = self.items[it]
        window_top = self.window_top
        for item in self.items:
            if item is not it:
                window_top -= item.height
                window_top -= self.style.spacing
            else:
                return window_top

    def get_box(self, batch, group):
        return get_box(
            self.window_left,
            self.window_right,
            self.window_bot,
            self.window_top,
            GL_TRIANGLES,
            self.style.bg_inactive.tup,
            batch,
            group)
