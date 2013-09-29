# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from logging import getLogger
from util import SaveableMetaclass
from calendar import Calendar, CAL_TYPE
from pyglet.text import Label
from pyglet.gl import GL_TRIANGLES


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


MAX_COLS = 3
ROWS_SHOWN = 240
SCROLL_FACTOR = 4


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

    def __len__(self):
        return len(self.skel)

    def __getattr__(self, attrn):
        if attrn in CharSheetItem.charsheet_atts:
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


class CharSheetPlaceTable(CharSheetTable):
    atrdic = {
        "skeleton": lambda self: self.character.placedict}
    colkeys = ["dimension", "place"]


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


class CharSheetCalendar(Calendar, CharSheetItem):
    atrdic = {
        "width": lambda self: self.charsheet.width,
        "row_height": lambda self: self.height / self.rows_shown}

    def __init__(self, charsheet, height, typ, *keys):
        self.charsheet = charsheet
        self.height = height
        Calendar.__init__(
            self, charsheet, ROWS_SHOWN, MAX_COLS,
            None, None, SCROLL_FACTOR, self.charsheet.style,
            typ, *keys)

    def __getattr__(self, attrn):
        if attrn in self.atrdic:
            return self.atrdic[attrn](self)
        elif attrn in CharSheetCalendar.atrdic:
            return CharSheetCalendar.atrdic[attrn](self)
        elif attrn in Calendar.atrdic:
            return Calendar.atrdic[attrn](self)
        else:
            return CharSheetItem.__getattr__(self, attrn)


class CharSheetThingCalendar(CharSheetCalendar):
    atrdic = {
        "dimension": lambda self:
        self.charsheet.closet.get_dimension(self._dimension),
        "thing": lambda self:
        self.charsheet.closet.get_thing(self._dimension, self._thing),
        "col_width": lambda self:
        self.width / MAX_COLS}

    def __init__(self, charsheet, height, *keys):
        self._dimension = keys[0]
        self._thing = keys[1]
        super(CharSheetThingCalendar, self).__init__(
            charsheet, height, CAL_TYPE["THING"], *keys)

    def __eq__(self, other):
        return (
            self.charsheet is other.charsheet and
            self._thing == other._thing and
            self._dimension == other._dimension)

    def __getattr__(self, attrn):
        if attrn in self.atrdic:
            return self.atrdic[attrn](self)
        elif attrn in CharSheetCalendar.atrdic:
            return CharSheetCalendar.atrdic[attrn](self)
        elif attrn in Calendar.atrdic:
            return Calendar.atrdic[attrn](self)
        else:
            return CharSheetItem.__getattr__(self, attrn)


class CharSheetPlaceCalendar(CharSheetCalendar):
    def __init__(self, charsheet, height, *keys):
        super(CharSheetPlaceCalendar, self).__init__(
            charsheet, height, CAL_TYPE["PLACE"], *keys)


class CharSheetPortalCalendar(CharSheetCalendar):
    def __init__(self, charsheet, height, *keys):
        super(CharSheetPortalCalendar, self).__init__(
            charsheet, height, CAL_TYPE["PORTAL"], *keys)


class CharSheetStatCalendar(CharSheetCalendar):
    def __init__(self, charsheet, height, *keys):
        super(CharSheetStatCalendar, self).__init__(
            charsheet, height, CAL_TYPE["STAT"], *keys)


class CharSheetSkillCalendar(CharSheetCalendar):
    def __init__(self, charsheet, height, *keys):
        super(CharSheetSkillCalendar, self).__init__(
            charsheet, height, CAL_TYPE["SKILL"], *keys)


SHEET_ITEM_TYPE = {
    "THINGTAB": 0,
    "PLACETAB": 1,
    "PORTALTAB": 2,
    "STATTAB": 3,
    "SKILLTAB": 4,
    "THINGCAL": 5,
    "PLACECAL": 6,
    "PORTALCAL": 7,
    "STATCAL": 8,
    "SKILLCAL": 9}


SHEET_ITEM_CLASS = {
    SHEET_ITEM_TYPE["THINGTAB"]: CharSheetThingTable,
    SHEET_ITEM_TYPE["PLACETAB"]: CharSheetPlaceTable,
    SHEET_ITEM_TYPE["PORTALTAB"]: CharSheetPortalTable,
    SHEET_ITEM_TYPE["STATTAB"]: CharSheetStatTable,
    SHEET_ITEM_TYPE["SKILLTAB"]: CharSheetSkillTable,
    SHEET_ITEM_TYPE["THINGCAL"]: CharSheetThingCalendar,
    SHEET_ITEM_TYPE["PLACECAL"]: CharSheetPlaceCalendar,
    SHEET_ITEM_TYPE["PORTALCAL"]: CharSheetPortalCalendar,
    SHEET_ITEM_TYPE["STATCAL"]: CharSheetStatCalendar,
    SHEET_ITEM_TYPE["SKILLCAL"]: CharSheetSkillCalendar}


class CharSheet(object):
    __metaclass__ = SaveableMetaclass
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
            []),
        (
            "charsheet_item",
            {"window": "TEXT NOT NULL DEFAULT 'Main'",
             "character": "TEXT NOT NULL",
             "idx": "INTEGER NOT NULL",
             "type": "INTEGER NOT NULL",
             "key0": "TEXT NOT NULL",
             "key1": "TEXT",
             "key2": "TEXT",
             "height": "INTEGER"},
            ("window", "character", "idx"),
            {"window, character": ("charsheet", "window, character")},
            ["CASE key1 WHEN NULL THEN type NOT IN ({0}) END".format(
                ", ".join([str(SHEET_ITEM_TYPE[typ]) for typ in (
                    "THINGTAB", "THINGCAL",
                    "PLACETAB", "PLACECAL",
                    "PORTALTAB", "PORTALCAL")])),
             "CASE key2 WHEN NULL THEN type<>{0} END".format(
                 str(SHEET_ITEM_TYPE["PORTALTAB"])),
             "CASE height WHEN NULL THEN type NOT IN ({0}) END".format(
                 ", ".join([str(SHEET_ITEM_TYPE[typ]) for typ in (
                     "THINGCAL", "PLACECAL", "PORTALCAL",
                     "STATCAL", "SKILLCAL")])),
             "idx>=0",
             "idx<={}".format(max(SHEET_ITEM_TYPE.viewvalues()))])
    ]

    rdfields = set(["visible", "interactive",
                    "left", "right", "bot", "top"])

    atrdic = {
        "closet": lambda self: self.window.closet,
        "batch": lambda self: self.window.batch,
        "group": lambda self: self.window.char_sheet_group,
        "window_left": lambda self: int(self.left * self.window.width),
        "window_bot": lambda self: int(self.bot * self.window.height),
        "window_top": lambda self: int(self.top * self.window.height),
        "window_right": lambda self: int(self.right * self.window.width),
        "width": lambda self: self.window_right - self.window_left,
        "_rowdict": lambda self: self.closet.skeleton[
            "charsheet"][self._window][self._character],
        "window": lambda self: self.closet.get_window(self._window),
        "character": lambda self: self.closet.get_character(self._character),
        "style": lambda self: self.closet.get_style(self._rowdict["style"])}

    def __init__(self, closet, window, character):
        s = super(CharSheet, self)
        s.__setattr__("closet", closet)
        s.__setattr__("_window", window)
        s.__setattr__("_character", character)
        s.__setattr__("lastdraw", [])

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

    def items(self):
        skel = self.window.closet.skeleton[
            "charsheet_item"][str(self.window)][str(self.character)]
        for rd in skel.itervalues():
            yield SHEET_ITEM_CLASS[rd["type"]](
                self, rd["height"], rd["key0"], rd["key1"], rd["key2"])

    def item_window_top(self, it):
        window_top = self.window_top
        for item in self.items():
            if item != it:
                window_top -= item.height
                window_top -= self.style.spacing
            else:
                return window_top

    def get_box(self, batch, group):
        l = self.window_left
        r = self.window_right
        b = self.window_bot
        t = self.window_top
        if (
                r < 0 or
                t < 0 or
                l > self.window.width or
                b > self.window.height):
            return
        return batch.add(
            6,
            GL_TRIANGLES,
            group,
            ('v2i', (l, b, l, t, r, t, l, b, r, b, r, t)),
            ('c4B', self.style.bg_inactive.tup * 6))

    def draw(self):
        drawn = []
        drawn.append(
            self.get_box(self.window.batch, self.window.calendar_group))
        drawn.extend([item.draw()
                      for item in self.items()])
        for done in self.lastdraw:
            try:
                done.delete()
            except AttributeError:
                pass
        self.lastdraw = drawn
