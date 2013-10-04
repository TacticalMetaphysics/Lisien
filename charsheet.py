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
ROWS_SHOWN = 50
SCROLL_FACTOR = 4
TOP_TICK = 0
LEFT_BRANCH = 0
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
        "rowheight": lambda self: self.style.fontsize + self.style.spacing,
        "_rowdict": lambda self:
        self.closet.skeleton["charsheet_item"][
            str(self.window)][str(self.character)][int(self)],
        "height": lambda self:
        self._rowdict["height"],
        "keys": lambda self:
        (self._rowdict["key0"], self._rowdict["key1"], self._rowdict["key2"])}

    def __init__(self, charsheet, idx):
        self.charsheet = charsheet
        self.idx = idx

    def __int__(self):
        return self.idx

    def __len__(self):
        return len(self.skel)

    def __eq__(self, other):
        return (
            self.charsheet is other.charsheet and
            int(self) == int(other))

    def __ne__(self, other):
        return (
            self.charsheet is not other.charsheet or
            int(self) != int(other))

    def __getattr__(self, attrn):
        if attrn in CharSheetItem.charsheet_atts:
            return getattr(self.charsheet, attrn)
        else:
            try:
                return CharSheetItem.atrdic[attrn](self)
            except KeyError:
                raise AttributeError(
                    "CharSheetItem does not have and cannot "
                    "compute attribute {}".format(attrn))


class CharSheetTable(CharSheetItem):
    atrdic = {
        "skeleton": lambda self:
        self.getskel()}

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            try:
                return CharSheetTable.atrdic[attrn](self)
            except KeyError:
                return CharSheetItem.__getattr__(self, attrn)

    def getskel(self):
        if self.keys[0] is None:
            return self.character_skel
        elif self.keys[1] is None:
            return self.character_skel[self.keys[0]]
        elif self.keys[2] is None:
            return self.character_skel[self.keys[0]][self.keys[1]]
        else:
            return self.character_skel[
                self.keys[0]][self.keys[1]][self.keys[2]]

    def iter_skeleton(self, branch, tick):
        for rd in self.skeleton.iterrows():
            if (
                    rd["branch"] == branch and
                    rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick)):
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
        celled = []
        for cell in row:
            celled.append(cell)
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

    def draw(self, batch, group, branch=None, tick=None):
        t = self.window_top - self.rowheight
        l = self.window_left
        for label in self.row_labels(
                self.colkeys, l, t, batch, group):
            yield label
        for row in self.iterrows(branch, tick):
            if t < self.window_bot + self.rowheight:
                return
            t -= self.rowheight
            for label in self.row_labels(
                    row, l, t, batch, group):
                yield label

    def hover(self, x, y):
        if (
                x > self.window_right and
                x < self.window_left and
                y > self.window_bot and
                y < self.window_top):
            return self


class CharSheetThingTable(CharSheetTable):
    atrdic = {
        "character_skel": lambda self:
        self.character.thingdict}
    colkeys = ["dimension", "thing", "location"]

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for dimension in self.character.thingdict:
                for thing in self.character.thingdict[dimension]:
                    for rd in self.character.thingdict[
                            dimension][thing][branch].iterrows():
                        yield rd
        elif self.keys[1] is None:
            dimension = self.keys[0]
            for thing in self.character.thingdict[dimension]:
                for rd in self.character.thingdict[
                        dimension][thing][branch].iterrows():
                    yield rd
        else:
            dimension = self.keys[0]
            thing = self.keys[1]
            for rd in self.character.thingdict[
                    dimension][thing][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch, tick):
        covered = set()
        for rd in self.get_branch_rd_iter(branch):
            if (rd["dimension"], rd["thing"]) in covered:
                continue
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                thing = self.closet.get_thing(
                    rd["dimension"], rd["thing"])
                rd2 = thing.locations[branch]
                prev = None
                r = None
                for (tick_from, rd3) in rd2.iteritems():
                    if tick_from > tick:
                        if prev is not None:
                            r = {
                                "dimension": rd["dimension"],
                                "thing": rd["thing"],
                                "location": prev["location"]}
                        break
                    prev = rd3
                if r is None:
                    if prev is None:
                        r = {
                            "dimension": rd["dimension"],
                            "thing": rd["thing"],
                            "location": "nowhere"}
                    else:
                        r = {
                            "dimension": rd["dimension"],
                            "thing": rd["thing"],
                            "location": prev["location"]}
                covered.add((rd["dimension"], rd["thing"]))
                yield r


class CharSheetPlaceTable(CharSheetTable):
    atrdic = {
        "character_skel": lambda self:
        self.character.placedict}
    colkeys = ["dimension", "place"]


class CharSheetPortalTable(CharSheetTable):
    atrdic = {
        "character_skel": lambda self:
        self.character.portaldict}
    colkeys = ["dimension", "origin", "destination"]


class CharSheetStatTable(CharSheetTable):
    atrdic = {
        "character_skel": lambda self:
        self.character.statdict}
    colkeys = ["stat", "value"]

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for stat in self.character.statdict:
                for rd in self.character.statdict[stat][branch].iterrows():
                    yield rd
        else:
            stat = self.keys[0]
            for rd in self.character.statdict[stat][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch, tick):
        covered = set()
        prev = None
        for rd in self.get_branch_rd_iter(branch):
            if rd["stat"] in covered:
                continue
            elif rd["tick_from"] == tick:
                covered.add(rd["stat"])
                prev = None
                yield rd
            elif rd["tick_from"] > tick:
                covered.add(rd["stat"])
                r = prev
                prev = None
                yield r
            prev = rd


class CharSheetSkillTable(CharSheetTable):
    atrdic = {
        "character_skel": lambda self:
        self.character.skilldict}
    colkeys = ["skill", "deck"]

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for skill in self.character.skilldict:
                for rd in self.character.skilldict[skill][branch].iterrows():
                    yield rd
        else:
            skill = self.keys[0]
            for rd in self.character.skilldict[skill][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch, tick):
        covered = set()
        prev = None
        for rd in self.get_branch_rd_iter(branch):
            if rd["skill"] in covered:
                continue
            elif rd["tick_from"] == tick:
                covered.add(rd["skill"])
                prev = None
                yield rd
            elif rd["tick_from"] > tick:
                covered.add(rd["skill"])
                r = prev
                prev = None
                yield r
            prev = rd


class CharSheetCalendar(Calendar, CharSheetItem):
    def __init__(self, charsheet, idx, cal_typ):
        self.charsheet = charsheet
        self.idx = idx
        Calendar.__init__(
            self, charsheet, ROWS_SHOWN, MAX_COLS,
            LEFT_BRANCH, SCROLL_FACTOR, self.charsheet.style,
            cal_typ, *self.keys)

    def __getattr__(self, attrn):
        if attrn in Calendar.atrdic:
            return Calendar.atrdic[attrn](self)
        else:
            return CharSheetItem.__getattr__(self, attrn)


class CharSheetThingCalendar(CharSheetCalendar):
    atrdic = {
        "dimension": lambda self:
        self.charsheet.closet.get_dimension(self.keys[0]),
        "thing": lambda self:
        self.charsheet.closet.get_thing(
            self.keys[0], self.keys[1]),
        "character_skel": lambda self:
        self.charsheet.thingdict[self.keys[0]][self.keys[1]],
        "col_width": lambda self:
        self.width / MAX_COLS}

    def __init__(self, charsheet, idx):
        super(CharSheetThingCalendar, self).__init__(
            charsheet, idx, CAL_TYPE["THING"])

    def __eq__(self, other):
        return (
            self.charsheet is other.charsheet and
            self.thing == other.thing and
            self.dimension == other.dimension)

    def __getattr__(self, attrn):
        if attrn in CharSheetThingCalendar.atrdic:
            return CharSheetThingCalendar.atrdic[attrn](self)
        else:
            return CharSheetCalendar.__getattr__(self, attrn)


class CharSheetPlaceCalendar(CharSheetCalendar):
    def __init__(self, charsheet, idx):
        super(CharSheetPlaceCalendar, self).__init__(
            charsheet, idx, CAL_TYPE["PLACE"])


class CharSheetPortalCalendar(CharSheetCalendar):
    def __init__(self, charsheet, idx):
        super(CharSheetPortalCalendar, self).__init__(
            charsheet, idx, CAL_TYPE["PORTAL"])


class CharSheetStatCalendar(CharSheetCalendar):
    def __init__(self, charsheet, idx):
        super(CharSheetStatCalendar, self).__init__(
            charsheet, idx, CAL_TYPE["STAT"])


class CharSheetSkillCalendar(CharSheetCalendar):
    def __init__(self, charsheet, idx):
        super(CharSheetSkillCalendar, self).__init__(
            charsheet, idx, CAL_TYPE["SKILL"])


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
             "key0": "TEXT",
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
        "group": lambda self: self.window.charsheet_group,
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
        s.__setattr__("top_ticks", {})
        s.__setattr__("offxs", {})
        s.__setattr__("offys", {})

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

    def __iter__(self):
        return self.values()

    def get_for_cal(self, cal, d, default):
        (k0, k1, k2) = cal.keys
        if k0 not in d:
            d[k0] = {}
        if k1 not in d[k0]:
            d[k0][k1] = {}
        if k2 not in d[k0][k1]:
            d[k0][k1][k2] = default
        return d[k0][k1][k2]

    def set_for_cal(self, cal, d, val):
        (k0, k1, k2) = cal.keys
        if k0 not in d:
            d[k0] = {}
        if k1 not in d[k0]:
            d[k0][k1] = {}
        d[k0][k1][k2] = val

    def get_cal_top_tick(self, cal):
        return self.get_for_cal(cal, self.top_ticks, TOP_TICK)

    def set_cal_top_tick(self, cal, tick):
        self.set_for_cal(cal, self.top_ticks, tick)

    def get_cal_offx(self, cal):
        return self.get_for_cal(cal, self.offxs, 0)

    def set_cal_offx(self, cal, offx):
        self.set_for_cal(cal, self.offxs, offx)

    def get_cal_offy(self, cal):
        return self.get_for_cal(cal, self.offys, 0)

    def set_cal_offy(self, cal, offy):
        self.set_for_cal(cal, self.offys, offy)

    def items(self):
        skel = self.window.closet.skeleton[
            "charsheet_item"][str(self.window)][str(self.character)]
        i = 0
        for rd in skel.itervalues():
            yield (i, SHEET_ITEM_CLASS[rd["type"]](
                self, rd["idx"]))
            i += 1

    def values(self):
        skel = self.window.closet.skeleton[
            "charsheet_item"][str(self.window)][str(self.character)]
        for rd in skel.itervalues():
            yield SHEET_ITEM_CLASS[rd["type"]](
                self, rd["idx"])

    def item_window_top(self, it):
        window_top = self.window_top
        for item in self.values():
            if item == it:
                return window_top
            else:
                window_top -= item.height
                window_top -= self.style.spacing

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

    def draw(self, batch, group):
        for item in self:
            for drawable in item.draw(batch, group):
                yield drawable

    def hover(self, x, y):
        for item in self:
            hovered = item.hover(x, y)
            if hovered is not None:
                return hovered

    def overlaps(self, x, y):
        return (
            self.window_left < x and
            x < self.window_right and
            self.window_bot < y and
            y < self.window_top)
