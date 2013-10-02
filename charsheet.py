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
        "height": lambda self: len(self) * self.rowheight}

    def __init__(self, charsheet, height, *keys):
        self.charsheet = charsheet
        self.height = height
        self.keys = keys

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

    def __eq__(self, other):
        return (
            self.charsheet is other.charsheet and
            self.height == other.height and
            self.keys == other.keys)

    def __ne__(self, other):
        return (
            self.charsheet is not other.charsheet or
            self.height != other.height or
            self.keys != other.keys)


class CharSheetTable(CharSheetItem):
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
        "skeleton": lambda self:
        self.character.thingdict[
            self.keys[0]]}
    colkeys = ["dimension", "thing", "location"]

    def __repr__(self):
        return "CharSheetThingTable({}, {}, {})".format(
            str(self.character), self.keys[0], self.keys[1])

    def iter_skeleton(self, branch, tick):
        for thing in self.skeleton:
            for (tick_from, rd) in self.skeleton[thing][branch].iteritems():
                if tick_from <= tick and (
                        rd["tick_to"] is None or
                        rd["tick_to"] >= tick):
                    # The iterators in the Skeleton class ensure that this
                    # will proceed in chronological order.
                    rd2 = self.closet.skeleton["thing_location"][
                        rd["dimension"]][rd["thing"]][branch]
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
                        r = {
                            "dimension": rd["dimension"],
                            "thing": rd["thing"],
                            "location": prev["location"]}
                    yield r
                    break


class CharSheetPlaceTable(CharSheetTable):
    atrdic = {
        "skeleton": lambda self:
        self.character.placedict[
            self.keys[0]][self.keys[1]]}
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
    def __init__(self, charsheet, height, typ, *keys):
        self.charsheet = charsheet
        self.height = height
        Calendar.__init__(
            self, charsheet, ROWS_SHOWN, MAX_COLS,
            LEFT_BRANCH, SCROLL_FACTOR, self.charsheet.style,
            typ, *keys)

    def __getattr__(self, attrn):
        if attrn in Calendar.atrdic:
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
        if attrn in Calendar.atrdic:
            return Calendar.atrdic[attrn](self)
        else:
            return super(CharSheetThingCalendar, self).__getattr__(attrn)


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
            "charsheet_viewport",
            {"dimension": "TEXT NOT NULL DEFAULT 'Physical'",
             "board": "INTEGER NOT NULL DEFAULT 0",
             "window": "TEXT NOT NULL DEFAULT 'Main'",
             "character": "TEXT NOT NULL",
             "left": "FLOAT NOT NULL DEFAULT 0.8",
             "bot": "FLOAT NOT NULL DEFAULT 0.0",
             "top": "FLOAT NOT NULL DEFAULT 1.0",
             "right": "FLOAT NOT NULL DEFAULT 1.0",
             "visible": "BOOLEAN NOT NULL DEFAULT 1",
             "interactive": "BOOLEAN NOT NULL DEFAULT 1",
             "style": "TEXT NOT NULL DEFAULT 'default_style'"},
            ("dimension", "board", "window", "character"),
            {"window": ("window", "name"),
             "style": ("style", "name"),
             "dimension, board": ("board", "dimension, idx"),
             "character": ("character", "name")
             },
            []),
        (
            "charsheet_item",
            {"dimension": "TEXT NOT NULL DEFAULT 'Physical'",
             "board": "INTEGER NOT NULL DEFAULT 0",
             "character": "TEXT NOT NULL",
             "idx": "INTEGER NOT NULL",
             "type": "INTEGER NOT NULL",
             "key0": "TEXT NOT NULL",
             "key1": "TEXT",
             "key2": "TEXT",
             "height": "INTEGER"},
            ("dimension", "board", "character", "idx"),
            {"dimension, board": ("board", "dimension, idx"),
             "character": ("character", "name")},
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
            "charsheet_viewport"][
            self._dimension][
            self._board][
            self._window][
            self._character],
        "dimension": lambda self: self.closet.get_dimension(self._dimension),
        "board": lambda self:
        self.closet.get_board(self._dimension, self._board),
        "window": lambda self: self.closet.get_window(self._window),
        "character": lambda self: self.closet.get_character(self._character),
        "style": lambda self: self.closet.get_style(self._rowdict["style"])}

    def __init__(self, closet, dimension, board, window, character):
        s = super(CharSheet, self)
        s.__setattr__("closet", closet)
        s.__setattr__("_dimension", dimension)
        s.__setattr__("_board", board)
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
        return self.items()

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
