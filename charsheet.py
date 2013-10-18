# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from logging import getLogger
from util import SaveableWidgetMetaclass
from calendar import Calendar, CAL_TYPE
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty,
    ListProperty)


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
SHEET_TO_CAL_TYPE = dict(
    [(SHEET_ITEM_TYPE[a], CAL_TYPE[a[:-3]]) for a in
     ("THINGCAL", "PLACECAL", "PORTALCAL", "STATCAL", "SKILLCAL")])


def generate_items(skel, keykey, valkey):
    for rd in skel.iterrows():
        yield (rd[keykey], rd[valkey])


class CharSheetTable(GridLayout):
    charsheet = ObjectProperty()
    keys = ListProperty()

    @property
    def skel(self):
        if self.keys[0] is None:
            return self.character_skel
        elif self.keys[1] is None:
            return self.character_skel[self.keys[0]]
        elif self.keys[2] is None:
            return self.character_skel[self.keys[0]][self.keys[1]]
        else:
            return self.character_skel[
                self.keys[0]][self.keys[1]][self.keys[2]]

    def __init__(self, **kwargs):
        GridLayout.__init__(self, **kwargs)
        self.cols = len(self.colkeys)
        self.row_default_height = (self.charsheet.style.fontsize
                                   + self.charsheet.style.spacing)
        self.row_force_default = True

        for key in self.colkeys:
            self.add_widget(Label(
                text=key,
                font_name=self.charsheet.style.fontface + '.ttf',
                font_size=self.charsheet.style.fontsize,
                color=self.charsheet.style.textcolor.rgba))
        for rd in self.iter_skeleton():
            for key in self.colkeys:
                self.add_widget(Label(
                    text=rd[key],
                    font_name=self.charsheet.style.fontface + '.ttf',
                    font_size=self.charsheet.style.fontsize,
                    color=self.charsheet.style.textcolor.rgba))

    def iter_skeleton(self, branch=None, tick=None):
        if branch is None:
            branch = self.charsheet.character.closet.branch
        if tick is None:
            tick = self.charsheet.character.closet.tick
        for rd in self.character_skel.iterrows():
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


class CharSheetThingTable(CharSheetTable):
    colkeys = ["dimension", "thing", "location"]

    @property
    def character_skel(self):
        return self.charsheet.character.thingdict

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for dimension in self.charsheet.character.thingdict:
                for thing in self.charsheet.character.thingdict[dimension]:
                    for rd in self.charsheet.character.thingdict[
                            dimension][thing][branch].iterrows():
                        yield rd
        elif self.keys[1] is None:
            dimension = self.keys[0]
            for thing in self.charsheet.character.thingdict[dimension]:
                for rd in self.charsheet.character.thingdict[
                        dimension][thing][branch].iterrows():
                    yield rd
        else:
            dimension = self.keys[0]
            thing = self.keys[1]
            for rd in self.charsheet.character.thingdict[
                    dimension][thing][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        if branch is None:
            branch = self.charsheet.character.closet.branch
        if tick is None:
            tick = self.charsheet.character.closet.tick
        covered = set()
        for rd in self.get_branch_rd_iter(branch):
            if (rd["dimension"], rd["thing"]) in covered:
                continue
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                thing = self.charsheet.character.closet.get_thing(
                    rd["dimension"], rd["thing"])
                rd2 = thing.locations[branch]
                prev = None
                r = None
                for (tick_from, rd3) in rd2.items():
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
    colkeys = ["dimension", "place"]

    @property
    def character_skel(self):
        return self.charsheet.character.placedict


class CharSheetPortalTable(CharSheetTable):
    colkeys = ["dimension", "origin", "destination"]

    @property
    def character_skel(self):
        return self.charsheet.character.portaldict


class CharSheetStatTable(CharSheetTable):
    colkeys = ["stat", "value"]

    @property
    def character_skel(self):
        return self.charsheet.character.statdict

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for stat in self.charsheet.character.statdict:
                for rd in self.charsheet.character.statdict[
                        stat][branch].iterrows():
                    yield rd
        else:
            stat = self.keys[0]
            for rd in self.charsheet.character.statdict[
                    stat][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        if branch is None:
            branch = self.charsheet.character.closet.branch
        if tick is None:
            tick = self.charsheet.character.closet.tick
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
    colkeys = ["skill", "deck"]

    @property
    def character_skel(self):
        return self.charsheet.character.skilldict

    def get_branch_rd_iter(self, branch):
        if self.keys[0] is None:
            for skill in self.charsheet.character.skilldict:
                for rd in self.charsheet.character.skilldict[
                        skill][branch].iterrows():
                    yield rd
        else:
            skill = self.keys[0]
            for rd in self.charsheet.character.skilldict[
                    skill][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        if branch is None:
            branch = self.charsheet.character.closet.branch
        if tick is None:
            tick = self.charsheet.character.closet.tick
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


class CharSheetCalendar(Calendar):
    charsheet = ObjectProperty()
    keys = ListProperty()

    def __init__(self, **kwargs):
        Calendar.__init__(
            self, rows_shown=ROWS_SHOWN, max_cols=MAX_COLS,
            left_branch=LEFT_BRANCH, top_tick=TOP_TICK,
            scroll_factor=SCROLL_FACTOR, **kwargs)


class CharSheetBg(Widget):
    pos = AliasProperty(
        lambda self: self.charsheet.pos,
        lambda self, v: None)

    size = AliasProperty(
        lambda self: self.charsheet.size,
        lambda self, v: None)

    def __init__(self, cs):
        self.charsheet = cs
        Widget.__init__(self)
        self.canvas.add(Rectangle(
            pos_hint=self.charsheet.pos_hint,
            size_hint=self.charsheet.size_hint))


class CharSheet(BoxLayout):
    __metaclass__ = SaveableWidgetMetaclass
    demands = ["character"]

    tables = [
        (
            "charsheet",
            {"character": "TEXT NOT NULL",
             "visible": "BOOLEAN NOT NULL DEFAULT 0",
             "interactive": "BOOLEAN NOT NULL DEFAULT 1",
             "x_hint": "FLOAT NOT NULL DEFAULT 0.8",
             "y_hint": "FLOAT NOT NULL DEFAULT 0.0",
             "w_hint": "FLOAT NOT NULL DEFAULT 0.2",
             "h_hint": "FLOAT NOT NULL DEFAULT 1.0",
             "style": "TEXT NOT NULL DEFAULT 'default_style'"},
            ("character",),
            {"character": ("character", "name"),
             "style": ("style", "name")},
            []),
        (
            "charsheet_item",
            {"character": "TEXT NOT NULL",
             "idx": "INTEGER NOT NULL",
             "type": "INTEGER NOT NULL",
             "key0": "TEXT",
             "key1": "TEXT",
             "key2": "TEXT",
             "height": "INTEGER"},
            ("character", "idx"),
            {"character": ("charsheet", "character")},
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
             "idx<={}".format(max(SHEET_ITEM_TYPE.values()))])
    ]
    character = ObjectProperty()
    rowdict = DictProperty()
    style = AliasProperty(
        lambda self: self.character.closet.get_style(
            self.rowdict["style"]),
        lambda self, v: None,
        bind=('rowdict',))

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self,
            orientation='vertical',
            pos_hint={'x': 0.8,
                      'y': 0.0},
            size_hint=(0.2, 1.0),
            spacing=10,
            **kwargs)
        rd = self.character.closet.skeleton[
            "charsheet"][unicode(self.character)]

        def upd_rd(*args):
            self.rowdict = dict(rd)

        upd_rd()
        rd.bind(touches=upd_rd)

        with self.canvas.before:
            Color(*self.style.bg_inactive.rgba)
            self.rect = Rectangle(pos=self.pos, size=self.size)

        def _update_rect(*args):
            self.rect.pos = self.pos
            self.rect.size = self.size
        self.bind(size=_update_rect, pos=_update_rect)
        for widget in self:
            self.add_widget(widget)

    def __iter__(self):
        for rd in self.character.closet.skeleton[
                "charsheet_item"][
                unicode(self.character)].iterrows():
            tabs = {
                SHEET_ITEM_TYPE["THINGTAB"]: CharSheetThingTable,
                SHEET_ITEM_TYPE["PLACETAB"]: CharSheetPlaceTable,
                SHEET_ITEM_TYPE["PORTALTAB"]: CharSheetPortalTable,
                SHEET_ITEM_TYPE["STATTAB"]: CharSheetStatTable,
                SHEET_ITEM_TYPE["SKILLTAB"]: CharSheetSkillTable
            }
            keylst = [rd["key0"], rd["key1"], rd["key2"]]
            if rd["type"] in tabs:
                yield tabs[rd["type"]](charsheet=self, keys=keylst)
            else:
                yield CharSheetCalendar(charsheet=self, keys=keylst,
                                        typ=SHEET_TO_CAL_TYPE[rd["type"]])
