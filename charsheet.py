# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from logging import getLogger
from util import SaveableWidgetMetaclass
from calendar import Calendar, CAL_TYPE
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Rectangle
from kivy.properties import AliasProperty, DictProperty


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


class CharSheetTable(GridLayout):
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

    def __init__(self, cs, *keys):
        self.charsheet = cs
        self.keys = keys
        GridLayout.__init__(
            self, cols=len(self.colkeys),
            row_default_height=self.charsheet.style.fontsize+self.charsheet.style.spacing,
            row_force_default=True)
        for key in self.colkeys:
            self.add_widget(Label(
                text=key,
                font_name=self.charsheet.style.fontface + '.ttf',
                font_size=self.charsheet.style.fontsize,
                color=self.charsheet.style.textcolor.tup))
        for rd in self.iter_skeleton():
            for key in self.colkeys:
                self.add_widget(Label(
                    text=rd[key],
                    font_name=self.charsheet.style.fontface + '.ttf',
                    font_size=self.charsheet.style.fontsize,
                    color=self.charsheet.style.textcolor.tup))

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

    def row_labels(self, row, l, t, batch, group):
        step = self.width / len(row)
        celled = []
        for cell in row:
            celled.append(cell)
            yield Label(
                cell,
                font_name=self.style.fontface,
                font_size=self.style.fontsize,
                color=self.style.textcolor.tup,
                size_hint=(l, t),
                valign='top')
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
    def __init__(self, charsheet, typ, *keys):
        Calendar.__init__(
            self, charsheet, ROWS_SHOWN, MAX_COLS, LEFT_BRANCH,
            TOP_TICK, SCROLL_FACTOR, typ, *keys)


class CharSheetThingCalendar(CharSheetCalendar):
    @property
    def thing(self):
        for rd in self.charsheet.character.thingdict.iterrows():
            r = self.charsheet.character.closet.get_thing(
                rd["dimension"], rd["thing"])
            break
        return r

    def __init__(self, charsheet, *keys):
        CharSheetCalendar.__init__(
            self, charsheet, CAL_TYPE["THING"], *keys)


class CharSheetPlaceCalendar(CharSheetCalendar):
    def __init__(self, charsheet, *keys):
        CharSheetCalendar.__init__(
            self, charsheet, SHEET_ITEM_TYPE["PLACECAL"], *keys)


class CharSheetPortalCalendar(CharSheetCalendar):
    def __init__(self, charsheet, *keys):
        CharSheetCalendar.__init__(
            self, charsheet, SHEET_ITEM_TYPE["PORTALCAL"], *keys)


class CharSheetStatCalendar(CharSheetCalendar):
    def __init__(self, charsheet, *keys):
        CharSheetCalendar.__init__(
            self, charsheet, SHEET_ITEM_TYPE["STATCAL"], *keys)


class CharSheetSkillCalendar(CharSheetCalendar):
    def __init__(self, charsheet, *keys):
        CharSheetCalendar.__init__(
            self, charsheet, SHEET_ITEM_TYPE["SKILLCAL"], *keys)


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
    rowdict = DictProperty()
    style = AliasProperty(
        lambda self: self.character.closet.get_style(
            self.rowdict["style"]),
        lambda self, v: None,
        bind=('rowdict',))

    def __init__(self, character):
        self.character = character
        rd = self.character.closet.skeleton[
            "charsheet"][unicode(self.character)]

        def upd_rd(*args):
            self.rowdict = dict(rd)

        upd_rd()
        rd.bind(touches=upd_rd)

        BoxLayout.__init__(
            self,
            orientation='vertical',
            pos_hint={'x': 0.8,
                      'y': 0.0},
            size_hint=(0.2, 1.0),
            spacing=10)
#        self.add_widget(CharSheetBg(self))
        for widget in self:
            self.add_widget(widget)

    def __iter__(self):
        for rd in self.character.closet.skeleton[
                "charsheet_item"][
                unicode(self.character)].iterrows():
            yield {
                SHEET_ITEM_TYPE["THINGTAB"]: CharSheetThingTable,
                SHEET_ITEM_TYPE["PLACETAB"]: CharSheetPlaceTable,
                SHEET_ITEM_TYPE["PORTALTAB"]: CharSheetPortalTable,
                SHEET_ITEM_TYPE["STATTAB"]: CharSheetStatTable,
                SHEET_ITEM_TYPE["SKILLTAB"]: CharSheetSkillTable,
                SHEET_ITEM_TYPE["THINGCAL"]: CharSheetThingCalendar,
                SHEET_ITEM_TYPE["PLACECAL"]: CharSheetPlaceCalendar,
                SHEET_ITEM_TYPE["PORTALCAL"]: CharSheetPortalCalendar,
                SHEET_ITEM_TYPE["STATCAL"]: CharSheetStatCalendar,
                SHEET_ITEM_TYPE["SKILLCAL"]: CharSheetSkillCalendar
                }[rd["type"]](self, rd["key0"], rd["key1"], rd["key2"])
