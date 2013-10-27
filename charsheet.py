# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from logging import getLogger
from kivybits import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.graphics import (
    Color,
    Line,
    Triangle)
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty,
    StringProperty)
import calendar


logger = getLogger(__name__)


SCROLL_FACTOR = 4
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
CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}
SHEET_TO_CAL_TYPE = dict(
    [(SHEET_ITEM_TYPE[a], CAL_TYPE[a[:-3]]) for a in
     ("THINGCAL", "PLACECAL", "PORTALCAL", "STATCAL", "SKILLCAL")])


def get_charsheet(item):
    while not isinstance(item, CharSheet):
        item = item.parent
    return item


def get_calendar(item):
    while not isinstance(item, Calendar):
        item = item.parent
    return item


class CalendarColumn(calendar.Column):
    branch = NumericProperty()

    def on_parent(self, instance, value):
        super(CalendarColumn, self).on_parent(instance, value)
        charsheet = get_charsheet(self)
        value.skel[self.branch].listener = self._trigger_layout

    def do_layout(self, *args):
        self.update()
        super(CalendarColumn, self).do_layout(*args)


class ThingCalendarColumn(CalendarColumn):
    def update(self, *args):
        if not hasattr(self, 'cells'):
            self.cells = {}
        calendar = self.parent
        thing = calendar.referent
        if self.branch not in thing.locations:
            return
        rditer = thing.locations[self.branch].iterrows()
        try:
            prev = next(rditer)
        except StopIteration:
            return
        done_for = set()
        for rd in rditer:
            if id(rd) not in self.cells:
                cc = self.add_cell(
                    prev["location"], prev["tick_from"], rd["tick_from"])
                self.cells[id(rd)] = cc
            else:
                cc = self.cells[id(rd)]
                cc.text = prev['location']
                cc.tick_from = prev['tick_from']
                cc.tick_to = rd['tick_from']
            done_for.add(id(rd))
            prev = rd
        if None in self.cells:
            indefcc = self.cells[None]
            indefcc.text = prev["location"]
            indefcc.tick_from = prev["tick_from"]
        else:
            indefcc = self.add_cell(prev["location"], prev["tick_from"])
            self.cells[None] = indefcc
        for cell in self.children:
            assert(cell in self.cells.viewvalues())
            cell.calendared()
        undone = set(self.cells.viewkeys()) - done_for - set([None])
        for ccid in undone:
            self.remove_widget(self.cells[ccid])
            del self.cells[ccid]


class PlaceCalendarColumn(CalendarColumn):
    pass


class PortalCalendarColumn(CalendarColumn):
    pass


class StatCalendarColumn(CalendarColumn):
    pass


class SkillCalendarColumn(CalendarColumn):
    pass


class Calendar(calendar.Calendar):
    lookahead = NumericProperty(100)
    cal_type = NumericProperty(0)
    referent = ObjectProperty(None)
    key0 = StringProperty()
    key1 = StringProperty(None, allownone=True)
    key2 = StringProperty(None, allownone=True)
    keys = ReferenceListProperty(key0, key1, key2)
    columns = DictProperty({})
    cal_types = {
        0: ThingCalendarColumn,
        1: PlaceCalendarColumn,
        2: PortalCalendarColumn,
        3: StatCalendarColumn,
        4: SkillCalendarColumn}
    connector = ObjectProperty(None)
    tl_width = 16
    tl_height = 8

    def __init__(self, **kwargs):
        super(Calendar, self).__init__(size_hint_y=None, **kwargs)

    def on_parent(self, instance, value):
        if value is None:
            if self.connector is not None:
                self.connector.unbind(hi_branch=self.update,
                                      hi_tick=self.update)
                self.connector = None
            return
        charsheet = get_charsheet(self)
        character = charsheet.character
        closet = character.closet
        ks = []
        for key in self.keys:
            if key not in (None, ''):
                ks.append(key)
        if self.connector is None:
            self.connector = character.closet.kivy_connector
        (line_points, wedge_points) = self.get_tl_points(
            self.connector.branch, self.connector.tick)
        self.tl_line = Line(points=line_points)
        self.tl_wedge = Triangle(points=wedge_points)
        self.tl_color = Color(1.0, 0.0, 0.0, 1.0)
        self.canvas.after.add(self.tl_color)
        self.canvas.after.add(self.tl_line)
        self.canvas.after.add(self.tl_wedge)
        if self.cal_type == 0:
            self.referent = closet.get_thing(*ks)
        elif self.cal_type == 1:
            self.referent = closet.get_place(*ks)
        elif self.cal_type == 2:
            self.referent = closet.get_portal(*ks)
        closet.kivy_connector.bind(branch=self.update, tick=self.uptick)
        self.update()

    def uptick(self, *args):
        (self.tl_line.points, self.tl_wedge.points) = self.get_tl_points(
            self.connector.branch, self.connector.tick)

    def update(self, *args):
        self.uptick(*args)
        constructor = self.cal_types[self.cal_type]
        for branch in self.skel:
            if branch not in self.columns:
                col = constructor(branch=branch)
                self.add_widget(col)
                self.columns[branch] = col
        super(Calendar, self).do_layout(*args)

    @property
    def skel(self):
        charsheet = get_charsheet(self)
        character = charsheet.character
        if self.cal_type == 0:
            return self.referent.locations
        elif self.cal_type == 1:
            return character.placedict
        elif self.cal_type == 2:
            return character.portaldict
        elif self.cal_type == 3:
            return character.statdict
        elif self.cal_type == 4:
            return character.skilldict

    def get_max_col_tick(self):
        return max((self.max_tick, self.height / self.tick_height,
                    self.min_ticks))

    def get_tl_points(self, branch, tick):
        l = self.x + (self.col_default_width + self.spacing[0]) * branch
        b = 0
        (r, t) = self.size
        try:
            c = self.tick_y(tick)
        except ZeroDivisionError:
            c = self.height
        line_points = (l, c, r, c)
        r = self.tl_width
        ry = self.tl_height / 2
        t = c + ry
        b = c - ry
        wedge_points = (l, t, r, c, l, b)
        return (line_points, wedge_points)


class BranchConnector(Widget):
    wedge_height = 8

    def on_parent(self, *args):
        self.x = (self.parent.parent_branch_col.window_right -
                  self.parent.parent_branch_col.style.spacing)
        self.y = (self.parent.calendar.tick_to_y(self.column.start_tick) +
                  self.parent.calendar.offy)
        with self.canvas:
            Color(*self.color)
            Line(points=self.get_line_points())
            Triangle(points=self.get_wedge_points())

    def get_line_points(self):
        y0 = self.y
        y2 = y0
        y1 = y0 + self.wedge_height
        x0 = self.x
        # x1 gotta be half way between x0 and the center of my column
        x2 = self.column.left + self.column.width / 2
        dx = x2 - x0
        x1 = x0 + dx
        return [x0, y0, x1, y0, x1, y1, x2, y1, x2, y2]

    def get_wedge_points(self):
        b = self.y
        t = b + self.wedge_height
        c = self.column.left + self.column.width / 2
        rx = self.wedge_width / 2
        l = c - rx
        r = c + rx
        return [c, b, r, t, l, t]


class CharSheetTable(GridLayout):
    key0 = StringProperty()
    key1 = StringProperty(None, allownone=True)
    key2 = StringProperty(None, allownone=True)
    keys = ReferenceListProperty(key0, key1, key2)
    charsheet = ObjectProperty(allownone=True)

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

    def on_parent(self, *args):
        self.charsheet = get_charsheet(self)
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
        closet = self.charsheet.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        for rd in self.iter_skeleton(branch, tick):
            yield [rd[key] for key in self.colkeys]


class CharSheetThingTable(CharSheetTable):
    colkeys = ["dimension", "thing", "location"]

    @property
    def character_skel(self):
        charsheet = get_charsheet(self)
        return charsheet.character.thingdict

    def get_branch_rd_iter(self, branch):
        charsheet = get_charsheet(self)
        if self.keys[0] is None:
            for dimension in charsheet.character.thingdict:
                for thing in charsheet.character.thingdict[dimension]:
                    for rd in charsheet.character.thingdict[
                            dimension][thing][branch].iterrows():
                        yield rd
        elif self.keys[1] is None:
            dimension = self.keys[0]
            for thing in charsheet.character.thingdict[dimension]:
                for rd in charsheet.character.thingdict[
                        dimension][thing][branch].iterrows():
                    yield rd
        else:
            dimension = self.keys[0]
            thing = self.keys[1]
            for rd in charsheet.character.thingdict[
                    dimension][thing][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        charsheet = get_charsheet(self)
        if branch is None:
            branch = charsheet.character.closet.branch
        if tick is None:
            tick = charsheet.character.closet.tick
        covered = set()
        for rd in self.get_branch_rd_iter(branch):
            if (rd["dimension"], rd["thing"]) in covered:
                continue
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                thing = charsheet.character.closet.get_thing(
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
        charsheet = get_charsheet(self)
        return charsheet.character.placedict


class CharSheetPortalTable(CharSheetTable):
    colkeys = ["dimension", "origin", "destination"]

    @property
    def character_skel(self):
        charsheet = get_charsheet(self)
        return charsheet.character.portaldict


class CharSheetStatTable(CharSheetTable):
    colkeys = ["stat", "value"]

    @property
    def character_skel(self):
        charsheet = get_charsheet(self)
        return charsheet.character.statdict

    def get_branch_rd_iter(self, branch):
        charsheet = get_charsheet(self)
        if self.keys[0] is None:
            for stat in charsheet.character.statdict:
                for rd in charsheet.character.statdict[
                        stat][branch].iterrows():
                    yield rd
        else:
            stat = self.keys[0]
            for rd in charsheet.character.statdict[
                    stat][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        charsheet = get_charsheet(self)
        if branch is None:
            branch = charsheet.character.closet.branch
        if tick is None:
            tick = charsheet.character.closet.tick
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
        charsheet = get_charsheet(self)
        return charsheet.character.skilldict

    def get_branch_rd_iter(self, branch):
        charsheet = get_charsheet(self)
        if self.keys[0] is None:
            for skill in charsheet.character.skilldict:
                for rd in charsheet.character.skilldict[
                        skill][branch].iterrows():
                    yield rd
        else:
            skill = self.keys[0]
            for rd in charsheet.character.skilldict[
                    skill][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        charsheet = get_charsheet(self)
        if branch is None:
            branch = charsheet.character.closet.branch
        if tick is None:
            tick = charsheet.character.closet.tick
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


class CharSheet(GridLayout):
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
             "key2": "TEXT"},
            ("character", "idx"),
            {"character": ("charsheet", "character")},
            ["CASE key1 WHEN NULL THEN type NOT IN ({0}) END".format(
                ", ".join([str(SHEET_ITEM_TYPE[typ]) for typ in (
                    "THINGTAB", "THINGCAL",
                    "PLACETAB", "PLACECAL",
                    "PORTALTAB", "PORTALCAL")])),
             "CASE key2 WHEN NULL THEN type<>{0} END".format(
                 str(SHEET_ITEM_TYPE["PORTALTAB"])),
             "idx>=0",
             "idx<={}".format(max(SHEET_ITEM_TYPE.values()))])
    ]
    character = ObjectProperty()
    rowdict = DictProperty()
    style = AliasProperty(
        lambda self: self.character.closet.get_style(
            self.rowdict["style"]),
        lambda self, v: None)
    tabs = {
        SHEET_ITEM_TYPE["THINGTAB"]: CharSheetThingTable,
        SHEET_ITEM_TYPE["PLACETAB"]: CharSheetPlaceTable,
        SHEET_ITEM_TYPE["PORTALTAB"]: CharSheetPortalTable,
        SHEET_ITEM_TYPE["STATTAB"]: CharSheetStatTable,
        SHEET_ITEM_TYPE["SKILLTAB"]: CharSheetSkillTable
    }

    def __init__(self, **kwargs):
        GridLayout.__init__(
            self,
            cols=1,
            pos_hint={'x': 0.7, 'y': 0.0},
            size_hint=(0.3, 1),
            spacing=10,
            **kwargs)

        rd = self.character.closet.skeleton[
            "charsheet"][unicode(self.character)]

        def upd_rd(*args):
            self.rowdict = dict(rd)
        upd_rd()
        rd.listener = upd_rd

        for rd in self.character.closet.skeleton[u"charsheet_item"][
                unicode(self.character)].iterrows():
            keylst = [rd["key0"], rd["key1"], rd["key2"]]
            if rd["type"] in self.tabs:
                self.add_widget(
                    self.tabs[rd["type"]](
                        size_hint=(None, None),
                        keys=keylst))
            else:
                view = ScrollView()
                cal = Calendar(
                    bg_color=self.style.bg_active.rgba,
                    text_color=self.style.textcolor.rgba,
                    font_name=self.style.fontface + '.ttf',
                    font_size=self.style.fontsize,
                    keys=keylst,
                    cal_type=SHEET_TO_CAL_TYPE[rd["type"]])
                cal.bind(minimum_height=cal.setter('height'))
                cal.bind(minimum_width=cal.setter('width'))
                self.add_widget(view)
                view.add_widget(cal)
