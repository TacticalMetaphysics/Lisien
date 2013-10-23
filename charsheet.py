# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from logging import getLogger
from util import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import (
    Color,
    Line,
    Rectangle,
    Triangle)
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty,
    ListProperty,
    NumericProperty,
    ReferenceListProperty,
    StringProperty)


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


class CalendarCell(RelativeLayout):
    bg_color = ObjectProperty(allownone=True)
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty(allownone=True)
    text = StringProperty()

    def __init__(self, **kwargs):
        RelativeLayout.__init__(
            self, size_hint=(None, None), **kwargs)
        print("Creating calendar cell with tick_from={}, "
              "tick_to={}, text={}".format(self.tick_from, self.tick_to,
                                           self.text))

    def on_parent(self, *args):
        charsheet = get_charsheet(self)
        style = charsheet.style
        self.bg_color = Color(*style.bg_active.rgba)
        self.canvas.before.add(self.bg_color)
        self.bg_rect = Rectangle(
            pos=self.pos,
            size=self.size)
        self.canvas.before.add(self.bg_rect)
        self.add_widget(Label(
            text=self.text,
            pos=self.pos,
            text_size=self.size,
            valign='top'))

    def do_layout(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        label = self.children[0]
        label.pos = self.pos
        label.text_size = self.size


class CalendarColumn(RelativeLayout):
    branch = NumericProperty()
    tl_line = ObjectProperty(allownone=True)
    tl_wedge = ObjectProperty(allownone=True)
    tl_color = ObjectProperty(allownone=True)
    tl_width = 16
    tl_height = 8

    @property
    def parent_branch_col(self):
        return self.parent.make_col(
            self.parent.parent.closet.skeleton[
                "timestream"][self.branch]["parent"])

    def on_parent(self, *args):
        charsheet = get_charsheet(self)
        closet = charsheet.character.closet
        self.do_layout()
        (line_points, wedge_points) = self.get_tl_points(closet.tick)
        if self.branch == closet.branch:
            self.tl_color = Color(1.0, 0.0, 0.0, 1.0)
        else:
            self.tl_color = Color(1.0, 0.0, 0.0, 0.0)
        self.canvas.after.add(self.tl_color)
        self.tl_line = Line(points=line_points)
        self.canvas.after.add(self.tl_line)
        self.tl_wedge = Triangle(points=wedge_points)
        self.canvas.after.add(self.tl_wedge)
        closet.bind(branch=self.upd_tl, tick=self.upd_tl)

    def __int__(self):
        return self.branch

    def __eq__(self, other):
        return hasattr(other, 'branch') and int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def upd_tl(self, *args):
        charsheet = get_charsheet(self)
        (line_points, wedge_points) = self.get_tl_points(
            charsheet.character.closet.tick)
        self.tl_line.points = line_points
        self.tl_wedge.points = wedge_points

    def get_tl_points(self, tick):
        (l, b) = self.to_parent(0, 0)
        (r, t) = self.to_parent(*self.size)
        try:
            c = self.parent.tick_y(tick)
        except ZeroDivisionError:
            c = self.height
        line_points = self.to_parent(l, c) + self.to_parent(r, c)
        r = self.tl_width
        ry = self.tl_height / 2
        t = c + ry
        b = c - ry
        wedge_points = self.to_parent(l, t) + self.to_parent(
            r, c) + self.to_parent(l, b)
        return (line_points, wedge_points)

    def do_layout(self, *args):
        calendar = get_calendar(self)
        charsheet = get_charsheet(self)
        closet = charsheet.character.closet
        tick_height = calendar.tick_height
        for cell in self.children:
            cell.pos = (0, calendar.tick_y(cell.tick_to))
            if cell.tick_from > closet.get_hi_tick():
                closet.set_hi_tick(cell.tick_from)
            if (
                    cell.tick_to is not None and
                    cell.tick_to > closet.get_hi_tick()):
                closet.set_hi_tick(cell.tick_to)
            if cell.tick_to is None:
                cell.size = (
                    self.width, (closet.get_hi_tick() -
                                 cell.tick_from) * tick_height)
            else:
                cell.size = (
                    self.width, (cell.tick_to - cell.tick_from) * tick_height)
            cell.do_layout()


class LocationCalendarColumn(CalendarColumn):
    thing = ObjectProperty()

    def on_parent(self, *args):
        charsheet = get_charsheet(self)
        closet = charsheet.character.closet
        self.thing = closet.get_thing(
            self.parent.keys[0], self.parent.keys[1])
        rowiter = self.thing.locations.iterrows()
        prev = next(rowiter)
        for rd in rowiter:
            tick_from = prev["tick_from"]
            if rd["tick_from"] is None:
                tick_to = closet.get_hi_tick()
            else:
                tick_to = rd["tick_from"]
                if tick_to > closet.get_hi_tick():
                    closet.set_hi_tick(tick_to)
            cc = CalendarCell(
                tick_from=tick_from, tick_to=tick_to, text=prev["location"])
            prev = rd
        cc = CalendarCell(
            tick_from=prev["tick_from"], tick_to=None, text=prev["location"])
        self.add_widget(cc)
        super(LocationCalendarColumn, self).on_parent(*args)


class PlaceCalendarColumn(CalendarColumn):
    pass


class PortalCalendarColumn(CalendarColumn):
    pass


class StatCalendarColumn(CalendarColumn):
    pass


class SkillCalendarColumn(CalendarColumn):
    pass


class Calendar(BoxLayout):
    cal_type = NumericProperty(0)
    scroll_factor = NumericProperty(4)
    tick_height = NumericProperty(10)
    col_default_width = NumericProperty(110)
    key0 = StringProperty()
    key1 = StringProperty(allownone=True)
    key2 = StringProperty(allownone=True)
    keys = ReferenceListProperty(key0, key1, key2)

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self,
            orientation='horizontal',
            height=200,
            size_hint=(None, None),
            **kwargs)

    def __eq__(self, other):
        return (
            other is not None and
            self.parent is other.parent and
            hasattr(other, 'cal_type') and
            self.cal_type == other.cal_type and
            self.skel == other.skel)

    def __ne__(self, other):
        return (
            other is None or
            self.parent is not other.parent or
            not hasattr(other, 'cal_type') or
            other.cal_type != self.cal_type or
            self.skel != other.skel)

    def tick_y(self, tick):
        charsheet = get_charsheet(self)
        hi_tick = charsheet.character.closet.timestream.hi_tick
        if tick is None:
            tick = hi_tick
        ticks_from_bot = hi_tick - tick
        return ticks_from_bot * self.tick_height

    def tick_y_hint(self, tick):
        return self.tick_y(tick) / float(self.height)

    def on_parent(self, *args):
        self.change_type(self.cal_type, self.keys)
        charsheet = get_charsheet(self)
        closet = charsheet.character.closet
        hi_branch = closet.timestream.hi_branch
        if not hasattr(self, 'skel'):
            return
        else:
            i = -1
            while i < hi_branch:
                i += 1
                col = self.make_col(i)
                self.add_widget(col)
        print("Added a calendar with {} columns.".format(len(self.children)))
        closet.timestream.bind(hi_branch=self.upd_width)
        self.do_layout()

    def upd_width(self, *args):
        charsheet = get_charsheet(self)
        branches = charsheet.character.closet.timestream.hi_branch + 1
        self.width = (self.col_default_width + self.spacing) * branches

    def change_type(self, cal_type, keys):
        charsheet = get_charsheet(self)
        self.cal_type = cal_type
        dk = {
            CAL_TYPE["THING"]: "thing",
            CAL_TYPE["PLACE"]: "place",
            CAL_TYPE["PORTAL"]: "portal",
            CAL_TYPE["STAT"]: "stat",
            CAL_TYPE["SKILL"]: "skill"}[cal_type]
        self.skel = charsheet.character.get_item_history(dk, *keys)
        self.keys = keys

    def make_col(self, branch):
        print("Making a calendar column for branch {}".format(branch))
        col_clas = {
            CAL_TYPE['THING']: LocationCalendarColumn,
            CAL_TYPE['PLACE']: PlaceCalendarColumn,
            CAL_TYPE['PORTAL']: PortalCalendarColumn,
            CAL_TYPE['STAT']: StatCalendarColumn,
            CAL_TYPE['SKILL']: SkillCalendarColumn
        }[self.cal_type]
        return col_clas(branch=branch, width=self.col_default_width)

    def do_layout(self, *args):
        for column in self.children:
            column.do_layout()
        super(Calendar, self).do_layout(*args)


class CharSheetTable(GridLayout):
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

    def on_parent(self, *args):
        charsheet = get_charsheet(self)
        self.cols = len(self.colkeys)
        self.row_default_height = (charsheet.style.fontsize
                                   + charsheet.style.spacing)
        self.row_force_default = True

        for key in self.colkeys:
            self.add_widget(Label(
                text=key,
                font_name=charsheet.style.fontface + '.ttf',
                font_size=charsheet.style.fontsize,
                color=charsheet.style.textcolor.rgba))
        for rd in self.iter_skeleton():
            for key in self.colkeys:
                self.add_widget(Label(
                    text=rd[key],
                    font_name=charsheet.style.fontface + '.ttf',
                    font_size=charsheet.style.fontsize,
                    color=charsheet.style.textcolor.rgba))

    def iter_skeleton(self, branch=None, tick=None):
        charsheet = get_charsheet(self)
        if branch is None:
            branch = charsheet.character.closet.branch
        if tick is None:
            tick = charsheet.character.closet.tick
        for rd in self.character_skel.iterrows():
            if (
                    rd["branch"] == branch and
                    rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick)):
                yield rd

    def iterrows(self, branch=None, tick=None):
        charsheet = get_charsheet(self)
        closet = charsheet.character.closet
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
        lambda self, v: None,
        bind=('rowdict',))
    tabs = {
        SHEET_ITEM_TYPE["THINGTAB"]: CharSheetThingTable,
        SHEET_ITEM_TYPE["PLACETAB"]: CharSheetPlaceTable,
        SHEET_ITEM_TYPE["PORTALTAB"]: CharSheetPortalTable,
        SHEET_ITEM_TYPE["STATTAB"]: CharSheetStatTable,
        SHEET_ITEM_TYPE["SKILLTAB"]: CharSheetSkillTable
    }

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self,
            orientation='vertical',
            pos_hint={'x': 0.7,
                      'y': 0.0},
            size_hint=(0.3, 1.0),
            spacing=10,
            **kwargs)

    def on_parent(self, *args):
        rd = self.character.closet.skeleton[
            "charsheet"][unicode(self.character)]

        def upd_rd(*args):
            self.rowdict = dict(rd)

        upd_rd()
        rd.bind(touches=upd_rd)

        for rd in self.character.closet.skeleton[u"charsheet_item"][
                unicode(self.character)].iterrows():
            keylst = [rd["key0"], rd["key1"], rd["key2"]]
            if rd["type"] in self.tabs:
                self.add_widget(
                    self.tabs[rd["type"]](
                        keys=keylst))
            else:
                self.add_widget(
                    Calendar(
                        keys=keylst,
                        typ=SHEET_TO_CAL_TYPE[rd["type"]],
                        scroll_factor=SCROLL_FACTOR))
