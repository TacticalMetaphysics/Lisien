# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from logging import getLogger
from kivybits import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.layout import Layout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    BoundedNumericProperty,
    ListProperty,
    ReferenceListProperty,
    StringProperty,
    BooleanProperty)


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


class ColorBox(BoxLayout):
    color = ListProperty()


class Cell(RelativeLayout):
    bg_color = ListProperty(None)
    text_color = ListProperty(None)
    font_name = StringProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    branch = NumericProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)
    text = StringProperty()
    calendar = ObjectProperty()
    rowid = NumericProperty()

    def __init__(self, **kwargs):
        calendar = kwargs["calendar"]
        for kwarg in ["bg_color", "text_color", "font_name", "font_size"]:
            if kwarg not in kwargs:
                kwargs[kwarg] = getattr(calendar, kwarg)
        super(Cell, self).__init__(
            **kwargs)


class Timeline(Widget):
    connector = ObjectProperty()
    col_width = NumericProperty()


class Calendar(Layout):
    cal_type = NumericProperty()
    bg_color = ListProperty()
    text_color = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    branch = NumericProperty(0)
    tick = BoundedNumericProperty(0, min=0)
    ticks_tall = NumericProperty(100)
    ticks_offscreen = NumericProperty(0)
    branches_offscreen = NumericProperty(2)
    spacing_x = NumericProperty(5)
    spacing_y = NumericProperty(5)
    branches_wide = NumericProperty(2)
    col_width = NumericProperty()
    tick_height = NumericProperty(10)
    xmov = NumericProperty(0)
    xcess = NumericProperty(0)
    ymov = NumericProperty(0)
    ycess = NumericProperty(0)
    dragging = BooleanProperty(False)
    keys = ListProperty()
    referent = ObjectProperty(None)
    skel = ObjectProperty(None)
    force_refresh = BooleanProperty(False)

    def on_parent(self, i, v):
        character = v.character
        closet = character.closet
        skeleton = closet.skeleton
        ks = []
        for key in v.keys:
            if key is None:
                break
            ks.append(key)
        self.keys = ks
        if self.cal_type == 0:
            (dimension, thing) = ks
            self.referent = closet.get_thing(dimension, thing)
            self.skel = skeleton["thing_location"][dimension][thing]
        elif self.cal_type == 1:
            (dimension, place) = ks
            self.referent = closet.get_place(dimension, place)
            self.skel = character.placedict[dimension][place]
        elif self.cal_type == 2:
            (dimension, orig, dest) = ks
            self.referent = closet.get_portal(dimension, orig, dest)
            self.skel = character.portaldict[dimension][orig][dest]
        elif self.cal_type == 3:
            stat = ks[0]
            self.skel = character.statdict[stat]
        elif self.cal_type == 4:
            skill = ks[0]
            self.skel = character.skilldict[skill]
        self.skel.listener = self.refresh_and_layout
        self.refresh_and_layout()
        self.bind(size=lambda i, v: self._trigger_layout(),
                  pos=lambda i, v: self._trigger_layout())

    def refresh_and_layout(self, *args):
        self.force_refresh = True
        self._trigger_layout()

    def branch_x(self, b):
        b -= self.branch
        return self.x + self.xmov + b * self.col_width

    def tick_y(self, t):
        if t is None:
            return self.y
        else:
            t -= self.tick
            return self.y + self.ymov + self.height - self.tick_height * t

    def refresh(self):
        minbranch = int(self.branch - self.branches_offscreen)
        maxbranch = int(
            self.branch + self.branches_wide + self.branches_offscreen)
        mintick = int(self.tick - self.ticks_offscreen)
        maxtick = int(self.tick + self.ticks_tall + self.ticks_offscreen)
        # I contain Cells.
        #
        # I should contain those that are visible, or nearly so.
        #
        # Remove those that are neither.
        for child in self.children:
            if (
                    child.branch < minbranch or
                    maxbranch < child.branch or
                    maxtick < child.tick_from or
                    (child.tick_to is not None and
                     child.tick_to < mintick)):
                self.remove_widget(child)
        # Find cells to show
        to_cover = {}
        content = {}
        for branch in xrange(minbranch, maxbranch):
            if branch not in self.skel:
                continue
            to_cover[branch] = set()
            content[branch] = {}
            rowiter = self.skel[branch].iterrows()
            prev = next(rowiter)
            for rd in rowiter:
                if (
                        prev["tick_from"] < maxtick and
                        rd["tick_from"] > mintick):
                    # I'll be showing this cell. Choose text for it
                    # based on my type.
                    if self.cal_type == 0:
                        text = prev["location"]
                    elif self.cal_type == 1:
                        text = prev["place"]
                    elif self.cal_type == 2:
                        text = "{}->{}".format(
                            prev["origin"], prev["destination"])
                    elif self.cal_type == 3:
                        text = prev["value"]
                    else:
                        text = ""
                    to_cover[branch].add(id(prev))
                    content[branch][id(prev)] = (
                        text, prev["tick_from"], rd["tick_from"])
                if rd["tick_from"] > maxtick:
                    break
                prev = rd
            # The last cell is infinitely long
            if prev["tick_from"] < maxtick:
                if self.cal_type == 0:
                    text = prev["location"]
                elif self.cal_type == 1:
                    text = prev["place"]
                elif self.cal_type == 2:
                    text = "{}->{}".format(
                        prev["origin"], prev["destination"])
                elif self.cal_type == 3:
                    text = prev["value"]
                else:
                    text = ""
                to_cover[branch].add(id(prev))
                content[branch][id(prev)] = (
                    text, prev["tick_from"], None)
        # I might already be showing some of these, though.
        #
        # Which ones don't I show?
        uncovered = {}
        covered = {}
        for child in self.children:
            if child.branch not in covered:
                covered[child.branch] = set()
            covered[child.branch].add(child.rowid)
        for (branch, coverage) in to_cover.iteritems():
            if branch not in covered:
                uncovered[branch] = coverage
            else:
                uncovered[branch] = coverage - covered[branch]
        # Construct cells for just the rowdicts that I'm not showing already
        for (branch, rowids) in uncovered.iteritems():
            n = 0
            for rowid in rowids:
                (text, tick_from, tick_to) = content[branch][rowid]
                cell = Cell(
                    calendar=self,
                    branch=branch,
                    text=text,
                    tick_from=tick_from,
                    tick_to=tick_to,
                    rowid=rowid)
                self.add_widget(cell)
                n += 1

    def do_layout(self, *largs):
        if self.parent is None:
            return
        branchwidth = self.col_width
        print(self.xmov)
        d_branch = int(self.xmov / branchwidth)
        tickheight = self.tick_height
        d_tick = int(self.ymov / tickheight)
        if abs(d_branch) >= 1 or abs(d_tick) >= 1:
            print("navigating branch+{} tick+{}".format(d_branch, d_tick))
            try:
                self.branch -= d_branch
            except ValueError:
                self.branch = 0
            self.xmov -= d_branch * (branchwidth + self.spacing_y)
            try:
                self.tick += d_tick
            except ValueError:
                self.tick = 0
            self.ymov -= d_tick * tickheight
            self.refresh()
        elif self.force_refresh:
            self.refresh()
            self.force_refresh = False
        for child in self.children:
            x = self.branch_x(child.branch)
            y = self.tick_y(child.tick_to)
            height = self.tick_y(child.tick_from) - y
            hs = self.spacing_y
            ws = self.spacing_x
            child.pos = (x + ws, y + hs)
            child.size = (branchwidth - ws, height - hs)

    def _touch_down(self, x, y, dx, dy):
        self.dragging = True

    def _touch_up(self, x, y, dx, dy):
        self.dragging = False
        self.xmov = 0
        self.xcess = 0
        self.ymov = 0
        self.ycess = 0
        self._trigger_layout()

    def on_touch_move(self, touch):
        if self.dragging:
            if self.xcess == 0:
                nuxmov = self.xmov + touch.dx
                if not (self.branch == 0 and nuxmov < 0):
                    self.xmov = nuxmov
                else:
                    self.xcess += touch.dx
            else:
                self.xcess += touch.dx
                if self.xcess > 0:
                    self.xcess = 0
            if self.ycess == 0:
                nuymov = self.ymov + touch.dy
                if not (self.tick == 0 and nuymov < 0):
                    self.ymov = nuymov
                else:
                    self.ycess += touch.dy
            else:
                self.ycess += touch.dy
                if self.ycess > 0:
                    self.ycess = 0
            self._trigger_layout()


def get_cal(cal_type, keys, bg_color, text_color, font_name, font_size):
    return Calendar(
        cal_type=cal_type,
        keys=keys,
        bg_color=bg_color,
        text_color=text_color,
        font_name=font_name,
        font_size=font_size)


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
    rowdict = DictProperty({})
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

    def on_parent(self, i, v):
        parent = v
        while not hasattr(parent, 'character'):
            parent = parent.parent
        self.character = parent.character
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
                self.add_widget(CalendarView(
                    character=self.character,
                    cal_type=SHEET_TO_CAL_TYPE[rd["type"]],
                    keys=keylst,
                    bg_color=self.style.bg_active.rgba,
                    text_color=self.style.textcolor.rgba,
                    font_name=self.style.fontface + '.ttf',
                    font_size=self.style.fontsize))

    def _touch_down(self, (x, y), dx, dy):
        for child in self.children:
            if child.collide_point(x, y):
                print("{} collides {},{}".format(child, x, y))
                child._touch_down(x, y, dx, dy)

    def _touch_up(self, (x, y), dx, dy):
        for child in self.children:
            if hasattr(child, '_touch_up'):
                child._touch_up(x, y, dx, dy)


class CharSheetView(RelativeLayout):
    character = ObjectProperty()

    def _touch_down(self, touch):
        stencil = self.children[0]
        charsheet = stencil.children[0]
        charsheet._touch_down(
            self.to_local(touch.x, touch.y),
            touch.dx, touch.dy)

    def _touch_up(self, touch):
        stencil = self.children[0]
        charsheet = stencil.children[0]
        charsheet._touch_up(
            self.to_local(touch.x, touch.y),
            touch.dx, touch.dy)


class CalendarView(RelativeLayout):
    cal_type = NumericProperty()
    keys = ListProperty()
    bg_color = ListProperty()
    text_color = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    character = ObjectProperty()

    def _touch_down(self, x, y, dx, dy):
        for child in self.children:
            if hasattr(child, '_touch_down'):
                child._touch_down(x, y, dx, dy)

    def _touch_up(self, x, y, dx, dy):
        for child in self.children:
            if hasattr(child, '_touch_up'):
                child._touch_up(x, y, dx, dy)
