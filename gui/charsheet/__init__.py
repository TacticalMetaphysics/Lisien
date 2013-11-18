# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from calendar import (
    CAL_TYPE,
    CalendarView)
from gui.kivybits import SaveableWidgetMetaclass
from table import (
    ThingTable,
    PlaceTable,
    PortalTable,
    StatTable,
    SkillTable)
from kivy.uix.gridlayout import GridLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty)


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
        SHEET_ITEM_TYPE["THINGTAB"]: ThingTable,
        SHEET_ITEM_TYPE["PLACETAB"]: PlaceTable,
        SHEET_ITEM_TYPE["PORTALTAB"]: PortalTable,
        SHEET_ITEM_TYPE["STATTAB"]: StatTable,
        SHEET_ITEM_TYPE["SKILLTAB"]: SkillTable
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
                        charsheet=self,
                        size_hint=(None, None),
                        keys=keylst))
            else:
                self.add_widget(CalendarView(
                    character=self.character,
                    cal_type=SHEET_TO_CAL_TYPE[rd["type"]],
                    keys=keylst,
                    bg_color=self.style.bg_active.rgba,
                    text_color=self.style.textcolor.rgba,
                    font_name=self.style.fontface,
                    font_size=self.style.fontsize))


class CharSheetView(RelativeLayout):
    character = ObjectProperty()

    @property
    def sheet(self):
        sheet = self.children[0]
        while not isinstance(sheet, CharSheet):
            sheet = sheet.children[0]
        return sheet

    def on_touch_down(self, touch):
        return self.sheet.on_touch_down(touch)

    def on_touch_move(self, touch):
        return self.sheet.on_touch_move(touch)

    def on_touch_up(self, touch):
        return self.sheet.on_touch_up(touch)
