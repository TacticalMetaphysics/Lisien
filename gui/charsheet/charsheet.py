# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from calendar import (
    CAL_TYPE,
    CalendarView)
from gui.kivybits import SaveableWidgetMetaclass
from table import (
    TableView)
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    ListProperty,
    NumericProperty,
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
    "SKILLCAL": 9,
    "THINGIMAGE": 10,
    "PLACEIMAGE": 11}
SHEET_TO_CAL_TYPE = dict(
    [(SHEET_ITEM_TYPE[a], CAL_TYPE[a[:-3]]) for a in
     ("THINGCAL", "PLACECAL", "PORTALCAL", "STATCAL", "SKILLCAL")])


class EditButton(ToggleButton):
    def collide_point(self, x, y):
        return super(EditButton, self).collide_point(*self.to_local(x, y))


class PawnImage(Image):
    character = ObjectProperty()
    keys = ListProperty()
    edbut = ObjectProperty()


class SpotImage(Image):
    character = ObjectProperty()
    keys = ListProperty()
    edbut = ObjectProperty()


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
            ["CASE key2 WHEN NULL THEN type<>{0} END".format(
                str(SHEET_ITEM_TYPE["PORTALTAB"])),
             "idx>=0",
             "idx<={}".format(max(SHEET_ITEM_TYPE.values()))])
    ]
    character = ObjectProperty()
    bone = ObjectProperty()
    style = ObjectProperty()
    completedness = NumericProperty()

    def on_parent(self, i, parent):
        character = self.character
        self.bone = character.closet.skeleton["charsheet"][unicode(character)]
        for bone in character.closet.skeleton[u"charsheet_item"][
                unicode(character)].iterbones():
            keylst = [bone["key0"], bone["key1"], bone["key2"]]
            if bone["type"] < 5:
                w = TableView(
                    character=character,
                    style=character.closet.get_style(self.bone["style"]),
                    item_type=bone["type"],
                    keys=keylst)
            elif bone["type"] < 10:
                # from the charsheet's perspective, the calendar's
                # background is the foreground.
                w = CalendarView(
                    character=character,
                    style=character.closet.get_style(self.bone["style"]),
                    item_type=bone["type"],
                    keys=keylst)
            elif bone["type"] == 10:
                w = PawnImage(
                    character=character,
                    keys=keylst)
            elif bone["type"] == 11:
                w = SpotImage(
                    character=character,
                    keys=keylst)
            eb = EditButton(
                text=character.closet.get_text('@edit'))
            w.edbut = eb
            self.add_widget(w)
            self.add_widget(eb)

    def on_touch_down(self, touch):
        for child in self.children:
            if child.on_touch_down(touch):
                return True

    def on_touch_move(self, touch):
        for child in self.children:
            child.on_touch_move(touch)

    def on_touch_up(self, touch):
        for child in self.children:
            child.on_touch_up(touch)
