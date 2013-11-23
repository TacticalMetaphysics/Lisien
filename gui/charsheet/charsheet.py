# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from calendar import (
    CAL_TYPE,
    CalendarLayout)
from gui.kivybits import SaveableWidgetMetaclass
from table import TableLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    ListProperty,
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


class Image(KivyImage):
    character = ObjectProperty()
    keys = ListProperty()
    edbut = ObjectProperty()

    def __init__(self, **kwargs):
        super(Image, self).__init__(**kwargs)
        self.size = self.texture.size


class PawnImage(Image):
    pass


class SpotImage(Image):
    pass


class CharSheet(GridLayout):
    """A display of some or all of the information making up a Character.

A CharSheet is a layout of vertically stacked widgets that are not
usable outside of a CharSheet. Those widgets are instances or
subclasses of Table, Calendar, or Image. In developer mode, each
widget has a toggle next to it that will enable editing the data
therein. The new data will be applied at the current branch and
tick.

    """
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

    def on_character(self, i, character):
        """Iterate over the bones under my name, and add widgets appropriate
to each of them.

Each widget gets kwargs character, style, item_type, and
keys. item_type is an integer, defined in SHEET_ITEM_TYPE,
representing both the widget class and the way it looks up its
data. keys identify a particular entity whose data will be displayed,
but they never include branch or tick--CharSheet will only display
things appropriate to the present, whenever that may be.

        """
        self.bone = character.closet.skeleton[u"charsheet"][unicode(character)]
        i = 0
        height = 0
        for bone in character.closet.skeleton[u"charsheet_item"][
                unicode(character)].iterbones():
            keylst = [bone["key0"], bone["key1"], bone["key2"]]
            eb = EditButton(
                text=character.closet.get_text('@edit'),
                group=unicode(self.character))
            if bone["type"] < 5:
                w = TableLayout(
                    character=character,
                    style=character.closet.get_style(self.bone["style"]),
                    item_type=bone["type"],
                    keys=keylst,
                    edbut=eb)
            elif bone["type"] < 10:
                w = CalendarLayout(
                    character=character,
                    style=character.closet.get_style(self.bone["style"]),
                    item_type=bone["type"],
                    keys=keylst,
                    edbut=eb)
            elif bone["type"] == 10:
                w = PawnImage(
                    character=character,
                    keys=keylst,
                    edbut=eb)
            elif bone["type"] == 11:
                w = SpotImage(
                    character=character,
                    keys=keylst,
                    edbut=eb)
            self.add_widget(w)
            self.add_widget(eb)
            self.rows_minimum[i] = w.height
            height += w.height + self.spacing[1]
            i += 1
        self.height = height

    def on_touch_down(self, touch):
        """If one of my children catches the touch, nobody else ought to, so
return True in that case."""
        for child in self.children:
            if child.on_touch_down(touch):
                return True

    def on_touch_move(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_move(touch)

    def on_touch_up(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_up(touch)


class CharSheetView(ScrollView):
    character = ObjectProperty()

    def on_touch_down(self, touch):
        super(CharSheetView, self).on_touch_down(touch)
        if self._touch:
            return True
