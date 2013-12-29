from kivy.properties import (
    NumericProperty,
    ListProperty,
    DictProperty,
    ObjectProperty)
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView

from LiSE.data import (
    THING_LOC_TAB,
    THING_STAT_TAB,
    PLACE_STAT_TAB,
    PORTAL_LOC_TAB,
    PORTAL_STAT_TAB,
    CHAR_STAT_TAB,
    ITEM_TYPE_TO_HEADERS,
    ITEM_TYPE_TO_FIELD_NAMES)


class TableCell(Label):
    text_getter = ObjectProperty()

    def upd_text(self, *args):
        self.text = self.text_getter()


class TableHeader(TableCell):
    pass


class TableBody(TableCell):
    pass


class TableContent(GridLayout):
    closet = ObjectProperty()
    headers_done = ListProperty([])
    items_done = DictProperty({})

    def do_layout(self, *args):
        for header in self.parent.headers:
            if header in self.headers_done:
                continue
            hwid = TableHeader(
                text_getter=lambda: self.closet.get_text(header))
            self.closet.register_text_listener(header, hwid.upd_text)
            hwid.upd_text()
            self.add_widget(hwid)
            self.headers_done.append(header)
        for item in self.parent.items:
            if item not in self.items_done:
                self.items_done[item] = []
            for fieldname in self.parent.fieldnames:
                if fieldname in self.items_done[item]:
                    continue
                bwid = TableBody(
                    text_getter=lambda: unicode(getattr(item, fieldname)))
                self.closet.register_time_listener(bwid.upd_text)
                bwid.upd_text()
                self.add_widget(bwid)
                self.items_done[item].append(fieldname)
            for stat in self.parent.stats:
                if stat in self.items_done[item]:
                    continue
                bwid = TableBody(
                    text_getter=lambda: unicode(item.get_stat(stat)))
                self.closet.register_time_listener(bwid.upd_text)
                bwid.upd_text()
                self.add_widget(bwid)
                self.items_done[item].append(stat)


class TableView(ScrollView):
    character = ObjectProperty()
    items = ListProperty()
    headers = ListProperty()
    fieldnames = ListProperty()
    stats = ListProperty()
    edbut = ObjectProperty()
