from kivy.properties import (
    NumericProperty,
    BooleanProperty,
    ListProperty,
    DictProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput


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

    def do_layout(self, *args):
        if len(self.headers) > len(self.fieldnames):
            raise ValueError(
                "There are too many headers.")
        elif len(self.fieldnames) > len(self.headers):
            raise ValueError(
                "There are too many fieldnames.")
        self.clear_widgets()
        for header in self.parent.headers:
            hwid = TableHeader(
                text_getter=lambda: self.closet.get_text(header))
            self.closet.register_text_listener(header, hwid.upd_text)
            self.add_widget(hwid)
        for item in self.parent.items:
            for fieldname in self.parent.fieldnames:
                bwid = TableBody(
                    text_getter=lambda: unicode(getattr(item, fieldname)))
                self.closet.register_time_listener(bwid.upd_text)
                bwid.upd_text()
                self.add_widget(bwid)
            for stat in self.parent.stats:
                bwid = TableBody(
                    text_getter=lambda: unicode(item.get_stat(stat)))
                self.closet.register_time_listener(bwid.upd_text)
                bwid.upd_text()
                self.add_widget(bwid)


class TableView(ScrollView):
    character = ObjectProperty()
    item_type = NumericProperty()
    items = ListProperty()
    headers = ListProperty()
    fieldnames = ListProperty()
    stats = ListProperty([])
    edbut = ObjectProperty()
