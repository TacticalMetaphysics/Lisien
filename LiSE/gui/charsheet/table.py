from kivy.properties import (
    AliasProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty)
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.listview import ListView
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.stencilview import StencilView
from kivy.clock import Clock


class TableCell(Label):
    """Basically just a Label that may be conveniently rewritten with a
    callback."""
    text_getter = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['size_hint_y'] = None
        self._trigger_upd_text = Clock.create_trigger(self.upd_text)
        super(TableCell, self).__init__(**kwargs)

    def upd_text(self, *args):
        self.text = self.text_getter()


class TableHeader(TableCell):
    """TableCell to put at the top of the table"""
    pass


class TableBody(TableCell):
    """TableCell to put in the rows of the table"""
    pass


class TableRow(BoxLayout):
    """Assembles appropriate TableBody for the fieldnames and statnames
    for the item"""
    item = ObjectProperty()
    tableview = ObjectProperty()
    closet = ObjectProperty()
    fieldnames = ListProperty()
    statnames = ListProperty()

    def __init__(self, **kwargs):
        kwargs['closet'] = kwargs['tableview'].character.closet
        kwargs['fieldnames'] = kwargs['tableview'].fieldnames
        kwargs['statnames'] = kwargs['tableview'].stats
        super(TableRow, self).__init__(**kwargs)
        for fieldname in self.fieldnames:
            bwid = TableBody(
                text_getter=lambda: unicode(getattr(self.item, fieldname)))
            self.closet.register_time_listener(bwid._trigger_upd_text)
            self.add_widget(bwid)
            bwid.upd_text()
        for statname in self.statnames:
            bwid = TableBody(
                text_getter=lambda: unicode(self.item.get_stat(statname)))
            self.closet.register_time_listener(bwid._trigger_upd_text)
            self.add_widget(bwid)
            bwid.upd_text()


class TableContent(StackLayout):
    """Contains a ListView to assemble the table."""
    closet = ObjectProperty()
    adapter = ObjectProperty()
    listview = ObjectProperty()

    def __init__(self, **kwargs):
        super(TableContent, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (self.closet and self.parent):
            Clock.schedule_once(self.finalize, 0)
            return
        head = BoxLayout()
        for header in self.parent.headers + self.parent.stats:
            hwid = TableHeader(
                text_getter=lambda: self.closet.get_text(header))
            self.closet.register_text_listener(header, hwid.upd_text)
            hwid.upd_text()
            head.add_widget(hwid)
        self.add_widget(head)
        self.adapter = ListAdapter(
            data=self.parent.items,
            args_converter=self.args_converter,
            cls=TableRow)
        self.listview = ListView(adapter=self.adapter)
        self.parent.bind(items=self.adapter.setter('data'))
        self.add_widget(self.listview)

    def args_converter(self, index, arg):
        return {
            'item': arg,
            'tableview': self.parent}

# TODO unify TableView and TableContent


class TableView(StencilView):
    charsheet = ObjectProperty()
    character = AliasProperty(
        lambda self: self.charsheet.character
        if self.charsheet else None,
        lambda self, v: None,
        bind=('charsheet',))
    mybone = ObjectProperty()
    headers = ListProperty()
    fieldnames = ListProperty()
    items = ListProperty()
    stats = ListProperty()
    i = NumericProperty()

    def __init__(self, **kwargs):
        super(TableView, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (
                self.charsheet and
                self.headers and
                (self.fieldnames or self.stats) and
                self.items):
            Clock.schedule_once(self.finalize, 0)
            return
        content = TableContent(
            closet=self.charsheet.character.closet,
            width=self.width,
            x=self.x,
            top=self.top)
        self.add_widget(content)
        self.bind(width=content.setter('width'),
                  x=content.setter('x'),
                  top=content.setter('top'))


class CharStatTableContent(GridLayout):
    closet = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['cols'] = 2
        super(CharStatTableContent, self).__init__(**kwargs)

    def repop(self, *args):
        self.clear_widgets()
        for stat in self.parent.stats:
            statwid = TableBody(
                text_getter=lambda: self.closet.get_text(stat))
            self.add_widget(statwid)
            valwid = TableBody(
                text_getter=lambda: unicode(
                    self.parent.character.get_stat(stat)))
            self.closet.register_time_listener(valwid.upd_text)


class CharStatTableView(StencilView):
    character = ObjectProperty()
    stats = ListProperty()
    i = NumericProperty()
