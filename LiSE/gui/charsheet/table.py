from kivy.properties import (
    AliasProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty
)
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.listview import ListView
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.stencilview import StencilView
from kivy.clock import Clock


class TableCell(object):
    """Basically just a Label that may be conveniently rewritten with a
    callback."""
    text_getter = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['size_hint_y'] = None
        self._trigger_upd_text = Clock.create_trigger(self.upd_text)
        super(TableCell, self).__init__(**kwargs)


class TableHeader(Label, TableCell):
    """TableCell to put at the top of the table"""
    def upd_text(self, *args):
        self.text = self.text_getter()


class TableBody(TextInput, TableCell):
    """TableCell to put in the rows of the table"""
    _touch = ObjectProperty()
    item = ObjectProperty()
    key = StringProperty()
    bone_setter = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['multiline'] = False
        super(TableBody, self).__init__(**kwargs)

    def on_text_validate(self, *args):
        self.bone_setter(self.key, self.text)
        self.focus = False
        self.upd_text()

    def upd_text(self, *args):
        self.text = ''
        self.hint_text = self.text_getter()


class TableRow(BoxLayout):
    """Assembles appropriate TableBody for the fieldnames and statnames
    for the item"""
    item = ObjectProperty()
    tab_type = StringProperty()
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
                item=self.item,
                key=fieldname,
                bone_setter=self.field_bone_setter,
                text_getter=lambda: str(getattr(self.item, fieldname)))
            self.closet.register_time_listener(bwid._trigger_upd_text)
            self.add_widget(bwid)
            bwid.upd_text()
        for statname in self.statnames:
            bwid = TableBody(
                item=self.item,
                key=statname,
                bone_setter=lambda k, v: self.item.set_stat(k, v),
                text_getter=lambda: str(self.item.get_stat(statname)))
            self.closet.register_time_listener(bwid._trigger_upd_text)
            self.add_widget(bwid)
            bwid.upd_text()

    def field_bone_setter(self, key, text):
        if self.tab_type == 'thing_tab' and key == 'location':
            self.item.set_location(text)
        elif self.tab_type == 'portal_tab':
            if key == 'origin':
                self.item.set_origin(text)
            elif key == 'destination':
                self.item.set_destination(text)
        else:
            raise ValueError("Don't know how to set {} of {}".format(
                key, type(self.item)))


class TableContent(GridLayout):
    closet = ObjectProperty()
    adapter = ObjectProperty()
    listview = ObjectProperty()
    tab_type = StringProperty()
    view = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['cols'] = 1
        super(TableContent, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (self.closet and self.view):
            Clock.schedule_once(self.finalize, 0)
            return
        head = BoxLayout()
        for header in self.view.headers + self.view.stats:
            hwid = TableHeader(
                text_getter=lambda: self.closet.get_text(header),
                size_hint_y=None,
                height=40)
            self.closet.register_text_listener(header, hwid.upd_text)
            hwid.upd_text()
            head.add_widget(hwid)
        self.add_widget(head)
        self.adapter = ListAdapter(
            data=self.view.items,
            args_converter=self.args_converter,
            cls=TableRow)
        self.listview = ListView(adapter=self.adapter)
        self.view.bind(items=self.adapter.setter('data'))
        self.add_widget(self.listview)

    def args_converter(self, index, arg):
        return {
            'item': arg,
            'tab_type': self.tab_type,
            'tableview': self.view,
            'size_hint_y': None,
            'height': 40}

# TODO unify TableView and TableContent


class TableView(StencilView):
    charsheet = ObjectProperty()
    character = AliasProperty(
        lambda self: self.charsheet.character
        if self.charsheet else None,
        lambda self, v: None,
        bind=('charsheet',))
    item_type = StringProperty()
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
        self.content = TableContent(
            closet=self.charsheet.character.closet,
            tab_type=self.item_type,
            view=self,
            size_hint_y=None,
            x=self.x,
            top=self.top,
            width=self.width)
        self.bind(x=self.content.setter('x'),
                  top=self.content.setter('top'),
                  width=self.content.setter('width'))
        self.add_widget(self.content)


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
