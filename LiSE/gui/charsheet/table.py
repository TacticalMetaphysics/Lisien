from kivy.properties import (
    AliasProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty,
    BooleanProperty
)
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.label import Label
from kivy.uix.listview import ListView
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
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
    tab_type = StringProperty()
    item = ObjectProperty()
    key = StringProperty()
    is_stat = BooleanProperty()

    def __init__(self, **kwargs):
        kwargs['multiline'] = False
        super(TableBody, self).__init__(**kwargs)

    def on_text_validate(self, *args):
        try:
            self.bone_setter(self.key, self.text)
        except ValueError:
            self.blink_error()
        self.focus = False
        self.upd_text()

    def blink_error(self):
        color = self.background_color
        self.background_color = [1, 0, 0, 1]

        def uncolor(*args):
            self.background_color = color
        Clock.schedule_once(uncolor, 1)

    def upd_text(self, *args):
        self.text = ''
        self.hint_text = self.text_getter()

    def bone_setter(self, key, value):
        if self.is_stat:
            self.item.set_stat(key, value)
        elif self.tab_type == 'thing_tab' and key == 'location':
            self.item.set_location(value, check_existence=True)
        elif self.tab_type == 'portal_tab':
            if key == 'origin':
                self.item.set_origin(value)
            elif key == 'destination':
                self.item.set_destination(value)
        else:
            raise ValueError("Don't know how to set {} of {}".format(
                key, type(self.item)))


class TableRow(BoxLayout):
    """Assembles appropriate TableBody for the fieldnames and statnames
    for the item"""
    charsheet = ObjectProperty()
    item = ObjectProperty()
    tab_type = StringProperty()
    tableview = ObjectProperty()
    closet = ObjectProperty()
    fieldnames = ListProperty()
    statnames = ListProperty()

    def __init__(self, **kwargs):
        kwargs['charsheet'] = kwargs['tableview'].charsheet
        kwargs['closet'] = kwargs['tableview'].character.closet
        kwargs['fieldnames'] = kwargs['tableview'].fieldnames
        kwargs['statnames'] = kwargs['tableview'].stats
        super(TableRow, self).__init__(**kwargs)

        if self.tab_type != 'char_tab':
            for fieldname in self.fieldnames:
                bwid = TableBody(
                    item=self.item,
                    key=fieldname,
                    is_stat=False,
                    tab_type=self.tab_type,
                    text_getter=lambda: str(getattr(self.item, fieldname)))
                self.closet.register_time_listener(bwid._trigger_upd_text)
                self.add_widget(bwid)
                bwid.upd_text()
        for statname in self.statnames:
            bwid = TableBody(
                item=self.item,
                key=statname,
                is_stat=True,
                tab_type=self.tab_type,
                text_getter=lambda: str(self.item.get_stat(statname)))
            self.closet.register_time_listener(bwid._trigger_upd_text)
            self.add_widget(bwid)
            bwid.upd_text()


class TableContent(GridLayout):
    charsheet = ObjectProperty()
    closet = ObjectProperty()
    adapter = ObjectProperty()
    listview = ObjectProperty()
    tab_type = StringProperty()
    view = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['cols'] = 1
        kwargs['closet'] = kwargs['charsheet'].character.closet
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
            charsheet=self.charsheet,
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
