# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Editors for textual data in the database.

The data is accessed via a "store" -- a mapping onto the table, used
like a dictionary. Each of the widgets defined here,
:class:`StringsEditor` and :class:`FuncsEditor`, displays a list of
buttons with which the user may select one of the keys in the store,
and edit its value in a text box.

Though they retrieve data the same way, these widgets have different
ways of saving data -- the contents of the :class:`FuncsEditor` input
will be compiled into Python bytecode, stored along with the source
code.

"""
from functools import partial

from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.togglebutton import ToggleButton

from kivy.properties import (
    AliasProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from .util import trigger


class StoreButton(ToggleButton):
    store = ObjectProperty()
    table = StringProperty('functions')
    name = StringProperty()
    source = StringProperty()


class StoreList(RecycleView):
    """Holder for a :class:`kivy.uix.listview.ListView` that shows what's
    in a store, using one of the StoreAdapter classes.

    """
    table = StringProperty()
    store = ObjectProperty()
    selection = ObjectProperty()

    def munge(self, name, source):
        return {
            'store': self.store,
            'table': self.table,
            'text': str(name),
            'name': name,
            'source': source,
            'on_state': self.select,
            'size_hint_y': None,
            'height': 30
        }

    def redata(self, *args):
        self.data = [self.munge(data) for data in self.iter_data()]
    _trigger_redata = trigger(redata)

    def select(self, inst, val):
        if val == 'down':
            self.selection = inst.name


class StoreEditor(BoxLayout):
    """StoreList on the left with its editor on the right."""
    table = StringProperty()
    store = ObjectProperty()
    storelist = ObjectProperty()
    data = ListProperty([])
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)
    selection = ListProperty([])
    oldsel = ListProperty([])
    name = StringProperty()
    source = StringProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            table=self._trigger_remake,
            store=self._trigger_remake
        )

    def remake(self, *args):
        if None in (self.store, self.table):
            return
        self.clear_widgets()
        self.storelist = self.list_cls(
            size_hint_x=0.4,
            table=self.table,
            store=self.store
        )
        self.storelist.bind(
            selection=self.changed_selection
        )
        self.bind(
            table=self.storelist.setter('table'),
            store=self.storelist.setter('store')
        )
        self.add_widget(self.storelist)
        self.add_editor()
    _trigger_remake = trigger(remake)

    def changed_selection(self, *args):
        if self.storelist.selection:
            self.selection = self.storelist.selection
            self.name = self.selection[0].name
            self.source = self.selection[0].source

    def add_editor(self, *args):
        """Construct whatever editor widget I use and add it to myself."""
        raise NotImplementedError

    def save(self, *args):
        """Write my editor's changes to disk."""
        raise NotImplementedError

    pawn_cfg = ObjectProperty()
    spot_cfg = ObjectProperty()


class StringInput(BoxLayout):
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)
    source = StringProperty()

    def _get_name(self):
        if 'stringname' not in self.ids:
            return ''
        return self.ids.stringname.text

    def _set_name(self, v, *args):
        if 'stringname' not in self.ids:
            Clock.schedule_once(partial(self._set_name, v), 0)
            return
        self.ids.stringname.text = v

    name = AliasProperty(_get_name, _set_name)

    def _get_source(self):
        if 'string' not in self.ids:
            return ''
        return self.ids.string.text

    def _set_source(self, v, *args):
        if 'string' not in self.ids:
            Clock.schedule_once(partial(self._set_source, v), 0)
            return
        self.ids.string.text = v

    source = AliasProperty(_get_source, _set_source)
