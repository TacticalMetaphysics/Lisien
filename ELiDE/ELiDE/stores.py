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
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView

from kivy.properties import (
    AliasProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from .util import trigger, RecycleToggleButton


class StoreButton(RecycleToggleButton):
    store = ObjectProperty()
    table = StringProperty('functions')
    name = StringProperty()
    source = StringProperty()
    select = ObjectProperty()

    def on_state(self, *args):
        if self.state == 'down':
            self.select(self)


class StoreList(RecycleView):
    """Holder for a :class:`kivy.uix.listview.ListView` that shows what's
    in a store, using one of the StoreAdapter classes.

    """
    table = StringProperty()
    store = ObjectProperty()
    selection = ObjectProperty()

    def __init__(self, **kwargs):
        self.bind(table=self._trigger_redata, store=self._trigger_redata)
        super().__init__(**kwargs)

    def munge(self, datum):
        i, (name, source) = datum
        return {
            'store': self.store,
            'table': self.table,
            'text': str(name),
            'name': name,
            'source': source,
            'select': self.select,
            'size_hint_y': None,
            'height': 30,
            'index': i
        }

    def redata(self, *args):
        if not self.table or not self.store:
            Clock.schedule_once(self.redata)
            return
        self.data = list(map(self.munge, enumerate(self.iter_data())))
    _trigger_redata = trigger(redata)

    def select(self, inst):
        self.selection = inst

    def iter_data(self):
        raise NotImplementedError



Builder.load_string("""
<StoreList>:
    viewclass: 'StoreButton'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        size_hint_y: None
        orientation: 'vertical'
""")