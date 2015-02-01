# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Widget for editing functions in a function store.

Has a scrolling menu of functions in the store on the left, an editor
view on the right, and it autosaves.

"""
from kivy.clock import Clock
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    StringProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.listview import ListView, ListItemButton
from kivy.adapters.listadapter import ListAdapter
from .codeinput import FunctionInput
from .stringinput import StringInput


class StoreButton(ListItemButton):
    store = ObjectProperty()
    table = StringProperty('function')
    name = ObjectProperty()
    source = StringProperty()


class StoreAdapter(ListAdapter):
    table = StringProperty('function')
    store = ObjectProperty()
    callback = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['data'] = []
        kwargs['cls'] = StoreButton
        kwargs['args_converter'] = lambda i, nametxt: {
            'store': self.store,
            'table': self.table,
            'name': nametxt[0],
            'source': nametxt[1],
            'on_press': lambda inst: self.callback(*nametxt)
        }
        kwargs['selection_mode'] = 'single'
        kwargs['allow_empty_selection'] = False
        self._trigger_redata = Clock.create_trigger(self.redata)
        self.bind(
            table=self._trigger_redata,
            store=self._trigger_redata
        )
        super().__init__(**kwargs)


class FuncStoreAdapter(StoreAdapter):
    def on_store(self, *args):
        if self.store is None:
            return
        self.store.listener(self._trigger_redata)

    def redata(self, *args):
        if self.store is None:
            return
        self.data = self.store.db.func_table_name_plaincode(self.table)


class StringStoreAdapter(StoreAdapter):
    def on_store(self, *args):
        if self.store is None:
            return
        self.store.listener(self._trigger_redata)
        self.store.lang_listener(self._trigger_redata)

    def redata(self, *args):
        if self.store is None:
            return
        self.data = self.store.db.string_table_lang_items(
            self.table, self.store.language
        )


class StoreList(Widget):
    table = StringProperty()
    store = ObjectProperty()
    selection = ListProperty()
    callback = ObjectProperty()

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self.bind(
            table=self._trigger_remake,
            store=self._trigger_remake
        )
        super().__init__(**kwargs)

    def changed_selection(self, adapter):
        self.selection = adapter.selection

    def remake(self, *args):
        if None in (self.store, self.table):
            return
        self.clear_widgets()
        self._adapter = self.adapter_cls(
            table=self.table,
            store=self.store,
            callback=self.callback
        )
        self._adapter.bind(
            on_selection_change=self.changed_selection
        )
        self.bind(
            table=self._adapter.setter('table'),
            store=self._adapter.setter('store'),
            callback=self._adapter.setter('callback')
        )
        self._listview = ListView(
            adapter=self._adapter,
            size_hint=self.size_hint,
            pos_hint=self.pos_hint,
            size=self.size,
            pos=self.pos
        )
        self.bind(
            size_hint=self._listview.setter('size_hint'),
            pos_hint=self._listview.setter('pos_hint'),
            size=self._listview.setter('size'),
            pos=self._listview.setter('pos')
        )
        self.add_widget(self._listview)


class FuncStoreList(StoreList):
    adapter_cls = FuncStoreAdapter


class StringStoreList(StoreList):
    adapter_cls = StringStoreAdapter


class StoreEditor(BoxLayout):
    table = StringProperty()
    store = ObjectProperty()
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)
    selection = ListProperty()

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self.bind(
            table=self._trigger_remake,
            store=self._trigger_remake
        )
        super().__init__(**kwargs)

    def remake(self, *args):
        if None in (self.store, self.table):
            return
        self.clear_widgets()
        self._list = self.list_cls(
            table=self.table,
            store=self.store,
            callback=self.save_and_load
        )
        self._list.bind(selection=self.setter('selection'))
        self.bind(
            table=self._list.setter('table'),
            store=self._list.setter('store')
        )
        self.add_widget(self._list)
        self.add_editor()

    def add_editor(self, *args):
        """Construct whatever editor widget I use and add it to myself."""
        raise NotImplementedError

    def save_and_load(self, name, source):
        """Save what's in the editor currently, and replace it with
        ``source``.

        """
        raise NotImplementedError


class StringsEditor(StoreEditor):
    list_cls = StringStoreList

    def add_editor(self, *args):
        if self.selection is None:
            Clock.schedule_once(self.add_editor, 0)
            return
        self._editor = StringInput(
            name=self.selection[0].name,
            text=self.selection[0].source,
            font_name=self.font_name,
            font_size=self.font_size
        )
        self.bind(
            font_name=self._editor.setter('font_name'),
            font_size=self._editor.setter('font_size'),
        )
        self.add_widget(self._editor)

    def on_selection(self, *args):
        self._editor.name = self.selection[0].name
        self._editor.text = self.selection[0].source


class FuncsEditor(StoreEditor):
    params = ListProperty(['engine', 'character'])
    subject_type = OptionProperty(
        'character', options=['character', 'thing', 'place', 'portal']
    )
    list_cls = FuncStoreList

    def on_subject_type(self, *args):
        self.params = {
            'character': ['engine', 'character'],
            'thing': ['engine', 'character', 'thing'],
            'place': ['engine', 'character', 'place'],
            'portal': ['engine', 'character', 'origin', 'destination']
        }[self.subject_type]

    def add_editor(self, *args):
        if None in (self.selection, self.params):
            Clock.schedule_once(self.add_editor, 0)
            return
        self._editor = FunctionInput(
            font_name=self.font_name,
            font_size=self.font_size,
            name=self.selection[0].name,
            params=self.params,
            source=self.selection[0].source
        )
        self.bind(
            font_name=self._editor.setter('font_name'),
            font_size=self._editor.setter('font_size'),
            params=self._editor.setter('params')
        )
        self.add_widget(self._editor)

    def on_selection(self, *args):
        self._editor.name = self.selection[0].name
        self._editor.source = self.selection[0].source
