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
from kivy.uix.floatlayout import FloatLayout
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
            'text': str(nametxt[0]),
            'name': nametxt[0],
            'source': nametxt[1],
            'on_press': lambda inst: self.callback(*nametxt),
            'size_hint_y': None,
            'height': 30
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
        self.data = list(self.store.db.func_table_name_plaincode(self.table))


class StringStoreAdapter(StoreAdapter):
    def on_store(self, *args):
        if self.store is None:
            return
        self.store.listener(self._trigger_redata)
        self.store.lang_listener(self._trigger_redata)

    def redata(self, *args):
        if self.store is None:
            return
        if self.selection:
            self._reselect = self.selection[0].text
        self.data = list(
            self.store.db.string_table_lang_items(
                self.table, self.store.language
            )
        )


class StoreList(FloatLayout):
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
            on_selection_change=self.changed_selection,
            data=self.reselect
        )
        self.bind(
            table=self._adapter.setter('table'),
            store=self._adapter.setter('store'),
            callback=self._adapter.setter('callback')
        )
        self._listview = ListView(
            adapter=self._adapter
        )
        self.add_widget(self._listview)

    def reselect(self, *args):
        if not hasattr(self._adapter, '_reselect'):
            return
        if self._adapter.selection[0].text == self._adapter._reselect:
            del self._adapter._reselect
            return
        grid = self._listview.children[0].children[0].children[0]
        for button in grid.children:
            if button.text == self._adapter._reselect:
                self._adapter.select_list([button], extend=False)
                return


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
            size_hint_x=0.4,
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
    name = StringProperty()
    source = StringProperty()

    def on_params(self, *args):
        if self.params == ['engine', 'character']:
            self.subject_type = 'character'
        elif self.params == ['engine', 'character', 'thing']:
            self.subject_type = 'thing'
        elif self.params == ['engine', 'character', 'place']:
            self.subject_type = 'place'
        elif self.params == ['engine', 'character', 'origin', 'destination']:
            self.subject_type = 'portal'
        else:
            raise ValueError(
                "Unsupported function signature: {}".format(self.params)
            )

    def add_editor(self, *args):
        if None in (self.selection, self.params):
            Clock.schedule_once(self.add_editor, 0)
            return
        self._editor = FunctionInput(
            font_name=self.font_name,
            font_size=self.font_size,
            params=self.params,
        )
        self.bind(
            font_name=self._editor.setter('font_name'),
            font_size=self._editor.setter('font_size'),
            name=self._editor.setter('name'),
            source=self._editor.setter('source')
        )
        self._editor.bind(params=self.setter('params'))
        self.add_widget(self._editor)

    def on_selection(self, *args):
        if self.selection == []:
            return
        self.name = self.selection[0].name
        self.source = self.selection[0].source
