# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
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
from collections import defaultdict
from functools import partial
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.adapters.models import SelectableDataItem
from kivy.properties import (
    AliasProperty,
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
from .util import trigger


class StoreDataItem(EventDispatcher, SelectableDataItem):
    """Stores ``name`` and ``source``, and remembers whether an item with
    its name is selected or not. If so, that means this very item
    should be selected as well -- probably it got destroyed and
    recreated when we were saving and loading.

    """
    name = ObjectProperty()
    source = StringProperty()
    selectedness = defaultdict(lambda: False)  # class property
    is_selected = AliasProperty(
        lambda self: self.selectedness[str(self.name)],
        lambda self, v: self.selectedness.__setitem__(str(self.name), v)
    )


class StoreButton(ListItemButton):
    """Really just a :class:`ListItemButton` with properties for some
    metadata I might want.

    """
    store = ObjectProperty()
    table = StringProperty('function')
    name = ObjectProperty()
    source = StringProperty()


class StoreAdapter(ListAdapter):
    """:class:`ListAdapter` used to make lists of :class:`StoreButton`.

    """
    table = StringProperty('function')
    store = ObjectProperty()
    loader = ObjectProperty()

    def __init__(self, **kwargs):
        """Initialize with empty ``data``, the :class:`StoreButton` ``cls``,
        an appropriate ``args_converter``,
        ``selection_mode``=``single`` and
        ``allow_empty_selection``=``False``.

        """
        kwargs['data'] = []
        kwargs['cls'] = StoreButton
        kwargs['args_converter'] = lambda i, storedata: {
            'store': self.store,
            'table': self.table,
            'text': str(storedata.name),
            'name': storedata.name,
            'source': storedata.source,
            'on_press': lambda inst: self.loader(),
            'size_hint_y': None,
            'height': 30
        }
        kwargs['selection_mode'] = 'single'
        kwargs['allow_empty_selection'] = False
        kwargs['propagate_selection_to_data'] = True
        super().__init__(**kwargs)

    def get_data(self):
        """Override this to return the appropriate data from my store in a
        list.

        """
        raise NotImplementedError


class FuncStoreAdapter(StoreAdapter):
    """:class:`StoreAdapter` that wraps a function store. Gets function
    names paired with their source code in plaintext.

    """

    def get_data(self, *args):
        """Get data from
        ``LiSE.query.QueryEngine.func_table_name_plaincode``.

        """
        return [
            StoreDataItem(name=k, source=v) for (k, v) in
            self.store.iterplain()
        ]


class StringStoreAdapter(StoreAdapter):
    """:class:`StoreAdapter` that wraps a string store. Gets string names
    paired with their plaintext.

    """

    def get_data(self, *args):
        """Get data from ``LiSE.query.QueryEngine.string_table_lang_items``.

        """
        return [
            StoreDataItem(name=k, source=v) for (k, v) in
            self.store.lang_items()
        ]


class StoreList(FloatLayout):
    """Holder for a :class:`kivy.uix.listview.ListView` that shows what's
    in a store, using one of the StoreAdapter classes.

    """
    table = StringProperty()
    store = ObjectProperty()
    selection = ListProperty([])
    saver = ObjectProperty()
    adapter = ObjectProperty()

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self._trigger_redata = Clock.create_trigger(self.redata)
        self.bind(
            table=self._trigger_remake,
            store=self._trigger_remake
        )
        super().__init__(**kwargs)

    def changed_selection(self, adapter):
        self.saver()
        self.selection = adapter.selection

    def remake(self, *args):
        """Make a new :class:`ListView`, add it to me, and then trigger
        ``redata`` to fill it with useful stuff.

        """
        if None in (self.store, self.table):
            return
        self.clear_widgets()
        self.adapter = self.adapter_cls(
            table=self.table,
            store=self.store,
            loader=self._trigger_redata
        )
        self.adapter.bind(
            on_selection_change=self.changed_selection
        )
        self.bind(
            table=self.adapter.setter('table'),
            store=self.adapter.setter('store')
        )
        self._listview = ListView(
            adapter=self.adapter
        )
        self.add_widget(self._listview)
        self._trigger_redata()

    def redata(self, *args):
        self.adapter.data = self.adapter.get_data()


class FuncStoreList(StoreList):
    adapter_cls = FuncStoreAdapter


class StringStoreList(StoreList):
    adapter_cls = StringStoreAdapter


class StoreEditor(BoxLayout):
    """StoreList on the left with its editor on the right."""
    table = StringProperty()
    store = ObjectProperty()
    storelist = ObjectProperty()
    adapter = ObjectProperty()
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

    def readapter(self, storelist, adapter):
        if not adapter:
            if self.adapter:
                self.adapter.unbind(data=setter('data'))
                self.data = []
            return
        if self.adapter:
            self.adapter.unbind(data=self.setter('data'))
            self.adapter = adapter
        adapter.bind(data=self.setter('data'))

    def remake(self, *args):
        if None in (self.store, self.table):
            return
        self.clear_widgets()
        self.storelist = self.list_cls(
            size_hint_x=0.4,
            table=self.table,
            store=self.store,
            saver=self.save
        )
        if self.storelist.adapter:
            self.adapter = self.storelist.adapter
            self.data = self.adapter.data
        self.storelist.bind(
            selection=self.changed_selection,
            adapter=self.readapter
        )
        self.readapter(self.storelist, self.storelist.adapter)
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

    def redata_reselect(self, *args):
        self.save()
        StoreDataItem.selectedness = defaultdict(lambda: False)
        StoreDataItem.selectedness[self.name] = True
        self.storelist._trigger_redata()
    _trigger_redata_reselect = trigger(redata_reselect)

    def add_editor(self, *args):
        """Construct whatever editor widget I use and add it to myself."""
        raise NotImplementedError

    def save(self, *args):
        """Write my editor's changes to disk."""
        raise NotImplementedError

    def _trigger_save(self, *args):
        Clock.unschedule(self.save)
        Clock.schedule_once(self.save, 0)

    pawn_cfg = ObjectProperty()
    spot_cfg = ObjectProperty()


class StringInput(BoxLayout):
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)
    name = StringProperty()
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


class StringsEditor(StoreEditor):
    list_cls = StringStoreList

    def add_editor(self, *args):
        if self.selection is None:
            Clock.schedule_once(self.add_editor, 0)
            return
        self._editor = StringInput(
            font_name=self.font_name,
            font_size=self.font_size,
            name=self.name,
            source=self.source
        )
        self.bind(
            font_name=self._editor.setter('font_name'),
            font_size=self._editor.setter('font_size'),
            name=self._editor.setter('name'),
            source=self._editor.setter('source')
        )
        self.add_widget(self._editor)

    def save(self, *args):
        self.source = self._editor.source
        if self.name != self._editor.name:
            del self.store[self.name]
            self.name = self._editor.name
        self.store[self.name] = self.source


class FuncsEditor(StoreEditor):
    params = ListProperty(['engine', 'character'])
    subject_type_params = {
        'character': ['engine', 'character'],
        'thing': ['engine', 'character', 'thing'],
        'place': ['engine', 'character', 'place'],
        'portal': ['engine', 'character', 'origin', 'destination']
    }
    subject_type = OptionProperty(
        'character', options=list(subject_type_params.keys())
    )
    list_cls = FuncStoreList

    def on_subject_type(self, *args):
        self.params = self.subject_type_params[self.subject_type]

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

    def save(self, *args):
        if '' in (self._editor.name, self._editor.source):
            return
        if (
                self.name == self._editor.name and
                self.source == self._editor.source
        ):
            return
        if self.name != self._editor.name:
            del self.store[self.name]
        self.name = self._editor.name
        self.source = self._editor.source
        Logger.debug(
            'saving function {}={}'.format(
                self.name,
                self.source
            )
        )
        self.store.set_source(self.name, self.source)
