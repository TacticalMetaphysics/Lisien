# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Widgets for editing the smallish functions you make in ELiDE.

Contains ``FuncsEditor``, a fancied-up ``CodeEditor``;
``FuncsEdBox``, a ``FuncsEditor`` with flair;
and a ``FuncsEdScreen`` for it to go in.

"""
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    StringProperty,
    ListProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from .stores import StoreAdapter, StoreDataItem, StoreEditor, StoreList
from .codeinput import FunctionInput
from .util import trigger


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


class FuncStoreList(StoreList):
    adapter_cls = FuncStoreAdapter


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

    @trigger
    def save(self, *args):
        if not hasattr(self, '_editor'):
            return
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


class FuncsEdBox(BoxLayout):
    funcs_ed = ObjectProperty()
    table = StringProperty()
    store = ObjectProperty()
    data = ListProperty()
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)
    toggle = ObjectProperty()

    def add_func(self, *args):
        if not self.ids.newfuncname.text:
            return
        newname = self.ids.newfuncname.text
        self.ids.newfuncname.text = ''
        self.funcs_ed.save()
        self.funcs_ed.name = newname
        self.ids.funcs_ed.source = 'def {}({}):\n    pass'.format(
            newname,
            ', '.join(self.funcs_ed.params)
        )
        self.funcs_ed._trigger_redata_reselect()

    def dismiss(self, *args):
        self.funcs_ed.save()
        self.toggle()

    def save(self, *args):
        self.funcs_ed.save()

    def subjtyp(self, val):
        if val == 'character':
            self.ids.char.active = True
        elif val == 'thing':
            self.ids.thing.active = True
        elif val == 'place':
            self.ids.place.active = True
        elif val == 'portal':
            self.ids.port.active = True

    def setchar(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'character'

    def setthing(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'thing'

    def setplace(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'place'

    def setport(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'portal'


class FuncsEdScreen(Screen):
    app = ObjectProperty()
    toggle = ObjectProperty()


Builder.load_string("""
<FuncsEdBox>:
    funcs_ed: funcs_ed
    data: funcs_ed.data if funcs_ed else []
    orientation: 'vertical'
    FuncsEditor:
        id: funcs_ed
        size_hint_y: 0.95
        table: root.table
        store: root.store
        font_name: root.font_name
        font_size: root.font_size
        on_subject_type: root.subjtyp(self.subject_type)
    BoxLayout:
        size_hint_y: 0.05
        Button:
            text: '+'
            on_press: root.add_func()
            size_hint_x: 0.2
        TextInput:
            id: newfuncname
            size_hint_x: 0.2
            hint_text: 'New function name'
        BoxLayout:
            size_hint_x: 0.4
            Widget:
                id: spacer
                size_hint_x: 0.05
            BoxLayout:
                size_hint_x: 0.3
                CheckBox:
                    id: char
                    group: 'subj_type'
                    size_hint_x: 0.2
                    active: True
                    on_active: root.setchar(self.active)
                Label:
                    text: 'Character'
                    size_hint_x: 0.8
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
            BoxLayout:
                size_hint_x: 0.23
                CheckBox:
                    id: thing
                    group: 'subj_type'
                    size_hint_x: 0.25
                    on_active: root.setthing(self.active)
                Label:
                    text: 'Thing'
                    size_hint_x: 0.75
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
            BoxLayout:
                size_hint_x: 0.23
                CheckBox:
                    id: place
                    group: 'subj_type'
                    size_hint_x: 0.25
                    on_active: root.setplace(self.active)
                Label:
                    text: 'Place'
                    size_hint_x: 0.75
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
            BoxLayout:
                size_hint_x: 0.24
                CheckBox:
                    id: port
                    group: 'subj_type'
                    size_hint_x: 0.25
                    on_active: root.setport(self.active)
                Label:
                    text: 'Portal'
                    size_hint_x: 0.75
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
        Button:
            text: 'Close'
            on_press: root.dismiss()
            size_hint_x: 0.2
<FuncsEdScreen>:
    TabbedPanel:
        default_tab: action
        TabbedPanelItem:
            id: trigger
            text: 'Trigger'
            on_state: triggers.save()
            FuncsEdBox:
                id: triggers
                toggle: root.toggle
                table: 'triggers'
                store: root.app.engine.trigger
                on_data: root.app.rules.rulesview._trigger_pull_triggers()
        TabbedPanelItem:
            id: prereq
            text: 'Prereq'
            on_state: prereqs.save()
            FuncsEdBox:
                id: prereqs
                toggle: root.toggle
                table: 'prereqs'
                store: root.app.engine.prereq
                on_data: root.app.rules.rulesview._trigger_pull_prereqs()
        TabbedPanelItem:
            id: action
            text: 'Action'
            on_state: actions.save()
            FuncsEdBox:
                id: actions
                toggle: root.toggle
                table: 'actions'
                store: root.app.engine.action
                on_data: root.app.rules.rulesview._trigger_pull_actions()
        TabbedPanelItem:
            id: other
            text: 'Other'
            on_state: others.save()
            FuncsEdBox:
                id: others
                toggle: root.toggle
                table: 'functions'
                store: root.app.engine.function
""")
