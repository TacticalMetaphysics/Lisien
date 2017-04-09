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
import re
import string
from functools import partial

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.togglebutton import ToggleButton

from kivy.properties import (
    AliasProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    StringProperty
)
from .util import trigger


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class RecycleToggleButton(ToggleButton, RecycleDataViewBehavior):
    index = NumericProperty()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        if is_selected and index == self.index:
            self.state = 'down'
        else:
            self.state = 'normal'


class StoreButton(RecycleToggleButton):
    store = ObjectProperty()
    table = StringProperty('functions')
    name = StringProperty()
    source = StringProperty()
    select = ObjectProperty()

    def on_parent(self, *args):
        if self.name == '+':
            self.state = 'down'
            self.select(self)

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
        i, name = datum
        return {
            'store': self.store,
            'table': self.table,
            'text': str(name),
            'name': name,
            'select': self.select,
            'index': i
        }

    def _iter_keys(self):
        yield '+'
        yield from sorted(self.store.keys())

    def redata(self, select_name=None, *args):
        if not self.table or not self.store:
            Clock.schedule_once(self.redata)
            return
        self.data = list(map(self.munge, enumerate(self._iter_keys())))
        if select_name:
            self._trigger_select_name(select_name)

    def _trigger_redata(self, select_name=None, *args):
        part = partial(self.redata, select_name, *args)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def select(self, inst):
        self.selection = inst
        for boxl in self.children:
            for child in boxl.children:
                if child != inst and child.state == 'down':
                    child.state = 'normal'

    def select_name(self, name, *args):
        for boxl in self.children:
            for child in boxl.children:
                if child.text == name:
                    child.state = 'down'
                    self.selection = child
                else:
                    child.state = 'normal'

    def _trigger_select_name(self, name):
        part = partial(self.select_name, name)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

class StringsEdScreen(Screen):
    toggle = ObjectProperty()
    language = StringProperty('eng')
    language_setter = ObjectProperty()

    def set_language(self, lang):
        # a little redundant
        self.language_setter(lang)
        self.language = lang


class Editor(BoxLayout):
    name_wid = ObjectProperty()
    store = ObjectProperty()
    # This next is the trigger on the EdBox, which may redata the StoreList
    _trigger_save = ObjectProperty()
    _trigger_delete = ObjectProperty()

    def save(self, *args):
        if not (self.name_wid and self.store):
            Logger.debug("{}: Not saving, missing name_wid or store".format(type(self).__name__))
            return
        if not (self.name_wid.text or self.name_wid.hint_text):
            Logger.debug("{}: Not saving, no name".format(type(self).__name__))
            return
        if self.name_wid.text and self.name_wid.text[0] in string.digits + string.whitespace + string.punctuation:
            # TODO alert the user to invalid name
            Logger.debug("{}: Not saving, invalid name".format(type(self).__name__))
            return
        do_redata = self.name_wid.hint_text == ''
        if self.name_wid.text not in self.store:
            do_redata = self.name_wid.text
        if (
            self.name_wid.text and
            self.name_wid.hint_text and
            self.name_wid.hint_text != self.name_wid.text and
            self.name_wid.hint_text in self.store
        ):
            del self.store[self.name_wid.hint_text]
            do_redata = self.name_wid.text
        if self.name_wid.text and (
            self.name_wid.text not in self.store or
            self.source != self.store[self.name_wid.text]
        ):
            Logger.debug("{}: Saving!".format(type(self).__name__))
            self.store[self.name_wid.text] = self.source
        else:
            self.store[self.name_wid.hint_text] = self.source
        return do_redata

    def delete(self, *args):
        key = self.name_wid.text or self.name_wid.hint_text
        if key not in self.store:
            # TODO feedback about missing key
            return
        del self.store[key]
        try:
            return min(kee for kee in self.store if kee > key)
        except ValueError:
            return '+'


class StringInput(Editor):
    def _get_name(self):
        if self.name_wid:
            return self.name_wid.text

    def _set_name(self, v, *args):
        if not self.name_wid:
            Clock.schedule_once(partial(self._set_name, v), 0)
            return
        self.name_wid.text = v

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


class EdBox(BoxLayout):
    storelist = ObjectProperty()
    editor = ObjectProperty()
    table = StringProperty()
    store = ObjectProperty()
    data = ListProperty()
    toggle = ObjectProperty()
    name = StringProperty('')

    def on_storelist(self, *args):
        self.storelist.bind(selection=self._pull_from_storelist)

    @trigger
    def _pull_from_storelist(self, *args):
        self.save()
        # The + button at the top is for adding an entry yet unnamed, so don't display hint text for it
        self.editor.name_wid.hint_text = self.name = self.storelist.selection.name.strip('+')
        self.editor.name_wid.text = ''
        try:
            self.editor.source = self.store[self.name]
        except KeyError:
            self.editor.source = self.get_default_text(self.name)
        if hasattr(self, '_lock_save'):
            del self._lock_save

    def dismiss(self, *args):
        self.save()
        self.toggle()

    def save(self, *args):
        if not self.editor:
            return
        if hasattr(self, '_lock_save'):
            return
        self._lock_save = True
        save_select = self.editor.save()
        if save_select:
            name = save_select if isinstance(save_select, str) else getattr(self, '_select_name', None)
            self.storelist.redata(name)
        else:
            del self._lock_save

    def _trigger_save(self, name=None):
        self._select_name = name
        Clock.unschedule(self.save)
        Clock.schedule_once(self.save, 0)

    def delete(self, *args):
        if not self.editor:
            return
        if hasattr(self, '_lock_save'):
            return
        self._lock_save = True
        del_select = self.editor.delete()
        if del_select:
            self.storelist.redata(del_select)
        else:
            del self._lock_save
    _trigger_delete = trigger(delete)


class StringNameInput(TextInput):
    _trigger_save = ObjectProperty()

    def on_focus(self, inst, val, *largs):
        if self.text and not val:
            self._trigger_save(self.text)


class StringsEdBox(EdBox):
    language = StringProperty('eng')

    @staticmethod
    def get_default_text(newname):
        return ''


sig_ex = re.compile('^ *def .+?\((.+)\):$')


class FunctionNameInput(TextInput):
    _trigger_save = ObjectProperty()

    def insert_text(self, s, from_undo=False):
        if self.text == '':
            if s[0] not in (string.ascii_letters + '_'):
                return
        return super().insert_text(
            ''.join(c for c in s if c in (string.ascii_letters + string.digits + '_'))
        )

    def on_focus(self, inst, val, *largs):
        if not val:
            self._trigger_save(self.text)


def sanitize_source(v, spaces=4):
    lines = v.split('\n')
    if not lines:
        return tuple(), ''
    firstline = lines[0].lstrip()
    if firstline == '' or firstline[0] == '@':
        del lines[0]
    if not lines:
        return tuple(), ''
    # how indented is it?
    for ch in lines[0]:
        if ch == ' ':
            spaces += 1
        elif ch == '\t':
            spaces += 4
        else:
            break
    params = tuple(
        parm.strip() for parm in
        sig_ex.match(lines[0]).group(1).split(',')
    )
    del lines[0]
    if not lines:
        return params, ''
    # hack to allow 'empty' functions
    if lines and lines[-1].strip() == 'pass':
        del lines[-1]
    return params, '\n'.join(line[spaces:] for line in lines)


class FuncEditor(Editor):
    """The editor widget for working with any particular function.

    Contains a one-line field for the function's name; a multi-line
    field for its code; and radio buttons to select its signature
    from among those permitted.

    """
    storelist = ObjectProperty()
    codeinput = ObjectProperty()
    _text = StringProperty()
    subject_type_params = {
        'character': ('engine', 'character'),
        'thing': ('engine', 'character', 'thing'),
        'place': ('engine', 'character', 'place'),
        'portal': ('engine', 'character', 'origin', 'destination')
    }
    params_subject_type = {v: k for k, v in subject_type_params.items()}
    subject_type = OptionProperty(
        'character', options=list(subject_type_params.keys())
    )

    def _subj_type_from_params(self, v):
        self.subject_type = self.params_subject_type[v]

    params = AliasProperty(
        lambda self: self.subject_type_params[self.subject_type],
        _subj_type_from_params,
        bind=('subject_type',)
    )

    def _get_source(self):
        code = self.get_default_text(self.name_wid.text or self.name_wid.hint_text)
        for line in self._text.split('\n'):
            code += (' ' * 4 + line + '\n')
        return code.rstrip(' \n\t')

    def _set_source(self, v):
        if not self.codeinput:
            Clock.schedule_once(partial(self._set_source, v), 0)
            return
        self.codeinput.unbind(text=self.setter('_text'))
        self.params, self.codeinput.text = sanitize_source(v)
        self.codeinput.bind(text=self.setter('_text'))

    source = AliasProperty(_get_source, _set_source, bind=('params', '_text'))

    def get_default_text(self, name):
        if not name or name == '+':
            name = 'a'
        return "def {}({}):\n".format(name, ', '.join(self.params))

    def on_codeinput(self, *args):
        self._text = self.codeinput.text
        self.codeinput.bind(text=self.setter('_text'))


class FuncsEdBox(EdBox):
    """Widget for editing the Python source of funcs to be used in LiSE sims.

    Contains a list of functions in the store it's about, next to a
    FuncEditor showing the source of the selected one, and some
    controls on the bottom that let you add, delete, and rename the function,
    or close the screen.

    """

    def get_default_text(self, newname):
        return self.editor.get_default_text(newname)

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
        self.editor.subject_type = 'character'

    def setthing(self, active):
        if not active:
            return
        self.editor.subject_type = 'thing'

    def setplace(self, active):
        if not active:
            return
        self.editor.subject_type = 'place'

    def setport(self, active):
        if not active:
            return
        self.editor.subject_type = 'portal'


class FuncsEdScreen(Screen):
    toggle = ObjectProperty()


Builder.load_string("""
#: import py3lexer pygments.lexers.Python3Lexer
<StoreButton>:
    size_hint_y: None
    height: 30
<StoreList>:
    viewclass: 'StoreButton'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        size_hint_y: None
        orientation: 'vertical'
<StringInput>:
    name_wid: stringname
    orientation: 'vertical'
    BoxLayout:
        size_hint_y: 0.05
        Label:
            id: title
            text: 'Title: '
            size_hint_x: None
            width: self.texture_size[0]
        StringNameInput:
            id: stringname
            multiline: False
            write_tab: False
            _trigger_save: root._trigger_save
        Button:
            text: 'del'
            size_hint_x: 0.1
            on_press: root._trigger_delete()
    TextInput:
        id: string
        disabled: root.store is None or (stringname.hint_text not in root.store and not stringname.text)
<StringsEdBox>:
    editor: strings_ed
    storelist: strings_list
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        StoreList:
            id: strings_list
            size_hint_x: 0.2
            table: root.table
            store: root.store
        StringInput:
            id: strings_ed
            store: root.store
            _trigger_save: root._trigger_save
            _trigger_delete: root._trigger_delete
<StringsEdScreen>:
    name: 'strings'
    BoxLayout:
        orientation: 'vertical'
        StringsEdBox:
            id: edbox
            toggle: root.toggle
            table: 'strings'
            store: app.engine.string
            language: root.language
        BoxLayout:
            size_hint_y: 0.05
            Button:
                text: 'Close'
                on_press: edbox.dismiss()
            Label:
                text_size: self.size
                halign: 'right'
                valign: 'middle'
                text: 'Language: '
            TextInput:
                id: language
                hint_text: root.language
                on_text_validate: root.set_language(self.text)
<Py3CodeInput@CodeInput>:
    lexer: py3lexer()
<FuncEditor>:
    codeinput: code
    name_wid: funname
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: funname.height
        Py3CodeInput:
            id: imafunction
            text: 'def'
            disabled: True
            size_hint: (None, None)
            height: self.line_height + self.font_size
            width: self.font_size * len(self.text)
            background_disabled_normal: ''
            disabled_foreground_color: self.foreground_color
        FunctionNameInput:
            id: funname
            size_hint: (0.4, None)
            height: self.line_height + self.font_size
            multiline: False
            write_tab: False
            _trigger_save: root._trigger_save
        Py3CodeInput:
            id: params
            text: '(' + ', '.join(root.params) + '):'
            disabled: True
            size_hint_y: None
            height: self.line_height + self.font_size
            background_disabled_normal: ''
            disabled_foreground_color: self.foreground_color
        Button:
            text: 'del'
            size_hint_x: 0.1
            on_press: root._trigger_delete()
    BoxLayout:
        orientation: 'horizontal'
        Label:
            canvas:
                Color:
                    rgba: params.background_color
                Rectangle:
                    pos: self.pos
                    size: self.size
                Color:
                    rgba: [1., 1., 1., 1.]
            # PEP8 standard indentation width is 4 spaces
            text: ' ' * 4
            size_hint_x: None
            width: self.texture_size[0]
        Py3CodeInput:
            id: code
<FuncsEdBox>:
    editor: funcs_ed
    storelist: funcs_list
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        BoxLayout:
            orientation: 'vertical'
            size_hint_x: 0.2
            StoreList:
                id: funcs_list
                table: root.table
                store: root.store
        FuncEditor:
            id: funcs_ed
            table: root.table
            store: root.store
            storelist: funcs_list
            _trigger_save: root._trigger_save
            _trigger_delete: root._trigger_delete
            on_subject_type: root.subjtyp(self.subject_type)
    BoxLayout:
        size_hint_y: 0.05
        Button:
            text: 'Close'
            on_press: root.dismiss()
            size_hint_x: 0.2
        Widget:
            id: spacer
        BoxLayout:
            size_hint_x: 0.6
            BoxLayout:
                size_hint_x: 0.32
                CheckBox:
                    id: char
                    group: 'subj_type'
                    size_hint_x: 0.2
                    on_active: root.setchar(self.active)
                Label:
                    text: 'Character'
                    size_hint_x: 0.8
                    text_size: self.size
                    halign: 'left'
                    valign: 'middle'
            BoxLayout:
                size_hint_x: 0.22
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
                size_hint_x: 0.22
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
<FuncsEdScreen>:
    name: 'funcs'
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
                store: app.engine.trigger
                on_data: app.rules.rulesview._trigger_pull_triggers()
        TabbedPanelItem:
            id: prereq
            text: 'Prereq'
            on_state: prereqs.save()
            FuncsEdBox:
                id: prereqs
                toggle: root.toggle
                table: 'prereqs'
                store: app.engine.prereq
                on_data: app.rules.rulesview._trigger_pull_prereqs()
        TabbedPanelItem:
            id: action
            text: 'Action'
            on_state: actions.save()
            FuncsEdBox:
                id: actions
                toggle: root.toggle
                table: 'actions'
                store: app.engine.action
                on_data: app.rules.rulesview._trigger_pull_actions()
""")