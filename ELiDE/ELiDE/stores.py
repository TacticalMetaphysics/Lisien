# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Editors for textual data in the database.

The data is accessed via a "store" -- a mapping onto the table, used
like a dictionary. Each of the widgets defined here,
:class:`StringsEditor` and :class:`FuncsEditor`, displays a list of
buttons with which the user may select one of the keys in the store,
and edit its value in a text box.

"""
import re
import string
from functools import partial
from ast import parse
from textwrap import indent, dedent

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.togglebutton import ToggleButton

from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from .util import trigger


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
            self.select(self.index)

    def on_state(self, *args):
        if self.state == 'down':
            self.select(self.index)


class StoreList(RecycleView):
    """Holder for a :class:`kivy.uix.listview.ListView` that shows what's
    in a store, using one of the StoreAdapter classes.

    """
    table = StringProperty()
    store = ObjectProperty()
    selection_name = StringProperty()
    boxl = ObjectProperty()

    def __init__(self, **kwargs):
        self.bind(table=self._trigger_redata)
        self._i2name = {}
        self._name2i = {}
        super().__init__(**kwargs)

    def on_store(self, *args):
        self.store.connect(self._trigger_redata)
        self.redata()

    def on_boxl(self, *args):
        self.boxl.bind(selected_nodes=self._pull_selection)

    def _pull_selection(self, *args):
        if not self.boxl.selected_nodes:
            return
        self.selection_name = self._i2name[self.boxl.selected_nodes[0]]

    def munge(self, datum):
        i, name = datum
        self._i2name[i] = name
        self._name2i[name] = i
        return {
            'store': self.store,
            'table': self.table,
            'text': str(name),
            'name': name,
            'select': self.ids.boxl.select_node,
            'index': i
        }

    def _iter_keys(self):
        yield '+'
        yield from sorted(self.store._cache.keys())

    def redata(self, *args, **kwargs):
        select_name = kwargs.get('select_name')
        if not self.table or self.store is None:
            Clock.schedule_once(self.redata)
            return
        self.data = list(map(self.munge, enumerate(self._iter_keys())))
        if select_name:
            self._trigger_select_name(select_name)

    def _trigger_redata(self, *args, **kwargs):
        part = partial(self.redata, *args, **kwargs)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def select_name(self, name, *args):
        self.boxl.select_node(self._name2i[name])

    def _trigger_select_name(self, name):
        part = partial(self.select_name, name)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)


class LanguageInput(TextInput):
    screen = ObjectProperty()

    def on_focus(self, instance, value, *largs):
        if not value:
            if self.screen.language != self.text:
                self.screen.language_setter(self.text)
                self.screen.language = self.text
            self.text = ''

class StringsEdScreen(Screen):
    toggle = ObjectProperty()
    language = StringProperty('eng')
    language_setter = ObjectProperty()
    edbox = ObjectProperty()

    def on_language(self, *args):
        if self.edbox is None:
            Clock.schedule_once(self.on_language, 0)
            return
        self.edbox.storelist.redata()

    def save(self, *args):
        if self.edbox is None:
            Clock.schedule_once(self.save, 0)
            return
        self.edbox.save()


class Editor(BoxLayout):
    name_wid = ObjectProperty()
    store = ObjectProperty()
    disable_text_input = BooleanProperty(False)
    # This next is the trigger on the EdBox, which may redata the StoreList
    _trigger_save = ObjectProperty()
    _trigger_delete = ObjectProperty()

    def save(self, *args):
        """Put text in my store, return True if it changed"""
        if self.name_wid is None or self.store is None:
            Logger.debug("{}: Not saving, missing name_wid or store".format(type(self).__name__))
            return
        if not (self.name_wid.text or self.name_wid.hint_text):
            Logger.debug("{}: Not saving, no name".format(type(self).__name__))
            return
        if self.name_wid.text and self.name_wid.text[0] in string.digits + string.whitespace + string.punctuation:
            # TODO alert the user to invalid name
            Logger.debug("{}: Not saving, invalid name".format(type(self).__name__))
            return
        if hasattr(self, '_do_parse'):
            try:
                parse(self.source)
            except SyntaxError:
                # TODO alert user to invalid source
                Logger.debug("{}: Not saving, couldn't parse".format(type(self).__name__))
                return
        do_redata = False
        if self.name_wid.text:
            if (
                self.name_wid.hint_text and
                self.name_wid.hint_text != self.name_wid.text and
                hasattr(self.store, self.name_wid.hint_text)
            ):
                delattr(self.store, self.name_wid.hint_text)
                do_redata = True
            if (
                not hasattr(self.store, self.name_wid.text) or
                getattr(self.store, self.name_wid.text) != self.source
            ):
                Logger.debug("{}: Saving!".format(type(self).__name__))
                setattr(self.store, self.name_wid.text, self.source)
                do_redata = True
        elif self.name_wid.hint_text:
            if (
                not hasattr(self.store, self.name_wid.hint_text) or
                getattr(self.store, self.name_wid.hint_text) != self.source
            ):
                Logger.debug("{}: Saving!".format(type(self).__name__))
                setattr(self.store, self.name_wid.hint_text, self.source)
                do_redata = True
        return do_redata

    def delete(self, *args):
        key = self.name_wid.text or self.name_wid.hint_text
        if not hasattr(self.store, key):
            # TODO feedback about missing key
            return
        delattr(self.store, key)
        try:
            return min(kee for kee in dir(self.store) if kee > key)
        except ValueError:
            return '+'


class StringInput(Editor):
    validate_name_input = ObjectProperty()

    def on_name_wid(self, *args):
        if not self.validate_name_input:
            Clock.schedule_once(self.on_name_wid, 0)
            return
        self.name_wid.bind(text=self.validate_name_input)

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
    disable_text_input = BooleanProperty(False)

    def on_storelist(self, *args):
        self.storelist.bind(selection_name=self._pull_from_storelist)

    @trigger
    def validate_name_input(self, *args):
        self.disable_text_input = not (self.valid_name(self.editor.name_wid.hint_text) or self.valid_name(self.editor.name_wid.text))

    @trigger
    def _pull_from_storelist(self, *args):
        self.save()
        # The + button at the top is for adding an entry yet unnamed, so don't display hint text for it
        self.editor.name_wid.hint_text = self.storelist.selection_name.strip('+')
        self.editor.name_wid.text = ''
        try:
            self.editor.source = getattr(
                self.store, self.editor.name_wid.hint_text
            )
        except AttributeError:
            self.editor.source = self.get_default_text(self.editor.name_wid.hint_text)
        self.disable_text_input = not self.valid_name(self.editor.name_wid.hint_text)
        if hasattr(self, '_lock_save'):
            del self._lock_save

    def dismiss(self, *args):
        self.save()
        self.toggle()

    def save(self, *args, name=None):
        if not self.editor:
            return
        if hasattr(self, '_lock_save'):
            return
        self._lock_save = True
        save_select = self.editor.save()
        if save_select:
            self.storelist.redata(select_name=name)
        else:
            del self._lock_save

    def _trigger_save(self, name=None):
        part = partial(self.save, name=name)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

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

    def on_store(self, *args):
        pass


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

    @staticmethod
    def valid_name(name):
        return name and name[0] != '+'


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


def munge_source(v):
    """Take Python source code, return its parameters and the rest of it dedented"""
    lines = v.split('\n')
    if not lines:
        return tuple(), ''
    firstline = lines[0].lstrip()
    while firstline == '' or firstline[0] == '@':
        del lines[0]
        firstline = lines[0].lstrip()
    if not lines:
        return tuple(), ''
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
    return params, dedent('\n'.join(lines))


class FuncEditor(Editor):
    """The editor widget for working with any particular function.

    Contains a one-line field for the function's name and a multi-line
    field for its code.

    """
    storelist = ObjectProperty()
    codeinput = ObjectProperty()
    params = ListProperty(['obj'])
    _text = StringProperty()
    _do_parse = True

    def _get_source(self):
        code = self.get_default_text(self.name_wid.text or self.name_wid.hint_text)
        if self._text:
            code += indent(self._text, ' ' * 4)
        else:
            code += ' ' * 4 + 'pass'
        return code.rstrip(' \n\t')

    def _set_source(self, v):
        if not self.codeinput:
            Clock.schedule_once(partial(self._set_source, v), 0)
            return
        self.codeinput.unbind(text=self.setter('_text'))
        self.params, self.codeinput.text = munge_source(str(v))
        self.codeinput.bind(text=self.setter('_text'))

    source = AliasProperty(_get_source, _set_source, bind=('_text', 'params'))

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
    FuncEditor showing the source of the selected one, and a close button.

    """

    def get_default_text(self, newname):
        return self.editor.get_default_text(newname)

    @staticmethod
    def valid_name(name):
        return name and name[0] not in string.digits + string.whitespace + string.punctuation


class FuncsEdScreen(Screen):
    toggle = ObjectProperty()

    def save(self, *args):
        self.ids.triggers.save()
        self.ids.prereqs.save()
        self.ids.actions.save()


Builder.load_string("""
#: import py3lexer pygments.lexers.Python3Lexer
<StoreButton>:
    size_hint_y: None
    height: 30
<StoreList>:
    viewclass: 'StoreButton'
    boxl: boxl
    SelectableRecycleBoxLayout:
        id: boxl
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
            on_release: root._trigger_delete()
    TextInput:
        id: string
        disabled: root.disable_text_input
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
            disable_text_input: root.disable_text_input
            validate_name_input: root.validate_name_input
            _trigger_save: root._trigger_save
            _trigger_delete: root._trigger_delete
<StringsEdScreen>:
    name: 'strings'
    edbox: edbox
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
                on_release: edbox.dismiss()
            Label:
                text_size: self.size
                halign: 'right'
                valign: 'middle'
                text: 'Language: '
            LanguageInput:
                id: language
                screen: root
                hint_text: root.language
                write_tab: False
                multiline: False
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
            on_text: root.validate_name_input(self.text)
        Py3CodeInput:
            id: params
            text: '({}):'.format(', '.join(root.params))
            disabled: True
            size_hint_y: None
            height: self.line_height + self.font_size
            background_disabled_normal: ''
            disabled_foreground_color: self.foreground_color
        Button:
            text: 'del'
            size_hint_x: 0.1
            on_release: root._trigger_delete()
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
            disabled: root.disable_text_input
<FuncsEdBox>:
    editor: funcs_ed
    storelist: funcs_list
    orientation: 'vertical'
    data: [(item['name'], self.store.get_source(item['name'])) for item in funcs_list.data[1:]]
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
            disable_text_input: root.disable_text_input
            validate_name_input: root.validate_name_input
            _trigger_save: root._trigger_save
            _trigger_delete: root._trigger_delete
    BoxLayout:
        size_hint_y: 0.05
        Button:
            text: 'Close'
            on_release: root.dismiss()
            size_hint_x: 0.2
        Widget:
            id: spacer
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
                on_data: app.rules.rulesview.set_functions('trigger', map(app.rules.rulesview.inspect_func, self.data))
        TabbedPanelItem:
            id: prereq
            text: 'Prereq'
            on_state: prereqs.save()
            FuncsEdBox:
                id: prereqs
                toggle: root.toggle
                table: 'prereqs'
                store: app.engine.prereq
                on_data: app.rules.rulesview.set_functions('prereq', map(app.rules.rulesview.inspect_func, self.data))
        TabbedPanelItem:
            id: action
            text: 'Action'
            on_state: actions.save()
            FuncsEdBox:
                id: actions
                toggle: root.toggle
                table: 'actions'
                store: app.engine.action
                on_data: app.rules.rulesview.set_functions('action', map(app.rules.rulesview.inspect_func, self.data))
""")
