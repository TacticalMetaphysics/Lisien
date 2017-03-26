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
from string import ascii_letters, digits
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

    def redata(self, *args):
        if not self.table or not self.store:
            Clock.schedule_once(self.redata)
            return
        self.data = list(map(self.munge, enumerate(sorted(self.store.keys()))))
    _trigger_redata = trigger(redata)

    def select(self, inst):
        self.selection = inst


class StringsEdScreen(Screen):
    engine = ObjectProperty()
    toggle = ObjectProperty()

    def add_string(self, *args):
        ed = self.ids.editor
        ed.save()
        newname = self.ids.strname.text
        if newname in self.engine.string:
            return
        self.engine.string[newname] = ed.source = ''
        assert(newname in self.engine.string)
        self.ids.strname.text = ''
        ed.name = newname


class StringInput(BoxLayout):
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


sig_ex = re.compile('^ *def .+?\((.+)\):$')


class FunctionNameInput(TextInput):
    def insert_text(self, s, from_undo=False):
        if self.text == '':
            if s[0] not in (ascii_letters + '_'):
                return
        return super().insert_text(
            ''.join(c for c in s if c in (ascii_letters + digits + '_'))
        )


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
        sig_ex.match(lines[0]).groups()[0].split(',')
    )
    del lines[0]
    if not lines:
        return params, ''
    # hack to allow 'empty' functions
    if lines and lines[-1].strip() == 'pass':
        del lines[-1]
    return params, '\n'.join(line[spaces:] for line in lines)


class FuncsEditor(BoxLayout):
    """The editor widget for working with any particular function.

    Contains a one-line field for the function's name; a multi-line
    field for its code; and radio buttons to select its signature
    from among those permitted.

    """
    name = StringProperty('')
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
        code = 'def ' + self.name + '(' + ', '.join(self.params) + '):\n'
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

    source = AliasProperty(_get_source, _set_source, bind=('name', 'params', '_text'))

    def on_codeinput(self, *args):
        self._text = self.codeinput.text
        self.codeinput.bind(text=self.setter('_text'))

    def on_storelist(self, *args):
        self.storelist.bind(selection=self._pull_func)

    def save(self, *args):
        if not (self.name and self.store):
            return
        if self.source != self.store.plain(self.name):
            Logger.debug('saving function {}'.format(self.name))
            self.store.set_source(self.name, self.source)
    _trigger_save = trigger(save)

    @trigger
    def _pull_func(self, *args):
        self.save()
        self.ids.funname.text = self.name = self.storelist.selection.name
        self.source = self.store.plain(self.name)


class FuncsEdBox(BoxLayout):
    """Widget for editing the Python source of funcs to be used in LiSE sims.

    Contains a list of functions in the store it's about, next to a
    FuncsEditor showing the source of the selected one, and some
    controls on the bottom that let you add, delete, and rename the function,
    or close the screen.

    """
    funcs_ed = ObjectProperty()
    table = StringProperty()
    store = ObjectProperty()
    data = ListProperty()
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

    def del_func(self, *args):
        raise NotImplementedError

    def rename_func(self, *args):
        raise NotImplementedError

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
    orientation: 'vertical'
    BoxLayout:
        size_hint_y: 0.05
        Label:
            text: 'Title: '
            size_hint_x: None
            width: self.texture_size[0]
        TextInput:
            id: stringname
    TextInput:
        id: string
<StringsEdScreen>:
    name: 'strings'
    BoxLayout:
        orientation: 'vertical'
        StringInput:
            id: editor
            table: 'strings'
            store: root.engine.string if root.engine else None
            size_hint_y: 0.95
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.05
            TextInput:
                id: strname
                hint_tex: 'New string name'
            Button:
                id: newstr
                text: 'New'
                on_press: root.add_string()
            Button:
                id: close
                text: 'Close'
                on_press: root.toggle()
<Py3CodeInput@CodeInput>:
    lexer: py3lexer()
<FuncsEditor>:
    codeinput: code
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
            size_hint_y: None
            height: self.line_height + self.font_size
            multiline: False
            write_tab: False
        Py3CodeInput:
            id: params
            text: '(' + ', '.join(root.params) + '):'
            disabled: True
            size_hint_y: None
            height: self.line_height + self.font_size
            background_disabled_normal: ''
            disabled_foreground_color: self.foreground_color
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
    funcs_ed: funcs_ed
    storelist: funcs_list
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        StoreList:
            id: funcs_list
            table: root.table
            store: root.store
            size_hint_x: 0.2
        FuncsEditor:
            id: funcs_ed
            table: root.table
            store: root.store
            storelist: funcs_list
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