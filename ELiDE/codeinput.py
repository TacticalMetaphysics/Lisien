from functools import partial
from string import ascii_letters, digits
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.uix.textinput import TextInput
from kivy.uix.codeinput import CodeInput
from kivy.uix.boxlayout import BoxLayout
from pygments import styles
from pygments.lexers import Python3Lexer


class ELiDECodeInput(CodeInput):
    lexer = ObjectProperty(Python3Lexer())


class FunctionInput(BoxLayout):
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)
    style_name = StringProperty('default')
    name = StringProperty()
    params = ListProperty(['foo', 'bar'])

    def _get_source(self):
        code = self.name + '(' + ', '.join(self.params) + '):\n'
        for line in self.ids.code.text.split('\n'):
            code += (' ' * 4 + line + '\n')
        return code

    def _set_source(self, v, *args):
        if 'code' not in self.ids:
            Clock.schedule_once(partial(self._set_source, v), 0)
            return
        lines = v.split('\n')
        del lines[0]
        self.ids.code.text = '\n'.join(line[4:] for line in lines)

    source = AliasProperty(_get_source, _set_source)

    def on_name(self, *args):
        if 'funname' not in self.ids:
            Clock.schedule_once(self.on_name, 0)
            return
        self.ids.funname.text = self.name


class FunctionNameInput(TextInput):
    def insert_text(self, s, from_undo=False):
        if self.text == '':
            if s[0] not in (ascii_letters + '_'):
                return
        return super().insert_text(
            ''.join(c for c in s if c in (ascii_letters + digits + '_'))
        )

kv = """
<FunctionInput>:
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: None
        height: funname.height
        ELiDECodeInput:
            id: imafunction
            text: 'def'
            font_name: root.font_name
            font_size: root.font_size
            style_name: root.style_name
            disabled: True
            size_hint: (None, None)
            height: self.line_height + self.font_size
            width: self.font_size * 2.5
            background_disabled_normal: ''
            disabled_foreground_color: self.foreground_color
        FunctionNameInput:
            id: funname
            font_name: root.font_name
            font_size: root.font_size
            size_hint_y: None
            height: self.line_height + self.font_size
            multiline: False
            write_tab: False
        ELiDECodeInput:
            id: params
            text: '(' + ', '.join(root.params) + '):'
            font_name: root.font_name
            font_size: root.font_size
            style_name: root.style_name
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
            font_name: root.font_name
            font_size: root.font_size
            # PEP8 standard indentation width is 4 spaces
            text: ' ' * 4
            size_hint_x: None
            width: self.texture_size[0]
        ELiDECodeInput:
            font_name: root.font_name
            font_size: root.font_size
            style_name: root.style_name
            id: code
"""
Builder.load_string(kv)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(ELiDEFunctionInput(header='def foo(bar, bas):', style=styles.get_style_by_name('fruity')))
