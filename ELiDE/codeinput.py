from string import ascii_letters, digits
from kivy.properties import (
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ELiDEFunctionInput(BoxLayout):
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)
    style_name = StringProperty('default')
    params = ListProperty(['foo', 'bar'])

    def get_func_code(self):
        code = self.header + '\n'
        for line in self.ids.code.text.split('\n'):
            code += (' ' * 4 + line + '\n')
        return code


class FunctionNameInput(TextInput):
    def insert_text(self, s, from_undo=False):
        if self.text == '':
            if s[0] not in (ascii_letters + '_'):
                return
        return super().insert_text(
            ''.join(c for c in s if c in (ascii_letters + digits + '_'))
        )

kv = """
<ELiDEFunctionInput>:
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
