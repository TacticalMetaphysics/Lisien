from kivy.properties import (
    ObjectProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.uix.codeinput import CodeInput
from kivy.uix.boxlayout import BoxLayout
from pygments.lexers import Python3Lexer


class ELiDECodeInput(CodeInput):
    lexer = ObjectProperty(Python3Lexer())


class ELiDEFunctionInput(BoxLayout):
    header = StringProperty('')


kv = """
<ELiDEFunctionInput>:
    orientation: 'vertical'
    Label:
        text: root.header
        size_hint_y: None
        height: self.texture_size[1]
    BoxLayout:
        orientation: 'horizontal'
        Label:
            # PEP8 standard indentation width is 4 spaces
            text: ' ' * 4
            size_hint_x: None
            width: self.texture_size[0]
        ELiDECodeInput:
            id: code
"""
Builder.load_string(kv)
