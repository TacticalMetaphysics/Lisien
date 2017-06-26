"""Generic dialog boxes and menus, for in front of a Board

Mostly these will be added as children of KvLayoutFront but you
could use them independently if you wanted.

"""
from kivy.properties import DictProperty, ListProperty, StringProperty, NumericProperty, VariableListProperty
from kivy.core.text import DEFAULT_FONT
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.lang import Builder


class Box(Widget):
    padding = VariableListProperty([6, 6, 6, 6])
    border = ListProperty([4, 4, 4, 4])
    font_size = StringProperty('15sp')
    font_name = StringProperty(DEFAULT_FONT)
    background = StringProperty(
        'atlas://data/images/defaulttheme/textinput')
    background_color = ListProperty([1, 1, 1, 1])
    foreground_color = ListProperty([0, 0, 0, 1])


class MessageBox(Box):
    """Looks like a TextInput but doesn't accept any input.

    Does support styled text with BBcode.

    """
    line_spacing = NumericProperty(0)
    text = StringProperty()


class DialogMenu(Box, BoxLayout):
    """Some buttons that make the game do things."""
    options = ListProperty()
    """List of pairs of (button_text, partial)"""
    funcs = DictProperty({})
    """Dict of functions to be used in place of string partials in the options"""

    def on_options(self, *args):
        self.clear_widgets()
        for txt, part in self.options:
            if not callable(part):
                part = self.funcs[part]
            self.add_widget(Button(text=txt, on_press=part))


class Dialog(BoxLayout):
    """MessageBox with a DialogMenu beneath it"""
    message_kwargs = DictProperty({})
    menu_kwargs = DictProperty({})

    # I'm worried that updating this way won't set the kivy.properties correctly
    def on_message_kwargs(self, *args):
        self.ids.msg.__dict__.update(self.message_kwargs)

    def on_menu_kwargs(self, *args):
        self.ids.menu.__dict__.update(self.menu_kwargs)


Builder.load_string("""
<MessageBox>:
    canvas.before:
        Color:
            rgba: self.background_color
        BorderImage:
            border: self.border
            pos: self.pos
            size: self.size
            source: self.background
    ScrollView:
        id: sv
        do_scroll_x: False
        size: root.width - root.border[1] - root.border[3], root.height - root.border[0] - root.border[2]
        Label:
            markup: True
            text: root.text
            font_name: root.font_name
            font_size: root.font_size
            line_spacing: root.line_spacing
            width: sv.width
            size_hint_y: None
            text_size: self.size
<DialogMenu>:
    canvas.before:
        Color:
            rgba: self.background_color
        BorderImage:
            border: self.border
            pos: self.pos
            size: self.size
            source: self.background
    orientation: 'vertical'
<Dialog>:
    MessageBox:
        id: msg
    ScrollView:
        DialogMenu:
            size_hint_y: None
            id: menu
""")


if __name__ == "__main__":
    from kivy.base import runTouchApp
    dia = Dialog(
        message_kwargs={'text': 'I am a dialog'},
        menu_kwargs={'options': [('one', lambda: None), ('two', lambda: None)]}
    )