
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.uix.widget import (
    Widget,
    WidgetMetaclass)
from kivy.properties import (
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty,
    BooleanProperty)
from kivy.clock import Clock
from kivy.lang import Builder

import LiSE
from LiSE.orm import SaveableMetaclass

from os import sep


class ClosetLabel(Label):
    """Mix-in class for various text-having widget classes, to make their
    text match some named string from the closet."""
    stringname = StringProperty()
    closet = ObjectProperty()
    symbolic = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(ClosetLabel, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (self.stringname and self.closet):
            Clock.schedule_once(self.finalize, 0)
            return
        self.closet.register_text_listener(self.stringname, self.retext)
        self.retext()

    def retext(self, *args):
        self.text = self.closet.get_text(self.stringname)

    def on_symbolic(self, *args):
        if self.symbolic:
            self.font_name = sep.join(
                [LiSE.__path__[-1], 'gui', 'assets', 'Entypo.ttf'])
            self.font_size = 30
        else:
            self.font_name = 'DroidSans'
            self.font_size = 16


class ClosetButton(Button, ClosetLabel):
    fun = ObjectProperty(None)
    arg = ObjectProperty(None)

    def on_release(self, *args):
        if self.fun is None:
            return
        if self.arg is None:
            self.fun()
        else:
            self.fun(self.arg)


class ClosetToggleButton(ClosetButton, ToggleButtonBehavior):
    pass


class LiSEWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    """A combination of :class:`~kivy.uix.widget.WidgetMetaclass`
    and :class:`~LiSE.util.SaveableMetaclass`.

    There is no additional functionality beyond what those metaclasses do."""
    def __new__(metaclass, clas, parents, attrs):
        q = WidgetMetaclass.__new__(
            LiSEWidgetMetaclass, clas, parents, attrs)
        r = SaveableMetaclass.__new__(
            LiSEWidgetMetaclass, clas, parents, dict(q.__dict__))
        if hasattr(r, 'kv') and not hasattr(r, 'kv_loaded'):
            Builder.load_string(r.kv)
            r.kv_loaded = True
        return r


class ClosetHintTextInput(TextInput):
    closet = ObjectProperty()
    failure_string = StringProperty()
    """String to use when the input failed to validate"""
    failure_color = ListProperty([1, 0, 0, 1])
    """Color to turn the input field when it fails"""
    failure_color_timeout = NumericProperty(0.5)
    """Time after which to turn the color back"""
    failure_string_timeout = NumericProperty(3)
    """Time after which to turn the hint_text back"""
    validator = ObjectProperty()
    """Boolean function for whether the input is acceptable"""
    kv = """
<ClosetHintTextInput>:
    hint_text: self.closet.get_text(self.stringname)\
    if self.closet and self.stringname else ''
"""
    __metaclass__ = LiSEWidgetMetaclass

    def validate(self):
        """If my text is valid, return True. Otherwise, communicate invalidity
        to the user, and return False.

        I'll communicate invalidity by blinking some other color,
        blanking out my text, and displaying an alternative hint_text
        for a little while.

        """
        if self.validator(self.text):
            return True
        else:
            self.text = ''
            oldcolor = self.color
            self.color = self.failure_color
            self.hint_text = self.closet.get_text(self.failure_string)

            def unfail_color():
                self.color = oldcolor
            Clock.schedule_once(unfail_color, self.failure_color_timeout)

            def unfail_text():
                self.hint_text = self.closet.get_text(self.stringname)
            Clock.schedule_once(unfail_text, self.failure_string_timeout)
            return False


class TouchlessWidget(Widget):
    def on_touch_down(self, touch):
        return

    def on_touch_move(self, touch):
        return

    def on_touch_up(self, touch):
        return

    def collide_point(self, x, y):
        return

    def collide_widget(self, w):
        return


class CueCard(TouchlessWidget):
    """Widget that looks like TextInput but doesn't take input and can't be
clicked.

This is used to display feedback to the user when it's not serious
enough to get a popup of its own.

    """
    closet = ObjectProperty()
    stringname = StringProperty()
    symbolic = BooleanProperty(False)
    completion = NumericProperty(0)

    def on_closet(self, *args):
        self.completion += 1

    def on_stringname(self, *args):
        self.completion += 1

    def on_completion(self, i, v):
        if v == 2:
            self.closet.register_text_listener(self.stringname, self.retext)
            self.revert_text()

    def revert_text(self):
        self.ids.l.text = self.closet.get_text(self.stringname)

    def retext(self, skel, k, v):
        if k == self.closet.language:
            self.text = v
