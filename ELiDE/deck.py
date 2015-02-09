# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Cards that can be assembled into piles, which can be fanned out or
stacked into decks, which can then be drawn from."""
from math import sqrt
from kivy.clock import Clock
from kivy.graphics import (
    Color,
    Rectangle
)
from kivy.properties import (
    AliasProperty,
    BoundedNumericProperty,
    DictProperty,
    ListProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty,
)
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image


golden = (1 + sqrt(5)) / 2


pos_hint_kwargs = (
    'pos_hint',
    'pos_hint_x',
    'pos_hint_y',
    'pos_hint_center_x',
    'pos_hint_center_y',
    'pos_hint_right',
    'pos_hint_top'
)

pos_kwargs = (
    'x',
    'y',
    'pos',
    'center',
    'center_x',
    'center_y',
    'right',
    'top'
)

pos_maybe_hint_kwargs = pos_hint_kwargs + pos_kwargs

size_hint_kwargs = (
    'size_hint',
    'size_hint_x',
    'size_hint_y'
)

size_kwargs = (
    'size',
    'width',
    'height'
)

size_maybe_hint_kwargs = size_hint_kwargs + size_kwargs

pos_size_maybe_hint_kwargs \
    = pos_maybe_hint_kwargs + size_maybe_hint_kwargs


def except_pos_size_kwargs(d):
    r = {}
    for (k, v) in d.items():
        if k not in pos_size_maybe_hint_kwargs:
            r[k] = v
    return r


def get_pos_hint_x(poshints, sizehintx):
    if 'x' in poshints:
        return poshints['x']
    elif sizehintx is not None:
        if 'center_x' in poshints:
            return (
                poshints['center_x'] -
                sizehintx / 2
            )
        elif 'right' in poshints:
            return (
                poshints['right'] -
                sizehintx
            )


def set_pos_hint_x(poshints, v):
    for k in ('center_x', 'right'):
        if k in poshints:
            del poshints[k]
    poshints['x'] = v
    poshints.dispatch()


def get_pos_hint_y(poshints, sizehinty):
    if 'y' in poshints:
        return poshints['y']
    elif sizehinty is not None:
        if 'center_y' in poshints:
            return (
                poshints['center_y'] -
                sizehinty / 2
            )
        elif 'top' in poshints:
            return (
                poshints['top'] -
                sizehinty
            )


def set_pos_hint_y(poshints, v):
    for k in ('center_y', 'top'):
        if k in poshints:
            del poshints[k]
    poshints['y'] = v
    poshints.dispatch()


def get_pos_hint_center_x(poshints, sizehintx):
    if 'center_x' in poshints:
        return poshints['center_x']
    elif sizehintx is not None:
        if 'x' in poshints:
            return (
                poshints['x'] + sizehintx / 2
            )
        elif 'right' in poshints:
            return (
                poshints['right'] - sizehintx / 2
            )


def set_pos_hint_center_x(poshints, v):
    for k in ('x', 'right'):
        if k in poshints:
            del poshints[k]
    poshints['center_x'] = v
    poshints.dispatch()


def get_pos_hint_center_y(poshints, sizehinty):
    if 'center_y' in poshints:
        return poshints['center_y']
    elif sizehinty is not None:
        if 'y' in poshints:
            return (
                poshints['y'] + sizehinty / 2
            )
        elif 'top' in poshints:
            return (
                poshints['y'] - sizehinty / 2
            )


def set_pos_hint_center_y(poshints, v):
    for k in ('y', 'top'):
        if k in poshints:
            del poshints[k]
    poshints['center_y'] = v
    poshints.dispatch()


def get_pos_hint_center(poshints, size_hint_x, size_hint_y):
    if 'center' in poshints:
        return poshints['center']
    cx = poshints['center_x'] if 'center_x' in poshints else None
    cy = poshints['center_y'] if 'center_y' in poshints else None
    if cx is None and size_hint_x is not None:
        if 'x' in poshints:
            cx = poshints['x'] + size_hint_x / 2
        elif 'right' in poshints:
            cx = poshints['right'] - size_hint_x / 2
    if cy is None and size_hint_y is not None:
        if 'y' in poshints:
            cy = poshints['y'] + size_hint_y / 2
        elif 'top' in poshints:
            cy = poshints['top'] - size_hint_y / 2
    return (cx, cy)


def set_pos_hint_center(poshints, v):
    for k in ('x', 'right', 'y', 'top', 'center_x', 'center_y'):
        if k in poshints:
            del poshints[k]
    poshints['center'] = v
    poshints.dispatch()


def get_pos_hint_right(poshints, sizehintx):
    if 'right' in poshints:
        return poshints['right']
    elif sizehintx is not None:
        if 'x' in poshints:
            return (
                poshints['x'] + sizehintx
            )
        elif 'center_x' in poshints:
            return (
                poshints['center_x'] +
                sizehintx / 2
            )


def set_pos_hint_right(poshints, v):
    for k in ('x', 'center_x'):
        if k in poshints:
            del poshints[k]
    poshints['right'] = v
    poshints.dispatch()


def get_pos_hint_top(poshints, sizehinty):
    if 'top' in poshints:
        return poshints['top']
    elif sizehinty is not None:
        if 'y' in poshints:
            return (
                poshints['y'] + sizehinty
            )
        elif 'center_y' in poshints:
            return (
                poshints['center_y'] +
                sizehinty / 2
            )


def set_pos_hint_top(poshints, v):
    for k in ('y', 'center_y'):
        if k in poshints:
            del poshints[k]
    poshints['top'] = v
    poshints.dispatch()


class ColorTextureBox(Widget):
    color = ListProperty([1, 1, 1, 1])
    texture = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.realinit()

    def realinit(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.realinit, 0)
            return
        self._color = Color(rgba=self.color)
        self.canvas.add(self._color)
        self._rect = Rectangle(
            texture=self.texture,
            pos=self.pos,
            size=self.size
        )
        self.canvas.add(self._rect)

    def on_color(self, *args):
        if not hasattr(self, '_color'):
            Clock.schedule_once(self.on_color, 0)
            return
        self._color.rgba = self.color

    def on_texture(self, *args):
        if not hasattr(self, '_rect'):
            Clock.schedule_once(self.on_texture, 0)
            return
        self._rect.texture = self.texture

    def on_pos(self, *args):
        if not hasattr(self, '_rect'):
            Clock.schedule_once(self.on_pos, 0)
            return
        self._rect.pos = self.pos

    def on_size(self, *args):
        if not hasattr(self, '_rect'):
            Clock.schedule_once(self.on_size, 0)
            return
        self._rect.size = self.size


class Card(RelativeLayout):
    foreground_source = StringProperty('')
    foreground_color = ListProperty([0, 1, 0, 1])
    foreground_image = ObjectProperty(None, allownone=True)
    foreground_texture = ObjectProperty(None, allownone=True)
    background_source = StringProperty('')
    background_color = ListProperty([0, 0, 1, 1])
    background_image = ObjectProperty(None, allownone=True)
    background_texture = ObjectProperty(None, allownone=True)
    art_source = StringProperty('')
    art_color = ListProperty([1, 0, 0, 1])
    art_image = ObjectProperty(None, allownone=True)
    art_texture = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self._trigger_remake()
        super().__init__(**kwargs)

    def remake(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.remake, 0)
            return

        self.background = ColorTextureBox(
            color=self.background_color,
            texture=self.background_texture
        )
        self.bind(
            background_color=self.background.setter('color'),
            background_texture=self.background.setter('texture')
        )
        self.add_widget(self.background)
        self.foreground = ColorTextureBox(
            pos_hint={
                'x': 0.025,
                'y': 0.025
            },
            size_hint=(0.95, 0.45),
            color=self.foreground_color,
            texture=self.foreground_texture
        )
        self.bind(
            foreground_color=self.foreground.setter('color'),
            foreground_texture=self.foreground.setter('texture')
        )
        self.add_widget(self.foreground)
        self.art = ColorTextureBox(
            pos_hint={
                'x': 0.025,
                'top': 0.975
            },
            size_hint=(0.95, 0.45),
            color=self.art_color,
            texture=self.art_texture
        )
        self.bind(
            art_color=self.art.setter('color'),
            art_texture=self.art.setter('texture')
        )
        self.add_widget(self.art)
        if (
                self.background_source and
                self.background_source != self.background_image.source
        ):
            self.background_image = Image(self.background_source)
        if (
                self.foreground_source and
                self.foreground_source != self.foreground_image.source
        ):
            self.foreground_image = Image(self.foreground_source)
        if self.art_source and self.art_source != self.art_image.source:
            self.art_image = Image(self.art_source)

    def on_background_image(self, *args):
        if self.background_image is not None:
            self.background_texture = self.background_image.texture

    def on_foreground_image(self, *args):
        if self.foreground_image is not None:
            self.foreground_texture = self.foreground_image.texture

    def on_art_image(self, *args):
        if self.art_image is not None:
            self.art_texture = self.art_image.texture


if __name__ == '__main__':
    from kivy.base import runTouchApp
    runTouchApp(Card(background_color=[1,0,0,1], foreground_color=[0,1,0,1], art_color=[0,0,1,1]))
