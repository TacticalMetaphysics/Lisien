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


class Card(RelativeLayout):
    foreground_source = StringProperty(None, allownone=True)
    foreground_texture = ObjectProperty(None, allownone=True)
    foreground_image = ObjectProperty(None, allownone=True)
    foreground_color = ListProperty([1, 1, 1, 1])
    foreground_size_hint_x = BoundedNumericProperty(
        0.95, min=0.0, max=1.0, allownone=True
    )
    foreground_size_hint_y = BoundedNumericProperty(
        0.45, min=0.0, max=1.0, allownone=True
    )
    foreground_size_hint = ReferenceListProperty(
        foreground_size_hint_x, foreground_size_hint_y
    )
    foreground_width = BoundedNumericProperty(100 * golden, min=1)
    foreground_height = BoundedNumericProperty(100, min=1)
    foreground_size = ReferenceListProperty(
        foreground_width, foreground_height
    )
    foreground_pos_hint = ObjectProperty({'x': 0.025, 'y': 0.025})
    foreground_pos_hint_x = AliasProperty(
        lambda self: get_pos_hint_x(
            self.foreground_pos_hint, self.foreground_size_hint_x
        ),
        lambda self, v: set_pos_hint_x(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_x')
    )
    foreground_pos_hint_y = AliasProperty(
        lambda self: get_pos_hint_y(
            self.foreground_pos_hint, self.foreground_size_hint_y
        ),
        lambda self, v: set_pos_hint_y(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_y')
    )
    foreground_pos_hint_center_x = AliasProperty(
        lambda self: get_pos_hint_center_x(
            self.foreground_pos_hint, self.foreground_size_hint_x
        ),
        lambda self, v: set_pos_hint_center_x(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_x')
    )
    foreground_pos_hint_center_y = AliasProperty(
        lambda self: get_pos_hint_center_y(
            self.foreground_pos_hint, self.foreground_size_hint_y
        ),
        lambda self, v: set_pos_hint_center_y(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_y')
    )

    def _set_foreground_pos_hint_center(self, v):
        for k in ('x', 'right', 'y', 'top'):
            if k in self.foreground_pos_hint:
                del self.foreground_pos_hint[k]
        (x, y) = v
        self.foreground_pos_hint['center_x'] = x
        self.foreground_pos_hint['center_y'] = y
        self.foreground_pos_hint.dispatch()

    foreground_pos_hint_center = AliasProperty(
        lambda self: (
            self.foreground_pos_hint_center_x,
            self.foreground_pos_hint_center_y
        ),
        _set_foreground_pos_hint_center,
        bind=('foreground_pos_hint', 'foreground_size_hint')
    )
    foreground_pos_hint_top = AliasProperty(
        lambda self: get_pos_hint_top(
            self.foreground_pos_hint, self.foreground_size_hint_y
        ),
        lambda self, v: set_pos_hint_top(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_y')
    )
    foreground_pos_hint_right = AliasProperty(
        lambda self: get_pos_hint_right(
            self.foreground_pos_hint, self.foreground_size_hint_x
        ),
        lambda self, v: set_pos_hint_right(
            self.foreground_pos_hint, v
        ),
        bind=('foreground_pos_hint', 'foreground_size_hint_x')
    )
    foreground_x = BoundedNumericProperty(16, min=0)
    foreground_y = BoundedNumericProperty(16, min=0)
    foreground_pos = ReferenceListProperty(foreground_x, foreground_y)
    _foreground_extra_kwargs = DictProperty({})

    def _get_foreground_kwargs(self):
        r = dict(self._foreground_extra_kwargs)
        r['pos_hint'] = self.foreground_pos_hint
        r['size_hint'] = self.foreground_size_hint
        r['pos'] = self.foreground_pos
        r['size'] = self.foreground_size
        return r

    def _set_foreground_kwargs(self, v):
        if 'size_hint' in v:
            self.foreground_size_hint = v['size_hint']
        if 'size_hint_x' in v:
            self.foreground_size_hint_x = v['size_hint_x']
        if 'size_hint_y' in v:
            self.foreground_size_hint_y = v['size_hint_y']
        if 'width' in v:
            self.foreground_width = v['width']
        if 'height' in v:
            self.foreground_height = v['height']
        if 'size' in v:
            self.foreground_size = v['size']
        for kwarg in pos_hint_kwargs:
            if kwarg in v:
                self.foreground_pos_hint[kwarg] = v[kwarg]
        self._foreground_extra_kwargs = except_pos_size_kwargs(v)

    foreground_kwargs = AliasProperty(
        _get_foreground_kwargs,
        _set_foreground_kwargs,
        bind=(
            '_foreground_extra_kwargs',
            'foreground_size_hint',
            'foreground_pos_hint',
            'foreground_size',
            'foreground_pos'
        )
    )

    background_source = StringProperty(None, allownone=True)
    background_texture = ObjectProperty(None, allownone=True)
    background_image = ObjectProperty(None, allownone=True)
    background_color = ListProperty([1, 1, 1, 1])
    background_size_hint_x = BoundedNumericProperty(
        1.0, min=0.0, max=1.0, allownone=True
    )
    background_size_hint_y = BoundedNumericProperty(
        1.0, min=0.0, max=1.0, allownone=True
    )
    background_size_hint = ReferenceListProperty(
        background_size_hint_x, background_size_hint_y
    )
    background_width = BoundedNumericProperty(100 * golden + 32, min=1)
    background_height = BoundedNumericProperty(148, min=1)
    background_size = ReferenceListProperty(
        background_width, background_height
    )
    background_pos_hint = ObjectProperty({})
    background_pos_hint_x = AliasProperty(
        lambda self: get_pos_hint_x(
            self.background_pos_hint, self.background_size_hint_x
        ),
        lambda self, v: set_pos_hint_x(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_x')
    )
    background_pos_hint_y = AliasProperty(
        lambda self: get_pos_hint_y(
            self.background_pos_hint, self.background_size_hint_y
        ),
        lambda self, v: set_pos_hint_y(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_y')
    )
    background_pos_hint_center_x = AliasProperty(
        lambda self: get_pos_hint_center_x(
            self.background_pos_hint, self.background_size_hint_x
        ),
        lambda self, v: set_pos_hint_center_x(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_x')
    )
    background_pos_hint_center_y = AliasProperty(
        lambda self: get_pos_hint_center_y(
            self.background_pos_hint, self.background_size_hint_y
        ),
        lambda self, v: set_pos_hint_center_y(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_y')
    )

    def _set_background_pos_hint_center(self, v):
        for k in ('x', 'right', 'y', 'top'):
            if k in self.background_pos_hint:
                del self.background_pos_hint[k]
        (x, y) = v
        self.background_pos_hint['center_x'] = x
        self.background_pos_hint['center_y'] = y
        self.background_pos_hint.dispatch()
    background_pos_hint_center = AliasProperty(
        lambda self: (
            self.background_pos_hint_center_x,
            self.background_pos_hint_center_y
        ),
        _set_background_pos_hint_center,
        bind=('background_pos_hint', 'background_size_hint')
    )
    background_pos_hint_top = AliasProperty(
        lambda self: get_pos_hint_top(
            self.background_pos_hint, self.background_size_hint_y
        ),
        lambda self, v: set_pos_hint_top(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_y')
    )
    background_pos_hint_right = AliasProperty(
        lambda self: get_pos_hint_right(
            self.background_pos_hint, self.background_size_hint_x
        ),
        lambda self, v: set_pos_hint_right(
            self.background_pos_hint, v
        ),
        bind=('background_pos_hint', 'background_size_hint_x')
    )
    background_x = BoundedNumericProperty(0, min=0)
    background_y = BoundedNumericProperty(0, min=0)
    background_pos = ReferenceListProperty(background_x, background_y)
    _background_extra_kwargs = DictProperty({})

    def _get_background_kwargs(self):
        r = dict(self._background_extra_kwargs)
        r['pos_hint'] = self.background_pos_hint
        r['size_hint'] = self.background_size_hint
        r['pos'] = self.background_pos
        r['size'] = self.background_size
        return r

    def _set_background_kwargs(self, v):
        if 'size_hint' in v:
            self.background_size_hint = v['size_hint']
        if 'size_hint_x' in v:
            self.background_size_hint_x = v['size_hint_x']
        if 'size_hint_y' in v:
            self.background_size_hint_y = v['size_hint_y']
        if 'width' in v:
            self.background_width = v['width']
        if 'height' in v:
            self.background_height = v['height']
        if 'size' in v:
            self.background_size = v['size']
        for kwarg in pos_hint_kwargs:
            if kwarg in v:
                self.background_pos_hint[kwarg] = v[kwarg]
        self._background_extra_kwargs = except_pos_size_kwargs(v)

    background_kwargs = AliasProperty(
        _get_background_kwargs,
        _set_background_kwargs,
        bind=(
            '_background_extra_kwargs',
            'background_size_hint',
            'background_pos_hint',
            'background_size',
            'background_pos'
        )
    )

    art_source = StringProperty(None, allownone=True)
    art_texture = ObjectProperty(None, allownone=True)
    art_image = ObjectProperty(None, allownone=True)
    art_color = ListProperty([1, 1, 1, 1])
    art_size_hint_x = BoundedNumericProperty(
        0.95, min=0, max=1, allownone=True
    )
    art_size_hint_y = BoundedNumericProperty(
        0.45, min=0, max=1, allownone=True
    )
    art_size_hint = ReferenceListProperty(
        art_size_hint_x, art_size_hint_y
    )
    art_pos_hint = ObjectProperty({'x': 0.025, 'top': 0.975})
    art_pos_hint_x = AliasProperty(
        lambda self: get_pos_hint_x(
            self.art_pos_hint, self.art_size_hint_x
        ),
        lambda self, v: set_pos_hint_x(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_x')
    )
    art_pos_hint_y = AliasProperty(
        lambda self: get_pos_hint_y(
            self.art_pos_hint, self.art_size_hint_y
        ),
        lambda self, v: set_pos_hint_y(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_y')
    )
    art_pos_hint_center_x = AliasProperty(
        lambda self: get_pos_hint_center_x(
            self.art_pos_hint, self.art_size_hint_x
        ),
        lambda self, v: set_pos_hint_center_x(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_x')
    )
    art_pos_hint_center_y = AliasProperty(
        lambda self: get_pos_hint_center_y(
            self.art_pos_hint, self.art_size_hint_y
        ),
        lambda self, v: set_pos_hint_center_y(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_y')
    )

    def _set_art_pos_hint_center(self, v):
        (x, y) = v
        self.art_pos_hint['center_x'] = x
        self.art_pos_hint['center_y'] = y
        self.art_pos_hint.dispatch()

    art_pos_hint_center = AliasProperty(
        lambda self: (
            self.art_pos_hint_center_x,
            self.art_pos_hint_center_y
        ),
        _set_art_pos_hint_center,
        bind=('art_pos_hint', 'art_size_hint')
    )
    art_pos_hint_top = AliasProperty(
        lambda self: get_pos_hint_top(
            self.art_pos_hint, self.art_size_hint_y
        ),
        lambda self, v: set_pos_hint_top(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_y')
    )
    art_pos_hint_right = AliasProperty(
        lambda self: get_pos_hint_right(
            self.art_pos_hint, self.art_size_hint_x
        ),
        lambda self, v: set_pos_hint_right(
            self.art_pos_hint, v
        ),
        bind=('art_pos_hint', 'art_size_hint_x')
    )
    art_width = BoundedNumericProperty(95, min=1)
    art_height = BoundedNumericProperty(232, min=1)
    art_size = ReferenceListProperty(art_width, art_height)
    art_x = BoundedNumericProperty(3, min=0)
    art_y = BoundedNumericProperty(132, min=0)
    art_pos = ReferenceListProperty(art_x, art_y)
    _art_extra_kwargs = DictProperty({})

    def _get_art_kwargs(self):
        r = dict(self._art_extra_kwargs)
        r['pos_hint'] = self.foreground_pos_hint
        r['size_hint'] = self.foreground_size_hint
        r['pos'] = self.foreground_pos
        r['size'] = self.foreground_size
        return r

    def _set_art_kwargs(self, v):
        if 'size_hint' in v:
            self.art_size_hint = v['size_hint']
        if 'size_hint_x' in v:
            self.art_size_hint_x = v['size_hint_x']
        if 'size_hint_y' in v:
            self.art_size_hint_y = v['size_hint_y']
        if 'width' in v:
            self.art_width = v['width']
        if 'height' in v:
            self.art_height = v['height']
        if 'size' in v:
            self.art_size = v['size']
        for kwarg in pos_hint_kwargs:
            if kwarg in v:
                self.art_pos_hint[kwarg] = v[kwarg]
        self._art_extra_kwargs = except_pos_size_kwargs(v)

    art_kwargs = AliasProperty(
        _get_art_kwargs,
        _set_art_kwargs,
        bind=(
            '_art_extra_kwargs',
            'art_size_hint',
            'art_pos_hint',
            'art_size',
            'art_pos'
        )
    )

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self.bind(on_parent=self._trigger_remake)
        super().__init__(**kwargs)

    def remake(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.remake, 0)
            return
        with self.canvas:
            self._bgcolor = Color(rgba=self.background_color)
            self._bgrect = Rectangle(**self.background_kwargs)
            self._fgcolor = Color(rgba=self.foreground_color)
            self._fgrect = Rectangle(**self.foreground_kwargs)
            self._artcolor = Color(rgba=self.art_color)
            self._artrect = Rectangle(**self.art_kwargs)
            Color(rgba=[1, 1, 1, 1])
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

    def on_background_texture(self, *args):
        if self.background_texture is None:
            return
        if not hasattr(self, '_bgrect'):
            Clock.schedule_once(self.on_background_texture, 0)
            return
        self._bgrect.texture = self.background_texture

    def on_foreground_texture(self, *args):
        if self.foreground_texture is None:
            return
        if not hasattr(self, '_fgrect'):
            Clock.schedule_once(self.on_foreground_texture, 0)
            return
        self._fgrect.texture = self.foreground_texture

    def on_art_texture(self, *args):
        if self.art_texture is None:
            return
        if not hasattr(self, '_artrect'):
            Clock.schedule_once(self.on_art_texture, 0)
            return
        self._artrect.texture = self.art_texture
