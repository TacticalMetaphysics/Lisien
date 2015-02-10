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
    BooleanProperty,
    DictProperty,
    ListProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image


class ColorTextureBox(FloatLayout):
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
        self._trigger_layout()

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


class Card(FloatLayout):
    foreground_source = StringProperty('')
    foreground_color = ListProperty([0, 1, 0, 1])
    foreground_image = ObjectProperty(None, allownone=True)
    background_source = StringProperty('')
    background_color = ListProperty([0, 0, 1, 1])
    background_image = ObjectProperty(None, allownone=True)
    art_source = StringProperty('')
    art_color = ListProperty([1, 0, 0, 1])
    art_image = ObjectProperty(None, allownone=True)
    show_art = BooleanProperty(True)
    headline_kwargs = DictProperty({
        'text': 'Headline',
        'markup': True,
        'size_hint': (None, None),
        'font_size': 18
    })
    headline_text = AliasProperty(
        lambda self: self.headline_kwargs['text'],
        lambda self, v: self.headline_kwargs.__setitem__('text', v),
        bind=('headline_kwargs',)
    )
    midline_kwargs = DictProperty({
        'text': 'Midline',
        'markup': True,
        'size_hint': (None, None),
        'font_size': 14
    })
    midline_text = AliasProperty(
        lambda self: self.midline_kwargs['text'],
        lambda self, v: self.midline_kwargs.__setitem__('text', v),
        bind=('midline_kwargs',)
    )
    footer_kwargs = DictProperty({
        'text': 'Footer',
        'markup': True,
        'size_hint': (None, None),
        'font_size': 10
    })
    footer_text = AliasProperty(
        lambda self: self.footer_kwargs['text'],
        lambda self, v: self.footer_kwargs.__setitem__('text', v),
        bind=('footer_kwargs',)
    )
    text_kwargs = DictProperty({
        'text': '',
        'markup': True,
        'font_name': 'DroidSans',
        'font_size': 12,
        'pos_hint': {'x': 0.01, 'top': 0.99},
        'valign': 'top'
    })
    text = AliasProperty(
        lambda self: self.text_kwargs['text'],
        lambda self, v: self.text_kwargs.__setitem__('text', v),
        bind=('text_kwargs',)
    )

    def __init__(self, **kwargs):
        self._trigger_remake = Clock.create_trigger(self.remake)
        self._trigger_remake()
        super().__init__(**kwargs)

    def remake(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.remake, 0)
            return
        self.clear_widgets()
        with self.canvas:
            self._color = Color(rgba=self.background_color)
            self._bgrect = Rectangle(
                size=self.size,
                pos=self.pos,
                texture=self.background_texture
            )
            Color(rgba=[1, 1, 1, 1])
        layout = BoxLayout(
            orientation='vertical',
            size_hint=(0.95, 0.95),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        self.bind(
            size=layout._trigger_layout,
            pos=layout._trigger_layout
        )
        self.add_widget(layout)
        self.headline = Label(**self.headline_kwargs)
        self.headline.size = self.headline.texture_size
        self.headline.bind(texture_size=self.headline.setter('size'))

        def upd_headline(*args):
            for (k, v) in self.headline_kwargs.items():
                getattr(self.headline, k).set(v)
        self.bind(headline_kwargs=upd_headline)
        layout.add_widget(self.headline)
        if self.show_art:
            self.art = ColorTextureBox(
                color=self.art_color,
                texture=self.art_texture,
                size_hint_y=0.45
            )
            self.bind(
                art_color=self.art.setter('color'),
                art_texture=self.art.setter('texture')
            )
            layout.add_widget(self.art)
        self.midline = Label(**self.midline_kwargs)
        self.midline.size = self.midline.texture_size
        self.midline.bind(texture_size=self.midline.setter('size'))

        def upd_midline(*args):
            for (k, v) in self.midline_kwargs.items():
                getattr(self.midline, k).set(v)
        self.bind(midline_kwargs=upd_midline)
        layout.add_widget(self.midline)
        self.foreground = ColorTextureBox(
            color=self.foreground_color,
            texture=self.foreground_texture,
            size_hint_y=0.45 if self.show_art else 0.9
        )
        self.bind(
            foreground_color=self.foreground.setter('color'),
            foreground_texture=self.foreground.setter('texture')
        )
        layout.add_widget(self.foreground)
        fgtext = Label(
            text=self.text,
            markup=True,
            font_name=self.font_name,
            font_size=self.font_size,
            size_hint=(None, None),
            text_size=self.foreground.size,
            pos_hint={'x': 0.01, 'top': 0.99},
            valign='top'
        )
        fgtext.size = fgtext.texture_size
        fgtext.bind(texture_size=fgtext.setter('size'))
        self.foreground.bind(
            size=fgtext.setter('text_size')
        )
        self.bind(
            text=fgtext.setter('text'),
            font_name=fgtext.setter('font_name'),
            font_size=fgtext.setter('font_size')
        )
        self.foreground.add_widget(fgtext)
        self.footer = Label(**self.footer_kwargs)
        self.footer.size = self.footer.texture_size
        self.footer.bind(texture_size=self.footer.setter('size'))

        def upd_footer(*args):
            for (k, v) in self.footer_kwargs.items():
                getattr(self.footer, k).set(v)
        self.bind(footer_kwargs=upd_footer)
        layout.add_widget(self.footer)

    def on_size(self, *args):
        if hasattr(self, '_bgrect'):
            self._bgrect.size = self.size

    def on_pos(self, *args):
        if hasattr(self, '_bgrect'):
            self._bgrect.pos = self.pos

    def on_background_source(self, *args):
        if self.background_source:
            self.background_image = Image(self.background_source)

    def on_background_image(self, *args):
        if self.background_image is not None:
            self.background_texture = self.background_image.texture

    def on_foreground_source(self, *args):
        if self.foreground_source:
            self.foreground_image = Image(self.foreground_source)

    def on_foreground_image(self, *args):
        if self.foreground_image is not None:
            self.foreground_texture = self.foreground_image.texture

    def on_art_source(self, *args):
        if self.art_source:
            self.art_image = Image(self.art_source)

    def on_art_image(self, *args):
        if self.art_image is not None:
            self.art_texture = self.art_image.texture

    def on_headline_kwargs(self, *args):
        if not hasattr(self, 'headline'):
            Clock.schedule_once(self.on_headline_kwargs, 0)
            return
        for (k, v) in self.headline_kwargs.items():
            if getattr(self.headline, k) != v:
                setattr(self.headline, k, v)

    def on_midline_kwargs(self, *args):
        if not hasattr(self, 'midline'):
            Clock.schedule_once(self.on_midline_kwargs, 0)
            return
        for (k, v) in self.midline_kwargs.items():
            if getattr(self.midline, k) != v:
                setattr(self.midline, k, v)

    def on_footer_kwargs(self, *args):
        if not hasattr(self, 'footer'):
            Clock.schedule_once(self.on_footer_kwargs, 0)
            return
        for (k, v) in self.footer_kwargs.items():
            if getattr(self.footer, k) != v:
                setattr(self.footer, k, v)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.modules import inspector
    card = Card(background_color=[1,0,0,1], foreground_color=[0,1,0,1], art_color=[0,0,1,1], text='Thequick brown fox jumps over the lazy dog')
    inspector.create_inspector(Window, card)
    runTouchApp(card)
