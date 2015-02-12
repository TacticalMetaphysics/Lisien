# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Cards that can be assembled into piles, which can be fanned out or
stacked into decks, which can then be drawn from."""
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.graphics import (
    Color,
    Rectangle
)
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    BoundedNumericProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    ReferenceListProperty,
    StringProperty,
)
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.layout import Layout
from kivy.uix.image import Image


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
    layout = ObjectProperty(None, allownone=True)
    dragging = BooleanProperty(False)
    collide_x = NumericProperty()
    collide_y = NumericProperty()
    collide_pos = ReferenceListProperty(collide_x, collide_y)
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
    show_headline = BooleanProperty(True)
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
    show_midline = BooleanProperty(True)
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
    show_footer = BooleanProperty(True)
    text_kwargs = DictProperty({
        'text': '',
        'markup': True,
        'shorten': True,
        'size_hint': (None, None),
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
        if self.show_headline:
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
        if self.show_midline:
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
        fgtext = Label(**self.text_kwargs)
        fgtext.size = fgtext.texture_size
        fgtext.bind(texture_size=fgtext.setter('size'))
        self.foreground.bind(
            size=fgtext.setter('text_size')
        )

        def upd_fgtext(*args):
            for (k, v) in self.text_kwargs.items():
                if getattr(fgtext, k) != v:
                    getattr(fgtext, k).set(v)
        self.bind(text_kwargs=upd_fgtext)
        self.foreground.add_widget(fgtext)
        if self.show_footer:
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

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.dragging = True
            touch.ud['card'] = self
            touch.grab(self.layout)
            touch.grab(self)
            self.collide_x = touch.x - self.x
            self.collide_y = touch.y - self.y
            if self.layout is not None:
                if (
                        self.layout.grabbed is not None and
                        self.layout.grabbed is not self
                ):
                    touch.ungrab(self)
                    self.dragging = False
                else:
                    self.layout.grabbed = self

    def on_touch_move(self, touch):
        if not self.dragging:
            touch.ungrab(self)
            return
        self.pos = (
            touch.x - self.collide_x,
            touch.y - self.collide_y
        )

    def on_touch_up(self, touch):
        if not self.dragging:
            return
        if self.layout is not None:
            self.layout.grabbed = None
            self.layout._trigger_layout()
            if self.layout.insertion_point is not None:
                self.parent.remove_widget(self)
                if self.layout.insertable:
                    self.layout.add_widget(
                        self, index=self.layout.insertion_point
                    )
                self.layout.insertion_point = None
        touch.ungrab(self)
        self.dragging = False


class DeckLayout(Layout):
    direction = OptionProperty(
        'descending', options=['ascending', 'descending']
    )
    card_size_hint_x = BoundedNumericProperty(0.1, min=0, max=1)
    card_size_hint_y = BoundedNumericProperty(0.2, min=0, max=1)
    card_size_hint = ReferenceListProperty(card_size_hint_x, card_size_hint_y)
    starting_pos_hint = DictProperty({'x': 0.05, 'top': 0.95})
    x_hint_step = NumericProperty(0.01)
    y_hint_step = NumericProperty(-0.01)
    hint_step = ReferenceListProperty(x_hint_step, y_hint_step)
    insertion_point = NumericProperty(None, allownone=True)
    grabbed = ObjectProperty(None, allownone=True)
    insertable = BooleanProperty(True)
    cards = ListProperty([])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            insertion_point=self._trigger_layout
        )

    def point_is_before_zeroth_card(self, zeroth, x, y):
        """While dragging a card, if you drag it past my zeroth card, you want
        to insert your card in position zero. This function detects
        this situation.

        """
        def ycmp():
            if self.y_hint_step == 0:
                return False
            elif self.y_hint_step > 0:
                # stacking upward
                return y < zeroth.y
            else:
                # stacking downward
                return y > zeroth.top
        if self.x_hint_step > 0:
            # stacking to the right
            if x < zeroth.x:
                return True
            return ycmp()
        elif self.x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x > zeroth.right:
                return True
            return ycmp()

    def point_is_after_last_card(self, last, x, y):
        def ycmp():
            if self.y_hint_step == 0:
                return False
            elif self.y_hint_step > 0:
                # stacking upward
                return y > last.top
            else:
                # stacking downward
                return y < last.y
        if self.x_hint_step > 0:
            # stacking to the right
            if x > last.right:
                return True
            return ycmp()
        elif self.x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x < last.x:
                return True
            return ycmp()

    def add_widget(self, wid, index=0):
        if isinstance(wid, Card):
            self.cards.append(wid)
            if wid.layout is not self:
                wid.layout = self
        super().add_widget(wid, index)
        self._trigger_layout()

    def remove_widget(self, wid):
        if isinstance(wid, Card):
            self.cards.remove(wid)
        super().remove_widget(wid)
        self._trigger_layout()

    def on_parent(self, *args):
        if self.parent is not None:
            self._trigger_layout()

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):
            return
        i = 0
        childs = [c for c in self.cards if not c.dragging]
        if self.direction == 'ascending':
            childs.reverse()
        for child in childs:
            if child.collide_point(*touch.pos):
                self.insertion_point = i
                return
            i += 1
        else:
            self.insertion_point = None
            if self.point_is_after_last_card(
                    childs[-1], *touch.pos
            ):
                self.insertion_point = len(self.cards)
            elif self.point_is_before_zeroth_card(
                    childs[0], *touch.pos
            ):
                self.insertion_point = 0

    def on_touch_up(self, *args):
        self.insertion_point = None
        super().on_touch_up(*args)

    def on_insertion_point(self, *args):
        Logger.debug(
            'DeckLayout: insertion point set to {}'.format(
                self.insertion_point
            )
        )

    def do_layout(self, *args):
        if self.size == [1, 1]:
            return
        childs = list(self.cards)
        if self.grabbed in childs:
            childs.remove(self.grabbed)
        inspt = self.insertion_point
        if self.direction == 'ascending':
            childs.reverse()
            if inspt is not None:
                if inspt == -1:
                    childs.insert(0, None)
                elif inspt == 0:
                    childs.append(None)
                else:
                    childs.insert(len(childs) - inspt - 1, None)
        else:
            if inspt is not None:
                childs.insert(inspt, None)
        pos_hint = dict(self.starting_pos_hint)
        (w, h) = self.size
        (x, y) = self.pos
        for child in childs:
            if child is not None and child is not self.grabbed:
                shw, shh = child.size_hint = self.card_size_hint
                child.pos_hint = pos_hint
                child.size = w * shw, h * shh
                for (k, v) in pos_hint.items():
                    if k == 'x':
                        child.x = x + v * w
                    elif k == 'right':
                        child.right = x + v * w
                    elif k == 'pos':
                        child.pos = x + v[0] * w, y + v[1] * h
                    elif k == 'y':
                        child.y = y + v * h
                    elif k == 'top':
                        child.top = y + v * h
                    elif k == 'center':
                        child.center = x + v[0] * w, y + v[1] * h
                    elif k == 'center_x':
                        child.center_x = x + v * w
                    elif k == 'center_y':
                        child.center_y = y + v * h
            for xkey in (
                    'x',
                    'center_x',
                    'right'
            ):
                if xkey in pos_hint:
                    pos_hint[xkey] += self.x_hint_step
            for ykey in (
                    'y',
                    'center_y',
                    'top'
            ):
                if ykey in pos_hint:
                    pos_hint[ykey] += self.y_hint_step
            if 'pos' in pos_hint:
                (phx, phy) = pos_hint['pos']
                phx += self.x_hint_step
                phy += self.y_hint_step
                pos_hint['pos'] = (phx, phy)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.modules import inspector
    deck0 = DeckLayout()
    deck1 = DeckLayout()
    for i in range(0, 10):
        deck0.add_widget(
            Card(
                background_color=[1, 0, 0, 1],
                foreground_color=[0, 1, 0, 1],
                art_color=[0, 0, 1, 1],
                text='The quick brown fox jumps over the lazy dog',
                show_art=False,
                show_midline=False,
                show_footer=False,
                headline_text='Card {}'.format(i)
            )
        )
        deck1.add_widget(
            Card(
                background_color=[0, 1, 0, 1],
                foreground_color=[1, 0, 0, 1],
                show_art=False,
                text='Lorem ipsum dolor sit amet',
                show_midline=False,
                show_footer=False,
                headline_text='Card {}'.format(i)
            )
        )

    class TestLayout(BoxLayout):
        def __init__(self, **kwargs):
            kwargs['orientation'] = 'vertical'
            super().__init__(**kwargs)
            self.add_widget(deck0)
            self.add_widget(deck1)

        def on_touch_move(self, touch):
            if 'card' in touch.ud:
                for deck in (deck0, deck1):
                    deck.dispatch('on_touch_move', touch)

    layout = TestLayout()
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
