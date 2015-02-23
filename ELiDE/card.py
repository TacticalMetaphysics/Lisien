# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Cards that can be assembled into piles, which can be fanned out or
stacked into decks, which can then be drawn from."""
from kivy.adapters.listadapter import ListAdapter
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    ReferenceListProperty,
    StringProperty,
)
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


class Card(FloatLayout):
    ud = DictProperty({})
    dragging = BooleanProperty(False)
    idx = NumericProperty()

    collide_x = NumericProperty()
    collide_y = NumericProperty()
    collide_pos = ReferenceListProperty(collide_x, collide_y)

    foreground = ObjectProperty()
    foreground_source = StringProperty('')
    foreground_color = ListProperty([1, 1, 1, 1])
    foreground_image = ObjectProperty(None, allownone=True)
    foreground_texture = ObjectProperty(None, allownone=True)

    background_source = StringProperty('')
    background_color = ListProperty([1, 1, 1, 1])
    background_image = ObjectProperty(None, allownone=True)
    background_texture = ObjectProperty(None, allownone=True)

    art = ObjectProperty()
    art_source = StringProperty('')
    art_color = ListProperty([1, 1, 1, 1])
    art_image = ObjectProperty(None, allownone=True)
    art_texture = ObjectProperty(None, allownone=True)
    show_art = BooleanProperty(True)

    headline = ObjectProperty()
    headline_text = StringProperty('Headline')
    headline_markup = BooleanProperty(True)
    headline_shorten = BooleanProperty(True)
    headline_font_name = StringProperty('DroidSans')
    headline_font_size = NumericProperty(18)
    headline_color = ListProperty([0, 0, 0, 1])

    midline = ObjectProperty()
    midline_text = StringProperty('')
    midline_markup = BooleanProperty(True)
    midline_shorten = BooleanProperty(True)
    midline_font_name = StringProperty('DroidSans')
    midline_font_size = NumericProperty(14)
    midline_color = ListProperty([0, 0, 0, 1])

    footer = ObjectProperty()
    footer_text = StringProperty('')
    footer_markup = BooleanProperty(True)
    footer_shorten = BooleanProperty(True)
    footer_font_name = StringProperty('DroidSans')
    footer_font_size = NumericProperty(10)
    footer_color = ListProperty([0, 0, 0, 1])

    text = StringProperty('')
    text_color = ListProperty([0, 0, 0, 1])
    markup = BooleanProperty(True)
    shorten = BooleanProperty(True)
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)

    def get_kwargs(self):
        kwargnames = (
            'ud',
            'foreground_source',
            'foreground_color',
            'background_source',
            'background_color',
            'art_source',
            'art_color',
            'show_art',
            'headline_text',
            'headline_markup',
            'headline_shorten',
            'headline_font_name',
            'headline_font_size',
            'headline_color',
            'midline_text',
            'midline_markup',
            'midline_font_name',
            'midline_font_size',
            'midline_color',
            'footer_text',
            'footer_markup',
            'footer_shorten',
            'footer_font_name',
            'footer_font_size',
            'footer_color',
            'text',
            'text_color',
            'markup',
            'shorten',
            'font_name',
            'font_size'
        )
        return {
            k: getattr(self, k)
            for k in kwargnames
        }

    def on_background_source(self, *args):
        if self.background_source:
            self.background_image = Image(source=self.background_source)

    def on_background_image(self, *args):
        if self.background_image is not None:
            self.background_texture = self.background_image.texture

    def on_foreground_source(self, *args):
        if self.foreground_source:
            self.foreground_image = Image(source=self.foreground_source)

    def on_foreground_image(self, *args):
        if self.foreground_image is not None:
            self.foreground_texture = self.foreground_image.texture

    def on_art_source(self, *args):
        if self.art_source:
            self.art_image = Image(source=self.art_source)

    def on_art_image(self, *args):
        if self.art_image is not None:
            self.art_texture = self.art_image.texture

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return
        if 'card' in touch.ud:
            return
        touch.grab(self)
        self.dragging = True
        touch.ud['card'] = self.get_kwargs()
        touch.ud['idx'] = self.idx
        touch.ud['layout'] = self.parent
        self.collide_x = touch.x - self.x
        self.collide_y = touch.y - self.y

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
        touch.ungrab(self)
        self.dragging = False


class DeckLayout(Layout):
    adapter = ObjectProperty()
    direction = OptionProperty(
        'ascending', options=['ascending', 'descending']
    )
    card_size_hint_x = NumericProperty()
    card_size_hint_y = NumericProperty()
    card_size_hint = ReferenceListProperty(card_size_hint_x, card_size_hint_y)
    starting_pos_hint = DictProperty()
    x_hint_step = NumericProperty()
    y_hint_step = NumericProperty()
    hint_step = ReferenceListProperty(x_hint_step, y_hint_step)
    insertion_point = NumericProperty(None, allownone=True)
    insertable = BooleanProperty()
    deletable = BooleanProperty()

    def on_parent(self, *args):
        if self.parent is not None:
            self._trigger_layout()

    def on_children(self, *args):
        self._trigger_layout()

    def on_insertion_point(self, *args):
        if self.insertion_point is not None:
            self._trigger_layout()

    def point_before_card(self, card, x, y):
        def ycmp():
            if self.y_hint_step == 0:
                return False
            elif self.y_hint_step > 0:
                # stacking upward
                return y < card.y
            else:
                # stacking downward
                return y > card.top
        if self.x_hint_step > 0:
            # stacking to the right
            if x < card.x:
                return True
            return ycmp()
        elif self.x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x > card.right:
                return True
            return ycmp()

    def point_after_card(self, card, x, y):
        def ycmp():
            if self.y_hint_step == 0:
                return False
            elif self.y_hint_step > 0:
                # stacking upward
                return y > card.top
            else:
                # stacking downward
                return y < card.y
        if self.x_hint_step > 0:
            # stacking to the right
            if x > card.right:
                return True
            return ycmp()
        elif self.x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x < card.x:
                return True
            return ycmp()

    def on_touch_move(self, touch):
        if 'card' not in touch.ud:
            return
        if not self.insertable and touch.ud['layout'] is not self:
            return
        childs = [c for c in self.children if not c.dragging]
        i = len(childs)
        if self.direction == 'descending':
            childs.reverse()
        for child in childs:
            if child.collide_point(*touch.pos):
                self.insertion_point = i
                return
            i -= 1
        else:
            if self.insertion_point in (0, len(self.children)):
                return
            if self.point_before_card(
                    self.children[0], *touch.pos
            ):
                self.insertion_point = len(self.children)
            elif self.point_after_card(
                    self.children[-1], *touch.pos
            ):
                self.insertion_point = 0

    def on_touch_up(self, touch):
        if 'card' not in touch.ud:
            return
        if not self.insertable and touch.ud['layout'] is not self:
            return
        if self.insertion_point is not None and self.collide_point(*touch.pos):
            if touch.ud['layout'].deletable:
                del touch.ud['layout'].adapter.data[touch.ud['idx']]
            self.adapter.data.insert(self.insertion_point, touch.ud['card'])
            self.insertion_point = None

    def do_layout(self, *args):
        if self.size == [1, 1]:
            return
        childs = list(self.children)
        inspt = self.insertion_point
        if inspt is not None and inspt > len(childs):
            childs.append(None)
        dragidx = None
        i = 0
        for child in childs:
            if child.dragging:
                dragidx = i
                break
            i += 1
        if dragidx is not None:
            del childs[dragidx]
        if inspt is not None:
            childs.insert(len(childs) - inspt, None)
        if self.direction == 'descending':
            childs.reverse()
        pos_hint = dict(self.starting_pos_hint)
        (w, h) = self.size
        (x, y) = self.pos
        for child in childs:
            if child is not None:
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


kv = """
<ColorTextureBox>:
    canvas:
        Color:
            rgba: root.color
        Rectangle:
            texture: root.texture
            pos: root.pos
            size: root.size
        Color:
            rgba: [1, 1, 1, 1]
<Card>:
    headline: headline
    midline: midline
    footer: footer
    art: art
    foreground: foreground
    ColorTextureBox:
        color: root.background_color
        texture: root.background_texture
        pos: root.pos
        size: root.size
        BoxLayout:
            size_hint: 0.9, 0.9
            pos_hint: {'x': 0.05, 'y': 0.05}
            orientation: 'vertical'
            Label:
                id: headline
                text: root.headline_text
                markup: root.headline_markup
                shorten: root.headline_shorten
                font_name: root.headline_font_name
                font_size: root.headline_font_size
                color: root.headline_color
                size_hint: (None, None)
                size: self.texture_size
            ColorTextureBox:
                id: art
                color: root.art_color
                texture: root.art_texture
                size_hint: (1, 1) if root.show_art else (None, None)
                size: (0, 0)
            Label:
                id: midline
                text: root.midline_text
                markup: root.midline_markup
                shorten: root.midline_shorten
                font_name: root.midline_font_name
                font_size: root.midline_font_size
                color: root.midline_color
                size_hint: (None, None)
                size: self.texture_size
            ColorTextureBox:
                id: foreground
                color: root.foreground_color
                texture: root.foreground_texture
                Label:
                    text: root.text
                    color: root.text_color
                    markup: root.markup
                    shorten: root.shorten
                    font_name: root.font_name
                    font_size: root.font_size
                    text_size: foreground.size
                    size_hint: (None, None)
                    size: self.texture_size
                    pos: foreground.pos
                    valign: 'top'
            Label:
                id: footer
                text: root.footer_text
                markup: root.footer_markup
                shorten: root.footer_shorten
                font_name: root.footer_font_name
                font_size: root.footer_font_size
                color: root.footer_color
                size_hint: (None, None)
                size: self.texture_size
"""
Builder.load_string(kv)


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.modules import inspector
    data0 = [
        {
            'background_color': [1, 0, 0, 1],
            'foreground_color': [0, 1, 0, 1],
            'text': 'The quick brown fox jumps over the lazy dog',
            'show_art': False,
            'midline_text': '',
            'midline_font_size': 0,
            'footer_text': str(i),
            'headline_text': 'Card {}'.format(i)
        }
        for i in range(0, 9)
    ]
    data1 = [
        {
            'background_color': [0, 1, 0, 1],
            'foreground_color': [1, 0, 0, 1],
            'text': 'Lorem ipsum dolor sit amet',
            'show_art': False,
            'midline_text': '',
            'midline_font_size': 0,
            'footer_text': str(i),
            'headline_text': 'Card {}'.format(i)
        }
        for i in range(0, 9)
    ]

    def args_converter(i, kv):
        kv['idx'] = i
        return kv
    adapter0 = ListAdapter(data=data0, cls=Card, args_converter=args_converter)
    adapter1 = ListAdapter(data=data1, cls=Card, args_converter=args_converter)
    layout = BoxLayout()
    layout.add_widget(DeckView(adapter=adapter0))
    layout.add_widget(DeckView(adapter=adapter1))
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
