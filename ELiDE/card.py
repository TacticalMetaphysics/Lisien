# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.clock import Clock
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
    BoundedNumericProperty
)
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


def get_pos_hint(poshints, sizehintx, sizehinty):
    return (
        get_pos_hint_x(poshints, sizehintx),
        get_pos_hint_y(poshints, sizehinty)
    )


class ColorTextureBox(FloatLayout):
    color = ListProperty([1, 1, 1, 1])
    texture = ObjectProperty(None, allownone=True)


class Card(FloatLayout):
    dragging = BooleanProperty(False)
    deck = NumericProperty()
    idx = NumericProperty()
    ud = DictProperty({})

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
        touch.ud['card'] = self
        touch.ud['idx'] = self.idx
        touch.ud['deck'] = self.deck
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


class DeckBuilderLayout(Layout):
    direction = OptionProperty(
        'ascending', options=['ascending', 'descending']
    )
    card_size_hint_x = BoundedNumericProperty(1, min=0, max=1)
    card_size_hint_y = BoundedNumericProperty(1, min=0, max=1)
    card_size_hint = ReferenceListProperty(card_size_hint_x, card_size_hint_y)
    starting_pos_hint = DictProperty({'x': 0, 'y': 0})
    card_x_hint_step = NumericProperty(0)
    card_y_hint_step = NumericProperty(-1)
    card_hint_step = ReferenceListProperty(card_x_hint_step, card_y_hint_step)
    deck_x_hint_step = NumericProperty(1)
    deck_y_hint_step = NumericProperty(0)
    deck_hint_step = ReferenceListProperty(deck_x_hint_step, deck_y_hint_step)
    decks = ListProperty([[]])  # list of lists of cards
    insertion_deck = BoundedNumericProperty(None, min=0, allownone=True)
    insertion_card = BoundedNumericProperty(None, min=0, allownone=True)

    def on_decks(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.on_decks, 0)
            return
        decknum = 0
        for deck in self.decks:
            cardnum = 0
            for card in deck:
                if not isinstance(card, Card):
                    raise TypeError("You must only put Card in decks")
                if card not in self.children:
                    self.add_widget(card)
                if card.deck != decknum:
                    card.deck = decknum
                if card.idx != cardnum:
                    card.idx = cardnum
                cardnum += 1
            decknum += 1
        self._trigger_layout()

    def point_before_card(self, card, x, y):
        def ycmp():
            if self.card_y_hint_step == 0:
                return False
            elif self.card_y_hint_step > 0:
                # stacking upward
                return y < card.y
            else:
                # stacking downward
                return y > card.top
        if self.card_x_hint_step > 0:
            # stacking to the right
            if x < card.x:
                return True
            return ycmp()
        elif self.card_x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x > card.right:
                return True
            return ycmp()

    def point_after_card(self, card, x, y):
        def ycmp():
            if self.card_y_hint_step == 0:
                return False
            elif self.card_y_hint_step > 0:
                # stacking upward
                return y > card.top
            else:
                # stacking downward
                return y < card.y
        if self.card_x_hint_step > 0:
            # stacking to the right
            if x > card.right:
                return True
            return ycmp()
        elif self.card_x_hint_step == 0:
            return ycmp()
        else:
            # stacking to the left
            if x < card.x:
                return True
            return ycmp()

    def on_touch_move(self, touch):
        if (
                'card' not in touch.ud or
                'layout' not in touch.ud or
                touch.ud['layout'] != self
        ):
            return
        if (
                touch.ud['layout'] == self and
                not hasattr(touch.ud['card'], '_topdecked')
        ):
            self.canvas.after.add(touch.ud['card'].canvas)
            touch.ud['card']._topdecked = True
        i = 0
        for deck in self.decks:
            cards = [card for card in deck if not card.dragging]
            maxidx = max(card.idx for card in cards)
            if self.direction == 'descending':
                cards.reverse()
            cards_collided = [
                card for card in cards if card.collide_point(*touch.pos)
            ]
            if cards_collided:
                collided = cards_collided.pop()
                for card in cards_collided:
                    if card.idx > collided.idx:
                        collided = card
                if collided.deck == touch.ud['deck']:
                    self.insertion_card = (
                        1 if collided.idx == 0 else
                        maxidx + 1 if collided.idx == maxidx else
                        collided.idx + 1 if collided.idx > touch.ud['idx']
                        else collided.idx
                    )
                else:
                    dropdeck = self.decks[collided.deck]
                    maxidx = max(card.idx for card in dropdeck)
                    self.insertion_card = (
                        1 if collided.idx == 0 else
                        maxidx + 1 if collided.idx == maxidx else
                        collided.idx + 1
                    )
                if self.insertion_deck != collided.deck:
                    self.insertion_deck = collided.deck
                return
            else:
                if self.insertion_deck == i:
                    if self.insertion_card in (0, len(deck)):
                        pass
                    elif self.point_before_card(
                            cards[0], *touch.pos
                    ):
                        self.insertion_card = 0
                    elif self.point_after_card(
                        cards[-1], *touch.pos
                    ):
                        self.insertion_card = cards[-1].idx
            i += 1

    def on_touch_up(self, touch):
        if (
                'card' not in touch.ud or
                'layout' not in touch.ud or
                touch.ud['layout'] != self
        ):
            return
        if hasattr(touch.ud['card'], '_topdecked'):
            self.canvas.after.remove(touch.ud['card'].canvas)
            del touch.ud['card']._topdecked
        if None not in (self.insertion_deck, self.insertion_card):
            # need to sync to adapter.data??
            card = self.decks[touch.ud['deck']][touch.ud['idx']]
            del self.decks[touch.ud['deck']][touch.ud['idx']]
            deck = self.decks[self.insertion_deck]
            if self.insertion_card > len(deck):
                deck.append(card)
            else:
                deck.insert(self.insertion_card, card)
            card.deck = self.insertion_deck
            card.idx = self.insertion_card
            self.insertion_deck = self.insertion_card = None
        self._trigger_layout()

    def on_insertion_card(self, *args):
        if self.insertion_card is not None:
            self._trigger_layout()

    def do_layout(self, *args):
        if self.size == [1, 1]:
            return
        for i in range(0, len(self.decks)):
            self.layout_deck(i)

    def layout_deck(self, i):
        def get_dragidx(cards):
            j = 0
            for card in cards:
                if card.dragging:
                    return j
                j += 1
        # Put a None in the card list in place of the card you're
        # hovering over, if you're dragging another card. This will
        # result in an empty space where the card will go if you drop
        # it now.
        cards = list(self.decks[i])
        dragidx = get_dragidx(cards)
        if dragidx is not None:
            del cards[dragidx]
        for card in cards:
            self.remove_widget(card)
        if self.insertion_deck == i and self.insertion_card is not None:
            insdx = self.insertion_card
            if dragidx is not None and insdx > dragidx:
                insdx -= 1
            cards.insert(insdx, None)
        if self.direction == 'descending':
            cards.reverse()
        # Work out the initial pos_hint for this deck
        (phx, phy) = get_pos_hint(self.starting_pos_hint, *self.card_size_hint)
        phx += self.deck_x_hint_step * i
        phy += self.deck_y_hint_step * i
        (w, h) = self.size
        (x, y) = self.pos
        # start assigning pos and size to cards
        for card in cards:
            if card is not None:
                (shw, shh) = self.card_size_hint
                card.pos = (
                    x + phx * w,
                    y + phy * h
                )
                card.size = (w * shw, h * shh)
                self.add_widget(card)
            phx += self.card_x_hint_step
            phy += self.card_y_hint_step


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
    canvas:
        Color:
            rgba: root.background_color
        Rectangle:
            texture: root.background_texture
            pos: root.pos
            size: root.size
        Color:
            rgba: [1, 1, 1, 1]
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
    deck0 = [
        Card(
            background_color=[0, 1, 0, 1],
            headline_text='Card {}'.format(i),
            art_color=[1, 0, 0, 1],
            midline_text='0deck',
            foreground_color=[0, 0, 1, 1],
            text='The quick brown fox jumps over the lazy dog',
            text_color=[1, 1, 1, 1],
            footer_text=str(i)
        )
        for i in range(0, 9)
    ]
    deck1 = [
        Card(
            background_color=[0, 0, 1, 1],
            headline_text='Card {}'.format(i),
            art_color=[0, 1, 0, 1],
            midline_text='1deck',
            foreground_color=[1, 0, 0, 1],
            text='Have a steak at the porter house bar',
            text_color=[1, 1, 0, 1],
            footer_text=str(i)
        )
        for i in range(0, 9)
    ]
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.modules import inspector
    layout = DeckBuilderLayout(
        card_size_hint=(0.15, 0.3),
        starting_pos_hint={'x': 0.1, 'top': 0.9},
        card_hint_step=(0.05, -0.1),
        deck_hint_step=(0.4, 0),
        decks=[deck0, deck1]
    )
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
