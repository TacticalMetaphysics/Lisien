# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Widget that looks like a trading card, and a layout within which it
can be dragged and dropped to some particular position within stacks
of other cards.

"""


from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    AliasProperty,
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
from kivy.graphics import InstructionGroup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.layout import Layout
from kivy.uix.stencilview import StencilView
from kivy.uix.image import Image
from kivy.uix.widget import Widget


dbg = Logger.debug


def get_pos_hint_x(poshints, sizehintx):
    """Return ``poshints['x']`` if available, or its computed equivalent
    otherwise.

    """
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
    """Return ``poshints['y']`` if available, or its computed equivalent
    otherwise.

    """
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
    """Return a tuple of ``(pos_hint_x, pos_hint_y)`` even if neither of
    those keys are present in the provided ``poshints`` -- they can be
    computed using the available keys together with ``size_hint_x``
    and ``size_hint_y``.

    """
    return (
        get_pos_hint_x(poshints, sizehintx),
        get_pos_hint_y(poshints, sizehinty)
    )


class ColorTextureBox(Widget):
    """A box, with a background of one solid color, an outline of another
    color, and possibly a texture covering the background.

    """
    color = ListProperty([1, 1, 1, 1])
    outline_color = ListProperty([0, 0, 0, 0])
    texture = ObjectProperty(None, allownone=True)


class Card(FloatLayout):
    """A trading card, similar to the kind you use to play games like
    _Magic: the Gathering_.

    Its appearance is determined by several properties, the most
    important being:

    * ``headline_text``, a string to be shown at the top of the card;
      may be styled with eg. ``headline_font_name`` or
      ``headline_color``

    * ``art_source``, the path to an image to be displayed below the
      headline; may be hidden by setting ``show_art`` to ``False``

    * ``midline_text``, similar to ``headline_text`` but appearing
      below the art

    * ``text``, shown in a box the same size as the art. Styleable
      like ``headline_text`` and you can customize the box with
      eg. ``foreground_color`` and ``foreground_source``

    * ``footer_text``, like ``headline_text`` but at the bottom

    :class:`Card` is particularly useful when put in a
    :class:`DeckLayout`, allowing the user to drag cards in between
    any number of piles, into particular positions within a particular
    pile, and so forth.

    """
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
    background_color = ListProperty([.7, .7, .7, 1])
    background_image = ObjectProperty(None, allownone=True)
    background_texture = ObjectProperty(None, allownone=True)

    outline_color = ListProperty([0, 0, 0, 1])
    content_outline_color = ListProperty([0, 0, 0, 0])
    foreground_outline_color = ListProperty([0, 0, 0, 1])
    art_outline_color = ListProperty([0, 0, 0, 0])

    art = ObjectProperty()
    art_source = StringProperty('')
    art_color = ListProperty([1, 1, 1, 1])
    art_image = ObjectProperty(None, allownone=True)
    art_texture = ObjectProperty(None, allownone=True)
    show_art = BooleanProperty(True)

    headline = ObjectProperty()
    headline_text = StringProperty('Headline')
    headline_markup = BooleanProperty(True)
    headline_font_name = StringProperty('Roboto-Regular')
    headline_font_size = NumericProperty(18)
    headline_color = ListProperty([0, 0, 0, 1])

    midline = ObjectProperty()
    midline_text = StringProperty('')
    midline_markup = BooleanProperty(True)
    midline_font_name = StringProperty('Roboto-Regular')
    midline_font_size = NumericProperty(14)
    midline_color = ListProperty([0, 0, 0, 1])

    footer = ObjectProperty()
    footer_text = StringProperty('')
    footer_markup = BooleanProperty(True)
    footer_font_name = StringProperty('Roboto-Regular')
    footer_font_size = NumericProperty(10)
    footer_color = ListProperty([0, 0, 0, 1])

    text = StringProperty('')
    text_color = ListProperty([0, 0, 0, 1])
    markup = BooleanProperty(True)
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)

    def on_background_source(self, *args):
        """When I get a new ``background_source``, load it as an
        :class:`Image` and store that in ``background_image``.

        """
        if self.background_source:
            self.background_image = Image(source=self.background_source)

    def on_background_image(self, *args):
        """When I get a new ``background_image``, store its texture in
        ``background_texture``.

        """
        if self.background_image is not None:
            self.background_texture = self.background_image.texture

    def on_foreground_source(self, *args):
        """When I get a new ``foreground_source``, load it as an
        :class:`Image` and store that in ``foreground_image``.

        """
        if self.foreground_source:
            self.foreground_image = Image(source=self.foreground_source)

    def on_foreground_image(self, *args):
        """When I get a new ``foreground_image``, store its texture in my
        ``foreground_texture``.

        """
        if self.foreground_image is not None:
            self.foreground_texture = self.foreground_image.texture

    def on_art_source(self, *args):
        """When I get a new ``art_source``, load it as an :class:`Image` and
        store that in ``art_image``.

        """
        if self.art_source:
            self.art_image = Image(source=self.art_source)

    def on_art_image(self, *args):
        """When I get a new ``art_image``, store its texture in
        ``art_texture``.

        """
        if self.art_image is not None:
            self.art_texture = self.art_image.texture

    def on_touch_down(self, touch):
        """If I'm the first card to collide this touch, grab it, store my
        metadata in its userdict, and store the relative coords upon
        me where the collision happened.

        """
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
        """If I'm being dragged, move so as to be always positioned the same
        relative to the touch.

        """
        if not self.dragging:
            touch.ungrab(self)
            return
        self.pos = (
            touch.x - self.collide_x,
            touch.y - self.collide_y
        )

    def on_touch_up(self, touch):
        """Stop dragging if needed."""
        if not self.dragging:
            return
        touch.ungrab(self)
        self.dragging = False

    def copy(self):
        """Return a new :class:`Card` just like me."""
        d = {}
        for att in (
                'deck',
                'idx',
                'ud',
                'foreground_source',
                'foreground_color',
                'foreground_image',
                'foreground_texture',
                'background_source',
                'background_color',
                'background_image',
                'background_texture',
                'outline_color',
                'content_outline_color',
                'foreground_outline_color',
                'art_outline_color',
                'art_source',
                'art_color',
                'art_image',
                'art_texture',
                'show_art',
                'headline_text',
                'headline_markup',
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
                'footer_font_name',
                'footer_font_size',
                'footer_color',
                'text',
                'text_color',
                'markup',
                'font_name',
                'font_size'
        ):
            v = getattr(self, att)
            if v is not None:
                d[att] = v
        return Card(**d)


class Foundation(ColorTextureBox):
    """An empty outline to indicate where a deck is when there are no
    cards in it.

    """
    color = ListProperty([])
    """Color of the outline"""
    deck = NumericProperty(0)
    """Index of the deck in the parent :class:`DeckLayout`"""

    def upd_pos(self, *args):
        """Ask the foundation where I should be, based on what deck I'm
        for.

        """
        self.pos = self.parent._get_foundation_pos(self.deck)

    def upd_size(self, *args):
        """I'm the same size as any given card in my :class:`DeckLayout`."""
        self.size = (
            self.parent.card_size_hint_x * self.parent.width,
            self.parent.card_size_hint_y * self.parent.height
        )


class DeckBuilderLayout(Layout):
    """Sizes and positions :class:`Card` objects based on their order
    within ``decks``, a list of lists where each sublist is a deck of
    cards.

    """
    direction = OptionProperty(
        'ascending', options=['ascending', 'descending']
    )
    """Should the beginning card of each deck appear on the bottom
    ('ascending'), or the top ('descending')?

    """
    card_size_hint_x = BoundedNumericProperty(1, min=0, max=1)
    """Each card's width, expressed as a proportion of my width."""
    card_size_hint_y = BoundedNumericProperty(1, min=0, max=1)
    """Each card's height, expressed as a proportion of my height."""
    card_size_hint = ReferenceListProperty(card_size_hint_x, card_size_hint_y)
    """Size hint of cards, relative to my size."""
    starting_pos_hint = DictProperty({'x': 0, 'y': 0})
    """Pos hint at which to place the initial card of the initial deck."""
    card_x_hint_step = NumericProperty(0)
    """Each time I put another card on a deck, I'll move it this much of
    my width to the right of the previous card.

    """
    card_y_hint_step = NumericProperty(-1)
    """Each time I put another card on a deck, I'll move it this much of
    my height above the previous card.

    """
    card_hint_step = ReferenceListProperty(card_x_hint_step, card_y_hint_step)
    """An offset, expressed in proportion to my size, applied to each
    successive card in a given deck.

    """
    deck_x_hint_step = NumericProperty(1)
    """When I start a new deck, it will be this far to the right of the
    previous deck, expressed as a proportion of my width.

    """
    deck_y_hint_step = NumericProperty(0)
    """When I start a new deck, it will be this far above the previous
    deck, expressed as a proportion of my height.

    """
    deck_hint_step = ReferenceListProperty(deck_x_hint_step, deck_y_hint_step)
    """Offset of each deck with respect to the previous, as a proportion
    of my size.

    """
    decks = ListProperty([[]])  # list of lists of cards
    """Put a list of lists of :class:`Card` objects here and I'll position
    them appropriately. Please don't use ``add_widget``.

    """
    deck_x_hint_offsets = ListProperty([])
    """An additional proportional x-offset for each deck, defaulting to 0."""
    deck_y_hint_offsets = ListProperty([])
    """An additional proportional y-offset for each deck, defaulting to 0."""
    foundation_color = ListProperty([1, 1, 1, 1])
    """Color to use for the outline showing where a deck is when it's
    empty.

    """
    insertion_deck = BoundedNumericProperty(None, min=0, allownone=True)
    """Index of the deck that a card is being dragged into."""
    insertion_card = BoundedNumericProperty(None, min=0, allownone=True)
    """Index within the current deck that a card is being dragged into."""
    _foundations = ListProperty([])
    """Private. A list of :class:`Foundation` widgets, one per deck."""

    def __init__(self, **kwargs):
        """Bind most of my custom properties to ``_trigger_layout``."""
        super().__init__(**kwargs)
        self.bind(
            card_size_hint=self._trigger_layout,
            starting_pos_hint=self._trigger_layout,
            card_hint_step=self._trigger_layout,
            deck_hint_step=self._trigger_layout,
            decks=self._trigger_layout,
            deck_x_hint_offsets=self._trigger_layout,
            deck_y_hint_offsets=self._trigger_layout,
            insertion_deck=self._trigger_layout,
            insertion_card=self._trigger_layout
        )

    def scroll_deck_x(self, decknum, scroll_x):
        """Move a deck left or right."""
        if decknum >= len(self.decks):
            raise IndexError("I have no deck at {}".format(decknum))
        if decknum >= len(self.deck_x_hint_offsets):
            self.deck_x_hint_offsets = list(self.deck_x_hint_offsets) + [0] * (
                decknum - len(self.deck_x_hint_offsets) + 1
            )
        self.deck_x_hint_offsets[decknum] += scroll_x
        self._trigger_layout()

    def scroll_deck_y(self, decknum, scroll_y):
        """Move a deck up or down."""
        if decknum >= len(self.decks):
            raise IndexError("I have no deck at {}".format(decknum))
        if decknum >= len(self.deck_y_hint_offsets):
            self.deck_y_hint_offsets = list(self.deck_y_hint_offsets) + [0] * (
                decknum - len(self.deck_y_hint_offsets) + 1
            )
        self.deck_y_hint_offsets[decknum] += scroll_y
        self._trigger_layout()

    def scroll_deck(self, decknum, scroll_x, scroll_y):
        """Move a deck."""
        self.scroll_deck_x(decknum, scroll_x)
        self.scroll_deck_y(decknum, scroll_y)

    def _get_foundation_pos(self, i):
        """Private. Get the absolute coordinates to use for a deck's
        foundation, based on the ``starting_pos_hint``, the
        ``deck_hint_step``, ``deck_x_hint_offsets``, and
        ``deck_y_hint_offsets``.

        """
        (phx, phy) = get_pos_hint(
            self.starting_pos_hint, *self.card_size_hint
        )
        phx += self.deck_x_hint_step * i + self.deck_x_hint_offsets[i]
        phy += self.deck_y_hint_step * i + self.deck_y_hint_offsets[i]
        x = phx * self.width + self.x
        y = phy * self.height + self.y
        return (x, y)

    def _get_foundation(self, i):
        """Return a :class:`Foundation` for some deck, creating it if
        needed.

        """
        if i >= len(self._foundations) or self._foundations[i] is None:
            oldfound = list(self._foundations)
            extend = i - len(oldfound) + 1
            if extend > 0:
                oldfound += [None] * extend
            width = self.card_size_hint_x * self.width
            height = self.card_size_hint_y * self.height
            found = Foundation(
                pos=self._get_foundation_pos(i), size=(width, height), deck=i
            )
            self.bind(
                pos=found.upd_pos,
                card_size_hint=found.upd_pos,
                deck_hint_step=found.upd_pos,
                size=found.upd_pos,
                deck_x_hint_offsets=found.upd_pos,
                deck_y_hint_offsets=found.upd_pos
            )
            self.bind(
                size=found.upd_size,
                card_size_hint=found.upd_size
            )
            oldfound[i] = found
            self._foundations = oldfound
        return self._foundations[i]

    def on_decks(self, *args):
        """Inform the cards of their deck and their index within the deck;
        extend the ``_hint_offsets`` properties as needed; and trigger
        a layout.

        """
        if None in (
                self.canvas,
                self.decks,
                self.deck_x_hint_offsets,
                self.deck_y_hint_offsets
        ):
            Clock.schedule_once(self.on_decks, 0)
            return
        self.clear_widgets()
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
        if len(self.deck_x_hint_offsets) < len(self.decks):
            self.deck_x_hint_offsets = list(self.deck_x_hint_offsets) + [0] * (
                len(self.decks) - len(self.deck_x_hint_offsets)
            )
        if len(self.deck_y_hint_offsets) < len(self.decks):
            self.deck_y_hint_offsets = list(self.deck_y_hint_offsets) + [0] * (
                len(self.decks) - len(self.deck_y_hint_offsets)
            )
        self._trigger_layout()

    def point_before_card(self, card, x, y):
        """Return whether ``(x, y)`` is somewhere before ``card``, given how I
        know cards to be arranged.

        If the cards are being stacked down and to the right, that
        means I'm testing whether ``(x, y)`` is above or to the left
        of the card.

        """
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
        """Return whether ``(x, y)`` is somewhere after ``card``, given how I
        know cards to be arranged.

        If the cards are being stacked down and to the right, that
        means I'm testing whether ``(x, y)`` is below or to the left
        of ``card``.

        """
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
        """If a card is being dragged, move other cards out of the way to show
        where the dragged card will go if you drop it.

        """
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
            touch.ud['card']._topdecked = InstructionGroup()
            touch.ud['card']._topdecked.add(touch.ud['card'].canvas)
            self.canvas.after.add(touch.ud['card']._topdecked)
        for i, deck in enumerate(self.decks):
            cards = [card for card in deck if not card.dragging]
            maxidx = max(card.idx for card in cards) if cards else 0
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
                else:
                    for j, found in enumerate(self._foundations):
                        if (
                                found is not None and
                                found.collide_point(*touch.pos)
                        ):
                            self.insertion_deck = j
                            self.insertion_card = 0
                            return

    def on_touch_up(self, touch):
        """If a card is being dragged, put it in the place it was just dropped
        and trigger a layout.

        """
        if (
                'card' not in touch.ud or
                'layout' not in touch.ud or
                touch.ud['layout'] != self
        ):
            return
        if hasattr(touch.ud['card'], '_topdecked'):
            self.canvas.after.remove(touch.ud['card']._topdecked)
            del touch.ud['card']._topdecked
        if None not in (self.insertion_deck, self.insertion_card):
            # need to sync to adapter.data??
            card = touch.ud['card']
            del card.parent.decks[card.deck][card.idx]
            for i in range(0, len(card.parent.decks[card.deck])):
                card.parent.decks[card.deck][i].idx = i
            deck = self.decks[self.insertion_deck]
            if self.insertion_card >= len(deck):
                deck.append(card)
            else:
                deck.insert(self.insertion_card, card)
            card.deck = self.insertion_deck
            card.idx = self.insertion_card
            self.decks[self.insertion_deck] = deck
            self.insertion_deck = self.insertion_card = None
        self._trigger_layout()

    def on_insertion_card(self, *args):
        """Trigger a layout"""
        if self.insertion_card is not None:
            self._trigger_layout()

    def do_layout(self, *args):
        """Layout each of my decks"""
        if self.size == [1, 1]:
            return
        for i in range(0, len(self.decks)):
            self.layout_deck(i)

    def layout_deck(self, i):
        """Stack the cards, starting at my deck's foundation, and proceeding
        by ``card_pos_hint``

        """
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
        if self.insertion_deck == i and self.insertion_card is not None:
            insdx = self.insertion_card
            if dragidx is not None and insdx > dragidx:
                insdx -= 1
            cards.insert(insdx, None)
        if self.direction == 'descending':
            cards.reverse()
        # Work out the initial pos_hint for this deck
        (phx, phy) = get_pos_hint(self.starting_pos_hint, *self.card_size_hint)
        phx += self.deck_x_hint_step * i + self.deck_x_hint_offsets[i]
        phy += self.deck_y_hint_step * i + self.deck_y_hint_offsets[i]
        (w, h) = self.size
        (x, y) = self.pos
        # start assigning pos and size to cards
        found = self._get_foundation(i)
        if found in self.children:
            self.remove_widget(found)
        self.add_widget(found)
        for card in cards:
            if card is not None:
                if card in self.children:
                    self.remove_widget(card)
                (shw, shh) = self.card_size_hint
                card.pos = (
                    x + phx * w,
                    y + phy * h
                )
                card.size = (w * shw, h * shh)
                self.add_widget(card)
            phx += self.card_x_hint_step
            phy += self.card_y_hint_step


class DeckBuilderView(DeckBuilderLayout, StencilView):
    """Just a :class:`DeckBuilderLayout` mixed with
    :class:`StencilView`.

    """
    pass


class ScrollBarBar(ColorTextureBox):
    """Tiny tweak to :class:`ColorTextureBox` to make it work within
    :class:`DeckBuilderScrollBar`

    """
    def on_touch_down(self, touch):
        """Tell my parent if I've been touched"""
        if self.parent is None:
            return
        if self.collide_point(*touch.pos):
            self.parent.bar_touched(self, touch)


class DeckBuilderScrollBar(FloatLayout):
    """A widget that looks a lot like one of the scrollbars on the sides
    of eg. :class:`kivy.uix.ScrollView`, which moves a single deck
    within a :class:`DeckBuilderLayout`.

    """
    orientation = OptionProperty(
        'vertical',
        options=['horizontal', 'vertical']
    )
    """Which way to scroll? Options are 'horizontal' and 'vertical'."""
    deckbuilder = ObjectProperty()
    """The :class:`DeckBuilderLayout` of the deck to scroll."""
    deckidx = NumericProperty(0)
    """The index of the deck to scroll, within its
    :class:`DeckBuilderLayout`'s ``decks`` property.

    """
    scrolling = BooleanProperty(False)
    """Has the user grabbed me?"""
    scroll_min = NumericProperty(-1)
    """How far left (if horizontal) or down (if vertical) I can move my
    deck, expressed as a proportion of the
    :class:`DeckBuilderLayout`'s width or height, respectively.

    """
    scroll_max = NumericProperty(1)
    """How far right (if horizontal) or up (if vertical) I can move my
    deck, expressed as a proportion of the
    :class:`DeckBuilderLayout`'s width or height, respectively.

    """

    scroll_hint = AliasProperty(
        lambda self: abs(self.scroll_max-self.scroll_min),
        lambda self, v: None,
        bind=('scroll_min', 'scroll_max')
    )
    """The distance between ``scroll_max`` and ``scroll_min``."""
    _scroll = NumericProperty(0)
    """Private. The current adjustment to the deck's ``pos_hint_x`` or
    ``pos_hint_y``.

    """

    def _get_scroll(self):
        zero = self._scroll - self.scroll_min
        return zero / self.scroll_hint

    def _set_scroll(self, v):
        if v < 0:
            v = 0
        if v > 1:
            v = 1
        normal = v * self.scroll_hint
        self._scroll = self.scroll_min + normal

    scroll = AliasProperty(
        _get_scroll,
        _set_scroll,
        bind=('_scroll', 'scroll_min', 'scroll_max')
    )
    """A number between 0 and 1 representing how far beyond ``scroll_min``
    toward ``scroll_max`` I am presently scrolled.

    """

    def _get_vbar(self):
        if self.deckbuilder is None:
            return (0, 1)
        vh = self.deckbuilder.height * (self.scroll_hint + 1)
        h = self.height
        if vh < h or vh == 0:
            return (0, 1)
        ph = max(0.01, h / vh)
        sy = min(1.0, max(0.0, self.scroll))
        py = (1 - ph) * sy
        return (py, ph)
    vbar = AliasProperty(
        _get_vbar,
        None,
        bind=('_scroll', 'scroll_min', 'scroll_max')
    )
    """A tuple of ``(y, height)`` for my scroll bar, if it's vertical."""

    def _get_hbar(self):
        if self.deckbuilder is None:
            return (0, 1)
        vw = self.deckbuilder.width * self.scroll_hint
        w = self.width
        if vw < w or vw == 0:
            return (0, 1)
        pw = max(0.01, w / vw)
        sx = min(1.0, max(0.0, self.scroll))
        px = (1 - pw) * sx
        return (px, pw)

    hbar = AliasProperty(
        _get_hbar,
        None,
        bind=('_scroll', 'scroll_min', 'scroll_max')
    )
    """A tuple of ``(x, width)`` for my scroll bar, if it's horizontal."""
    bar_color = ListProperty([.7, .7, .7, .9])
    """Color to use for the scroll bar when scrolling. RGBA format."""
    bar_inactive_color = ListProperty([.7, .7, .7, .2])
    """Color to use for the scroll bar when not scrolling. RGBA format."""
    bar_texture = ObjectProperty(None, allownone=True)
    """Texture for the scroll bar, normally ``None``."""

    def __init__(self, **kwargs):
        """Arrange to be laid out whenever I'm scrolled or the range of my
        scrolling changes.

        """
        super().__init__(**kwargs)
        self.bind(
            _scroll=self._trigger_layout,
            scroll_min=self._trigger_layout,
            scroll_max=self._trigger_layout
        )

    def do_layout(self, *args):
        """Put the bar where it's supposed to be, and size it in proportion to
        the size of the scrollable area.

        """
        if 'bar' not in self.ids:
            Clock.schedule_once(self.do_layout)
            return
        if self.orientation == 'horizontal':
            self.ids.bar.size_hint_x = self.hbar[1]
            self.ids.bar.pos_hint = {'x': self.hbar[0], 'y': 0}
        else:
            self.ids.bar.size_hint_y = self.vbar[1]
            self.ids.bar.pos_hint = {'x': 0, 'y': self.vbar[0]}
        super().do_layout(*args)

    def upd_scroll(self, *args):
        """Update my own ``scroll`` property to where my deck is actually
        scrolled.

        """
        att = 'deck_{}_hint_offsets'.format(
            'x' if self.orientation == 'horizontal' else 'y'
        )
        self._scroll = getattr(self.deckbuilder, att)[self.deckidx]

    def on_deckbuilder(self, *args):
        """Bind my deckbuilder to update my ``scroll``, and my ``scroll`` to
        update my deckbuilder.

        """
        if self.deckbuilder is None:
            return
        att = 'deck_{}_hint_offsets'.format(
            'x' if self.orientation == 'horizontal' else 'y'
        )
        offs = getattr(self.deckbuilder, att)
        if len(offs) <= self.deckidx:
            Clock.schedule_once(self.on_deckbuilder, 0)
            return
        self.bind(scroll=self.handle_scroll)
        self.deckbuilder.bind(**{att: self.upd_scroll})
        self.upd_scroll()
        self.deckbuilder._trigger_layout()

    def handle_scroll(self, *args):
        """When my ``scroll`` changes, tell my deckbuilder how it's scrolled
        now.

        """
        if 'bar' not in self.ids:
            Clock.schedule_once(self.handle_scroll, 0)
            return
        att = 'deck_{}_hint_offsets'.format(
            'x' if self.orientation == 'horizontal' else 'y'
        )
        offs = list(getattr(self.deckbuilder, att))
        if len(offs) <= self.deckidx:
            Clock.schedule_once(self.on_scroll, 0)
            return
        offs[self.deckidx] = self._scroll
        setattr(self.deckbuilder, att, offs)
        self.deckbuilder._trigger_layout()

    def bar_touched(self, bar, touch):
        """Start scrolling, and record where I started scrolling."""
        self.scrolling = True
        self._start_bar_pos_hint = get_pos_hint(bar.pos_hint, *bar.size_hint)
        self._start_touch_pos_hint = (
            touch.x / self.width,
            touch.y / self.height
        )
        self._start_bar_touch_hint = (
            self._start_touch_pos_hint[0] - self._start_bar_pos_hint[0],
            self._start_touch_pos_hint[1] - self._start_bar_pos_hint[1]
        )
        touch.grab(self)

    def on_touch_move(self, touch):
        """Move the scrollbar to the touch, and update my ``scroll``
        accordingly.

        """
        if not self.scrolling or 'bar' not in self.ids:
            touch.ungrab(self)
            return
        touch.push()
        touch.apply_transform_2d(self.parent.to_local)
        touch.apply_transform_2d(self.to_local)
        if self.orientation == 'horizontal':
            hint_right_of_bar = (touch.x - self.ids.bar.x) / self.width
            hint_correction = hint_right_of_bar - self._start_bar_touch_hint[0]
            self.scroll += hint_correction
        else:  # self.orientation == 'vertical'
            hint_above_bar = (touch.y - self.ids.bar.y) / self.height
            hint_correction = hint_above_bar - self._start_bar_touch_hint[1]
            self.scroll += hint_correction
        touch.pop()

    def on_touch_up(self, touch):
        """Stop scrolling."""
        self.scrolling = False


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
            rgba: root.outline_color
        Line:
            points: [self.x, self.y, self.right, self.y, self.right, self.top, self.x, self.top, self.x, self.y]
        Color:
            rgba: [1, 1, 1, 1]
<Foundation>:
    color: [0, 0, 0, 0]
    outline_color: [1, 1, 1, 1]
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
            rgba: root.outline_color
        Line:
            points: [self.x, self.y, self.right, self.y, self.right, self.top, self.x, self.top, self.x, self.y]
        Color:
            rgba: [1, 1, 1, 1]
    BoxLayout:
        size_hint: 0.9, 0.9
        pos_hint: {'x': 0.05, 'y': 0.05}
        orientation: 'vertical'
        canvas:
            Color:
                rgba: root.content_outline_color
            Line:
                points: [self.x, self.y, self.right, self.y, self.right, self.top, self.x, self.top, self.x, self.y]
            Color:
                rgba: [1, 1, 1, 1]
        Label:
            id: headline
            text: root.headline_text
            markup: root.headline_markup
            font_name: root.headline_font_name
            font_size: root.headline_font_size
            color: root.headline_color
            size_hint: (None, None)
            size: self.texture_size
        ColorTextureBox:
            id: art
            color: root.art_color
            texture: root.art_texture
            outline_color: root.art_outline_color if root.show_art else [0, 0, 0, 0]
            size_hint: (1, 1) if root.show_art else (None, None)
            size: (0, 0)
        Label:
            id: midline
            text: root.midline_text
            markup: root.midline_markup
            font_name: root.midline_font_name
            font_size: root.midline_font_size
            color: root.midline_color
            size_hint: (None, None)
            size: self.texture_size
        ColorTextureBox:
            id: foreground
            color: root.foreground_color
            outline_color: root.foreground_outline_color
            texture: root.foreground_texture
            Label:
                text: root.text
                color: root.text_color
                markup: root.markup
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
            font_name: root.footer_font_name
            font_size: root.footer_font_size
            color: root.footer_color
            size_hint: (None, None)
            size: self.texture_size
<DeckBuilderScrollBar>:
    ScrollBarBar:
        id: bar
        color: root.bar_color if root.scrolling else root.bar_inactive_color
        texture: root.bar_texture
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
            show_art=False,
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
    builder = DeckBuilderLayout(
        card_size_hint=(0.15, 0.3),
        pos_hint={'x': 0, 'y': 0},
        starting_pos_hint={'x': 0.1, 'top': 0.9},
        card_hint_step=(0.05, -0.1),
        deck_hint_step=(0.4, 0),
        decks=[deck0, deck1],
        deck_y_hint_offsets=[0, 1]
    )
    layout = BoxLayout(orientation='horizontal')
    left_bar = DeckBuilderScrollBar(
        deckbuilder=builder,
        orientation='vertical',
        size_hint_x=0.1,
        deckidx=0
    )
    right_bar = DeckBuilderScrollBar(
        deckbuilder=builder,
        orientation='vertical',
        size_hint_x=0.1,
        deckidx=1
    )
    layout.add_widget(left_bar)
    layout.add_widget(builder)
    layout.add_widget(right_bar)
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
