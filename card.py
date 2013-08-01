# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
from util import SaveableMetaclass, PatternHolder, phi, dictify_row, BranchTicksIter

"""Views on Effects and EffectDecks that look like cards--you know,
rectangles with letters and pictures on them."""


class Card:
    __metaclass__ = SaveableMetaclass
    tables = [
        (
            "card",
            {
                "effect": "text not null",
                "display_name": "text not null",
                "image": "text",
                "text": "text",
                "style": "text not null default 'BigLight'"},
            ("effect",),
            {
                "image": ("img", "name"),
                "style": ("style", "name"),
                "effect": ("effect", "name")},
            [])]

    def __init__(self, db, effect, display_name, image, text, style):
        self.db = db
        self._display_name = display_name
        self._effect = str(effect)
        self.img = image
        self._text = text
        self.style = style
        self.db.carddict[str(self)] = self

    def __getattr__(self, attrn):
        if attrn == 'text':
            if self._text is None:
                return ''
            elif self._text[0] == '@':
                return self.db.get_text(self._text[1:])
            else:
                return self._text
        elif attrn == 'display_name':
            if self._display_name[0] == '@':
                return self.db.get_text(self._display_name[1:])
            else:
                return self._display_name
        elif attrn == 'effect':
            return self.db.effectdict[self._effect]
        elif attrn in ('width', 'height', 'x', 'y'):
            if not hasattr(self, 'widget'):
                return 0
            else:
                return getattr(self.widget, attrn)
        else:
            raise AttributeError("Card has no attribute {0}".format(attrn))

    def __str__(self):
        return self._effect

    def get_tabdict(self):
        return {
            "card": {
                "display_name": self._display_name,
                "effect": str(self.effect),
                "img": str(self.img),
                "text": self._text,
                "style": str(self.style)}
        }

    def get_keydict(self):
        return {
            "card": {
                "name": self.name}}

class TextHolder:
    def __init__(self, cardwidget):
        self.cardwidget = cardwidget
        self.bgimage = None
        self.bgsprite = None
        self.label = None

    def __getattr__(self, attrn):
        if attrn == "width":
            return self.cardwidget.width - 4 * self.cardwidget.style.spacing
        elif attrn == "height":
            if isinstance(
                    self.cardwidget.base.img,
                    pyglet.image.AbstractImage):
                return (
                    self.cardwidget.height / 2 - 4
                    * self.cardwidget.style.spacing)
            else:
                return (
                    self.cardwidget.height - 4
                    * self.cardwidget.style.spacing)
        elif attrn == "window_left":
            return self.cardwidget.x + 2 * self.cardwidget.style.spacing
        elif attrn == "window_right":
            return self.window_left + self.width
        elif attrn == "window_bot":
            return (
                self.cardwidget.window_bot + 2
                * self.cardwidget.style.spacing)
        elif attrn == "window_top":
            return self.window_bot + self.height
        elif attrn == "text_left":
            return self.window_left + self.cardwidget.style.spacing
        elif attrn == "text_bot":
            return self.window_bot + self.cardwidget.style.spacing
        elif attrn == "text_width":
            return self.width - self.cardwidget.style.spacing
        elif attrn == "text_height":
            return self.height - self.cardwidget.style.spacing
        else:
            return getattr(self.cardwidget, attrn)


class CardWidget:
    def __init__(self, base, hand):
        self.base = base
        self.hand = hand
        self.db = self.base.db
        self.window = self.hand.window
        self.grabpoint = None
        self.visible = True
        self.interactive = True
        self.hovered = False
        self.floating = False
        self.tweaks = 0
        self.pats = PatternHolder(self.base.style)
        self.bgimage = None
        self.bgsprite = None
        self.textholder = TextHolder(self)
        self.imgsprite = None

    def __int__(self):
        return self.hand.deck.index(self.base.effect)

    def __getattr__(self, attrn):
        if attrn == 'x':
            return self.hand.window_left + self.width * int(self)
        elif attrn == 'y':
            return self.hand.window_bot
        elif attrn == 'hovered':
            return self.gw.hovered is self
        elif attrn == 'pressed':
            return self.gw.pressed is self
        elif attrn == 'grabbed':
            return self.gw.grabbed is self
        elif attrn == 'img':
            return self.base.img
        elif attrn == 'display_name':
            return self.base.display_name
        elif attrn == 'text':
            return self.base.text
        elif attrn == 'style':
            return self.base.style
        elif attrn == 'width':
            if self.img is None:
                # Just a default width for now
                return 128
            else:
                # The width of the image plus some gutterspace on each side
                return self.img.width + self.style.spacing * 2
        elif attrn == 'height':
            return int(self.width * phi)
        elif attrn == 'window_left':
            return self.x
        elif attrn == 'window_right':
            return self.x + self.width
        elif attrn == 'window_bot':
            return self.y
        elif attrn == 'window_top':
            return self.y + self.height
        elif attrn == 'widget':
            return self
        elif hasattr(self.base, attrn):
            return getattr(self.base, attrn)
        else:
            raise AttributeError(
                "CardWidget has no attribute {0}".format(attrn))

    def __hash__(self):
        return hash(self.get_state_tup())

    def overlaps(self, x, y):
        return (
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def save(self):
        self.base.save()

    def toggle_visibility(self):
        self.visible = not self.visible
        self.tweaks += 1

    def hide(self):
        if self.visible:
            self.toggle_visibility()

    def show(self):
        if not self.visible:
            self.toggle_visibility()

    def hovered(self):
        print "card hovered"
        self.hovered = True

    def unhovered(self):
        print "card unhovered"
        self.hovered = False

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        if self.grabpoint is None:
            self.oldx = x
            self.oldy = y
            self.floating = True
            self.grabpoint = (x - self.x, y - self.y)
            self.hand.adjust()
        (grabx, graby) = self.grabpoint
        self.x = x - grabx + dx
        self.y = y - graby + dy

    def dropped(self, x, y, button, modifiers):
        if (
                x > self.hand.window_left and
                x < self.hand.window_right and
                y > self.hand.window_bot and
                y < self.hand.window_top):
            min_x = self.hand.window_right
            max_x = 0
            for card in self.hand:
                if card.window_left < min_x:
                    min_x = card.window_left
                if card.window_right > max_x:
                    max_x = card.window_right
                if (
                        x > card.window_left and
                        x < card.window_bot):
                    print "Dropped {0} on {1}".format(str(self), str(card))
                    self.hand.remove(self)
                    self.hand.insert(self.hand.index(card), self)
                    break
            # I am either to the left of all cards in hand, or to the right.
            # Which edge am I closer to?
            space_left = x - min_x
            space_right = max_x - x
            self.hand.remove(self)
            if space_left < space_right:
                self.hand.insert(0, self)
            else:
                self.hand.append(self)
        self.floating = False
        self.grabpoint = None
        self.hand.adjust()

    def get_state_tup(self):
        return (
            self.display_name,
            self.text,
            self.img,
            self.visible,
            self.interactive,
            self.hovered,
            self.tweaks)


class HandIterator:
    def __init__(self, hand):
        self.db = hand.db
        self.hand = hand
        self.deckiter = iter(hand.deck.effects)

    def __iter__(self):
        return self

    def next(self):
        effect = self.deckiter.next()
        card = self.db.carddict[str(effect)]
        if not hasattr(card, 'widget'):
            card.widget = CardWidget(card, self.hand)
        return card.widget

class Hand:
    """A view onto an EffectDeck that shows every card in it, in the same order."""
    __metaclass__ = SaveableMetaclass
    tables = [
        ("hand",
         {"window": "text not null default 'Main'",
          "effect_deck": "text not null",
          "left": "float not null",
          "right": "float not null",
          "top": "float not null",
          "bot": "float not null",
          "style": "text not null default 'BigLight'",
          "visible": "boolean default 1",
          "interactive": "boolean default 1"},
         ("window", "effect_deck"),
         {"window": ("window", "name"),
          "effect_deck": ("effect_deck", "deck"),
          "style": ("style", "name")},
         [])]

    def __init__(self, window, deck, left, right, top, bot, style,
                 visible, interactive):
        self.window = window
        self.db = window.db
        self.deck = deck
        self.left_prop = left
        self.right_prop = right
        self.top_prop = top
        self.bot_prop = bot
        self.style = style
        self.visible = bool(visible)
        self.interactive = bool(interactive)
        self.oldstate = None

    def __hash__(self):
        return hash(self.get_state_tup())

    def __iter__(self):
        return HandIterator(self)

    def __getattr__(self, attrn):
        if attrn == "board":
            return self.window.board
        elif attrn == "hovered":
            return self.window.hovered is self
        elif attrn == "window_left":
            try:
                return int(self.left_prop * self.window.width)
            except AttributeError:
                return 0
        elif attrn == "window_right":
            try:
                return int(self.right_prop * self.window.width)
            except AttributeError:
                return 0
        elif attrn == "window_bot":
            try:
                return int(self.bot_prop * self.window.height)
            except AttributeError:
                return 0
        elif attrn == "window_top":
            try:
                return int(self.top_prop * self.window.height)
            except AttributeError:
                return 0
        elif attrn == "on_screen":
            return (
                self.window_right > 0 and
                self.window_left < self.window.width and
                self.window_top > 0 and
                self.window_bot < self.window.height)
        else:
            raise AttributeError(
                "Hand has no attribute named {0}".format(attrn))

    def __len__(self):
        return len(self.deck)

    def __str__(self):
        return self.name

    def _translate_index(self, idx):
        if idx > 0 and idx < len(self):
            return idx
        elif idx >= len(self):
            raise IndexError(
                "Index {0} in Hand {1} out of range.".format(
                    idx, self.name))
        while idx < 0:
            idx += len(self)
        return idx

    def get_state_tup(self):
        cardbits = []
        for card in self:
            cardbits.extend(iter(card.get_state_tup()))
        return (
            tuple(cardbits),
            self.visible,
            self.interactive,
            self.window_left,
            self.window_right,
            self.window_bot,
            self.window_top)

    def append(self, card):
        eff = card.effect
        self.deck.append(eff)

    def insert(self, i, card):
        eff = card.effect
        self.deck.insert(i, eff)

    def remove(self, card):
        eff = card.effect
        self.deck.remove(eff)

    def index(self, card):
        return self.deck.index(card.effect)

    def pop(self, i=None):
        return self.deck.pop(i).card

    def adjust(self):
        if len(self) == 0:
            return
        windobot = self.window_bot + self.style.spacing
        prev_right = self.window_left
        for card in iter(self):
            print "Adjusting card {0}".format(str(card))
            if card.widget is not None and card.widget.floating:
                continue
            card.hand = self
            x = prev_right + self.style.spacing
            if card.widget is None:
                card.widget = CardWidget(card, self)
            else:
                card.widget.x = x
                card.widget.y = windobot
            prev_right = x + card.widget.width

    def get_tabdict(self):
        return {
            "hand": [
                {"window": str(self.window),
                 "effect_deck": str(self.deck),
                 "left": self.left_prop,
                 "right": self.right_prop,
                 "top": self.top_prop,
                 "bot": self.bot_prop,
                 "style": str(self.style)}]}

    def overlaps(self, x, y):
        return (
            self.visible and
            self.interactive and
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)
