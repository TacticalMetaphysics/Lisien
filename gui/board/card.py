# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, phi
from gui.kivybits import SaveableWidgetMetaclass
from kivy.graphics import InstructionGroup, Rectangle
from kivy.uix.image import Image
from kivy.uix.label import Label


"""Views on Effects and EffectDecks that look like cards--you know,
rectangles with letters and pictures on them."""


class Card(object):
    __metaclass__ = SaveableMetaclass
    atrdic = {
        "_text": lambda self: self._rowdict["text"],
        "_display_name": lambda self: self._rowdict["display_name"],
        "image": lambda self: self.closet.get_img(self._rowdict["image"]),
        "img": lambda self: self.image,
        "text": lambda self: self.gett(),
        "display_name": lambda self: self.getd()}

    tables = [
        (
            "card",
            {
                "effect": "text not null",
                "display_name": "text not null",
                "image": "text",
                "text": "text"},
            ("effect",),
            {
                "image": ("img", "name"),
                "effect": ("effect", "name")},
            [])]

    def __init__(self, closet, effect):
        self.closet = closet
        self.effect = effect
        self.closet.carddict[str(self)] = self
        self._rowdict = self.closet.skeleton["card"][str(self.effect)]

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError("Card has no attribute {0}".format(attrn))

    def __str__(self):
        return str(self.effect)

    def gett(self):
        if self._text is None:
            return ''
        elif self._text[0] == '@':
            return self.closet.get_text(self._text[1:])
        else:
            return self._text

    def getd(self):
        if self._display_name is None:
            return ''
        elif self._display_name[0] == '@':
            return self.closet.get_text(self._display_name[1:])
        else:
            return self._display_name


class TextHolder:
    atrdic = {
        "width": lambda self:
        (self.cardwidget.width - 4) * self.cardwidget.style.spacing,
        "height": lambda self: self.getheight(),
        "window_left": lambda self:
        (self.cardwidget.x + 2) * self.cardwidget.style.spacing,
        "window_right": lambda self: self.window_left + self.width,
        "window_bot": lambda self:
        (self.cardwidget.window_bot + 2) * self.cardwidget.style.spacing,
        "window_top": lambda self: self.window_bot + self.height,
        "text_left": lambda self:
        self.window_left + self.cardwidget.style.spacing,
        "text_bot": lambda self:
        self.window_bot + self.cardwidget.style.spacing,
        "text_width": lambda self:
        self.width - self.cardwidget.style.spacing,
        "text_height": lambda self:
        self.height - self.cardwidget.style.spacing}

    def __init__(self, cardwidget):
        self.cardwidget = cardwidget

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "TextHolder instance has no attribute named {0}".format(attrn))

    @property
    def height(self):
        if self.cardwidget.base.img is not None:
            return (
                self.cardwidget.height / 2 - 4
                * self.cardwidget.style.spacing)
        else:
            return (
                self.cardwidget.height - 4
                * self.cardwidget.style.spacing)

    def get_instructions(self, group=None):
        if group is None:
            group = InstructionGroup()
        w = self.width
        h = self.height
        group.add(Rectangle(pos=(self.window_left, self.window_bot),
                            size=()))
        group.add(Label(
            text=self.card.text,
            color=self.style.text_active.tup,
            font_name=self.style.fontface,
            font_size=self.style.fontsize,
            text_size=(w, h),
            valign='top'))
        return group


class CardWidget(Image):
    __metaclass__ = SaveableWidgetMetaclass
    atrdic = {
        "x": lambda self:
        self.hand.window_left + self.width * int(self),
        "y": lambda self: self.hand.window_bot,
        "hovered": lambda self: self.gw.hovered is self,
        "pressed": lambda self: self.gw.pressed is self,
        "grabbed": lambda self: self.gw.grabbed is self,
        "img": lambda self: self.base.img,
        "display_name": lambda self: self.base.display_name,
        "text": lambda self: self.base.text,
        "style": lambda self: self.base.style,
        "width": lambda self: self.getwidth(),
        "height": lambda self: int(self.width * phi),
        "window_left": lambda self: self.x,
        "window_right": lambda self: self.x + self.width,
        "window_bot": lambda self: self.y,
        "window_top": lambda self: self.y + self.height,
        "widget": lambda self: self,
        "state": lambda self: self.get_state_tup,
        "closet": lambda self: self.base.closet}

    def __init__(self, base, hand):
        self.base = base
        self.hand = hand
        self.window = self.hand.window
        self.visible = True
        self.interactive = True
        self.hovered = False
        self.floating = False
        self.textholder = TextHolder(self)

    def __int__(self):
        return self.hand.deck.index(self.base.effect)

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "CardWidget has no attribute {0}".format(attrn))

    def __hash__(self):
        return hash(self.get_state_tup())

    def getwidth(self):
        if self.img is None:
            # Just a default width for now
            return 128
        else:
            # The width of the image plus some gutterspace on each side
            return self.img.width + self.style.spacing * 2

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
        self.hovered = True

    def unhovered(self):
        self.hovered = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.grabpoint is None:
            self.oldx = x
            self.oldy = y
            self.floating = True
            self.grabpoint = (x - self.x, y - self.y)
            self.hand.adjust()
        (grabx, graby) = self.grabpoint
        self.x = x - grabx + dx
        self.y = y - graby + dy
        return self

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
            self.window_left,
            self.window_bot,
            self.tweaks)

    def get_instructions(self, group=None):
        if group is None:
            group = InstructionGroup()
        group.add(Rectangle(pos=(self.window_left, self.window_bot),
                            size=(self.width, self.height)))
        self.textholder.get_instructions(group)
        return group


class HandIterator:
    def __init__(self, hand, carddict):
        self.hand = hand
        self.carddict = carddict
        self.deckiter = iter(hand.deck.effects)

    def __iter__(self):
        return self

    def __next__(self):
        effect = next(self.deckiter)
        card = self.carddict[str(effect)]
        if not hasattr(card, 'widget'):
            card.widget = CardWidget(card, self.hand)
        return card.widget


class Hand(object):
    __metaclass__ = SaveableMetaclass
    """A view onto an EffectDeck that shows every card in it, in the same
order."""
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

    def __init__(self, window, deck):
        self.window = window
        self.closet = self.window.closet
        self.deck = deck

    def __hash__(self):
        return hash(self.get_state_tup())

    def __iter__(self):
        return HandIterator(self, self.window.carddict)

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.closet.skeleton[str(self.window)][str(self.deck)]
        elif attrn in (
                "left_prop", "right_prop", "top_prop",
                "bot_prop", "visible", "interactive"):
            return self._rowdict[attrn]
        elif attrn == "style":
            return self.closet.get_style(self._rowdict["style"])
        elif attrn == "board":
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
        return str(self.deck)

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

    def overlaps(self, x, y):
        return (
            self.visible and
            self.interactive and
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def get_instructions(self, group=None):
        if group is None:
            group = InstructionGroup()
        for card in self:
            card.get_instructions(group)
