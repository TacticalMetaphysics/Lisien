# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
import re
import pyglet
from logging import getLogger


"""Simple menu widgets"""


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


ON_CLICK_RE = re.compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuItem:
    """A thing in a menu that you can click to make something happen."""
    tables = [
        ('menu_item',
         {"window": "text not null default 'Main'",
          'menu': 'text not null',
          'idx': 'integer not null',
          'text': 'text',
          'icon': 'text',
          'on_click': 'text not null',
          'closer': "boolean not null default 1"},
         ("window", "menu", "idx"),
         {"window, menu": ("menu", "window, name")},
         [])]

    def geticon(self):
        if self._icon is not None:
            return self.closet.get_img(self._icon)

    def gettxt(self):
        if self._text is None:
            return None
        elif self._text[0] == '@':
            return self.closet.get_text(self._text[1:])
        else:
            return self._text

    def lwl(self):
            if self.icon is None:
                return self.window_left
            else:
                return (
                    self.window_left +
                    self.icon.width +
                    self.menu.style.spacing)

    atrdic = {
        "closer": lambda self: self._rowdict["closer"],
        "_text": lambda self: self._rowdict["text"],
        "_icon": lambda self: self._rowdict["icon"],
        "_on_click": lambda self: self._rowdict["on_click"],
        "icon": lambda self: self.geticon(),
        "text": lambda self: self.gettxt(),
        "hovered": lambda self: self.window.hovered is self,
        "pressed": lambda self: self.window.pressed is self,
        "window_left": lambda self: (
            self.menu.window_left + self.menu.style.spacing),
        "label_window_left": lambda self: self.lwl(),
        "window_right": lambda self: (
            self.menu.window_right - self.menu.style.spacing),
        "label_window_right": lambda self: self.window_right,
        "width": lambda self: self.window_right - self.window_left,
        "height": lambda self: (
            self.menu.style.fontsize + self.menu.style.spacing),
        "window_top": lambda self: self.menu.window_top - (
            self.idx * self.height),
        "window_bot": lambda self: self.window_top - self.height,
        "rx": lambda self: self.width / 2,
        "ry": lambda self: self.height / 2,
        "r": lambda self: {True: self.rx, False: self.ry}[self.rx > self.ry]}

    visible = True
    interactive = True

    def __init__(self, menu, idx):
        """Return a menu item in the given board, the given menu; at the given
index in that menu; with the given text; which executes the given
effect deck when clicked; closes or doesn't when clicked; starts
visible or doesn't; and starts interactive or doesn't.

With db, register in db's menuitemdict.

        """
        self.menu = menu
        self.closet = self.menu.closet
        self.batch = self.menu.batch
        self.window = self.menu.window
        self.idx = idx
        self._rowdict = self.closet.skeleton["menu_item"][
            str(self.window)][str(self.menu)][int(self)]
        while len(self.menu.items) <= self.idx:
            self.menu.items.append(None)
        self.menu.items[self.idx] = self
        (funname, argstr) = re.match("(.+)\((.*)\)", self._on_click).groups()
        (fun, argre) = self.closet.func[funname]
        try:
            on_click_arg_tup = re.match(argre, argstr).groups()
        except:
            on_click_arg_tup = tuple()
        self.calls = 0

        def on_click_fun(self):
            self.calls += 1
            t = (self,) + on_click_arg_tup
            return fun(*t)

        self.on_click = on_click_fun

        self.old_window_left = 0
        self.old_window_bot = 0
        self.old_label_text = ''
        self.old_label_color = (0, 0, 0, 0)
        self.old_label_x = 0
        self.old_label_y = 0
        self.label = None
        self.sprite = None

    def __int__(self):
        return self.idx

    def __getattr__(self, attrn):
        return self.atrdic[attrn](self)

    def onclick(self, x, y, button, modifiers):
        return self.on_click(self)

    def overlaps(self, x, y):
        return (
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def delete(self):
        if self.label is not None:
            try:
                self.label.delete()
            except AttributeError:
                pass
            self.label = None
        if self.sprite is not None:
            try:
                self.sprite.delete()
            except AttributeError:
                pass
            self.sprite = None

    def draw(self):
        b = self.window_bot
        l = self.window_left
        if self.icon is not None:
            try:
                if self.old_window_left != l:
                    self.sprite.x = l
                    self.old_window_left = l
                if self.old_window_bot != b:
                    self.sprite.y = b
                    self.old_window_bot = b
            except AttributeError:
                self.sprite = pyglet.sprite.Sprite(
                    self.icon.tex,
                    l, b,
                    batch=self.batch,
                    group=self.window.menu_fg_group)
        if self.text not in ('', None):
            txt = self.text
            color = self.menu.style.textcolor.tup
            l = self.label_window_left
            try:
                if self.old_label_text != txt:
                    self.label.text = self.text
                    self.old_label_text = txt
                if self.old_label_color != color:
                    self.label.color = self.menu.style.textcolor.tup
                    self.old_label_color = color
                if self.old_label_x != l:
                    self.label.x = l
                    self.old_label_x = l
                if self.old_label_y != b:
                    self.label.y = b
                    self.old_label_y = b
            except AttributeError:
                self.label = pyglet.text.Label(
                    self.text,
                    self.menu.style.fontface,
                    self.menu.style.fontsize,
                    color=self.menu.style.textcolor.tup,
                    x=l,
                    y=b,
                    batch=self.batch,
                    group=self.window.menu_fg_group)


class Menu:
    """Container for MenuItems; not interactive unto itself."""
    tables = [
        ('menu',
         {"window": "text not null default 'Main'",
          'name': 'text not null',
          'left': "float not null default 0.1",
          'bot': "float not null default 0.0",
          'top': 'float not null default 1.0',
          'right': 'float not null default 0.2',
          'style': "text not null default 'SmallDark'"},
         ("window", 'name'),
         {"window": ("window", "name"),
          "style": ("style", "name")},
         [])]
    atrdic = {
        "left_prop": lambda self: self._rowdict["left"],
        "right_prop": lambda self: self._rowdict["right"],
        "top_prop": lambda self: self._rowdict["top"],
        "bot_prop": lambda self: self._rowdict["bot"],
        "style": lambda self: (
            self.closet.get_style(self._rowdict["style"])),
        "hovered": lambda self: self.window.hovered is self,
        "window_left": lambda self: int(self.window.width * self.left_prop),
        "window_right": lambda self: int(self.window.width * self.right_prop),
        "window_top": lambda self: int(self.window.height * self.top_prop),
        "window_bot": lambda self: int(self.window.height * self.bot_prop),
        "width": lambda self: self.window_right - self.window_left,
        "height": lambda self: self.window_top - self.window_bot,
        "rx": lambda self: self.width / 2,
        "ry": lambda self: self.height / 2,
        "r": lambda self: {True: self.rx, False: self.ry
                           }[self.rx > self.ry],
        "state": lambda self: self.get_state_tup()}
    interactive = True

    def __init__(self, window, name):
        """Return a menu in the given board, with the given name, bounds,
style, and flags main_for_window and visible.

Bounds are proportional with respect to the lower left corner of the
window. That is, they are floats, never below 0.0 nor above 1.0, and
they express a portion of the window's width or height.

main_for_window prevents the menu from ever being hidden. visible
determines if you can see it at the moment.

With db, register with db's menudict.

        """
        self.window = window
        self.name = name
        self.batch = self.window.batch
        self.closet = self.window.closet
        self._rowdict = self.closet.skeleton[
            "menu"][str(self.window)][str(self)]
        self.closet = self.window.closet
        self.active_pattern = pyglet.image.SolidColorImagePattern(
            self.style.bg_active.tup)
        self.inactive_pattern = pyglet.image.SolidColorImagePattern(
            self.style.bg_inactive.tup)
        self.rowheight = self.style.fontsize + self.style.spacing
        self.items = []
        self.sprite = None
        self.oldwidth = None
        self.oldheight = None
        self.oldcoords = None
        self.pressed = False
        self.freshly_adjusted = False
        self.visible = False
        self.image = self.inactive_pattern.create_image(
            self.width, self.height)
        self._rowdict = self.closet.skeleton["menu"][
            str(self.window)][str(self)]

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        return self.atrdic[attrn](self)

    def __eq__(self, other):
        """Return true if the names and boards match"""
        return (
            self.name == other.name and
            self.board == other.board)

    def __getitem__(self, i):
        """Return an item herein"""
        return self.items[i]

    def __setitem__(self, i, to):
        """Set a menuitem"""
        self.items[i] = to

    def __delitem__(self, i):
        """Delete a menuitem"""
        return self.items.__delitem__(i)

    def __contains__(self, mi):
        return mi in self.items

    def append(self, mi):
        self.items.append(mi)

    def remove(self, mi):
        self.items.remove(mi)

    def index(self, mi):
        return self.items.index(mi)

    def adjust(self):
        """Assign absolute coordinates to myself and all my items."""
        i = 0
        for item in self.items:
            item.top_from_top = i * self.rowheight
            item.bot_from_top = item.top_from_top + self.rowheight
            item.window_top = self.window_top - item.top_from_top
            item.window_bot = item.window_top - self.rowheight
            i += 1

    def overlaps(self, x, y):
        return (
            (self.visible or str(self) == self.window.main_menu_name) and
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def hover(self, x, y):
        for item in self.items:
            if item.overlaps(x, y):
                return item

    def draw_sprite(self):
        self.sprite = pyglet.sprite.Sprite(
            self.image, self.window_left, self.window_bot,
            batch=self.batch, group=self.window.menu_bg_group)

    def delete_sprite(self):
        if self.sprite is not None:
            try:
                self.sprite.delete()
            except AttributeError:
                pass
            self.sprite = None

    def delete_items(self):
        for item in self.items:
            item.delete()

    def delete(self):
        self.delete_items()
        self.delete_sprite()

    def draw(self):
        if self.visible or str(self) == self.window.main_menu_name:
            coords = (self.window_left, self.window_bot)
            w = self.width
            h = self.height
            if w != self.oldwidth or h != self.oldheight:
                self.image = self.inactive_pattern.create_image(
                    self.width, self.height)
                old_sprite = self.sprite
                self.draw_sprite()
                try:
                    old_sprite.delete()
                except AttributeError:
                    pass
            elif self.sprite is None:
                self.draw_sprite()
            elif self.oldcoords != coords:
                self.sprite.set_position(*coords)
            for item in self.items:
                item.draw()
            self.oldwidth = w
            self.oldheight = h
            self.oldcoords = coords
