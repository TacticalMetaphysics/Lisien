# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
import re
import pyglet
from pyglet.graphics import OrderedGroup


"""Simple menu widgets"""


__metaclass__ = SaveableMetaclass


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
        self.rumor = self.menu.rumor
        self.batch = self.menu.batch
        self.group = self.menu.labelgroup
        self.window = self.menu.window
        self.idx = idx
        while len(self.menu.items) <= self.idx:
            self.menu.items.append(None)
        self.menu.items[self.idx] = self
        (funname, argstr) = re.match("(.+)\((.*)\)", self._on_click).groups()
        (fun, argre) = self.rumor.func[funname]
        try:
            on_click_arg_tup = re.match(argre, argstr).groups()
        except:
            on_click_arg_tup = tuple()
        self.calls = 0

        def on_click_fun(self):
            if self.calls == 1:
                pass
            print "menuitem function {0} called "
            "for the {1}th time".format(
                self._on_click, self.calls)
            self.calls += 1
            t = (self,) + on_click_arg_tup
            return fun(*t)

        self.on_click = on_click_fun

    def __int__(self):
        return self.idx

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.rumor.tabdict["menu_item"][
                str(self.window)][str(self.menu)][int(self)]
        elif attrn == "closer":
            return self._rowdict["closer"]
        elif attrn == "_text":
            return self._rowdict["text"]
        elif attrn == "_icon":
            return self._rowdict["icon"]
        elif attrn == "_on_click":
            return self._rowdict["on_click"]
        elif attrn == "icon":
            if self._icon is not None:
                return self.rumor.get_img(self._icon)
        elif attrn == 'text':
            if self._text is None:
                return None
            elif self._text[0] == '@':
                return self.rumor.get_text(self._text[1:])
            else:
                return self._text
        elif attrn == 'hovered':
            return self.window.hovered is self
        elif attrn == 'pressed':
            return self.window.pressed is self
        elif attrn == 'window_left':
            return self.menu.window_left + self.menu.style.spacing
        elif attrn == 'label_window_left':
            if self.icon is None:
                return self.window_left
            else:
                return (
                    self.window_left +
                    self.icon.width +
                    self.menu.style.spacing)
        elif attrn == 'window_right':
            return self.menu.window_right - self.menu.style.spacing
        elif attrn == 'label_window_right':
            return self.window_right
        elif attrn == 'width':
            return self.window_right - self.window_left
        elif attrn == 'height':
            return self.menu.style.fontsize + self.menu.style.spacing
        elif attrn == 'window_top':
            return self.menu.window_top - self.idx * self.height
        elif attrn == 'window_bot':
            return self.window_top - self.height
        elif attrn == 'rx':
            return self.width / 2
        elif attrn == 'ry':
            return self.height / 2
        elif attrn == 'r':
            if self.rx > self.ry:
                return self.rx
            else:
                return self.ry
        else:
            raise AttributeError(
                "MenuItem instance has no such attribute: " +
                attrn)

    def onclick(self, x, y, button, modifiers):
        return self.on_click(self)

    def overlaps(self, x, y):
        return (
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def draw(self):
        try:
            self.label.delete()
        except:
            pass
        if self.menu.visible or self.window.main_menu_name == str(self.menu):
            if self.icon is not None:
                try:
                    self.sprite.x = self.window_left
                    self.sprite.y = self.window_bot
                except:
                    self.sprite = pyglet.sprite.Sprite(
                        self.icon.tex,
                        self.window_left,
                        self.window_bot,
                        batch=self.batch,
                        group=self.group)
            if self.text not in ('', None):
                try:
                    self.label.text = self.text
                    self.label.color = self.menu.style.textcolor.tup
                    self.label.x = self.label_window_left
                    self.label.y = self.window_bot
                except:
                    self.label = pyglet.text.Label(
                        self.text,
                        self.menu.style.fontface,
                        self.menu.style.fontsize,
                        color=self.menu.style.textcolor.tup,
                        x=self.window_left,
                        y=self.window_bot,
                        batch=self.batch,
                        group=self.group)
            else:
                try:
                    self.label.delete()
                except:
                    pass

    def get_tabdict(self):
        return {
            "menu_item": [{
                "window": str(self.window),
                "menu": str(self.menu),
                "idx": self.idx,
                "text": self._text,
                "on_click": self._on_click,
                "closer": self.closer}]
        }


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
        self.batch = self.window.batch
        self.supergroup = OrderedGroup(0, self.window.menugroup)
        self.bggroup = OrderedGroup(0, self.supergroup)
        self.labelgroup = OrderedGroup(1, self.supergroup)
        self.rumor = self.window.rumor
        self.name = name
        self.active_pattern = pyglet.image.SolidColorImagePattern(
            self.style.bg_active.tup)
        self.inactive_pattern = pyglet.image.SolidColorImagePattern(
            self.style.bg_inactive.tup)
        self.rowheight = self.style.fontsize + self.style.spacing
        self.items = []
        self.sprite = None
        self.pressed = False
        self.freshly_adjusted = False
        self.visible = False

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.rumor.tabdict["menu"][str(self.window)][str(self)]
        elif attrn == "left_prop":
            return self._rowdict["left"]
        elif attrn == "right_prop":
            return self._rowdict["right"]
        elif attrn == "top_prop":
            return self._rowdict["top"]
        elif attrn == "bot_prop":
            return self._rowdict["bot"]
        elif attrn == "style":
            return self.rumor.get_style(self._rowdict["style"])
        elif attrn == 'hovered':
            return self.window.hovered is self
        elif attrn == 'window_left':
            if self.window is None:
                return 0
            else:
                return int(self.window.width * self.left_prop)
        elif attrn == 'window_bot':
            if self.window is None:
                return 0
            else:
                return int(self.window.height * self.bot_prop)
        elif attrn == 'window_top':
            if self.window is None:
                return 0
            else:
                return int(self.window.height * self.top_prop)
        elif attrn == 'window_right':
            if self.window is None:
                return 0
            else:
                return int(self.window.width * self.right_prop)
        elif attrn == 'width':
            return self.window_right - self.window_left
        elif attrn == 'height':
            return self.window_top - self.window_bot
        elif attrn == 'rx':
            return int(
                (self.window.width * self.right_prop -
                 self.window.width * self.left_prop)
                / 2)
        elif attrn == 'ry':
            return int(
                (self.window.height * self.top_prop -
                 self.window.height * self.bot_prop)
                / 2)
        elif attrn == 'r':
            if self.rx > self.ry:
                return self.rx
            else:
                return self.ry
        else:
            raise AttributeError(
                "Menu instance has no such attribute: " +
                attrn)

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

    def get_state_tup(self):
        """Return a tuple containing everything you need to decide how to draw
me"""
        return (
            self,
            self.window_left,
            self.window_bot,
            self.window_top,
            self.window_right,
            self.style,
            self.visible,
            self.grabpoint,
            self.pressed,
            self.tweaks)

    def get_tabdict(self):
        return {
            "menu": [{
                "window": str(self.window),
                "name": str(self),
                "left": self.left_prop,
                "bottom": self.bot_prop,
                "top": self.top_prop,
                "right": self.right_prop,
                "style": str(self.style)
            }]}

    def save(self):
        for it in self.items:
            it.save()
        self.coresave()

    def draw(self):
        for item in self.items:
            item.draw()
        try:
            self.sprite.delete()
        except:
            pass
        if self.visible or str(self) == self.window.main_menu_name:
            image = self.inactive_pattern.create_image(
                self.width, self.height)
            self.sprite = pyglet.sprite.Sprite(
                image, self.window_left, self.window_bot,
                batch=self.batch, group=self.bggroup)
