# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
import logging
from util import (
    SaveableMetaclass,
    fortyfive,
    TabdictIterator)
from math import atan, cos, sin
from arrow import Arrow
from menu import Menu, MenuItem
from card import Hand
from board import BoardViewport
from picpicker import PicPicker
from calendar import Calendar


class SaveableWindowMetaclass(
        pyglet.window._WindowMetaclass, SaveableMetaclass):
    pass


logger = logging.getLogger(__name__)

platform = pyglet.window.get_platform()

display = platform.get_default_display()

screen = display.get_default_screen()


class ScissorOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent, window, left, top, bot, right):
        super(ScissorOrderedGroup, self).__init__(order, parent)
        self.window = window
        self.left_prop = left
        self.top_prop = top
        self.bot_prop = bot
        self.right_prop = right

    def set_state(self):
        l = int(self.left_prop * self.window.width)
        b = int(self.bot_prop * self.window.height)
        r = int(self.right_prop * self.window.width)
        t = int(self.top_prop * self.window.height)
        w = r - l
        h = t - b
        pyglet.gl.glScissor(l, b, w, h)
        pyglet.gl.glEnable(pyglet.gl.GL_SCISSOR_TEST)

    def unset_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_SCISSOR_TEST)


class TransparencyGroup(pyglet.graphics.Group):
    def set_state(self):
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)

    def unset_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_BLEND)


class TransparencyOrderedGroup(
        pyglet.graphics.OrderedGroup,
        TransparencyGroup):
    pass


class MousySpot:
    """A spot-like object that's always at the last known position
of the mouse."""
    x = 0
    y = 0

    def __getattr__(self, attrn):
        if attrn == "window_x":
            return self.x
        elif attrn == "window_y":
            return self.y
        elif attrn in ("coords", "window_coords"):
            return (self.x, self.y)
        else:
            raise AttributeError


class ViewportIter:
    def __init__(self, dimensiondict):
        self.dimiter = dimensiondict.itervalues()
        self.boarditer = iter(self.dimiter.next().boards)
        self.viewiter = iter(self.boarditer.next().viewports)

    def __iter__(self):
        return self

    def next(self):
        try:
            r = self.viewiter.next()
            if r is not None:
                return r
        except StopIteration:
            try:
                self.viewiter = iter(self.boarditer.next().viewports)
            except StopIteration:
                self.boarditer = iter(self.dimiter.next().boards)
                self.viewiter = iter(self.boarditer.next().viewports)
        return self.next()


class GameWindow(pyglet.window.Window):
    __metaclass__ = SaveableWindowMetaclass
    tables = [
        ("window",
         {"name": "text not null default 'Main'",
          "min_width": "integer not null default 1280",
          "min_height": "integer not null default 800",
          "arrowhead_size": "integer not null default 10",
          "arrow_width": "float not null default 1.4",
          "main_menu": "text not null default 'Main'"},
         ("name",),
         {"main_menu": ("menu", "name")},
         [])]

    def __init__(
            self, rumor, name):
        """Initialize the game window, its groups, and some state tracking."""
        config = screen.get_best_config()
        pyglet.window.Window.__init__(self, config=config)
        self.edge_order = 1
        self.hand_order = 1
        self.batch = pyglet.graphics.Batch()
        self.biggroup = pyglet.graphics.Group()
        self.boardgroup = pyglet.graphics.OrderedGroup(0, self.biggroup)
        self.calgroup = TransparencyOrderedGroup(4, self.biggroup)
        self.handgroup = pyglet.graphics.OrderedGroup(5, self.biggroup)
        self.menugroup = pyglet.graphics.OrderedGroup(6, self.biggroup)
        self.pickergroup = ScissorOrderedGroup(
            8, self.biggroup, self, 0.3, 0.6, 0.3, 0.6)
        self.topgroup = pyglet.graphics.OrderedGroup(65535, self.biggroup)
        self.name = name
        self.rumor = rumor
        self.dimensiondict = self.rumor.get_dimensions(
            [rd["dimension"] for rd in
             TabdictIterator(self.rumor.tabdict[
                 "board_viewport"][str(self)])])
        for rd in TabdictIterator(self.rumor.tabdict[
                "board_viewport"][str(self)]):
            self.rumor.get_board(rd["dimension"], rd["board"])
            dimension = self.dimensiondict[rd["dimension"]]
            boardi = rd["board"]
            viewi = rd["idx"]
            board = dimension.boards[boardi]
            board.viewports[viewi] = BoardViewport(
                self.rumor, self, dimension, board, viewi)
        stylenames = set()
        handnames = set()
        for rd in TabdictIterator(
                self.rumor.tabdict["menu"][str(self)]):
            stylenames.add(rd["style"])
        if str(self) in self.rumor.tabdict["hand"]:
            for rd in TabdictIterator(
                    self.rumor.tabdict["hand"][str(self)]):
                stylenames.add(rd["style"])
                handnames.add(rd["name"])
        for rd in TabdictIterator(
                self.rumor.tabdict["calendar"][str(self)]):
            stylenames.add(rd["style"])
        self.rumor.get_styles(stylenames)
        carddict = self.rumor.get_cards_in_hands(handnames)
        imagenames = set()
        for rd in TabdictIterator(
                self.rumor.tabdict["menu_item"][str(self)]):
            if rd["icon"] is not None:
                imagenames.add(rd["icon"])
        for rd in TabdictIterator(carddict):
            if rd["image"] is not None:
                imagenames.add(rd["image"])
        self.rumor.get_imgs(imagenames)
        self.menudict = {}
        for rd in TabdictIterator(
                self.rumor.tabdict["menu"][str(self)]):
            menu = Menu(self, rd["name"])
            for mird in TabdictIterator(
                    self.rumor.tabdict["menu_item"][
                        str(self)][str(menu)]):
                MenuItem(menu, mird["idx"])
            self.menudict[str(menu)] = menu
        if str(self) in self.rumor.tabdict["hand"]:
            effect_deck_names = set()
            for rd in TabdictIterator(
                    self.rumor.tabdict["hand"][str(self)]):
                effect_deck_names.add(rd["deck"])
            effect_decks = self.rumor.get_effect_decks(effect_deck_names)
            self.handdict = {}
            for rd in TabdictIterator(
                    self.rumor.tabdict["hand"][str(self)]):
                effd = effect_decks[rd["effect_deck"]]
                self.handdict[rd["effect_deck"]] = Hand(self, effd)
        if hasattr(self, 'handdict'):
            self.carddict = self.rumor.get_cards_in_hands(self.handdict.keys())
        self.calendars = []
        for rd in TabdictIterator(
                self.rumor.tabdict["calendar"][str(self)]):
            while len(self.calendars) <= rd["idx"]:
                self.calendars.append(None)
            self.calendars[rd["idx"]] = Calendar(self, rd["idx"])
        self.mouspot = MousySpot()
        self.squareoff = self.arrowhead_size * sin(fortyfive)
        self.picker = None
        self.hover_iter_getters = [
            lambda: iter(self.calendars),
            self.menudict.itervalues,
            lambda: self.viewports,
            lambda: (self.picker,)]
        if hasattr(self, 'handdict'):
            self.hover_iter_getters.append(self.handdict.itervalues)
        self.pressed = None
        self.hovered = None
        self.grabbed = None
        self.portal_from = None
        self.thing_pic = None
        self.thing_pic_sprite = None
        self.place_pic = None
        self.place_pic_sprite = None
        self.placing = False
        self.thinging = False
        self.portaling = False
        self.selected = set()
        self.keep_selected = False
        self.prev_view_bot = 0

        orbimg = self.rumor.imgdict['default_spot']
        rx = orbimg.width / 2
        ry = orbimg.height / 2
        self.create_place_cursor = (
            pyglet.window.ImageMouseCursor(orbimg, rx, ry))
        self.create_place_cursor.rx = rx
        self.create_place_cursor.ry = ry
        self.drawn_board = None
        self.drawn_edges = None
        self.edge_order = 1
        self.floaty_portal = None

        self.time_travel_target = None

        self.dxdy_hist_max = 10
        self.dx_hist = [0] * self.dxdy_hist_max
        self.dy_hist = [0] * self.dxdy_hist_max

        self.timeline = None

        self.last_age = -1
        self.last_timeline_y = -1

        self.dxdy_hist_counter = 0

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.rumor.tabdict["window"][str(self)]
        elif attrn in ("min_width", "min_height",
                       "arrowhead_size", "arrow_width"):
            return self._rowdict[attrn]
        elif attrn == "main_menu_name":
            return self._rowdict["main_menu"]
        elif attrn == 'viewports':
            return ViewportIter(self.dimensiondict)
        elif attrn == 'menus':
            return self.menudict.itervalues()
        elif attrn == 'hands':
            if hasattr(self, 'handdict'):
                return self.handdict.itervalues()
            else:
                return []
        elif attrn == 'dx':
            return sum(self.dx_hist)
        elif attrn == 'dy':
            return sum(self.dy_hist)
        elif attrn == 'offset_x':
            return -1 * self.view_left
        elif attrn == 'offset_y':
            return -1 * self.view_bot
        elif attrn == 'arrow_girth':
            return self.arrow_width * 2
        else:
            raise AttributeError(
                "AbstractGameWindow has no attribute named {0}".format(attrn))

    def __str__(self):
        return self.name

    def update(self, dt):
        (x, y) = self.mouspot.coords

        def get_hovered():
            for get in self.hover_iter_getters:
                for hoverable in get():
                    if (
                            hoverable is not None and
                            hasattr(hoverable, 'overlaps') and
                            hoverable.overlaps(x, y)):
                        if hasattr(hoverable, 'hover'):
                            self.hovered = hoverable.hover(x, y)
                        else:
                            self.hovered = hoverable
                        return
        get_hovered()

        if self.portal_from is None:
            try:
                (self.floaty_portal.orig.x,
                 self.floaty_portal.orig.y) = self.floaty_coords()
            except:
                pass
        try:
            self.place_pic_sprite.set_position(
                x - self.place_pic.rx, y - self.place_pic.ry)
        except:
            pass
        try:
            self.thing_pic_sprite.set_position(
                x - self.thing_pic.rx, y - self.thing_pic.ry)
        except:
            pass

    def on_draw(self):
        (width, height) = self.get_size()
        if (
                width < self.min_width or
                height < self.min_height):
            self.set_minimum_size(self.min_width, self.min_height)
        if self.picker is not None:
            self.picker.draw(self.batch, self.pickergroup)
        for menu in self.menus:
            menu.draw()
        for calendar in self.calendars:
            if calendar is not None:
                calendar.draw()
        for hand in self.hands:
            hand.draw()
        for viewport in self.viewports:
            viewport.draw()
        # well, I lied. I was really only adding those things to the batch.
        # NOW I'll draw them.
        self.batch.draw()

    def on_mouse_press(self, x, y, button, modifiers):
        """If there's something already highlit, and the mouse is
still over it when pressed, it's been half-way clicked; remember this."""
        self.pressed = self.hovered
        print "pressed {0}".format(repr(self.pressed))

    def on_mouse_release(self, x, y, button, modifiers):
        """If something was being dragged, drop it. If something was being
pressed but not dragged, it's been clicked. Otherwise do nothing."""
        if self.grabbed is not None:
            if hasattr(self.grabbed, 'dropped'):
                self.grabbed.dropped(x, y, button, modifiers)
            self.grabbed = None
            return
        if (
                self.pressed not in self.selected and
                not self.keep_selected):
            for sel in iter(self.selected):
                if hasattr(sel, 'unselect'):
                    sel.unselect()
            self.selected = set()
        if self.pressed is not None:
            if self.pressed.overlaps(x, y):
                if hasattr(self.pressed, 'selectable'):
                    if hasattr(self.pressed, 'select'):
                        self.pressed.select()
                    print "selected {0}".format(repr(self.pressed))
                    self.selected.add(self.pressed)
                    if hasattr(self.pressed, 'reciprocate'):
                        reciprocal = self.pressed.reciprocate()
                        if reciprocal is not None:
                            self.selected.add(reciprocal)
                if hasattr(self.pressed, 'onclick'):
                    self.pressed.onclick(x, y, button, modifiers)
        if self.place_pic is not None:
            if self.placing:
                self.place_pic_sprite = pyglet.sprite.Sprite(
                    self.place_pic.tex,
                    x,
                    y - self.place_pic.height,
                    batch=self.batch,
                    group=self.spotgroup)
                self.placing = False
            else:
                try:
                    self.place_pic_sprite.delete()
                except:
                    pass
                pl = self.rumor.make_generic_place(self.dimension)
                sp = self.board.get_spot(pl)
                sp.set_coords(x + self.view_left, y + self.view_bot)
                sp.set_img(self.place_pic)
                self.place_pic = None
                logger.debug("made generic place: %s", str(pl))
            return
        if self.thing_pic is not None:
            if self.thinging:
                self.thing_pic_sprite = pyglet.sprite.Sprite(
                    self.thing_pic.tex,
                    x,
                    y - self.thing_pic.height,
                    batch=self.batch,
                    group=self.pawngroup)
                self.thinging = False
            else:
                try:
                    self.thing_pic_sprite.delete()
                except:
                    pass
                sp = self.board.get_spot_at(
                    x + self.view_left, y + self.view_bot)
                if sp is not None:
                    pl = sp.place
                    th = self.rumor.make_generic_thing(self.dimension, pl)
                    self.board.make_pawn(th)
                    th.pawns[int(self.board)].set_img(self.thing_pic)
                    logger.debug("made generic thing: %s", str(th))
                self.thing_pic = None
            return
        if self.portaling:
            if self.portal_from is None:
                if hasattr(self.pressed, 'place'):
                    print "portaling from {0}".format(self.pressed)
                    self.portal_from = self.pressed
                    self.floaty_portal.orig = self.portal_from
                    return
                else:
                    self.portaling = False
                    self.portal_from = None
                    try:
                        self.floaty_portal.delete()
                    except:
                        pass
                    return
            else:
                if (
                        hasattr(self.pressed, 'place') and
                        hasattr(self.portal_from, 'place') and
                        self.pressed.place != self.portal_from.place):
                    port = self.rumor.make_portal(
                        self.portal_from.place,
                        self.pressed.place)
                    self.board.make_arrow(port)
                self.portaling = False
                self.portal_from = None
                self.floaty_portal.delete()
                return
        if (
                self.pressed is not None and
                self.pressed in self.selected):
            self.selected.remove(self.pressed)
        if hasattr(self.selected, 'unselect'):
                self.selected.unselect()
        if not self.keep_selected:
            for it in iter(self.selected):
                if hasattr(it, 'unselect'):
                    it.unselect()
            self.selected = set()
        if hasattr(self.grabbed, 'dropped'):
            self.grabbed.dropped(x, y, button, modifiers)
        if hasattr(self.pressed, 'selectable'):
            self.selected.add(self.pressed)
            if hasattr(self.pressed, 'select'):
                self.pressed.select()
        self.grabbed = None
        self.pressed = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """If the thing previously pressed has a
move_with_mouse method, use it.
     """
        if self.grabbed is None:
            self.grabbed = self.pressed
        elif hasattr(self.grabbed, 'move_with_mouse'):
            self.grabbed.move_with_mouse(x, y, dx, dy, buttons, modifiers)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # for now, this only does anything if you're moused over
        # the calendar
        for calendar in self.calendars:
            if calendar.overlaps(x, y):
                sf = calendar.scroll_factor
                calendar.scrolled_to += scroll_y * sf
                return
        if self.picker is not None:
            if self.picker.overlaps(x, y):
                while scroll_y > 0:
                    self.picker.scroll_up_once()
                    scroll_y -= 1
                while scroll_y < 0:
                    self.picker.scroll_down_once()
                    scroll_y += 1

    def on_mouse_motion(self, x, y, dx, dy):
        """Find the widget, if any, that the mouse is over,
and highlight it.
        """
        self.mouspot.x = x
        self.mouspot.y = y
        self.dx_hist[self.dxdy_hist_counter % self.dxdy_hist_max] = dx
        self.dy_hist[self.dxdy_hist_counter % self.dxdy_hist_max] = dy
        self.dxdy_hist_counter += 1

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.DELETE:
            self.delete_selection()

    def delete_selection(self):
        for dead in iter(self.selected):
            dead.delete()
        self.selected = set()

    def draw_line(self, points, color, group, verts=None):
        colors = color * 2
        if verts is None:
            verts = self.batch.add(
                2,
                pyglet.gl.GL_LINES,
                group,
                ('v2i', tuple(points)),
                ('c4B', tuple(colors)))
        else:
            verts.vertices = list(points)
            verts.colors = list(colors)
        return verts

    def draw_box(
        self, left, top, right, bot,
            color, group, verts=(None, None, None, None)):
        return (
            self.draw_line(
                (left, bot, left, top),
                color,
                group,
                verts[0]),
            self.draw_line(
                (left, top, right, top),
                color,
                group,
                verts[1]),
            self.draw_line(
                (right, top, right, bot),
                color,
                group,
                verts[2]),
            self.draw_line(
                (right, bot, left, bot),
                color,
                group,
                verts[3]))

    def draw_menu(self, menu):
        menu.draw(self.batch, self.menugroup)
        for menu_item in menu:
            if menu_item.label is not None:
                try:
                    menu_item.label.delete()
                except (AttributeError, AssertionError):
                    pass
            menu_item.draw(self.batch, menu.labelgroup)

    def sensible_calendar_for(self, something):
        """Return a calendar appropriate for representing some schedule-dict
associated with the argument."""
        return self.calendars[0]

    def create_place(self):
        self.picker = PicPicker(
            self, 0.3, 0.6, 0.3, 0.6,
            self.calendars[0].style, 'place_pic', 'placing')

    def create_thing(self):
        self.picker = PicPicker(
            self, 0.3, 0.6, 0.3, 0.6,
            self.calendars[0].style, 'thing_pic', 'thinging')

    def create_portal(self):
        boguspot = MousySpot()
        (boguspot.x, boguspot.y) = self.floaty_coords()
        self.floaty_portal = Arrow(self.board, boguspot, self.mouspot)
        self.portaling = True

    def floaty_coords(self):
        dx = self.dx
        dy = self.dy
        length = self.arrowhead_size * 2
        x = self.mouspot.x
        y = self.mouspot.y
        if dx == 0:
            if dy > 0:
                return (x, y - length)
            else:
                return (x, y + length)
        elif dy == 0:
            if dx > 0:
                return (x - length, y)
            else:
                return (x + length, y)
        else:
            xco = 1
            yco = 1
            if dx < 0:
                xco = -1
            if dy < 0:
                yco = -1
            x *= xco
            dx *= xco
            y *= yco
            dy *= yco
            theta = atan(float(dy) / float(dx))
            xleft = int(x - cos(theta) * length)
            ybot = int(y - sin(theta) * length)
            return (xleft * xco, ybot * yco)

    def get_tabdict(self):
        return {
            "window": [{
                "name": str(self),
                "min_width": self.min_width,
                "min_height": self.min_height,
                "dimension": str(self.dimension),
                "board": int(self.board),
                "arrowhead_size": self.arrowhead_size,
                "arrow_width": self.arrow_width,
                "view_left": self.view_left,
                "view_bot": self.view_bot,
                "main_menu": self.main_menu_name}]}
