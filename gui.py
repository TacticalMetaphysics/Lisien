# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
import ctypes
import math
import logging
from util import SaveableMetaclass
from math import atan, pi, sin, cos, hypot
from arrow import Arrow
from menu import Menu, MenuItem
from card import Hand
from calendar import Calendar
from picpicker import PicPicker

from collections import OrderedDict

logger = logging.getLogger(__name__)

ninety = math.pi / 2

fortyfive = math.pi / 4

threesixty = math.pi * 2


def average(*args):
    n = len(args)
    return sum(args)/n


line_len_rise_run = hypot


def line_len(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return hypot(rise, run)


def slope_theta_rise_run(rise, run):
    try:
        return atan(rise/run)
    except ZeroDivisionError:
        if rise >= 0:
            return ninety
        else:
            return -1 * ninety


def slope_theta(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return slope_theta_rise_run(rise, run)


def opp_theta_rise_run(rise, run):
    try:
        return atan(run/rise)
    except ZeroDivisionError:
        if run >= 0:
            return ninety
        else:
            return -1 * ninety


def opp_theta(ox, oy, dx, dy):
    rise = dy - oy
    run = dx - ox
    return opp_theta_rise_run(rise, run)


def truncated_line(leftx, boty, rightx, topy, r, from_start=False):
    # presumes pointed up and right
    if r == 0:
        return (leftx, boty, rightx, topy)
    rise = topy - boty
    run = rightx - leftx
    length = line_len_rise_run(rise, run) - r
    theta = slope_theta_rise_run(rise, run)
    if from_start:
        leftx = rightx - math.cos(theta) * length
        boty = topy - math.sin(theta) * length
    else:
        rightx = leftx + math.cos(theta) * length
        topy = boty + math.sin(theta) * length
    return (leftx, boty, rightx, topy)


def extended_line(leftx, boty, rightx, topy, r):
    return truncated_line(leftx, boty, rightx, topy, -1 * r)


def trimmed_line(leftx, boty, rightx, topy, trim_start, trim_end):
    et = truncated_line(leftx, boty, rightx, topy, trim_end)
    return truncated_line(et[0], et[1], et[2], et[3], trim_start, True)


def wedge_offsets_core(theta, opp_theta, taillen):
    top_theta = theta - fortyfive
    bot_theta = pi - fortyfive - opp_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    return (
        xoff1, yoff1, xoff2, yoff2)


def wedge_offsets_rise_run(rise, run, taillen):
    # theta is the slope of a line bisecting the ninety degree wedge.
    theta = slope_theta_rise_run(rise, run)
    opp_theta = opp_theta_rise_run(rise, run)
    return wedge_offsets_core(theta, opp_theta, taillen)


def wedge_offsets_slope(slope, taillen):
    theta = atan(slope)
    opp_theta = atan(1/slope)
    return wedge_offsets_core(theta, opp_theta, taillen)


def get_line_width():
    see = ctypes.c_float()
    pyglet.gl.glGetFloatv(pyglet.gl.GL_LINE_WIDTH, see)
    return float(see.value)


def set_line_width(w):
    wcf = ctypes.c_float(w)
    pyglet.gl.glLineWidth(wcf)


class ScissorOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent, window, left, top, bot, right):
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


class BoldLineOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent=None, width=1.0):
        self.width = float(width)
        pyglet.graphics.OrderedGroup.__init__(self, order, parent)

    def set_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_LINE_SMOOTH)
        set_line_width(self.width)


class SmoothBoldLineOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent=None, width=1.0):
        self.width = float(width)
        pyglet.graphics.OrderedGroup.__init__(self, order, parent)

    def set_state(self):
        set_line_width(self.width)
        pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)


class TransparencyGroup(pyglet.graphics.Group):
    def set_state(self):
        pyglet.gl.glEnable(pyglet.gl.GL_BLEND)

    def unset_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_BLEND)


class TransparencyOrderedGroup(
        pyglet.graphics.OrderedGroup,
        TransparencyGroup):
    pass


class WindowSaver:
    __metaclass__ = SaveableMetaclass
    tables = [
        ("window",
         {"name": "text not null default 'Main'",
          "min_width": "integer not null default 1280",
          "min_height": "integer not null default 800",
          "dimension": "text not null default 'Physical'",
          "board": "integer not null default 0",
          "arrowhead_size": "integer not null default 10",
          "arrow_width": "float not null default 1.4",
          "view_left": "integer not null default 0",
          "view_bot": "integer not null default 0",
          "main_menu": "text not null default 'Main'"},
         ("name",),
         {"dimension, board": ("board", "dimension, i"),
          "main_menu": ("menu", "name")},
         ["view_left>=0", "view_bot>=0"])]

    def __init__(self, win):
        self.win = win

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


class GameWindow(pyglet.window.Window):
    """Instantiates a Pyglet window and displays the given board in it."""
    edge_order = 1

    def __init__(self, rumor, name, min_width, min_height, dimension,
                 boardnum, arrowhead_size, arrow_width, view_left, view_bot,
                 main_menu, hand_rows, cal_rows, menu_rows, menu_item_rows):
        """Initialize the game window, its groups, and some state tracking."""
        super(GameWindow, self).__init__()
        self.rumor = rumor
        self.name = name
        self.min_width = min_width
        self.min_height = min_height
        self.set_minimum_size(self.min_width, self.min_height)
        self.dimension = dimension
        self.hands_by_name = OrderedDict()
        self.calendars = []
        self.menus_by_name = OrderedDict()
        self.main_menu_name = main_menu
        self.board = self.rumor.get_board(boardnum, self)
        self.arrowhead_size = arrowhead_size
        self.arrow_width = arrow_width
        self.squareoff = self.arrowhead_size * math.sin(fortyfive)
        self.view_left = view_left
        self.view_bot = view_bot
        self.saver = WindowSaver(self)

        for row in menu_rows:
            self.menus_by_name[row[0]] = Menu(
                self, row[0], row[1], row[2], row[3], row[4],
                self.rumor.styledict[row[5]])
        for row in menu_item_rows:
            self.menus_by_name[row[0]].items[row[1]] = MenuItem(
                self.menus_by_name[row[0]],
                row[1],
                row[2],
                row[4],
                row[3])
        self.hands_by_name = OrderedDict()
        for row in hand_rows:
            self.hands_by_name[row[0]] = Hand(
                self, self.rumor.effectdeckdict[row[0]],
                row[1], row[2], row[3], row[4],
                self.rumor.styledict[row[5]], row[6], row[7])
        self.calendars = []
        for row in cal_rows:
            self.calendars.append(
                Calendar(
                    self, row[0], row[1], row[2], row[3], row[4],
                    self.rumor.styledict[row[5]], row[6], row[7], row[8],
                    row[9]))
        self.picker = None
        self.thing_pic = None
        self.place_pic = None

        self.biggroup = pyglet.graphics.Group()
        self.boardgroup = pyglet.graphics.OrderedGroup(0, self.biggroup)
        self.edgegroup = pyglet.graphics.OrderedGroup(1, self.biggroup)
        self.spotgroup = pyglet.graphics.OrderedGroup(2, self.biggroup)
        self.pawngroup = pyglet.graphics.OrderedGroup(3, self.biggroup)
        self.higroup = pyglet.graphics.OrderedGroup(4, self.biggroup)
        self.calgroup = TransparencyOrderedGroup(5, self.biggroup)
        self.celgroup = TransparencyOrderedGroup(6, self.biggroup)
        self.labelgroup = pyglet.graphics.OrderedGroup(7, self.biggroup)
        self.pickergroup = ScissorOrderedGroup(
            8, self.biggroup, self, 0.3, 0.3, 0.3, 0.3)
        self.pickerbggroup = pyglet.graphics.OrderedGroup(0, self.pickergroup)
        self.pickerfggroup = pyglet.graphics.OrderedGroup(1, self.pickergroup)
        self.topgroup = pyglet.graphics.OrderedGroup(65535, self.biggroup)
        self.linegroups = {}
        self.bggd = {}
        self.fggd = {}

        self.pressed = None
        self.hovered = None
        self.grabbed = None
        self.selected = set()
        self.edge_order = 1
        self.keep_selected = False
        self.prev_view_bot = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.dxdy_hist_ct = 0
        dxdy_hist_max = 10
        self.dx_hist = [0] * dxdy_hist_max
        self.dy_hist = [0] * dxdy_hist_max

        self.batch = pyglet.graphics.Batch()

        self.drawn_board = None
        self.drawn_edges = None
        self.timeline = None

        self.onscreen = set([None])
        self.last_age = -1
        self.last_timeline_y = -1

        orbimg = self.rumor.imgdict['default_spot']
        rx = orbimg.width / 2
        ry = orbimg.height / 2
        self.create_place_cursor = (
            pyglet.window.ImageMouseCursor(orbimg, rx, ry))
        self.create_place_cursor.rx = rx
        self.create_place_cursor.ry = ry
        self.portaling = False
        self.portal_from = None
        self.portal_triple = ((None, None), (None, None), (None, None))
        for menu in self.menus:
            menu.adjust()

    def __getattr__(self, attrn):
        if attrn == 'hands':
            return self.hands_by_name.itervalues()
        elif attrn == 'calendars':
            return self.calendars_by_name.itervalues()
        elif attrn == 'menus':
            return self.menus_by_name.itervalues()
        elif attrn == 'main_menu':
            return self.menus_by_name[self.main_menu_name]
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
            try:
                return getattr(self.saver, attrn)
            except AttributeError:
                raise AttributeError(
                    "GameWindow has no attribute named {0}".format(attrn))

    def __str__(self):
        return self.name

    def on_draw(self):
        """Draw the background image; all spots, pawns, and edges on the
d; all visible menus; and the calendar, if it's visible."""
        # draw the image picker
        if self.picker is not None:
            newstate = self.picker.get_state_tup()
        else:
            newstate = None
        if newstate is not None and newstate not in self.onscreen:
            self.onscreen.discard(self.picker.oldstate)
            self.onscreen.add(newstate)
            self.picker.oldstate = newstate
            self.picker.delete()
            if self.picker.visible:
                if self.picker.hovered:
                    image = self.picker.bgpat_active.create_image(
                        self.picker.width, self.picker.height)
                else:
                    image = self.picker.bgpat_inactive.create_image(
                        self.picker.width, self.picker.height)
                    self.picker.sprite = pyglet.sprite.Sprite(
                        image,
                        self.picker.window_left,
                        self.picker.window_bot,
                        batch=self.batch,
                        group=self.pickerbggroup)
                    self.picker.layout()
                    for pixrow in self.picker.pixrows:
                        for pic in pixrow:
                            if pic.in_picker:
                                try:
                                    pic.sprite.x = pic.window_left
                                    pic.sprite.y = pic.window_bot
                                except AttributeError:
                                    pic.sprite = pyglet.sprite.Sprite(
                                        pic.tex,
                                        pic.window_left,
                                        pic.window_bot,
                                        batch=self.batch,
                                        group=self.pickerfggroup)
                                else:
                                    pic.delete()
        # draw the spots, representing places
        for spot in self.board.spots:
            if str(spot.place) == 'myroom':
                pass
            newstate = spot.get_state_tup()
            if newstate in self.onscreen:
                continue
            self.onscreen.discard(spot.oldstate)
            self.onscreen.add(newstate)
            spot.oldstate = newstate
            if spot.visible:
                l = spot.window_left
                r = spot.window_right
                b = spot.window_bot
                t = spot.window_top
                if spot in self.selected:
                    yelo = (255, 255, 0, 0)
                    spot.box_edges = self.draw_box(
                        l, t, r, b, yelo, self.higroup, spot.box_edges)
                else:
                    for vertls in spot.box_edges:
                        try:
                            vertls.delete()
                        except (AttributeError, AssertionError):
                            pass
                    spot.box_edges = (None, None, None, None)
                try:
                    spot.sprite.x = spot.window_left
                    spot.sprite.y = spot.window_bot
                except AttributeError:
                    spot.sprite = pyglet.sprite.Sprite(
                        spot.img.tex,
                        spot.window_left,
                        spot.window_bot,
                        batch=self.batch,
                        group=self.spotgroup)
            else:
                try:
                    spot.sprite.delete()
                except (AttributeError, AssertionError):
                    pass
        # draw the edges, representing portals
        if self.portaling:
            if self.portal_from is not None:
                for pair in self.portal_triple:
                    for vex in pair:
                        try:
                            vex.delete()
                        except:
                            pass
                self.portal_triple = self.connect_arrow(
                    self.portal_from.window_x,
                    self.portal_from.window_y,
                    self.last_mouse_x,
                    self.last_mouse_y,
                    self.portal_triple)
            else:
                dx = self.dx
                dy = self.dy
                length = self.arrowhead_size * 2
                x = self.last_mouse_x
                y = self.last_mouse_y
                if dx == 0:
                    if dy > 0:
                        coords = (x, y - length, x, y)
                    else:
                        coords = (x, y + length, x, y)
                elif dy == 0:
                    if dx > 0:
                        coords = (x - length, y, x, y)
                    else:
                        coords = (x + length, y, x, y)
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
                    theta = atan(float(dy)/float(dx))
                    xleft = int(x - cos(theta) * length)
                    ybot = int(y - sin(theta) * length)
                    coords = (xleft * xco, ybot * yco, x * xco, y * yco)
                    (x1, y1, x2, y2) = coords
                    self.portal_triple = self.connect_arrow(
                        x1, y1, x2, y2, 0, self.portal_triple)
        for edge in self.board.arrows:
            newstate = edge.get_state_tup()
            if newstate in self.onscreen:
                continue
            self.onscreen.discard(edge.oldstate)
            self.onscreen.add(newstate)
            edge.oldstate = newstate
            if edge.orig.visible or edge.dest.visible:
                order = edge.order
                if edge in self.selected:
                    order += 9000
                    for pair in edge.vertices:
                        for unit in pair:
                            try:
                                unit.delete()
                            except:
                                pass
                    edge.vertices = (
                        (None, None), (None, None), (None, None))
                edge.vertices = self.connect_arrow(
                    edge.orig.window_x,
                    edge.orig.window_y,
                    edge.dest.window_x,
                    edge.dest.window_y,
                    order,
                    edge.vertices,
                    edge.dest.r,
                    edge.highlit)
            else:
                for pair in edge.vertices:
                    for twopair in pair:
                        try:
                            twopair.delete()
                        except:
                            pass
        # draw the pawns, representing things
        for pawn in self.board.pawns:
            newstate = pawn.get_state_tup()
            if newstate in self.onscreen:
                continue
            self.onscreen.discard(pawn.oldstate)
            self.onscreen.add(newstate)
            pawn.oldstate = newstate
            if pawn.visible:
                l = pawn.window_left
                r = pawn.window_right
                b = pawn.window_bot
                t = pawn.window_top
                try:
                    pawn.sprite.x = pawn.window_left
                    pawn.sprite.y = pawn.window_bot
                except AttributeError:
                    pawn.sprite = pyglet.sprite.Sprite(
                        pawn.img.tex,
                        l, b,
                        batch=self.batch,
                        group=self.pawngroup)
                if pawn in self.selected:
                    yelo = (255, 255, 0, 0)
                    pawn.box_edges = self.draw_box(
                        l, t, r, b, yelo, self.higroup, pawn.box_edges)
                else:
                    for edge in pawn.box_edges:
                        try:
                            edge.delete()
                        except (AttributeError, AssertionError):
                            pass
                    pawn.box_edges = (None, None, None, None)
            else:
                try:
                    pawn.sprite.delete()
                except (AttributeError, AssertionError):
                    pass
                for edge in pawn.box_edges:
                    try:
                        edge.delete()
                    except (AttributeError, AssertionError):
                        pass
     # draw the menus, really just their backgrounds for the moment
        for (menuname, menu) in self.menus_by_name.iteritems():
            menustate = menu.get_state_tup()
            if menustate not in self.onscreen:
                self.onscreen.add(menustate)
                self.onscreen.discard(menu.oldstate)
                menu.oldstate = menustate
                if menu.visible or menuname == self.main_menu_name:
                    image = (
                        menu.inactive_pattern.create_image(
                            menu.width,
                            menu.height))
                    menu.sprite = pyglet.sprite.Sprite(
                        image, menu.window_left, menu.window_bot,
                        batch=self.batch, group=self.calgroup)
                else:
                    try:
                        menu.sprite.delete()
                    except:
                        pass
            for item in menu:
                newstate = item.get_state_tup()
                if newstate in self.onscreen:
                    continue
                self.onscreen.add(newstate)
                self.onscreen.discard(item.oldstate)
                item.oldstate = newstate
                if menu.visible or menuname == self.main_menu_name:
                    item.label = pyglet.text.Label(
                        item.text,
                        menu.style.fontface,
                        menu.style.fontsize,
                        color=menu.style.textcolor.tup,
                        x=item.window_left,
                        y=item.window_bot,
                        batch=self.batch,
                        group=self.labelgroup)
                else:
                    try:
                        item.label.delete()
                    except:
                        pass

      # draw the calendar
        for calendar in self.calendars:
            newstate = calendar.get_state_tup()
            if newstate not in self.onscreen:
                self.onscreen.add(newstate)
                self.onscreen.discard(calendar.oldstate)
                calendar.oldstate = newstate
                for calcol in calendar.cols:
                    if calcol.width != calcol.old_width:
                        calcol.old_image = (
                            calcol.inactive_pattern.create_image(
                                calcol.width, calcol.height))
                        calcol.old_width = calcol.width
                        image = calcol.old_image
                        calcol.sprite = pyglet.sprite.Sprite(
                            image,
                            calcol.window_left,
                            calcol.window_bot,
                            batch=self.batch,
                            group=self.calgroup)
                    for cel in calcol.cells:
                        if cel.visible:
                            if self.hovered is cel:
                                image = cel.active_pattern.create_image(
                                    cel.width, cel.height).texture
                            else:
                                image = cel.inactive_pattern.create_image(
                                    cel.width, cel.height).texture
                            cel.sprite = pyglet.sprite.Sprite(
                                image,
                                cel.window_left,
                                cel.window_bot,
                                batch=self.batch,
                                group=self.celgroup)
                            y = cel.window_top - cel.label_height
                            if cel.label is None:
                                cel.label = pyglet.text.Label(
                                    cel.text,
                                    cel.style.fontface,
                                    cel.style.fontsize,
                                    color=cel.style.textcolor.tup,
                                    width=cel.width,
                                    height=cel.height,
                                    x=cel.window_left,
                                    y=y,
                                    multiline=True,
                                    batch=self.batch,
                                    group=self.labelgroup)
                            else:
                                cel.label.x = cel.window_left
                                cel.label.y = y
                        else:
                            try:
                                cel.label.delete()
                            except AttributeError:
                                pass
                            try:
                                cel.sprite.delete()
                            except AttributeError:
                                pass
                            cel.label = None
                            cel.sprite = None
        if self.last_age != self.rumor.tick:
            # draw the time line on top of the calendar
            for calendar in self.calendars:
                try:
                    calendar.timeline.delete()
                except (AttributeError, AssertionError):
                    pass
                if not (calendar.visible and len(calendar.cols) > 0):
                    continue
                top = calendar.window_top
                left = calendar.window_left
                right = calendar.window_right
                starting = calendar.scrolled_to
                age = self.rumor.tick
                age_from_starting = age - starting
                age_offset = age_from_starting * calendar.row_height
                y = top - age_offset
                color = (255, 0, 0)
                if (
                        calendar.visible and
                        y > calendar.window_bot):
                    calendar.timeline = self.batch.add(
                        2, pyglet.graphics.GL_LINES, self.topgroup,
                        ('v2i', (left, y, right, y)),
                        ('c3B', color * 2))
            self.last_age = self.rumor.tick
        # draw any and all hands
        for hand in self.hands:
            # No state management yet because the hand itself has
            # no graphics. The cards in it do.
            if not (hand.visible and hand.on_screen):
                continue
            for card in hand:
                ctxth = card.textholder
                redrawn = (card.bgimage is None or
                           ctxth.bgimage is None or
                           card.bgimage.width != card.width or
                           card.bgimage.height != card.height)
                if redrawn:
                    card.bgimage = (
                        card.pats.bg_inactive.create_image(
                            card.width, card.height))
                    ctxth.bgimage = (
                        card.pats.bg_active.create_image(
                            ctxth.width,
                            ctxth.height))
                    try:
                        card.bgsprite.delete()
                    except (AttributeError, AssertionError):
                        pass
                    try:
                        ctxth.bgsprite.delete()
                    except (AttributeError, AssertionError):
                        pass
                    card.bgsprite = None
                    ctxth.bgsprite = None
                if card.visible:
                    if card.bgsprite is None:
                        card.bgsprite = pyglet.sprite.Sprite(
                            card.bgimage,
                            card.window_left,
                            card.window_bot,
                            batch=self.batch,
                            group=self.calgroup)
                    else:
                        if card.bgsprite.x != card.window_left:
                            card.bgsprite.x = card.window_left
                        if card.bgsprite.y != card.window_bot:
                            card.bgsprite.y = card.window_bot
                        if redrawn:
                            card.bgsprite.image = card.bgimage
                    if ctxth.bgsprite is None:
                        ctxth.bgsprite = pyglet.sprite.Sprite(
                            ctxth.bgimage,
                            ctxth.window_left,
                            ctxth.window_bot,
                            batch=self.batch,
                            group=self.celgroup)
                    else:
                        if ctxth.bgsprite.x != ctxth.window_left:
                            ctxth.bgsprite.x = ctxth.window_left
                        if ctxth.bgsprite.y != ctxth.window_bot:
                            ctxth.bgsprite.y = ctxth.window_bot
                        if redrawn:
                            ctxth.bgsprite.image = ctxth.bgimage
                    if ctxth.label is None:
                        ctxth.label = pyglet.text.Label(
                            card.text,
                            ctxth.style.fontface,
                            ctxth.style.fontsize,
                            anchor_y='bottom',
                            x=ctxth.text_left,
                            y=ctxth.text_bot,
                            width=ctxth.text_width,
                            height=ctxth.text_height,
                            multiline=True,
                            batch=self.batch,
                            group=self.labelgroup)
                    else:
                        if (
                                ctxth.label.x !=
                                ctxth.text_left):
                            ctxth.label.x = (
                                ctxth.text_left)
                        if (
                                ctxth.label.y !=
                                ctxth.text_bot):
                            ctxth.label.y = (
                                ctxth.text_bot)
                        if (
                                ctxth.label.width !=
                                ctxth.text_width):
                            ctxth.label.width = (
                                ctxth.text_width)
                        if (
                                ctxth.label.height !=
                                ctxth.text_height):
                            ctxth.label.height = (
                                ctxth.text_height)
                    if isinstance(card.img, pyglet.image.AbstractImage):
                        x = card.window_left + card.style.spacing
                        y = ctxth.window_top + card.style.spacing
                        if card.imgsprite is None:
                            card.imgsprite = pyglet.sprite.Sprite(
                                card.img,
                                x, y,
                                batch=self.batch,
                                group=self.celgroup)
                        else:
                            if card.imgsprite.x != x:
                                card.imgsprite.x = x
                            if card.imgsprite.y != y:
                                card.imgsprite.y = y
                else:  # card not visible
                    for dead in (
                            card.bgsprite,
                            card.imgsprite,
                            ctxth.bgsprite,
                            ctxth.label):
                        if dead is not None:
                            try:
                                dead.delete()
                            except:
                                pass
                    card.bgsprite = None
                    card.imgsprite = None
                    ctxth.bgsprite = None
                    ctxth.label = None
        # draw the background image
        if self.drawn_board is None:
            self.drawn_board = pyglet.sprite.Sprite(
                self.board.wallpaper.tex,
                self.offset_x,
                self.offset_y,
                batch=self.batch, group=self.boardgroup)
        else:
            if self.drawn_board.x != self.offset_x:
                self.drawn_board.x = self.offset_x
            if self.drawn_board.y != self.offset_y:
                self.drawn_board.y = self.offset_y
        # well, I lied. I was really only adding those things to the batch.
        # NOW I'll draw them.
        self.batch.draw()

    def on_mouse_press(self, x, y, button, modifiers):
        """If there's something already highlit, and the mouse is
still over it when pressed, it's been half-way clicked; remember this."""
        logger.debug("mouse pressed at %d, %d", x, y)
        self.pressed = self.hovered

    def on_mouse_release(self, x, y, button, modifiers):
        """If something was being dragged, drop it. If something was being
pressed but not dragged, it's been clicked. Otherwise do nothing."""
        logger.debug("mouse released at %d, %d", x, y)
        if self.place_pic is not None:
            pl = self.rumor.make_generic_place(self.dimension)
            sp = self.board.get_spot(pl)
            sp.set_coords(x + self.view_left, y + self.view_bot)
            sp.set_img(self.place_pic)
            self.set_mouse_cursor()
            self.place_pic = None
            logger.debug("made generic place: %s", str(pl))
        elif self.thing_pic is not None:
            sp = self.board.get_spot_at(x + self.view_left, y + self.view_bot)
            if sp is not None:
                pl = sp.place
                th = self.rumor.make_generic_thing(self.dimension, pl)
                th.set_img(self.thing_pic)
                self.board.make_pawn(th)
            self.set_mouse_cursor()
            self.thing_pic = None
            logger.debug("made generic thing: %s", str(th))
        elif self.portaling:
            if self.portal_from is None:
                if hasattr(self.pressed, 'place'):
                    self.portal_from = self.pressed
                    logger.debug(
                        "Making a portal from %s...",
                        str(self.portal_from.place))
                else:
                    self.portaling = False
                    self.portal_from = None
                    for line in self.portal_triple:
                        if line is not None:
                            for edge in line:
                                try:
                                    edge.delete()
                                except:
                                    pass
                    self.portal_triple = (
                        (None, None), (None, None), (None, None))
            else:
                if (
                        hasattr(self.pressed, 'place') and
                        hasattr(self.portal_from, 'place') and
                        self.pressed.place != self.portal_from.place):
                    logger.debug("...to %s", str(self.pressed.place))
                    port = self.rumor.make_portal(
                        self.portal_from.place,
                        self.pressed.place)
                    a = Arrow(self.board, port)
                    port.arrows = []
                    while len(port.arrows) <= int(self.board):
                        port.arrows.append(None)
                    port.arrows[int(self.board)] = a
                self.portaling = False
                self.portal_from = None
                for line in self.portal_triple:
                    for edge in line:
                        try:
                            edge.delete()
                        except:
                            pass
                self.portal_triple = (
                    (None, None), (None, None), (None, None))
        elif self.grabbed is not None:
            if hasattr(self.grabbed, 'dropped'):
                self.grabbed.dropped(x, y, button, modifiers)
        elif self.pressed is not None:
            if not self.keep_selected:
                for sel in iter(self.selected):
                    sel.tweaks += 1
                self.selected = set()
            if self.pressed.overlaps(x, y):
                logger.debug("%s clicked", str(self.pressed))
                if hasattr(self.pressed, 'selectable'):
                    self.pressed.selected()
                    logger.debug("Selecting it.")
                    self.selected.add(self.pressed)
                    self.pressed.tweaks += 1
                    if hasattr(self.pressed, 'reciprocate'):
                        reciprocal = self.pressed.reciprocate()
                        if reciprocal is not None:
                            self.selected.add(reciprocal)
                            reciprocal.tweaks += 1
                if hasattr(self.pressed, 'onclick'):
                    self.pressed.onclick()
        else:
            for sel in iter(self.selected):
                sel.tweaks += 1
                if hasattr(sel, 'unselected'):
                    sel.unselected()
            self.selected = set()
        self.pressed = None
        self.grabbed = None

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """If the thing previously pressed has a
move_with_mouse method, use it.
     """
        if self.grabbed is None:
            if (
                    self.pressed is not None and
                    x > self.pressed.window_left and
                    x < self.pressed.window_right and
                    y > self.pressed.window_bot and
                    y < self.pressed.window_top and
                    hasattr(self.pressed, 'move_with_mouse')):
                self.grabbed = self.pressed
            else:
                self.view_left -= dx
                if (
                        self.view_left +
                        self.width >
                        self.board.wallpaper.width):
                    self.view_left = (
                        self.img.width -
                        self.width)
                elif self.view_left < 0:
                    self.view_left = 0
                self.view_bot -= dy
                if (
                        self.view_bot +
                        self.height >
                        self.board.wallpaper.height):
                    self.view_bot = (
                        self.img.height -
                        self.height)
                elif self.view_bot < 0:
                    self.view_bot = 0
                if self.pressed is not None:
                    self.pressed = None
                self.grabbed = None
        else:
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
        self.last_mouse_x = x
        self.last_mouse_y = y
        del self.dx_hist[0]
        del self.dy_hist[0]
        self.dx_hist.append(dx)
        self.dy_hist.append(dy)
        if self.hovered is None:
            if (
                    self.picker is not None and
                    self.picker.overlaps(x, y)):
                for pixrow in self.picker.pixrows:
                    for pic in pixrow:
                        if pic.overlaps(x, y):
                            self.hovered = pic
                            return
                self.hovered = self.picker
                return
            for hand in self.hands:
                if hand.overlaps(x, y):
                    for card in hand:
                        if (
                                x > card.window_left and
                                x < card.window_right):
                            self.hovered = card
                            card.tweaks += 1
                            return
            for menu in self.menus:
                if menu.overlaps(x, y):
                    for item in menu.items:
                        if (
                                y > item.window_bot and
                                y < item.window_top):
                            self.hovered = item
                            item.tweaks += 1
                            return
            for pawn in self.board.pawns:
                if pawn.overlaps(x, y):
                    self.hovered = pawn
                    pawn.tweaks += 1
                    return
            for spot in self.board.spots:
                if spot.overlaps(x, y):
                    self.hovered = spot
                    spot.tweaks += 1
                    return
            for edge in self.board.arrows:
                if edge.touching(x, y):
                    if (
                            self.pressed is None or
                            edge.order > self.pressed.order):
                        self.pressed = edge
        else:
            if not self.hovered.overlaps(x, y):
                self.hovered.tweaks += 1
                self.hovered = None

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.DELETE:
            self.delete_selection()

    def create_place(self):
        self.picker = PicPicker(self, 0.3, 0.3, 0.3, 0.3,
                                self.calendars[0].style, 'place_pic')

    def create_thing(self):
        self.picker = PicPicker(self, 0.3, 0.3, 0.3, 0.3,
                                self.calendars[0].style, 'thing_pic')

    def create_portal(self):
        if not hasattr(self, 'portaled'):
            self.portaled = 0
        self.portaled += 1
        self.portaling = True

    def delete_selection(self):
        for dead in iter(self.selected):
            dead.delete()
        self.selected = set()

    def connect_arrow(
            self, ox, oy, dx, dy,
            order,
            old_triple=((None, None), (None, None), (None, None)),
            center_shrink=0,
            highlight=False):
        supergroup = pyglet.graphics.OrderedGroup(order, self.edgegroup)
        bggroup = SmoothBoldLineOrderedGroup(
            0, supergroup, self.arrow_girth)
        fggroup = BoldLineOrderedGroup(
            1, supergroup, self.arrow_width)
        # xs and ys should be integers.
        #
        # results will be called l, c, r for left tail, center, right tail
        if dy < oy:
            yco = -1
        else:
            yco = 1
        if dx < ox:
            xco = -1
        else:
            xco = 1
        (leftx, boty, rightx, topy) = truncated_line(
            float(ox * xco), float(oy * yco),
            float(dx * xco), float(dy * yco), center_shrink+1)
        taillen = float(self.arrowhead_size)
        rise = topy - boty
        run = rightx - leftx
        if rise == 0:
            xoff1 = cos(fortyfive) * taillen
            yoff1 = xoff1
            xoff2 = xoff1
            yoff2 = -1 * yoff1
        elif run == 0:
            xoff1 = sin(fortyfive) * taillen
            yoff1 = xoff1
            xoff2 = -1 * xoff1
            yoff2 = yoff1
        else:
            (xoff1, yoff1, xoff2, yoff2) = wedge_offsets_rise_run(
                rise, run, taillen)
        x1 = int(rightx - xoff1) * xco
        x2 = int(rightx - xoff2) * xco
        y1 = int(topy - yoff1) * yco
        y2 = int(topy - yoff2) * yco
        endx = int(rightx) * xco
        endy = int(topy) * yco
        if highlight:
            bgcolor = (255, 255, 0, 0)
            fgcolor = (0, 0, 0, 0)
        else:
            bgcolor = (64, 64, 64, 64)
            fgcolor = (255, 255, 255, 0)
        lpoints = (x1, y1, endx, endy)
        cpoints = (ox, oy, endx, endy)
        rpoints = (x2, y2, endx, endy)
        lbg = self.draw_line(
            lpoints, bgcolor, bggroup, old_triple[0][0])
        cbg = self.draw_line(
            cpoints, bgcolor, bggroup, old_triple[1][0])
        rbg = self.draw_line(
            rpoints, bgcolor, bggroup, old_triple[2][0])
        lfg = self.draw_line(
            lpoints, fgcolor, fggroup, old_triple[0][1])
        rfg = self.draw_line(
            rpoints, fgcolor, fggroup, old_triple[2][1])
        cfg = self.draw_line(
            cpoints, fgcolor, fggroup, old_triple[1][1])
        return ((lbg, lfg), (cbg, cfg), (rbg, rfg))

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
        for menu_item in menu:
            if menu_item.label is not None:
                try:
                    menu_item.label.delete()
                except (AttributeError, AssertionError):
                    pass

    def on_close(self):
        self.dimension.save()
        self.board.save()
        for hand in self.hands_by_name.itervalues():
            hand.save()
        for cal in self.calendars:
            cal.save()
        for menu in self.menus_by_name.itervalues():
            menu.save()
        self.rumor.c.execute(
            "DELETE FROM window WHERE name=?", (str(self),))
        save_these = (
            str(self),
            self.min_width,
            self.min_height,
            str(self.dimension),
            int(self.board),
            self.arrowhead_size,
            self.arrow_width,
            self.view_left,
            self.view_bot,
            self.main_menu_name)
        self.rumor.c.execute(
            "INSERT INTO window (name, min_width, "
            "min_height, dimension, board, arrowhead_size, "
            "arrow_width, view_left, view_bot, main_menu) "
            "VALUES ({0})".format(
                ", ".join(["?"] * len(save_these))),
            save_these)
        self.rumor.conn.commit()
        self.rumor.conn.close()
        super(GameWindow, self).on_close()

    def sensible_calendar_for(self, something):
        """Return a calendar appropriate for representing some schedule-dict
associated with the argument."""
        return self.calendars[0]
