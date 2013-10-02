# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from util import (
    TerminableImg,
    TerminableInteractivity)
from pyglet.sprite import Sprite
from pyglet.gl import GL_LINES
from logging import getLogger


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class Pawn(TerminableImg, TerminableInteractivity):
    """A token to represent something that moves about between places."""
    atrdic = {
        "imagery": lambda self: self.get_imagery(),
        "interactivity": lambda self: self.get_interactivity(),
        "img": lambda self: self.get_img(),
        "visible": lambda self: self.img is not None,
        "coords": lambda self: self.get_coords(),
        "x": lambda self: self.coords[0],
        "y": lambda self: self.coords[1],
        "width": lambda self: self.img.width,
        "height": lambda self: self.img.height,
        "rx": lambda self: self.width / 2,
        "ry": lambda self: self.height / 2,
        "r": lambda self: {True: self.rx, False: self.ry}[self.rx > self.ry]}

    def __init__(self, closet, dimension, board, thing):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        self.closet = closet
        self.dimension = dimension
        self.board = board
        self.thing = thing
        self.grabpoint = None
        self.sprite = None
        self.oldstate = None
        self.tweaks = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.selectable = True
        self.vertlist = None

    def __str__(self):
        return str(self.thing)

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Pawn instance has no such attribute: " +
                attrn)

    def __setattr__(self, attrn, val):
        if attrn == "img":
            self.set_img(val)
        elif attrn == "interactive":
            self.set_interactive(val)
        else:
            super(Pawn, self).__setattr__(attrn, val)

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.imagery[branch][tick_from] = {
            "dimension": str(self.dimension),
            "thing": str(self.thing),
            "board": str(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "img": str(img)}

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.interactivity[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "thing": str(self.thing),
            "branch": branch,
            "tick_from": tick_from}

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            origspot = self.board.get_spot(loc.orig)
            destspot = self.board.get_spot(loc.dest)
            oc = origspot.get_coords(branch, tick)
            dc = destspot.get_coords(branch, tick)
            if None in (oc, dc):
                return self.cheat_coords
            (ox, oy) = oc
            (dx, dy) = dc
            prog = self.thing.get_progress(branch, tick)
            odx = dx - ox
            ody = dy - oy
            self.cheat_coords = (int(ox + odx * prog),
                                 int(oy + ody * prog))
            return self.cheat_coords
        elif str(loc) in self.board.spotdict:
            spot = self.board.get_spot(loc)
            return spot.get_coords()

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)


class PawnWidget:
    selectable = True

    def get_board_coords(self):
        loc = self.pawn.thing.location
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            # actually SpotWidgets
            origspot = self.viewport.spotdict[str(loc.orig)]
            destspot = self.viewport.spotdict[str(loc.dest)]
            (ox, oy) = origspot.coords
            (dx, dy) = destspot.coords
            prog = self.pawn.thing.get_progress()
            odx = dx - ox
            ody = dy - oy
            return (int(ox + odx * prog) + self.drag_offset_x,
                    int(oy + ody * prog) + self.drag_offset_y)
        elif str(loc) in self.viewport.spotdict:
            (x, y) = self.viewport.spotdict[str(loc)].coords
            return (x + self.drag_offset_x,
                    y + self.drag_offset_y)

    atrdic = {
        "coords": lambda self: self.get_board_coords(),
        "board_x": lambda self: self.coords[0],
        "board_y": lambda self: self.coords[1],
        "board_left": lambda self: self.board_x,
        "board_right": lambda self: self.board_x + self.pawn.width,
        "board_top": lambda self: self.board_y + self.pawn.height,
        "board_bot": lambda self: self.board_y,
        "window_left": lambda self: self.board_left + self.viewport.offset_x,
        "window_right": lambda self: self.board_right + self.viewport.offset_x,
        "window_bot": lambda self: self.board_bot + self.viewport.offset_y,
        "window_top": lambda self: self.board_top + self.viewport.offset_y,
        "selected": lambda self: self in self.window.selected,
        "hovered": lambda self: self is self.window.hovered,
        "pressed": lambda self: self is self.window.pressed,
        "grabbed": lambda self: self is self.window.grabbed,
        "in_view": lambda self: (
            self.window_right > 0 and
            self.window_left < self.window.width and
            self.window_top > 0 and
            self.window_bot < self.window.height),
        "calendars": lambda self: [
            cal for cal in self.window.calendars if
            cal.thing == self.thing]}

    def __init__(self, viewport, pawn):
        self.pawn = pawn
        self.closet = self.pawn.closet
        self.viewport = viewport
        self.batch = self.viewport.batch
        self.window = self.viewport.window
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.old_window_left = None
        self.old_window_bot = None
        self.old_points = None

    def __getattr__(self, attrn):
        if attrn in self.atrdic:
            return self.atrdic[attrn](self)
        elif attrn in (
                "img", "visible", "interactive",
                "width", "height", "thing"):
            return getattr(self.pawn, attrn)
        else:
            raise AttributeError(
                "PawnWidget instance has no attribute " + attrn)

    def __str__(self):
        return str(self.pawn)

    def get_cals(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        return

    def hover(self, x, y):
        return self

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.drag_offset_x += dx
        self.drag_offset_y += dy
        return self

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my thing doesn't have anything else to
do, make it journey there.

If it DOES have anything else to do, make the journey in another branch.

        """
        spotto = None
        for spot in self.viewport.board.spots:
            if (
                    self.window_left < spot.x and
                    spot.x < self.window_right and
                    self.window_bot < spot.y and
                    spot.y < self.window_top):
                spotto = spot
                break
        if spotto is not None:
            self.thing.journey_to(spotto.place)
            # This is a silly hack.
#            for cal in self.calendars:
#                cal.refresh()
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass
        try:
            self.vertlist.delete()
        except:
            pass

    def draw(self):
        if (
                self.pawn.get_coords() is None or
                self.img is None):
            self.delete()
            return
        try:
            wl = self.window_left
            wb = self.window_bot
            if self.old_window_left != wl:
                self.sprite.x = self.window_left
                self.old_window_left = wl
            if self.old_window_bot != wb:
                self.sprite.y = self.window_bot
                self.old_window_bot = wb
        except AttributeError:
            self.sprite = Sprite(
                self.img.tex,
                self.window_left,
                self.window_bot,
                batch=self.batch,
                group=self.window.pawn_group)
        if self.selected:
            yelo = (255, 255, 0, 255)
            colors = yelo * 4
            points = (
                self.window_left, self.window_top,
                self.window_right, self.window_top,
                self.window_right, self.window_bot,
                self.window_left, self.window_bot)
            if points != self.old_points:
                try:
                    self.vertlist.vertices = points
                except:
                    self.vertlist = self.batch.add_indexed(
                        4,
                        GL_LINES,
                        self.window.pawn_group,
                        (0, 1, 1, 2, 2, 3, 3, 0),
                        ('v2i', points),
                        ('c4B', colors))
                self.old_points = points
        else:
            try:
                self.vertlist.delete()
            except:
                pass
            self.vertlist = None

    def overlaps(self, x, y):
        return (
            self.window_left < x and x < self.window_right and
            self.window_bot < y and y < self.window_top)

    def pass_focus(self):
        return self.viewport
