# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from util import SaveableWidgetMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.properties import AliasProperty, DictProperty, ObjectProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image


"""Class for user's view on gameworld, and support functions."""


class Wallpaper(Image):
    board = ObjectProperty()

    def __init__(self, **kwargs):
        starttex = kwargs["board"].get_texture()
        Image.__init__(self, texture=starttex, size=starttex.size, **kwargs)
        self.board.bind(rowdict=self.upd_texture)

    def upd_texture(self, instance, value):
        self.texture = self.board.get_texture()
        self.size = self.texture.size


class Board(ScrollView):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [(
        "board",
        {"dimension": "text not null default 'Physical'",
         "wallpaper": "text not null default 'default_wallpaper'",
         "x": "float not null default 0.0",
         "y": "float not null default 0.0"},
        ("dimension",),
        {"wallpaper": ("img", "name")},
        ["x>=0", "y>=0", "x<=1", "y<=1"])]
    arrow_width = 1.4
    arrowhead_size = 10
    auto_bring_to_front = False
    closet = ObjectProperty()
    dimension = ObjectProperty()
    spotdict = DictProperty({})
    pawndict = DictProperty({})
    arrowdict = DictProperty({})
    rowdict = DictProperty({})
    scroll_x = AliasProperty(
        lambda self: self._get_scroll_x(),
        lambda self, v: self._set_scroll_x(v))
    scroll_y = AliasProperty(
        lambda self: self._get_scroll_y(),
        lambda self, v: self._set_scroll_y(v))

    def __init__(self, **kwargs):
        if kwargs["dimension"].__class__ in (str, unicode):
            kwargs["dimension"] = kwargs["closet"].get_dimension(
                kwargs["dimension"])
        walln = kwargs["closet"].skeleton["board"][unicode(
            kwargs["dimension"])]["wallpaper"]
        walltex = kwargs["closet"].get_texture(walln)
        ScrollView.__init__(
            self, size=walltex.size,
            size_hint=(None, None), **kwargs)
        self.last_touch = None
        self.closet.boarddict[unicode(self.dimension)] = self
        self.upd_rowdict()
        wall = Wallpaper(board=self)
        self.closet.skeleton["board"][unicode(self.dimension)].bind(
            touches=self.upd_rowdict)
        content = RelativeLayout()
        self.add_widget(content)
        content.add_widget(wall)
        if (
                "spot_coords" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "spot_coords"]):
            for rd in self.dimension.closet.skeleton[
                    "spot_coords"][unicode(self.dimension)].iterrows():
                place = self.dimension.get_place(rd["place"])
                spot = Spot(board=self, place=place)
                self.spotdict[unicode(place)] = spot
        if (
                "pawn_img" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "pawn_img"]):
            for rd in self.dimension.closet.skeleton[
                    "pawn_img"][unicode(self.dimension)].iterrows():
                thing = self.dimension.get_thing(rd["thing"])
                pawn = Pawn(board=self, thing=thing)
                self.pawndict[unicode(thing)] = pawn
        for portal in self.dimension.portals:
            arrow = Arrow(board=self, portal=portal)
            self.arrowdict[unicode(portal)] = arrow
            content.add_widget(arrow)
        for spot in self.spotdict.itervalues():
            content.add_widget(spot)
        for pawn in self.pawndict.itervalues():
            content.add_widget(pawn)

    def __str__(self):
        return str(self.dimension)

    def _get_scroll_x(self):
        return self.closet.skeleton["board"][unicode(self)]["x"]

    def _set_scroll_x(self, v):
        self.closet.skeleton["board"][unicode(self)]["x"] = v

    def _get_scroll_y(self):
        return self.closet.skeleton["board"][unicode(self)]["y"]

    def _set_scroll_y(self, v):
        self.closet.skeleton["board"][unicode(self)]["y"] = v

    def upd_rowdict(self, *args):
        self.rowdict = dict(self.closet.skeleton["board"][unicode(self)])

    def get_texture(self):
        return self.closet.get_texture(self.rowdict["wallpaper"])

    def new_branch(self, parent, branch, tick):
        for spot in self.spotdict.itervalues():
            spot.new_branch(parent, branch, tick)
        for pawn in self.pawndict.itervalues():
            pawn.new_branch(parent, branch, tick)

    def on_touch_down(self, touch):
        collidable_iters = [
            self.pawndict.itervalues(),
            self.spotdict.itervalues(),
            self.arrowdict.itervalues()]
        for it in collidable_iters:
            for that in it:
                if that.collide_point(touch.x, touch.y):
                    that.dragging = True
                    if isinstance(that, Spot):
                        loc = that.place
                        for thing in loc.get_contents():
                            thingn = unicode(thing)
                            if thingn in self.pawndict:
                                pawn = self.pawndict[thingn]
                                that.bind(
                                    transform=pawn.extra_translate)
        super(Board, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        collidable_iters = [
            self.pawndict.itervalues(),
            self.spotdict.itervalues(),
            self.arrowdict.itervalues()]
        for it in collidable_iters:
            for that in it:
                if that.dragging:
                    if isinstance(that, Spot):
                        loc = that.place
                        for thing in loc.get_contents():
                            thingn = unicode(thing)
                            if thingn in self.pawndict:
                                pawn = self.pawndict[thingn]
                                pawn.unbind(transform=pawn.extra_translate)
                    that.dragging = False
                    that.on_drop()
