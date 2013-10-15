# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from util import SaveableWidgetMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.graphics import Rectangle
from kivy.properties import AliasProperty, StringProperty, DictProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image


"""Class for user's view on gameworld, and support functions."""


class Wallpaper(Image):
    def __init__(self, board, **kwargs):
        self.board = board
        starttex = self.board.get_texture()
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
    rowdict = DictProperty({})
    scroll_x = AliasProperty(
        lambda self: self._get_scroll_x(),
        lambda self, v: self._set_scroll_x(v))
    scroll_y = AliasProperty(
        lambda self: self._get_scroll_y(),
        lambda self, v: self._set_scroll_y(v))
    

    def __init__(self, closet, dimension, **kwargs):
        self.closet = closet
        self.dimension = dimension
        self.spotdict = {}
        self.pawndict = {}
        self.arrowdict = {}
        self.selected = set()
        self.upd_rowdict()
        self.closet.skeleton["board"][unicode(self.dimension)].bind(
            touches=self.upd_rowdict)
        wall = Wallpaper(self)
        ScrollView.__init__(self, size=wall.size, **kwargs)
        content = RelativeLayout(size=wall.size, size_hint=(None, None))
        content.add_widget(wall)
        if (
                "spot_coords" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "spot_coords"]):
            for rd in self.dimension.closet.skeleton[
                    "spot_coords"][unicode(self.dimension)].iterrows():
                place = self.dimension.get_place(rd["place"])
                spot = Spot(self, place)
                self.spotdict[unicode(place)] = spot
        if (
                "pawn_img" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "pawn_img"]):
            for rd in self.dimension.closet.skeleton[
                    "pawn_img"][unicode(self.dimension)].iterrows():
                thing = self.dimension.get_thing(rd["thing"])
                pawn = Pawn(self, thing)
                self.pawndict[unicode(thing)] = pawn
        for portal in self.dimension.portals:
            arrow = Arrow(self, portal)
            self.arrowdict[unicode(portal)] = arrow
            content.add_widget(arrow)
        for spot in self.spotdict.itervalues():
            content.add_widget(spot)
        for pawn in self.pawndict.itervalues():
            content.add_widget(pawn)
        self.add_widget(content)
    
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
