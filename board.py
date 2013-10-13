# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from util import SaveableWidgetMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.uix.scatter import Scatter
from kivy.properties import AliasProperty
from kivy.uix.relativelayout import RelativeLayout


"""Class for user's view on gameworld, and support functions."""





class Board(Scatter):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [(
        "board",
        {"dimension": "text not null default 'Physical'",
         "wallpaper": "text not null default 'default_wallpaper'",
         "left": "integer not null default 0",
         "bot": "integer not null default 0"},
        ("dimension",),
        {"wallpaper": ("img", "name")},
        [])]
    arrow_width = 1.4
    arrowhead_size = 10
    dxdy_hist_max = 10
    wallpaper = AliasProperty(
        lambda self: self.children[0],
        lambda self, v: None)
    

    def __init__(self, closet, dimension, **kwargs):
        self.closet = closet
        self.dimension = dimension
        wallpaper = self.closet.get_img(
            self.closet.skeleton["board"][str(self)]["wallpaper"])
        self.spotdict = {}
        self.pawndict = {}
        self.selected = set()
        Scatter.__init__(self)
        self.add_widget(wallpaper)
        wallpaper.add_widget(RelativeLayout(size=wallpaper.size))
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
            self.wallpaper.children[0].add_widget(arrow)
        for spot in self.spotdict.itervalues():
            self.wallpaper.children[0].add_widget(spot)
        for pawn in self.pawndict.itervalues():
            self.wallpaper.children[0].add_widget(pawn)
        self.move()

    def __str__(self):
        return str(self.dimension)

    def get_pos(self):
        rd = self.closet.skeleton["board"][str(self)]
        x = rd["left"]
        y = rd["bot"]
        return (x, y)

    def set_pos(self, x, y):
        rd = self.closet.skeleton["board"][str(self)]
        rd["left"] = x
        rd["bot"] = y

    def move(self):
        for spot in self.spotdict.itervalues():
            spot.move()
