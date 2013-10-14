# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from util import SaveableWidgetMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.properties import (
    AliasProperty,
    ReferenceListProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty)
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image


"""Class for user's view on gameworld, and support functions."""


class Wallpaper(Image):
    texture = AliasProperty(
        lambda self: self.board.get_texture(),
        lambda self, v: None)
    width = AliasProperty(
        lambda self: self.texture.width,
        lambda self, v: None,
        bind=('texture',))
    height = AliasProperty(
        lambda self: self.texture.height,
        lambda self, v: None,
        bind=('texture',))
    size = ReferenceListProperty(width, height)
    norm_image_size = AliasProperty(
        lambda self: self.size,
        lambda self, v: None,
        bind=('size',))
    def __init__(self, board, **kwargs):
        self.board = board
        Image.__init__(self, **kwargs)
        self.bind(texture=self.board.rowdict)


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
    spotdict = DictProperty({})
    pawndict = DictProperty({})
    arrowdict = DictProperty({})
    closet = ObjectProperty()
    dimension = ObjectProperty()
    rowdict = AliasProperty(
        lambda self: self.get_rowdict(),
        lambda self, v: None)

    def __init__(self, **kwargs):
        self.selected = set()
        ScrollView.__init__(self, scroll_hint=(None, None), scroll_y=0.0, **kwargs)
        Clock.schedule_once(self.populate, 0)

    def get_texture(self):
        if self.closet is not None and self.rowdict is not None:
            return self.closet.get_texture(self.rowdict["wallpaper"])

    def get_rowdict(self):
        if self.dimension != None:
            return self.closet.skeleton["board"][unicode(self.dimension)]

    def populate(self, *args):
        self.bind(rowdict=self.rowdict.touches)
        self.size = self.get_texture().size
        content = RelativeLayout(size=self.get_texture().size,
                                 size_hint=(None, None))
        wall = Wallpaper(self, size_hint=(None, None))
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
        self.add_widget(content)
