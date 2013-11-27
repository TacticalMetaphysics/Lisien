# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    DictProperty,
    NumericProperty,
    ObjectProperty)
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from spot import Spot
from arrow import Arrow
from card import Card
from pawn import Pawn

"""Class for user's view on gameworld, and support functions."""


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
    bone = ObjectProperty()
    offx = NumericProperty(0)
    offy = NumericProperty(0)
    wallwidth = NumericProperty(0)
    wallheight = NumericProperty(0)
    dragging = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        if kwargs["dimension"].__class__ in (str, unicode):
            kwargs["dimension"] = kwargs["closet"].get_dimension(
                kwargs["dimension"])
        ScrollView.__init__(self, scroll_y=0, **kwargs)
        self.last_touch = None
        self.closet.boarddict[unicode(self.dimension)] = self
        self.upd_bone()
        self.closet.skeleton["board"][unicode(
            self.dimension)].listener = self.upd_bone
        tex = self.get_texture()
        (self.wallwidth, self.wallheight) = tex.size
        content = RelativeLayout(size_hint=(None, None), size=tex.size)
        content.add_widget(Image(pos=(0, 0), texture=tex, size=tex.size))
        self.add_widget(content)
        if (
                "spot_coords" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "spot_coords"]):
            for bone in self.dimension.closet.skeleton[
                    "spot_coords"][unicode(self.dimension)].iterbones():
                place = self.dimension.get_place(bone.place)
                spot = Spot(board=self, place=place)
                self.spotdict[unicode(place)] = spot
        if (
                "pawn_img" in self.closet.skeleton and
                unicode(self.dimension) in self.dimension.closet.skeleton[
                    "pawn_img"]):
            for bone in self.dimension.closet.skeleton[
                    "pawn_img"][unicode(self.dimension)].iterbones():
                thing = self.dimension.get_thing(bone.thing)
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

    def __unicode__(self):
        return unicode(self.dimension)

    def __repr__(self):
        return "Board({})".format(self)

    def upd_bone(self, *args):
        self.bone = self.closet.skeleton["board"][unicode(self)]

    def get_texture(self):
        return self.closet.get_texture(self.bone.wallpaper)

    def get_spot(self, loc):
        if loc is None:
            return None
        if not hasattr(loc, 'v'):
            # I think this isn't always raising when I expect it to
            raise TypeError("Spots should only be made for Places")
        if unicode(loc) not in self.spotdict:
            self.spotdict[unicode(loc)] = Spot(board=self, place=loc)
        return self.spotdict[unicode(loc)]

    def get_arrow(self, loc):
        if loc is None:
            return None
        if not hasattr(loc, 'origin'):
            raise TypeError("Arrows should only be made for Portals")
        if unicode(loc) not in self.arrowdict:
            self.arrowdict[unicode(loc)] = Arrow(board=self, portal=loc)
        return self.arrowdict[unicode(loc)]

    def new_branch(self, parent, branch, tick):
        for spot in self.spotdict.itervalues():
            spot.new_branch(parent, branch, tick)
        for pawn in self.pawndict.itervalues():
            pawn.new_branch(parent, branch, tick)

    def on_touch_down(self, touch):
        for preemptor in ("charsheet", "menu"):
            if preemptor in touch.ud:
                return
        return super(Board, self).on_touch_down(touch)
