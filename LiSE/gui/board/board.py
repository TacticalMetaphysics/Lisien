# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    DictProperty,
    NumericProperty,
    ObjectProperty)
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from spot import Spot
from arrow import Arrow
from pawn import Pawn


class Board(ScrollView):
    """A graphical view onto a facade, resembling a game board."""
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ("board", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "observed": "text not null default 'Physical'",
                "host": "text not null default 'Physical'",
                "wallpaper": "text not null default 'default_wallpaper'",
                "x": "float not null default 0.0",
                "y": "float not null default 0.0",
                "arrow_width": "float not null default 1.4",
                "arrowhead_size": "integer not null default 10",
                "arrow_bg": "text not null default 'black'",
                "arrow_fg": "text not null default 'white'"},
            "primary_key": ("observer", "observed"),
            "foreign_keys": {
                "arrow_bg": ("color", "name"),
                "arrow_fg": ("color", "name")},
            "checks": ("x>=0", "y>=0", "x<=1", "y<=1",
                       "arrow_width>0", "arrowhead_size>0")})]
    facade = ObjectProperty()
    bone = ObjectProperty(None, allownone=True)
    content = ObjectProperty(None)
    completion = NumericProperty(0)

    spotdict = DictProperty({})
    pawndict = DictProperty({})
    arrowdict = DictProperty({})

    def on_facade(self, i, v):
        self.completion += 1

    def on_parent(self, i, v):
        self.completion += 1

    def on_completion(self, i, v):
        if v == 2:
            self.finalize()

    def finalize(self):
        self.closet.boarddict[unicode(self.dimension)] = self
        self.upd_bone()
        self.closet.skeleton["board"][unicode(
            self.character)].listener = self.upd_bone
        tex = self.closet.get_texture(self.bone.wallpaper)
        content = RelativeLayout(
            size_hint=(None, None),
            size=tex.size)
        content.add_widget(Image(pos=(0, 0), texture=tex, size=tex.size))
        super(Board, self).add_widget(content)
        for bone in self.facade.closet.skeleton[u"spot"][
                unicode(self.facade)].iterbones():
            place = self.facade.get_place(bone.place)
            self.spotdict[bone.place] = Spot(board=self, place=place)
        for bone in self.facade.closet.skeleton[u"pawn"][
                unicode(self.facade)].iterbones():
            thing = self.facade.get_thing(bone.thing)
            self.pawndict[bone.thing] = Pawn(board=self, thing=thing)
        for bone in self.facade.closet.skeleton[u"portal"][
                unicode(self.facade)].iterbones():
            portal = self.facade.get_portal(bone.name)
            self.arrowdict[bone.name] = Arrow(board=self, portal=portal)
            content.add_widget(self.arrowdict[bone.name])
        for spot in self.spotdict.itervalues():
            content.add_widget(spot)
        for pawn in self.pawndict.itervalues():
            content.add_widget(pawn)

    def __str__(self):
        return str(self.facade)

    def __unicode__(self):
        return unicode(self.facade)

    def __repr__(self):
        return "Board({})".format(self)

    def add_widget(self, w):
        return self.children[0].add_widget(w)

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
