# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.graphics import Rectangle
from kivy.properties import (
    AliasProperty,
    DictProperty,
    NumericProperty,
    ObjectProperty)
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from spot import Spot
from arrow import Arrow
from pawn import Pawn


class Board(FloatLayout):
    """A graphical view onto a facade, resembling a game board."""
    __metaclass__ = SaveableWidgetMetaclass
    demands = ["thing", "img"]
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
            "primary_key": ("observer", "observed", "host"),
            "foreign_keys": {
                "arrow_bg": ("color", "name"),
                "arrow_fg": ("color", "name")},
            "checks": ("x>=0", "y>=0", "x<=1", "y<=1",
                       "arrow_width>0", "arrowhead_size>0")})]
    facade = ObjectProperty()
    host = ObjectProperty()
    wallpaper = ObjectProperty()

    def _set_bone(self, bone):
        self.facade.closet.skeleton[u"board"][
            unicode(self.facade.observer)][
            unicode(self.facade.observed)][
            unicode(self.host)] = bone

    def _set_x(self, x):
        bone = self.bone._replace(x=x)
        self._set_bone(bone)

    def _set_y(self, y):
        bone = self.bone._replace(y=y)
        self._set_bone(bone)

    spotdict = DictProperty({})
    pawndict = DictProperty({})
    arrowdict = DictProperty({})

    def on_spotdict(self, i, v):
        for vv in v.itervalues():
            if vv.parent not in (None, self):
                pass

    @property
    def bone(self):
        return self.facade.closet.skeleton[u"board"][
            unicode(self.facade.observer)][
            unicode(self.facade.observed)][
            unicode(self.host)]

    @property
    def arrow_width(self):
        return self.bone.arrow_width

    @property
    def arrowhead_size(self):
        return self.bone.arrowhead_size

    def skelset(self, skel, namefield, bone):
        if bone.observer not in skel:
            skel[bone.observer] = {}
        if 'observed' in bone._fields:
            if bone.observed not in skel[
                    bone.observer]:
                skel[bone.observer][bone.observed] = {}
            if bone.host not in skel[bone.observer][bone.observed]:
                skel[bone.observer][bone.observed][bone.host] = {}
            skel = skel[bone.observer][bone.observed][bone.host]
        else:
            if bone.host not in skel[bone.observer]:
                skel[bone.observer][bone.host] = {}
            skel = skel[bone.observer][bone.host]
        if getattr(bone, namefield) not in skel:
            skel[getattr(bone, namefield)] = []
        if 'layer' in bone._fields:
            if bone.layer not in skel[getattr(bone, namefield)]:
                skel[getattr(bone, namefield)][bone.layer] = []
            skel = skel[getattr(bone, namefield)][bone.layer]
        else:
            skel = skel[getattr(bone, namefield)]
        if bone.branch not in skel:
            skel[bone.branch] = []
        skel[bone.branch][bone.tick] = bone

    def finalize(self, *args):
        obsrvr = unicode(self.facade.observer)
        obsrvd = unicode(self.facade.observed)
        host = unicode(self.host)
        tex = self.host.closet.get_texture('default_wallpaper')
        self.wallpaper = tex
        self.size = self.wallpaper.size
        if obsrvr not in self.facade.closet.board_d:
            self.facade.closet.board_d[obsrvr] = {}
        if obsrvd not in self.facade.closet.board_d[obsrvr]:
            self.facade.closet.board_d[obsrvr][obsrvd] = {}
        self.facade.closet.board_d[obsrvr][obsrvd][host] = self
        # Regardless of what the facade *shows*, create spots, pawns,
        # and portals for everything in the host, just in case I need
        # to show them.
        for bone in self.facade.closet.skeleton[u"spot"].iterbones():
            if bone.host == host and bone.place not in self.spotdict:
                char = self.facade.closet.get_character(bone.host)
                place = char.get_place(bone.place)
                self.spotdict[bone.place] = Spot(board=self, place=place)
        for bone in self.facade.closet.skeleton[u"portal"][obsrvd].iterbones():
            if bone.host == host:
                port = self.facade.observed.get_portal(bone.name)
                self.arrowdict[bone.name] = Arrow(
                    board=self, portal=port)
        for bone in self.facade.closet.skeleton[u"pawn"].iterbones():
            if bone.host == host and bone.thing not in self.pawndict:
                char = self.facade.closet.get_character(bone.observed)
                try:
                    thing = char.get_thing(bone.thing)
                except KeyError:
                    thing = char.make_thing(bone.thing)
                self.pawndict[bone.thing] = Pawn(board=self, thing=thing)
        for arrow in self.arrowdict.itervalues():
            self.add_widget(arrow)
        for spot in self.spotdict.itervalues():
            self.add_widget(spot)
        for pawn in self.pawndict.itervalues():
            self.add_widget(pawn)

    def __str__(self):
        return str(self.facade)

    def __unicode__(self):
        return unicode(self.facade)

    def __repr__(self):
        return "Board({})".format(self)

    def get_texture(self):
        return self.facade.closet.get_texture(self.bone.wallpaper)

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
            for bone in spot.new_branch(parent, branch, tick):
                yield bone
        for pawn in self.pawndict.itervalues():
            for bone in pawn.new_branch(parent, branch, tick):
                yield bone
