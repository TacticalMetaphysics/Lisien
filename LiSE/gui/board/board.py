# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    AliasProperty,
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
    completion = NumericProperty(0)

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

    scroll_x = AliasProperty(
        lambda self: self.bone.x,
        _set_x)
    scroll_y = AliasProperty(
        lambda self: self.bone.y,
        _set_y)

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

    def on_facade(self, i, v):
        self.completion += 1

    def on_host(self, i, v):
        self.completion += 1

    def on_parent(self, i, v):
        self.completion += 1

    def on_completion(self, i, v):
        if v == 3:
            self.finalize()

    def finalize(self):
        branch = self.facade.closet.branch
        tick = self.facade.closet.tick
        obsrvr = unicode(self.facade.observer)
        obsrvd = unicode(self.facade.observed)
        host = unicode(self.host)
        if obsrvr not in self.facade.closet.board_d:
            self.facade.closet.board_d[obsrvr] = {}
        if obsrvd not in self.facade.closet.board_d[obsrvr]:
            self.facade.closet.board_d[obsrvr][obsrvd] = {}
        self.facade.closet.board_d[obsrvr][obsrvd][host] = self
        tex = self.facade.closet.get_texture(self.bone.wallpaper)
        content = RelativeLayout(
            size_hint=(None, None),
            size=tex.size)
        content.add_widget(Image(pos=(0, 0), texture=tex, size=tex.size))
        super(Board, self).add_widget(content)
        # Regardless of what the facade *shows*, create spots, pawns,
        # and portals for everything in the host, just in case I need
        # to show them.
        for bone in self.facade.closet.skeleton[u"spot"].iterbones():
            if bone.host == host and bone.place not in self.spotdict:
                char = self.facade.closet.get_character(bone.host)
                place = char.get_place(bone.place)
                self.spotdict[bone.place] = Spot(board=self, place=place)
        for bone in self.facade.closet.skeleton[u"portal"].iterbones():
            if bone.host == host and bone.name not in self.arrowdict:
                char = self.facade.closet.get_character(bone.character)
                try:
                    port = char.get_portal(bone.name)
                except KeyError:
                    boen = char.closet.skeleton[u"portal_loc"][
                        unicode(char)][bone.name][branch].value_during(tick)
                    port = char.make_portal(
                        boen.origin, boen.destination, bone.host,
                        bone.name, boen.branch, boen.tick)
                self.arrowdict[bone.name] = Arrow(board=self, portal=port)
                content.add_widget(self.arrowdict[bone.name])
        for bone in self.facade.closet.skeleton[u"pawn"].iterbones():
            if bone.host == host and bone.thing not in self.pawndict:
                char = self.facade.closet.get_character(bone.observed)
                try:
                    thing = char.get_thing(bone.thing)
                except KeyError:
                    thing = char.make_thing(bone.thing)
                self.pawndict[bone.thing] = Pawn(board=self, thing=thing)
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
            spot.new_branch(parent, branch, tick)
        for pawn in self.pawndict.itervalues():
            pawn.new_branch(parent, branch, tick)

    def on_touch_down(self, touch):
        for preemptor in ("charsheet", "menu"):
            if preemptor in touch.ud:
                return
        if not self._touch:
            for pawn in self.pawndict.itervalues():
                if pawn.collide_point(touch.x, touch.y):
                    self._touch = touch
                    break
        if not self._touch:
            for spot in self.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    self._touch = touch
                    break
        if self.parent.dummyspot is not None:
            self.parent.dummyspot.pos = (touch.x, touch.y)
        return super(Board, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.parent.dummyspot is not None:
            self.parent.dummyspot.pos = (touch.x, touch.y)
        return super(Board, self).on_touch_move(touch)
