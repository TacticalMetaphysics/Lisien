# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ObjectProperty)
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import Clock
from spot import Spot
from arrow import Arrow
from pawn import Pawn


class BoardLayout(RelativeLayout):
    def add_widget(self, w):
        super(BoardLayout, self).add_widget(w)
        if hasattr(w, 'handle_time'):
            w.handle_time(*self.parent.host.closet.time)
        if hasattr(w, 'upd_texs'):
            w.upd_texs()


class Board(FloatLayout):
    """A graphical view onto a facade, resembling a game board."""
    __metaclass__ = SaveableWidgetMetaclass
    demands = ["thing", "graphic_img"]
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
    spotdict = DictProperty({})
    spotlayout = ObjectProperty()
    pawndict = DictProperty({})
    pawnlayout = ObjectProperty()
    arrowdict = DictProperty({})
    arrowlayout = ObjectProperty()
    bone = ObjectProperty()

    def _set_arrow_width(self, w):
        self.bone = self.bone._replace(arrow_width=w)
        self._trigger_set_bone()

    arrow_width = AliasProperty(
        lambda self: self.bone.arrow_width,
        _set_arrow_width,
        bind=('bone',))

    def _set_arrowhead_size(self, v):
        self.bone = self.bone._replace(arrowhead_size=v)
        self._trigger_set_bone()

    arrowhead_size = AliasProperty(
        lambda self: self.bone.arrowhead_size,
        _set_arrowhead_size,
        bind=('bone',))

    def _set_bone(self, *args):
        self.host.closet.set_bone(self.bone)

    def __init__(self, **kwargs):
        kwargs['size_hint'] = (None, None)
        self._trigger_set_bone = Clock.create_trigger(self._set_bone)
        return super(Board, self).__init__(**kwargs)

    def finalize(self, *args):
        def repos_all(*args):
            for each in (
                    self.wallpaper,
                    self.arrowlayout,
                    self.spotlayout,
                    self.pawnlayout):
                each.pos = self.pos
        if not self.facade and self.host:
            Clock.schedule_once(self.finalize, 0)
            return
        bone = self.host.closet.skeleton[u'board'][
            unicode(self.facade.observer)][unicode(self.facade.observed)][
            unicode(self.host)]
        self.bone = bone
        tex = self.host.closet.get_img(bone.wallpaper).texture
        self.size = tex.size
        self.wallpaper = Image(
            texture=tex,
            pos=self.pos,
            size=self.size)
        self.add_widget(self.wallpaper)
        self.arrowlayout = BoardLayout(pos=self.pos, size=self.size)
        self.add_widget(self.arrowlayout)
        self.spotlayout = BoardLayout(pos=self.pos, size=self.size)
        self.add_widget(self.spotlayout)
        self.pawnlayout = BoardLayout(pos=self.pos, size=self.size)
        self.add_widget(self.pawnlayout)
        self.bind(pos=repos_all)
        if bone.observer not in self.facade.closet.board_d:
            self.facade.closet.board_d[bone.observer] = {}
        if bone.observed not in self.facade.closet.board_d[bone.observer]:
            self.facade.closet.board_d[bone.observer][bone.observed] = {}
        self.facade.closet.board_d[bone.observer][bone.observed][
            bone.host] = self
        # Regardless of what the facade *shows*, create spots, pawns,
        # and portals for everything in the host, just in case I need
        # to show them.
        for spotbone in self.facade.closet.skeleton[u"spot"].iterbones():
            if (
                    spotbone.host == bone.host and
                    spotbone.place not in self.spotdict):
                char = self.facade.closet.get_character(spotbone.host)
                place = char.get_place(spotbone.place)
                self.spotdict[spotbone.place] = Spot(board=self, place=place)
        for portbone in self.facade.closet.skeleton[u"portal"][
                bone.observed].iterbones():
            if portbone.host == bone.host:
                port = self.facade.observed.get_portal(portbone.name)
                self.arrowdict[portbone.name] = Arrow(
                    board=self, portal=port)
        for pawnbone in self.facade.closet.skeleton[u"pawn"].iterbones():
            if (
                    pawnbone.host == bone.host and
                    pawnbone.thing not in self.pawndict):
                char = self.facade.closet.get_character(bone.observed)
                try:
                    thing = char.get_thing(pawnbone.thing)
                except KeyError:
                    thing = char.make_thing(pawnbone.thing)
                self.pawndict[pawnbone.thing] = Pawn(board=self, thing=thing)
        for arrow in self.arrowdict.itervalues():
            self.arrowlayout.add_widget(arrow)
        for spot in self.spotdict.itervalues():
            self.spotlayout.add_widget(spot)
        for pawn in self.pawndict.itervalues():
            self.pawnlayout.add_widget(pawn)

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
