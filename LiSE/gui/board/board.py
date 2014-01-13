# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    DictProperty,
    ObjectProperty)
from kivy.logger import Logger
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

    def on_touch_down(self, touch):
        if touch.grab_current in self.children:
            return
        for child in self.children:
            if child.x > touch.x:
                continue
            if child.y > touch.y:
                continue
            if child.right < touch.x:
                continue
            if child.top < touch.y:
                continue
            return child.on_touch_down(touch)


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
    final = BooleanProperty()
    scroll_x = AliasProperty(
        lambda self: self.bone.x,
        lambda self, v: self.host.closet.set_bone(self.bone._replace(x=v)),
        cache=False)
    scroll_y = AliasProperty(
        lambda self: self.bone.x,
        lambda self, v: self.host.closet.set_bone(self.bone._replace(y=v)),
        cache=False)

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

    def __init__(self, **kwargs):
        kwargs['size_hint'] = (None, None)
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
        bone = self.bone
        self.scroll_x = bone.x
        self.scroll_y = bone.y
        tex = self.host.closet.get_img('default_wallpaper').texture
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


class BoardView(ScrollView):
    app = ObjectProperty()
    board = ObjectProperty()

    def on_board(self, *args):
        self.scroll_x = self.board.scroll_x
        self.scroll_y = self.board.scroll_y
        self.add_widget(self.board)

    def on_touch_down(self, touch):
        for preemptor in 'menu', 'charsheet', 'portaling':
            if preemptor in touch.ud:
                self.do_scroll_x = self.do_scroll_y = False
        if self.do_scroll_x:
            self.do_scroll_x = self.do_scroll_y = (
                not self.board.pawnlayout.on_touch_down(touch))
        if self.do_scroll_x:
            self.do_scroll_x = self.do_scroll_y = (
                not self.board.spotlayout.on_touch_down(touch))
        return super(BoardView, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        self.do_scroll_x = self.do_scroll_y = True
        return super(BoardView, self).on_touch_up(touch)
