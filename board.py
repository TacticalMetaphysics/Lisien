# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.uix.scatterlayout import ScatterLayout
from kivy.properties import ObjectProperty


"""Class for user's view on gameworld, and support functions."""


__metaclass__ = SaveableWidgetMetaclass


class Board(ScatterLayout):
    closet = ObjectProperty()
    dimension = ObjectProperty()

    def __init__(self, **kwargs):
        self.closet = kwargs["closet"]
        self.dimension = kwargs["dimension"]
        ScatterLayout.__init__(self, **kwargs)

    def build(self):
        if (
                "spot_coords" in self.closet.skeleton and
                str(self.dimension) in self.dimension.closet.skeleton[
                    "spot_coords"]):
            for rd in self.dimension.closet.skeleton[
                    "spot_coords"][str(self.dimension)].iterrows():
                place = self.dimension.get_place(rd["place"])
                spot = Spot(board=self, place=place)
                self.add_widget(spot)
        if (
                "pawn_img" in self.closet.skeleton and
                str(self.dimension) in self.dimension.closet.skeleton[
                    "pawn_img"]):
            for rd in self.dimension.closet.skeleton[
                    "pawn_img"][str(self.dimension)].iterrows():
                thing = self.dimension.get_thing(rd["thing"])
                pawn = Pawn(self, thing)
                self.add_widget(pawn)
        for portal in self.dimension.portals:
            arrow = Arrow(board=self, portal=portal)
            self.add_widget(arrow)
