# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty
)
from kivy.graphics import (
    InstructionGroup,
    Color,
    Line
)
from ELiDE.kivygarden.texturestack import ImageStack


class PawnSpot(ImageStack):
    board = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    linecolor = ListProperty()
    use_boardspace = True

    def on_linecolor(self, *args):
        if hasattr(self, 'color'):
            self.color.rgba = self.linecolor
            return

        def upd_box_points(*args):
            self.box.points = [
                self.x, self.y,
                self.right, self.y,
                self.right, self.top,
                self.x, self.top,
                self.x, self.y
            ]
        self.grp = InstructionGroup()
        self.color = Color(*self.linecolor)
        self.grp.add(self.color)
        self.box = Line()
        upd_box_points()
        self.bind(
            pos=upd_box_points,
            size=upd_box_points
        )
        self.grp.add(self.box)
        self.grp.add(Color(1., 1., 1.))
        self.canvas.after.add(self.grp)
