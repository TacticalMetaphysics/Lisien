# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty
)


class EditButton(ToggleButton):
    def collide_point(self, x, y):
        return super(EditButton, self).collide_point(*self.to_local(x, y))


class ItemView(RelativeLayout):
    character = ObjectProperty()
    item_type = NumericProperty()
    keys = ListProperty()
    style = ObjectProperty()
    editing = BooleanProperty(False)

    def add_widget(self, w):
        # make sure the EditButton stays on top
        edbut = None
        if not isinstance(w, EditButton):
            for child in self.children:
                if isinstance(child, EditButton):
                    edbut = child
                    self.remove_widget(edbut)
                    break
        super(ItemView, self).add_widget(w)
        if edbut is not None:
            self.add_widget(edbut)
