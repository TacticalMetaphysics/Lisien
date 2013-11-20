# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)


class EditButton(ToggleButton):
    def collide_point(self, x, y):
        return super(EditButton, self).collide_point(*self.to_local(x, y))


class ItemView(RelativeLayout):
    item_type = NumericProperty()
    keys = ListProperty()
    bg_color_inactive = ListProperty()
    text_color_inactive = ListProperty()
    bg_color_active = ListProperty()
    text_color_active = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    character = ObjectProperty()

    @property
    def connector(self):
        return self.character.closet.kivy_connector

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
