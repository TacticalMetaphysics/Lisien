# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.properties import ObjectProperty
from kivy.uix import TextInput


class Renamer(TextInput):
    referent = ObjectProperty()

    def on_text_validate(self, *args):
        pass
