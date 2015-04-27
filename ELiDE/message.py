# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""A widget to display some styled text to the player, containing
"hyperlinks" that have some specific effect on the world state when
clicked.

Unlike most ELiDE widgets, this one is meant to always be under the
direct control of the developer, and does not itself represent
anything about the state of the simulation.

"""
from kivy.uix.label import Label


class Message(Label):
    pass
