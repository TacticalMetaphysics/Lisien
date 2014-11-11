# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Widget to display information on the present state of the selected
entity.

By default it's only big enough to hold the entity's name and a few
icons. It's got a splitter on top that lets you make it bigger.

"""

from kivy.uix.splitter import Splitter
from kivy.uix.treeview import TreeView, TreeNode
from kivy.uix.textinput import TextInput
from kivy.properties import (
    ListProperty,
    OptionProperty,
    StringProperty
)
from kivy.clock import Clock


class DataTreeNode(TreeNode, TextInput):
    """TreeNode that is a TextInput which, when its ``update`` method is
    called, looks into the LiSE engine to update its current value to
    whatever it's about."""
    entity_type = OptionProperty(
        'Thing',
        options=['Thing', 'Place', 'Portal', 'Character']
    )
    name = StringProperty()
    
