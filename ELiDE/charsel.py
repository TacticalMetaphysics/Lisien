# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.properties import ObjectProperty
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.listview import ListView, ListItemButton


"""Widget that lets you switch characters or create a new one."""


class CharListView(ListView):
    layout = ObjectProperty()
    set_char = ObjectProperty()

    def __init__(self, **kwargs):
        if 'adapter' not in kwargs:
            kwargs['adapter'] = ListAdapter(
                data=[],
                selection_mode='single',
                allow_empty_selection=True,
                cls=ListItemButton,
                args_converter=lambda i, char: {
                    'size_hint_y': None,
                    'height': 30,
                    'text': char
                }
            )
        super().__init__(**kwargs)

    def on_adapter(self, *args):
        def selchange(*args):
            if (
                    hasattr(self.layout, '_popover') and
                    self.layout.character_name !=
                    self.adapter.selection[0].text
            ):
                self.set_char(
                    self.layout.engine.character[
                        self.adapter.selection[0].text
                    ]
                )
        self.adapter.bind(
            on_selection_change=selchange
        )
