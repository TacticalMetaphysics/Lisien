# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Graphical view upon a Facade.

"""
from kivy.clock import Clock
from kivy.uix.listview import (
    ListView,
    CompositeListItem,
    ListItemButton,
    ListItemLabel
)
from kivy.adapters.dictadapter import DictAdapter
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty,
    StringProperty,
    DictProperty
)


class CharSheet(Widget):
    engine = ObjectProperty()
    character = ObjectProperty()
    data = DictProperty()

    def __init__(self, **kwargs):
        self._trigger_redata = Clock.create_trigger(self._redata)
        super().__init__(**kwargs)
        self.adapter = DictAdapter(
            data=self.data,
            cls=CompositeListItem,
            sorted_keys=sorted(self.data.keys()),
            args_converter=lambda i, t: {
                'text': t[1],
                'size_hint_y': None,
                'height': 30,
                'representing_cls': ListItemLabel,
                'cls_dicts': [
                    {
                        'cls': ListItemButton,
                        'kwargs': {'text': t[0]}
                    }, {
                        'cls': ListItemLabel,
                        'kwargs': {'text': t[1]}
                    }
                ]
            }
        )
        self.view = ListView(adapter=self.adapter)
        self.bind(
            pos=self.view.setter('pos'),
            size=self.view.setter('size')
        )
        self.add_widget(self.view)
        # Do a full update any time anything about my character
        # changes, and whenever the game-time changes.
        #
        # Terribly inefficient; optimize please.
        for cat in ('place', 'portal', 'thing'):
            for event in ('on_set_{}', 'on_del_{}', 'on_set_{}_item', 'on_del_{}_item'):
                getattr(self.character, event.format(cat)).append(self._trigger_redata)
        def do_trigger_redata():
            self._trigger_redata()
        do_trigger_redata.__name__ = self.character.name + "_trigger_redata"
        self.engine.on_time(do_trigger_redata)

    def _items(self):
        return self.character.items()

    def _redata(self, *args):
        d = {}
        for (k, v) in self._items():
            d[k] = (k, v)
        self.adapter.data = d
        self.adapter.sorted_keys = sorted(d.keys())
        self.view._trigger_populate()


class ThingSheet(CharSheet):
    thing = StringProperty()

    def _items(self):
        return self.character.thing[self.thing].items()


class PlaceSheet(CharSheet):
    place = StringProperty()

    def _items(self):
        return self.character.place[self.place].items()


class PortalSheet(CharSheet):
    origin = StringProperty()
    destination = StringProperty()

    def _items(self):
        return self.character.portal[self.origin][self.destination].items()
