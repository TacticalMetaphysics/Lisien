# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Widget to enable browsing rules and the functions that make them."""
from functools import partial
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.adapters import ListAdapter
from kivy.uix.listview import ListView, ListItemButton
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, DictProperty


class RulesView(Widget):
    subject = ObjectProperty()
    func_adapter = DictProperty({})

    def get_func_data(self, store):
        return list(
            self.engine.function.db.func_table_name_plaincode(store)
        )

    def get_func_adapter(self, store):
        if store not in self.func_adapter:
            self.func_adapter[store] = ListAdapter(
                data=self.get_func_data(store),
                cls=ListItemButton,
                args_converter=lambda i, (name, code): {
                    'text': name,
                    'on_press': lambda inst:
                    self.show_func_editor(
                        store,
                        name,
                        code
                    )
                },
                selection_mode='single',
                allow_empty_selection=True
            )
        return self.func_adapter[store]

    def refresh_func_adapter(self, store, *args):
        self.get_func_adapter(store).data = self.get_func_data(store)

    def on_engine(self):
        if self.engine is None:
            return
        self._func_view_trigger = ListView(
            adapter=self.get_func_adapter('trigger')
        )
        self._trigger_refresh_trigger = Clock.create_trigger(
            partial(self.refresh_func_adapter, 'trigger')
        )
        self._func_view_prereq = ListView(
            adapter=self.get_func_adapter('prereq')
        )
        self._trigger_refresh_prereq = Clock.create_trigger(
            partial(self.refresh_func_adapter, 'prereq')
        )
        self._func_view_action = ListView(
            adapter=self.get_func_adapter('action')
        )
        self._trigger_refresh_action = Clock.create_trigger(
            partial(self.refresh_func_adapter, 'action')
        )
