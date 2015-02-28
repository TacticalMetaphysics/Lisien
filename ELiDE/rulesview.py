# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from inspect import getsource
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.listview import ListView, ListItemButton
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from .card import Card, DeckBuilderView, DeckBuilderScrollBar
Factory.register('Card', cls=Card)
Factory.register('DeckBuilderView', cls=DeckBuilderView)
Factory.register('DeckBuilderScrollBar', cls=DeckBuilderScrollBar)


dbg = Logger.debug


class RulesList(ListView):
    rulebook = ObjectProperty()
    rulesview = ObjectProperty()

    def __init__(self, **kwargs):
        if 'adapter' not in kwargs:
            kwargs['adapter'] = ListAdapter(
                data=[],
                selection_mode='single',
                cls=ListItemButton,
                args_converter=lambda i, rule: {
                    'size_hint_y': None,
                    'height': 30,
                    'text': rule.name,
                    'on_press': lambda inst:
                    self.set_rule(rule)
                }
            )
        super().__init__(**kwargs)

    def on_rulebook(self, *args):
        if self.rulebook is None:
            return
        self.adapter.data = list(self.rulebook)

        @self.rulebook.listener
        def upd_adapter_data(rb):
            self.adapter.data = list(rb)

    def set_rule(self, rule):
        self.rulesview.rule = rule


class RulesView(FloatLayout):
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rule = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if self.canvas is None:
            Clock.schedule_once(self.finalize, 0)
            return

        deck_builder_kwargs = {
            'pos_hint': {'x': 0, 'y': 0},
            'starting_pos_hint': {'x': 0.05, 'top': 0.95},
            'card_size_hint': (0.3, 0.4),
            'card_hint_step': (0, -0.1),
            'deck_x_hint_step': 0.4
        }
        self._box = BoxLayout()
        self.add_widget(self._box)
        self._list = RulesList(
            rulebook=self.rulebook,
            rulesview=self,
            size_hint_x=0.33
        )
        self.bind(rulebook=self._list.setter('rulebook'))
        self._box.add_widget(self._list)
        self._tabs = TabbedPanel(
            do_default_tab=False
        )
        self._box.add_widget(self._tabs)

        self._action_tab = TabbedPanelItem(text='Actions')
        self._tabs.add_widget(self._action_tab)
        self._action_builder = DeckBuilderView(**deck_builder_kwargs)
        self._scroll_left_action = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'x': 0, 'y': 0},
            deckbuilder=self._action_builder,
            deckidx=0
        )
        self._scroll_right_action = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'right': 1, 'y': 0},
            deckbuilder=self._action_builder,
            deckidx=1
        )
        self._action_layout = FloatLayout()
        self._action_tab.add_widget(self._action_layout)
        self._action_layout.add_widget(self._action_builder)
        self._action_layout.add_widget(self._scroll_left_action)
        self._action_layout.add_widget(self._scroll_right_action)

        self._trigger_tab = TabbedPanelItem(text='Triggers')
        self._tabs.add_widget(self._trigger_tab)
        self._trigger_builder = DeckBuilderView(**deck_builder_kwargs)
        self._scroll_left_trigger = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'x': 0, 'y': 0},
            deckbuilder=self._trigger_builder,
            deckidx=0
        )
        self._scroll_right_trigger = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'right': 1, 'y': 0},
            deckbuilder=self._trigger_builder,
            deckidx=1
        )
        self._trigger_layout = FloatLayout()
        self._trigger_tab.add_widget(self._trigger_layout)
        self._trigger_layout.add_widget(self._trigger_builder)
        self._trigger_layout.add_widget(self._scroll_left_trigger)
        self._trigger_layout.add_widget(self._scroll_right_trigger)

        self._prereq_tab = TabbedPanelItem(text='Prereqs')
        self._tabs.add_widget(self._prereq_tab)
        self._prereq_builder = DeckBuilderView(**deck_builder_kwargs)
        self._scroll_left_prereq = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'x': 0, 'y': 0},
            deckbuilder=self._prereq_builder,
            deckidx=0
        )
        self._scroll_right_prereq = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'right': 1, 'y': 0},
            deckbuilder=self._prereq_builder,
            deckidx=1
        )
        self._prereq_layout = FloatLayout()
        self._prereq_tab.add_widget(self._prereq_layout)
        self._prereq_layout.add_widget(self._prereq_builder)
        self._prereq_layout.add_widget(self._scroll_left_prereq)
        self._prereq_layout.add_widget(self._scroll_right_prereq)

    def on_rule(self, *args):
        if self.rule is None:
            dbg('RulesView: no rule')
            return
        for attrn in '_trigger_builder', '_prereq_builder', '_action_builder':
            if not hasattr(self, attrn):
                dbg('RulesView: no {}'.format(attrn))
                Clock.schedule_once(self.on_rule, 0)
                return
        unused_triggers = [
            Card(
                ud={'type': 'trigger'},
                headline_text=name,
                show_art=False,
                midline_text='Trigger',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.trigger.db.func_table_name_plaincode('trigger')
        ]
        used_triggers = [
            Card(
                ud={'type': 'trigger'},
                headline_text=trigger.__name__,
                show_art=False,
                midline_text='Trigger',
                text=getsource(trigger),
                show_footer=False
            )
            for trigger in self.rule.triggers
        ]
        self._trigger_builder.decks = [unused_triggers, used_triggers]
        unused_prereqs = [
            Card(
                ud={'type': 'prereq'},
                headline_text=name,
                show_art=False,
                midline_text='Prereq',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.prereq.db.func_table_name_plaincode('prereq')
        ]
        used_prereqs = [
            Card(
                ud={'type': 'prereq'},
                headline_text=prereq.__name__,
                show_art=False,
                midline_text='Prereq',
                text=getsource(prereq),
                show_footer=False
            )
            for prereq in self.rule.prereqs
        ]
        self._prereq_builder.decks = [unused_prereqs, used_prereqs]
        unused_actions = [
            Card(
                ud={'type': 'action'},
                headline_text=name,
                show_art=False,
                midline_text='Action',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.action.db.func_table_name_plaincode('action')
        ]
        used_actions = [
            Card(
                ud={'type': 'action'},
                headline_text=action.__name__,
                show_art=False,
                midline_text='Action',
                text=getsource(action),
                show_footer=False
            )
            for action in self.rule.actions
        ]
        self._action_builder.decks = [unused_actions, used_actions]
