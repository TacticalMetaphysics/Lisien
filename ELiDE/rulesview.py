# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from collections import OrderedDict

from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import AliasProperty, ObjectProperty, StringProperty
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.listview import ListView, ListItemButton
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from .card import Card, DeckBuilderView, DeckBuilderScrollBar


# TODO:
# 1. Make it more obvious whose rules you are editing
# 2. Allow re-ordering of the rules


dbg = Logger.debug


class RuleButton(ListItemButton):
    rule = ObjectProperty()


class RulesList(ListView):
    rulebook = ObjectProperty()
    rulesview = ObjectProperty()

    def __init__(self, **kwargs):
        if 'adapter' not in kwargs:
            kwargs['adapter'] = ListAdapter(
                data=[],
                selection_mode='single',
                allow_empty_selection=False,
                cls=RuleButton,
                args_converter=lambda i, rule: {
                    'size_hint_y': None,
                    'height': 30,
                    'text': rule.name,
                    'rule': rule
                }
            )
        super().__init__(**kwargs)

    def on_adapter(self, *args):
        self.adapter.bind(
            on_selection_change=lambda inst:
            self.set_rule(self.adapter.selection[0].rule)
        )

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

    def _get_headline_text(self):
        # This shows the entity whose rules you're editing if you
        # haven't assigned a different rulebook from usual. Otherwise
        # it shows the name of the rulebook. I'd like it to show
        # *both*.
        if self.rulebook is None:
            return ''
        rn = self.rulebook.name
        if not isinstance(rn, tuple):
            return str(rn)
        if len(rn) == 2:
            (char, node) = rn
            character = self.engine.character[char]
            if node in character.thing:
                return "Character: {}, Thing: {}".format(*rn)
            else:
                return "Character: {}, Place: {}".format(*rn)
        elif len(rn) == 3:
            return "Character: {}, Portal: {}->{}".format(*rn)
        else:
            return str(rn)
    headline_text = AliasProperty(
        _get_headline_text,
        lambda self, v: None,
        bind=('rulebook',)
    )

    def __init__(self, **kwargs):
        self._trigger_upd_rule_triggers = Clock.create_trigger(
            self.upd_rule_triggers
        )
        self._trigger_upd_unused_triggers = Clock.create_trigger(
            self.upd_unused_triggers
        )
        self._trigger_upd_rule_prereqs = Clock.create_trigger(
            self.upd_rule_prereqs
        )
        self._trigger_upd_unused_prereqs = Clock.create_trigger(
            self.upd_unused_prereqs
        )
        self._trigger_upd_rule_actions = Clock.create_trigger(
            self.upd_rule_actions
        )
        self._trigger_upd_unused_actions = Clock.create_trigger(
            self.upd_unused_actions
        )
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
        self._headline = Label(
            size_hint_y=0.05,
            pos_hint={
                'top': 1,
                'center_x': 0.5
            },
            text=self.headline_text
        )
        self.bind(headline_text=self._headline.setter('text'))
        self.add_widget(self._headline)
        self._box = BoxLayout(
            size_hint_y=0.95,
            pos_hint={'top': 0.95}
        )
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
            deckidx=0,
            scroll_min=0
        )
        self._scroll_right_action = DeckBuilderScrollBar(
            size_hint_x=0.01,
            pos_hint={'right': 1, 'y': 0},
            deckbuilder=self._action_builder,
            deckidx=1,
            scroll_min=0
        )
        self._action_layout = FloatLayout()
        self._action_tab.add_widget(self._action_layout)
        self._action_layout.add_widget(self._action_builder)
        self._action_layout.add_widget(self._scroll_left_action)
        self._action_layout.add_widget(self._scroll_right_action)
        self._action_layout.add_widget(
            Label(
                text='Used',
                pos_hint={'center_x': 0.1, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )
        self._action_layout.add_widget(
            Label(
                text='Unused',
                pos_hint={'center_x': 0.5, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )

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
        self._trigger_layout.add_widget(
            Label(
                text='Used',
                pos_hint={'center_x': 0.1, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )
        self._trigger_layout.add_widget(
            Label(
                text='Unused',
                pos_hint={'center_x': 0.5, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )
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
        self._prereq_layout.add_widget(
            Label(
                text='Used',
                pos_hint={'center_x': 0.1, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )
        self._prereq_layout.add_widget(
            Label(
                text='Unused',
                pos_hint={'center_x': 0.5, 'center_y': 0.98},
                size_hint=(None, None)
            )
        )

    def on_rule(self, *args):
        def getname(o):
            return o if isinstance(o, str) else o.__name__

        if self.rule is None:
            dbg('RulesView: no rule')
            return
        for attrn in '_trigger_builder', '_prereq_builder', '_action_builder':
            if not hasattr(self, attrn):
                dbg('RulesView: no {}'.format(attrn))
                Clock.schedule_once(self.on_rule, 0)
                return
        self._trigger_builder.clear_widgets()
        self._prereq_builder.clear_widgets()
        self._action_builder.clear_widgets()
        unused_triggers = [
            Card(
                ud={
                    'type': 'trigger',
                    'funcname': name
                },
                headline_text=name,
                show_art=False,
                midline_text='Trigger',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.trigger.iterplain()
        ]
        used_triggers = [
            Card(
                ud={
                    'type': 'trigger',
                    'funcname': getname(trigger),
                },
                headline_text=getname(trigger),
                show_art=False,
                midline_text='Trigger',
                text=self.engine.trigger.plain(getname(trigger)),
                show_footer=False
            )
            for trigger in self.rule.triggers
        ]
        self._trigger_builder.unbind(decks=self._trigger_upd_unused_triggers)
        self._trigger_builder.unbind(decks=self._trigger_upd_rule_triggers)
        self._trigger_builder.decks = [used_triggers, unused_triggers]
        self._trigger_builder.bind(decks=self._trigger_upd_rule_triggers)
        self._trigger_builder.bind(decks=self._trigger_upd_unused_triggers)
        unused_prereqs = [
            Card(
                ud={
                    'type': 'prereq',
                    'funcname': name
                },
                headline_text=name,
                show_art=False,
                midline_text='Prereq',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.prereq.iterplain()
        ]
        used_prereqs = [
            Card(
                ud={
                    'type': 'prereq',
                    'funcname': getname(prereq)
                },
                headline_text=getname(prereq),
                show_art=False,
                midline_text='Prereq',
                text=self.engine.prereq.plain(getname(prereq)),
                show_footer=False
            )
            for prereq in self.rule.prereqs
        ]
        self._prereq_builder.unbind(decks=self._trigger_upd_unused_prereqs)
        self._prereq_builder.unbind(decks=self._trigger_upd_rule_prereqs)
        self._prereq_builder.decks = [used_prereqs, unused_prereqs]
        self._prereq_builder.bind(decks=self._trigger_upd_rule_prereqs)
        self._prereq_builder.bind(decks=self._trigger_upd_unused_prereqs)
        unused_actions = [
            Card(
                ud={
                    'type': 'action',
                    'funcname': name
                },
                headline_text=name,
                show_art=False,
                midline_text='Action',
                text=source,
                show_footer=False
            )
            for (name, source) in
            self.engine.action.iterplain()
        ]
        used_actions = [
            Card(
                ud={
                    'type': 'action',
                    'funcname': getname(action)
                },
                headline_text=getname(action),
                show_art=False,
                midline_text='Action',
                text=self.engine.action.plain(getname(action)),
                show_footer=False
            )
            for action in self.rule.actions
        ]
        self._action_builder.unbind(decks=self._trigger_upd_rule_actions)
        self._action_builder.unbind(decks=self._trigger_upd_unused_actions)
        self._action_builder.decks = [used_actions, unused_actions]
        self._action_builder.bind(decks=self._trigger_upd_rule_actions)
        self._action_builder.bind(decks=self._trigger_upd_unused_actions)

    def upd_rule_actions(self, *args):
        actions = [
            card.ud['funcname'] for card in
            self._action_builder.decks[0]
        ]
        if self.rule.actions != actions:
            self.rule.actions = actions

    def upd_unused_actions(self, *args):
        """Make sure to have exactly one copy of every valid action in the
        "unused" pile on the right.

        Doesn't read from the database.

        """
        self._action_builder.unbind(decks=self._trigger_upd_unused_actions)
        actions = OrderedDict()
        cards = list(self._action_builder.decks[1])
        cards.reverse()
        for card in cards:
            actions[card.ud['funcname']] = card
        for card in self._action_builder.decks[0]:
            if card.ud['funcname'] not in actions:
                actions[card.ud['funcname']] = card.copy()
        unused = list(actions.values())
        unused.reverse()
        self._action_builder.decks[1] = unused
        self._action_builder.bind(decks=self._trigger_upd_unused_actions)

    def upd_rule_prereqs(self, *args):
        prereqs = [
            card.ud['funcname'] for card in
            self._prereq_builder.decks[0]
        ]
        if self.rule.prereqs != prereqs:
            self.rule.prereqs = prereqs

    def upd_unused_prereqs(self, *args):
        """Make sure to have exactly one copy of every valid prereq in the
        "unused" pile on the right.

        Doesn't read from the database.

        """
        self._prereq_builder.unbind(decks=self._trigger_upd_unused_prereqs)
        prereqs = OrderedDict()
        cards = list(self._prereq_builder.decks[1])
        cards.reverse()
        for card in cards:
            prereqs[card.ud['funcname']] = card
        for card in self._prereq_builder.decks[0]:
            if card.ud['funcname'] not in prereqs:
                prereqs[card.ud['funcname']] = card.copy()
        unused = list(prereqs.values())
        unused.reverse()
        self._prereq_builder.decks[1] = unused
        self._prereq_builder.bind(decks=self._trigger_upd_unused_prereqs)

    def upd_rule_triggers(self, att, *args):
        triggers = [
            card.ud['funcname'] for card in
            self._trigger_builder.decks[0]
        ]
        if self.rule.triggers != triggers:
            self.rule.triggers = triggers

    def upd_unused_triggers(self, *args):
        """Make sure to have exactly one copy of every valid prereq in the
        "unused" pile on the right.

        Doesn't read from the database.

        """
        self._trigger_builder.unbind(decks=self._trigger_upd_unused_triggers)
        triggers = OrderedDict()
        cards = list(self._trigger_builder.decks[1])
        cards.reverse()
        for card in cards:
            triggers[card.ud['funcname']] = card
        for card in self._trigger_builder.decks[0]:
            if card.ud['funcname'] not in triggers:
                triggers[card.ud['funcname']] = card.copy()
        unused = list(triggers.values())
        unused.reverse()
        self._trigger_builder.decks[1] = unused
        self._trigger_builder.bind(decks=self._trigger_upd_unused_triggers)


class RulesBox(BoxLayout):
    engine = ObjectProperty()
    rulesview = ObjectProperty()
    new_rule_name = StringProperty()
    new_rule = ObjectProperty()
    toggle_rules_view = ObjectProperty()


Builder.load_string("""
<RulesBox>:
    orientation: 'vertical'
    new_rule_name: rulename.text
    rulesview: rulesview
    RulesView:
        id: rulesview
        engine: root.engine
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.05
        TextInput:
            id: rulename
            hint_text: 'New rule name'
            write_tab: False
        Button:
            text: '+'
            on_press: root.new_rule()
        Button:
            text: 'Close'
            on_press: root.toggle_rules_view()
""")
