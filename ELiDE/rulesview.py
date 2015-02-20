# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from inspect import getsource
from kivy.logger import Logger
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.properties import (
    ListProperty,
    ObjectProperty,
    OptionProperty
)
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.listview import ListView, ListItemButton

from .card import DeckScrollView
Factory.register('DeckScrollView', cls=DeckScrollView)


class RulesView(FloatLayout):
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rule = ObjectProperty()
    triggers_data = ListProperty()
    prereqs_data = ListProperty()
    actions_data = ListProperty()
    rule_triggers_data = ListProperty()
    rule_prereqs_data = ListProperty()
    rule_actions_data = ListProperty()
    inserting = OptionProperty(
        'none', options=['trigger', 'prereq' 'action']
    )

    def on_triggers_data(self, *args):
        Logger.debug(
            "RulesView: got triggers_data {}".format(
                self.triggers_data
            )
        )

    def on_prereqs_data(self, *args):
        Logger.debug(
            "RulesView: got prereqs_data {}".format(
                self.prereqs_data
            )
        )

    def on_actions_data(self, *args):
        Logger.debug(
            "RulesView: got actions_data {}".format(
                self.actions_data
            )
        )

    def on_rule_triggers_data(self, *args):
        Logger.debug(
            "RulesView: got rule_triggers_data {}".format(
                self.rule_triggers_data
            )
        )

    def on_rule_prereqs_data(self, *args):
        Logger.debug(
            "RulesView: got rule_prereqs_data {}".format(
                self.rule_prereqs_data
            )
        )

    def on_rule_actions_data(self, *args):
        Logger.debug(
            "RulesView: got rule_actions_data {}".format(
                self.rule_actions_data
            )
        )

    def on_touch_move(self, touch):
        if 'card' in touch.ud:
            card = touch.ud['card']
            if self.inserting != card.ud['type']:
                self.inserting = card.ud['type']
        else:
            if self.inserting != 'none':
                self.inserting = 'none'
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.inserting != 'none':
            self.inserting = 'none'
        return super().on_touch_up(touch)

    def on_engine(self, *args):
        if self.engine is None:
            return
        self.triggers_data = list(
            self.engine.trigger.db.func_table_name_plaincode('trigger')
        )
        self.prereqs_data = list(
            self.engine.prereq.db.func_table_name_plaincode('prereq')
        )
        self.actions_data = list(
            self.engine.action.db.func_table_name_plaincode('action')
        )

    def on_rule(self, *args):
        if self.rule is None:
            return
        self.rule_triggers_data = [
            (trigger.__name__, getsource(trigger))
            for trigger in self.rule.triggers
        ]
        self.rule_prereqs_data = [
            (prereq.__name__, getsource(prereq))
            for prereq in self.rule.prereqs
        ]
        self.rule_actions_data = [
            (action.__name__, getsource(action))
            for action in self.rule.actions
        ]


class RulesList(ListView):
    rulebook = ObjectProperty()

    def __init__(self, **kwargs):
        if 'adapter' not in kwargs:
            kwargs['adapter'] = ListAdapter(
                data=[],
                selection_mode='single',
                cls=ListItemButton,
                args_converter=lambda i, rule: {
                    'text': rule.name,
                    'on_press': lambda inst:
                    self.parent.setter('rule')(rule)
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


kv = """
<RulesView>:
    rule: list.adapter.selection[0] if list.adapter and list.adapter.selection else None
    BoxLayout:
        RulesList:
            id: list
            rulebook: root.rulebook
        BoxLayout:
            id: triggers
            orientation: 'vertical'
            Label:
                text: 'Triggers'
                size_hint_y: None
                height: self.texture_size[1]
            DeckScrollView:
                id: trigdeck
                data: root.rule_triggers_data
                insertable: root.inserting == 'trigger'
                deletable: True
            DeckScrollView:
                id: trigcoll
                data: root.triggers_data
        BoxLayout:
            id: prereqs
            orientation: 'vertical'
            Label:
                text: 'Prereqs'
                size_hint_y: None
                height: self.texture_size[1]
            DeckScrollView:
                id: preqdeck
                data: root.rule_prereqs_data
                insertable: root.inserting == 'prereq'
                deletable: True
            DeckScrollView:
                id: preqcoll
                data: root.prereqs_data
        BoxLayout:
            id: actions
            orientation: 'vertical'
            Label:
                text: 'Actions'
                size_hint_y: None
                height: self.texture_size[1]
            DeckScrollView:
                id: actdeck
                data: root.rule_actions_data
                insertable: root.inserting == 'action'
                deletable: True
            DeckScrollView:
                id: actcoll
                data: root.actions_data
"""
Builder.load_string(kv)
