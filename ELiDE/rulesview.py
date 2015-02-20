# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.clock import Clock
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
    rule_triggers_data = ListProperty()
    rule_prereqs_data = ListProperty()
    rule_actions_data = ListProperty()
    inserting = OptionProperty(
        'none', options=['trigger', 'prereq' 'action']
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


kv = """
<RulesView>:
    rule: list.selection[0] if list.selection else None
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
                size_hint_y: None
                height: 200 * len(root.rule_triggers_data)
            DeckScrollView:
                id: trigcoll
                data: root.triggers_data
                size_hint_y: None
                height: 200 * len(root.triggers_data)
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
                size_hint_y: None
                height: 200 * len(root.rule_prereqs_data)
            DeckScrollView:
                id: preqcoll
                data: root.prereqs_data
                size_hint_y: None
                height: 200 * len(root.rule_prereqs_data)
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
                size_hint_y: None
                height: 200 * len(root.rule_actions_data)
            DeckScrollView:
                id: actcoll
                data: root.actions_data
                size_hint_y: None
                height: 200 * len(root.actions_data)
"""
Builder.load_string(kv)
