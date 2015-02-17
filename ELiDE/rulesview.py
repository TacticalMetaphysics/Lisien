# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.lang import Builder
from kivy.properties import (
    BooleanProperty,
    ObjectProperty
)
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.listview import ListView


class RulesView(FloatLayout):
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rule = ObjectProperty()


class RulesList(ListView):
    rulebook = ObjectProperty()
    rule_triggers_insertable = BooleanProperty(False)
    rule_prereqs_insertable = BooleanProperty(False)
    rule_actions_insertable = BooleanProperty(False)


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
                insertable: root.rule_triggers_insertable
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
                insertable: root.rule_prereqs_insertable
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
                insertable: root.rule_actions_insertable
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
