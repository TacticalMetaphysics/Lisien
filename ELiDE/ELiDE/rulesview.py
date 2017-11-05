# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com
from collections import OrderedDict
from inspect import signature

from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import AliasProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.screenmanager import Screen

from .card import Card, DeckBuilderView, DeckBuilderScrollBar
from .util import trigger


# TODO:
# 1. Make it more obvious whose rules you are editing
# 2. Allow re-ordering of the rules


dbg = Logger.debug


# How do these get instantiated?
class RuleButton(ToggleButton, RecycleDataViewBehavior):
    rulesview = ObjectProperty()
    ruleslist = ObjectProperty()
    rule = ObjectProperty()

    def on_state(self, *args):
        if self.state == 'down':
            self.rulesview.rule = self.rule
            for button in self.ruleslist.children[0].children:
                if button != self:
                    button.state = 'normal'


class RulesList(RecycleView):
    rulebook = ObjectProperty()
    rulesview = ObjectProperty()

    def __init__(self, **kwargs):
        self.bind(rulebook=self._trigger_redata)
        super().__init__(**kwargs)

    def redata(self, *args):
        self.data = [
            {'rulesview': self.rulesview, 'rule': rule, 'index': i, 'ruleslist': self}
            for i, rule in enumerate(self.rulebook)
        ]
    _trigger_redata = trigger(redata)


class RulesView(FloatLayout):
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rule = ObjectProperty(allownone=True)

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
            if node in {
                    'character', 'avatar', 'character_thing',
                    'character_place', 'character_node', 'character_portal'
            }:
                return "Character: {}, rulebook: {}".format(char, node)
            elif node in character.thing:
                return "Character: {}, Thing: {}".format(*rn)
            elif node in character.place:
                return "Character: {}, Place: {}".format(*rn)
            else:
                raise KeyError("Node {} not present in character".format(node))
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

        for functyp in 'trigger', 'prereq', 'action':
            tab = TabbedPanelItem(text=functyp.capitalize())
            setattr(self, '_{}_tab'.format(functyp), tab)
            self._tabs.add_widget(getattr(self, '_{}_tab'.format(functyp)))
            builder = DeckBuilderView(**deck_builder_kwargs)
            setattr(self, '_{}_builder'.format(functyp), builder)
            builder.bind(decks=getattr(self, '_trigger_push_{}s'.format(functyp)))
            scroll_left = DeckBuilderScrollBar(
                size_hint_x=0.01,
                pos_hint={'x': 0, 'y': 0},
                deckbuilder=builder,
                deckidx=0,
                scroll_min=0
            )
            setattr(self, '_scroll_left_' + functyp, scroll_left)
            scroll_right = DeckBuilderScrollBar(
                size_hint_x=0.01,
                pos_hint={'right': 1, 'y': 0},
                deckbuilder=builder,
                deckidx=1,
                scroll_min=0
            )
            setattr(self, '_scroll_right_' + functyp, scroll_right)
            layout = FloatLayout()
            setattr(self, '_{}_layout'.format(functyp), layout)
            tab.add_widget(layout)
            layout.add_widget(builder)
            layout.add_widget(scroll_left)
            layout.add_widget(scroll_right)
            layout.add_widget(
                Label(
                    text='Used',
                    pos_hint={'center_x': 0.1, 'center_y': 0.98},
                    size_hint=(None, None)
                )
            )
            layout.add_widget(
                Label(
                    text='Unused',
                    pos_hint={'center_x': 0.5, 'center_y': 0.98},
                    size_hint=(None, None)
                )
            )
            self.bind(rule=getattr(self, '_trigger_pull_{}s'.format(functyp)))

    def redata(self, *args):
        self._list.redata()

    def _pull_functions(self, what):
        allfuncs = list(map(self._inspect_func, getattr(self.engine, what)._cache.items()))
        rulefuncnames = getattr(self.rule, what+'s')
        unused = [
            Card(
                ud={
                    'type': what,
                    'funcname': name,
                    'signature': sig
                },
                headline_text=name,
                show_art=False,
                midline_text=what.capitalize(),
                text=source
            )
            for (name, source, sig) in allfuncs if name not in rulefuncnames
        ]
        used = [
            Card(
                ud={
                    'type': what,
                    'funcname': name,
                },
                headline_text=name,
                show_art=False,
                midline_text=what.capitalize(),
                text=str(getattr(getattr(self.engine, what), name))
            )
            for name in rulefuncnames
        ]
        return used, unused

    def pull_triggers(self, *args):
        self._trigger_builder.decks = self._pull_functions('trigger')
    _trigger_pull_triggers = trigger(pull_triggers)

    def pull_prereqs(self, *args):
        self._prereq_builder.decks = self._pull_functions('prereq')
    _trigger_pull_prereqs = trigger(pull_prereqs)

    def pull_actions(self, *args):
        self._action_builder.decks = self._pull_functions('action')
    _trigger_pull_actions = trigger(pull_actions)

    def _inspect_func(self, namesrc):
        (name, src) = namesrc
        glbls = {}
        lcls = {}
        exec(src, glbls, lcls)
        assert name in lcls
        func = lcls[name]
        return name, src, signature(func)

    def update_builders(self, *args):
        for attrn in '_trigger_builder', '_prereq_builder', '_action_builder':
            if not hasattr(self, attrn):
                dbg('RulesView: no {}'.format(attrn))
                Clock.schedule_once(self.update_builders, 0)
                return
        self._trigger_builder.clear_widgets()
        self._prereq_builder.clear_widgets()
        self._action_builder.clear_widgets()
        if self.rule is None:
            dbg('RulesView: no rule')
            return
        if hasattr(self, '_list'):
            self._list.redata()
        self.pull_triggers()
        self.pull_prereqs()
        self.pull_actions()
    _trigger_update_builders = trigger(update_builders)

    def _upd_unused(self, what):
        """Make sure to have exactly one copy of every valid function in the
        "unused" pile on the right.

        Doesn't read from the database.

        """
        builder = getattr(self, '_{}_builder'.format(what))
        updtrig = getattr(self, '_trigger_upd_unused_{}s'.format(what))
        builder.unbind(decks=updtrig)
        funcs = OrderedDict()
        cards = list(self._action_builder.decks[1])
        cards.reverse()
        for card in cards:
            funcs[card.ud['funcname']] = card
        for card in self._action_builder.decks[0]:
            if card.ud['funcname'] not in funcs:
                funcs[card.ud['funcname']] = card.copy()
        unused = list(funcs.values())
        unused.reverse()
        builder.decks[1] = unused
        builder.bind(decks=updtrig)

    def upd_unused_actions(self, *args):
        self._upd_unused('action')
    _trigger_upd_unused_actions = trigger(upd_unused_actions)

    def upd_unused_triggers(self, *args):
        self._upd_unused('trigger')
    _trigger_upd_unused_triggers = trigger(upd_unused_triggers)

    def upd_unused_prereqs(self, *args):
        self._upd_unused('prereq')
    _trigger_upd_unused_prereqs = trigger(upd_unused_prereqs)

    def _push_funcs(self, what):
        funcs = [
            card.ud['funcname'] for card in
            getattr(self, '_{}_builder'.format(what)).decks[0]
        ]
        funlist = getattr(self.rule, what+'s')
        if funlist != funcs:
            setattr(self.rule, what+'s', funcs)

    def push_actions(self, *args):
        self._push_funcs('action')
    _trigger_push_actions = trigger(push_actions)

    def push_prereqs(self, *args):
        self._push_funcs('prereq')
    _trigger_push_prereqs = trigger(push_prereqs)

    def push_triggers(self, att, *args):
        self._push_funcs('trigger')
    _trigger_push_triggers = trigger(push_triggers)


class RulesScreen(Screen):
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rulesview = ObjectProperty()
    new_rule_name = StringProperty()
    toggle = ObjectProperty()

    def new_rule(self, *args):
        if self.new_rule_name in self.engine.rule:
            # TODO: feedback to say you already have such a rule
            return
        new_rule = self.engine.rule.new_empty(self.new_rule_name)
        assert(new_rule is not None)
        self.rulebook.append(new_rule)
        self.rulesview.redata()
        self.ids.rulesview.rule = new_rule
        self.ids.rulename.text = ''


Builder.load_string("""
<RuleButton>:
    text: self.rule.name if self.rule else ''
<RulesList>:
    viewclass: 'RuleButton'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        size_hint_y: None
        orientation: 'vertical'
<RulesScreen>:
    name: 'rules'
    new_rule_name: rulename.text
    rulesview: rulesview
    BoxLayout:
        orientation: 'vertical'
        RulesView:
            id: rulesview
            engine: root.engine
            rulebook: root.rulebook
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
                on_press: root.toggle()
""")
