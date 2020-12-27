# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from collections import OrderedDict
from inspect import signature

from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.widget import Widget
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
    """A button to select a rule to edit"""
    rulesview = ObjectProperty()
    ruleslist = ObjectProperty()
    rule = ObjectProperty()

    def on_state(self, *args):
        """If I'm pressed, unpress all other buttons in the ruleslist"""
        # This really ought to be done with the selection behavior
        if self.state == 'down':
            self.rulesview.rule = self.rule
            for button in self.ruleslist.children[0].children:
                if button != self:
                    button.state = 'normal'


class RulesList(RecycleView):
    """A list of rules you might want to edit

    Presented as buttons, which you can click to select one rule at a time.

    """
    rulebook = ObjectProperty()
    rulesview = ObjectProperty()

    def on_rulebook(self, *args):
        """Make sure to update when the rulebook changes"""
        if self.rulebook is None:
            return
        self.rulebook.connect(self._trigger_redata, weak=False)
        self.redata()

    def redata(self, *args):
        """Make my data represent what's in my rulebook right now"""
        if self.rulesview is None:
            Clock.schedule_once(self.redata, 0)
            return
        data = [
            {'rulesview': self.rulesview, 'rule': rule, 'index': i, 'ruleslist': self}
            for i, rule in enumerate(self.rulebook)
        ]
        self.data = data

    def _trigger_redata(self, *args, **kwargs):
        if hasattr(self, '_scheduled_redata'):
            Clock.unschedule(self._scheduled_redata)
        self._scheduled_redata = Clock.schedule_once(self.redata, 0)


class RulesView(Widget):
    """The view to edit a rule

    Presents three tabs, one each for trigger, prereq, and action. Each has a
    deckbuilder in it with a column of used functions and a column of unused actions.

    """
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    entity = ObjectProperty()
    rule = ObjectProperty(allownone=True)

    def on_rule(self, *args):
        """Make sure to update when the rule changes"""
        if self.rule is None:
            return
        self.rule.connect(self._listen_to_rule)

    def _listen_to_rule(self, rule, **kwargs):
        if rule is not self.rule:
            rule.disconnect(self._listen_to_rule)
            return
        if 'triggers' in kwargs:
            self.pull_triggers()
        if 'prereqs' in kwargs:
            self.pull_prereqs()
        if 'actions' in kwargs:
            self.pull_actions()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        """Add my tabs"""
        if not self.canvas:
            Clock.schedule_once(self.finalize, 0)
            return

        deck_builder_kwargs = {
            'pos_hint': {'x': 0, 'y': 0},
            'starting_pos_hint': {'x': 0.05, 'top': 0.95},
            'card_size_hint': (0.3, 0.4),
            'card_hint_step': (0, -0.1),
            'deck_x_hint_step': 0.4
        }

        self._tabs = TabbedPanel(
            size=self.size,
            pos=self.pos,
            do_default_tab=False
        )
        self.bind(
            size=self._tabs.setter('size'),
            pos=self._tabs.setter('pos')
        )
        self.add_widget(self._tabs)

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

    def get_functions_cards(self, what, allfuncs):
        """Return a pair of lists of Card widgets for used and unused functions.

        :param what: a string: 'trigger', 'prereq', or 'action'
        :param allfuncs: a sequence of functions' (name, sourcecode, signature)

        """
        if not self.rule:
            return [], []
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

    def set_functions(self, what, allfuncs):
        """Set the cards in the ``what`` builder to ``allfuncs``

        :param what: a string, 'trigger', 'prereq', or 'action'
        :param allfuncs: a sequence of triples of (name, sourcecode, signature) as taken by my
        ``get_function_cards`` method.

        """
        setattr(getattr(self, '_{}_builder'.format(what)), 'decks', self.get_functions_cards(what, allfuncs))

    def _pull_functions(self, what):
        return self.get_functions_cards(what, list(map(self.inspect_func, getattr(self.engine, what)._cache.items())))

    def pull_triggers(self, *args):
        """Refresh the cards in the trigger builder"""
        self._trigger_builder.decks = self._pull_functions('trigger')
    _trigger_pull_triggers = trigger(pull_triggers)

    def pull_prereqs(self, *args):
        """Refresh the cards in the prereq builder"""
        self._prereq_builder.decks = self._pull_functions('prereq')
    _trigger_pull_prereqs = trigger(pull_prereqs)

    def pull_actions(self, *args):
        """Refresh the cards in the action builder"""
        self._action_builder.decks = self._pull_functions('action')
    _trigger_pull_actions = trigger(pull_actions)

    def inspect_func(self, namesrc):
        """Take a function's (name, sourcecode) and return a triple of (name, sourcecode, signature)"""
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

        :param what: a string, 'trigger', 'prereq', or 'action'

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
        if not self.rule:
            Logger.debug("RulesView: not pushing {} for lack of rule".format(what))
            return
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


class RulesBox(BoxLayout):
    """A BoxLayout containing a RulesList and a RulesView

    As well as an input for a new rule name; a button to add a new rule by that
    name; and a close button.

    Currently has no way to rename rules (2018-08-15)

    """
    engine = ObjectProperty()
    rulebook = ObjectProperty()
    rulebook_name = StringProperty()
    entity = ObjectProperty()
    entity_name = StringProperty()
    new_rule_name = StringProperty()
    toggle = ObjectProperty()
    ruleslist = ObjectProperty()
    rulesview = ObjectProperty()

    def on_ruleslist(self, *args):
        if not self.ruleslist.children:
            Clock.schedule_once(self.on_ruleslist, 0)
            return
        self.ruleslist.children[0].bind(children=self._upd_ruleslist_selection)

    def new_rule(self, *args):
        if self.new_rule_name in self.engine.rule:
            # TODO: feedback to say you already have such a rule
            return
        self._new_rule_name = self.new_rule_name
        new_rule = self.engine.rule.new_empty(self.new_rule_name)
        assert(new_rule is not None)
        self.rulebook.append(new_rule)
        self.ruleslist.redata()
        self.ids.rulename.text = ''

    def _upd_ruleslist_selection(self, *args):
        if not hasattr(self, '_new_rule_name'):
            return
        for child in self.ruleslist.children[0].children:
            if child.text == self._new_rule_name:
                child.state = 'down'
            else:
                child.state = 'normal'


class RulesScreen(Screen):
    """Screen containing a RulesBox for one rulebook"""
    engine = ObjectProperty()
    entity = ObjectProperty()
    rulebook = ObjectProperty()
    toggle = ObjectProperty()
    rulesview = ObjectProperty()

    def new_rule(self, *args):
        self.children[0].new_rule()


class CharacterRulesScreen(Screen):
    """Screen with TabbedPanel for all the character-rulebooks"""
    engine = ObjectProperty()
    character = ObjectProperty()
    toggle = ObjectProperty()

    def _get_rulebook(self, rb):
        return {
            'character': self.character.rulebook,
            'avatar': self.character.avatar.rulebook,
            'character_thing': self.character.thing.rulebook,
            'character_place': self.character.place.rulebook,
            'character_portal': self.character.portal.rulebook
        }[rb]

    def finalize(self, *args):
        assert not hasattr(self, '_finalized')
        if not (
            self.engine and self.toggle and self.character
        ):
            Clock.schedule_once(self.finalize, 0)
            return
        self._tabs = TabbedPanel(do_default_tab=False)
        for rb, txt in (
                ('character', 'character'),
                ('avatar', 'avatar'),
                ('character_thing', 'thing'),
                ('character_place', 'place'),
                ('character_portal', 'portal')
        ):
            tab = TabbedPanelItem(text=txt)
            setattr(self, '_{}_tab'.format(rb), tab)
            box = RulesBox(
                engine=self.engine,
                rulebook=self._get_rulebook(rb),
                entity=self.character,
                toggle=self.toggle
            )
            tab.add_widget(box)
            self._tabs.add_widget(tab)
        self.add_widget(self._tabs)
        self._finalized = True

    def on_character(self, *args):
        if not hasattr(self, '_finalized'):
            self.finalize()
            return
        for rb in (
            'character', 'avatar', 'character_thing',
            'character_place', 'character_portal'
        ):
            tab = getattr(self, '_{}_tab'.format(rb))
            tab.content.entity = self.character
            tab.content.rulebook = self._get_rulebook(rb)
            # Currently there's no way to assign a new rulebook to an entity
            # in ELiDE, so I don't need to account for that, but what if the
            # rulebook changes as a result of some code running in the LiSE core?
            # 2018-08-13


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
<RulesBox>:
    new_rule_name: rulename.text
    ruleslist: ruleslist
    rulesview: rulesview
    rulebook_name: str(self.rulebook.name) if self.rulebook is not None else ''
    entity_name: str(self.entity.name) if self.entity is not None else ''
    orientation: 'vertical'
    Label:
        text: root.entity_name + '    -    ' + root.rulebook_name
        size_hint_y: 0.05
    BoxLayout:
        orientation: 'horizontal'
        RulesList:
            id: ruleslist
            rulebook: root.rulebook
            entity: root.entity
            rulesview: rulesview
            size_hint_x: 0.2
        RulesView:
            id: rulesview
            engine: root.engine
            rulebook: root.rulebook
            entity: root.entity
            size_hint_x: 0.8
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.05
        TextInput:
            id: rulename
            hint_text: 'New rule name'
            write_tab: False
        Button:
            text: '+'
            on_release: root.new_rule()
        Button:
            text: 'Close'
            on_release: root.toggle()
<RulesScreen>:
    name: 'rules'
    rulesview: box.rulesview
    RulesBox:
        id: box
        engine: root.engine
        rulebook: root.rulebook
        entity: root.entity
        toggle: root.toggle
<CharacterRulesScreen>:
    name: 'charrules'
""")
