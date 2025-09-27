# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
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
from operator import attrgetter

from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.logger import Logger
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.screenmanager import Screen
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from .card import Card, DeckBuilderScrollBar, DeckBuilderView
from .stores import FuncEditor
from .util import devour, logwrap, store_kv


def trigger(func):
	return triggered()(func)


dbg = Logger.debug


class RuleButton(ToggleButton):
	"""A button to select a rule to edit"""

	rulesview = ObjectProperty()
	ruleslist = ObjectProperty()
	rule = ObjectProperty()

	@logwrap(section="RuleButton")
	def on_state(self, *args):
		"""If I'm pressed, unpress all other buttons in the ruleslist"""
		# This really ought to be done with the selection behavior
		if self.state == "down":
			self.rulesview.rule = self.rule
			for button in self.ruleslist.iter_rule_buttons():
				if button != self:
					button.state = "normal"


class RuleButtonBox(BoxLayout, RecycleDataViewBehavior):
	rulesview = ObjectProperty()
	ruleslist = ObjectProperty()
	rule = ObjectProperty()
	button = ObjectProperty()
	index = NumericProperty()

	@trigger
	def move_rule_up(self, *_):
		i = int(self.index)
		if i <= 0:
			return
		self.ruleslist.rulebook.move_rule_back(i)

	@trigger
	def move_rule_down(self, *_):
		i = int(self.index)
		if i >= len(self.ruleslist.rulebook):
			return
		self.ruleslist.rulebook.move_rule_forward(i)
		self.index = i + 1


class RulesList(RecycleView):
	"""A list of rules you might want to edit

	Presented as buttons, which you can click to select one rule at a time.

	"""

	rulebook = ObjectProperty()
	rulesview = ObjectProperty()

	@logwrap(section="RulesList")
	def on_rulebook(self, *_):
		"""Make sure to update when the rulebook changes"""
		if self.rulebook is None:
			return
		Logger.debug("RulesList: got rulebook")
		self.rulebook.connect(self._trigger_redata, weak=False)
		self.redata()

	@logwrap(section="RulesList")
	def on_rulesview(self, *_):
		if self.rulebook is None:
			Logger.debug("RulesList: got rulesview before rulebook")
			return
		self.redata()

	@logwrap(section="RulesList")
	def redata(self, *_):
		"""Make my data represent what's in my rulebook right now"""
		if self.rulesview is None:
			Logger.debug(
				"RulesList: delaying redata until rulesview available"
			)
			self._trigger_redata()
			return
		data = [
			{
				"rulesview": self.rulesview,
				"rule": rule,
				"index": i,
				"ruleslist": self,
			}
			for i, rule in enumerate(self.rulebook)
		]
		self.data = data

	@logwrap(section="RulesList")
	def _trigger_redata(self, *_, **__):
		if hasattr(self, "_scheduled_redata"):
			Clock.unschedule(self._scheduled_redata)
		self._scheduled_redata = Clock.schedule_once(self.redata, 0)

	def iter_rule_buttons(self):
		if not self.children:
			return
		for rules_box in self.children[0].children:
			yield rules_box.button


class RulesView(Widget):
	"""The view to edit a rule

	Presents three tabs, one each for trigger, prereq, and action. Each has a
	deckbuilder in it with a column of used functions and a column of unused
	actions.

	"""

	rulebook = ObjectProperty()
	entity = ObjectProperty()
	rule = ObjectProperty(allownone=True)

	@property
	def engine(self):
		return App.get_running_app().engine

	@logwrap(section="RulesView")
	def on_rule(self, *args):
		"""Make sure to update when the rule changes"""
		if self.rule is None:
			return
		self.rule.connect(self._listen_to_rule)

	@logwrap(section="RulesView")
	def _listen_to_rule(self, rule, **kwargs):
		if rule is not self.rule:
			rule.disconnect(self._listen_to_rule)
			return
		if "triggers" in kwargs:
			self.pull_triggers()
		if "prereqs" in kwargs:
			self.pull_prereqs()
		if "actions" in kwargs:
			self.pull_actions()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.finalize()

	def unbind_all(self, *_):
		binds = App.get_running_app()._bindings
		for att in ("size", "pos"):
			for uid in devour(binds["RulesView", att]):
				self.unbind_uid(att, uid)
		for functyp in "trigger", "prereq", "action":
			for uid in devour(binds["RulesView", "rule", functyp]):
				self.unbind_uid("rule", uid)
			for uid in devour(binds["RulesView", functyp, "decks"]):
				getattr(self, f"_{functyp}_builder").unbind_uid("decks", uid)

	@logwrap(section="RulesView")
	def finalize(self, *_):
		"""Add my tabs"""
		assert not getattr(self, "_finalized", False), "Already finalized"
		if not self.canvas:
			Clock.schedule_once(self.finalize, 0)
			return
		app = App.get_running_app()
		binds = app._bindings

		deck_builder_kwargs = {
			"pos_hint": {"x": 0, "y": 0},
			"starting_pos_hint": {"x": 0.05, "top": 0.95},
			"card_size_hint": (0.3, 0.4),
			"card_hint_step": (0, -0.1),
			"deck_x_hint_step": 0.4,
		}

		self._tabs = TabbedPanel(
			size=self.size, pos=self.pos, do_default_tab=False
		)
		for att in ("size", "pos"):
			for uid in devour(binds["RulesView", att]):
				self.unbind_uid(att, uid)
			binds["RulesView", att].add(
				self.fbind(att, self._tabs.setter(att))
			)
		self.add_widget(self._tabs)

		for functyp in "trigger", "prereq", "action":
			tab = TabbedPanelItem(text=functyp.capitalize())
			setattr(self, "_{}_tab".format(functyp), tab)
			self._tabs.add_widget(getattr(self, "_{}_tab".format(functyp)))
			builder = DeckBuilderView(**deck_builder_kwargs)
			setattr(self, "_{}_builder".format(functyp), builder)
			for uid in devour(binds["RulesView", functyp, "decks"]):
				builder.unbind_uid("decks", uid)
			uid = builder.fbind(
				"decks", getattr(self, f"_trigger_push_{functyp}s")
			)
			setattr(
				self,
				f"_trigger_push_{functyp}s_uid",
				uid,
			)
			binds["RulesView", functyp, "decks"].add(uid)
			scroll_left = DeckBuilderScrollBar(
				size_hint_x=0.01,
				pos_hint={"x": 0, "y": 0},
				deckbuilder=builder,
				deckidx=0,
				scroll_min=0,
			)
			setattr(self, "_scroll_left_" + functyp, scroll_left)
			scroll_right = DeckBuilderScrollBar(
				size_hint_x=0.01,
				pos_hint={"right": 1, "y": 0},
				deckbuilder=builder,
				deckidx=1,
				scroll_min=0,
			)
			setattr(self, "_scroll_right_" + functyp, scroll_right)
			layout = FloatLayout()
			setattr(self, "_{}_layout".format(functyp), layout)
			tab.add_widget(layout)
			layout.add_widget(builder)
			layout.add_widget(scroll_left)
			layout.add_widget(scroll_right)
			layout.add_widget(
				Label(
					text="Used",
					pos_hint={"center_x": 0.1, "center_y": 0.98},
					size_hint=(None, None),
				)
			)
			layout.add_widget(
				Label(
					text="Unused",
					pos_hint={"center_x": 0.5, "center_y": 0.98},
					size_hint=(None, None),
				)
			)
			for uid in devour(binds["RulesView", "rule", functyp]):
				self.unbind_uid("rule", uid)
			binds["RulesView", "rule", functyp].add(
				self.fbind("rule", getattr(self, f"_trigger_pull_{functyp}s"))
			)
		app._unbinders.append(self.unbind_all)
		self._finalized = True

	@logwrap(section="RulesView")
	def _edit_something(self, card: Card):
		what_store = card.ud["type"]
		what_function = card.headline_text
		Logger.info(f"EditButton: {what_store}.{what_function}")
		if not hasattr(self, "rule_func_editor_modal"):
			self.rule_func_editor_modal = ModalView()
			self.rule_func_editor = FuncEditor(
				size_hint_y=0.9, deletable=False
			)

			def save_and_dismiss_modal(*args):
				changed = self.rule_func_editor.save()
				Logger.info(
					f"EditButton: {what_store}.{what_function} {'not ' if not changed else ''}changed"
				)
				if changed:
					card.text = self.rule_func_editor.source
				self.rule_func_editor_modal.dismiss()

			self.rule_func_editor_save_button = Button(
				text="Save", on_press=save_and_dismiss_modal
			)
			self.rule_func_editor_cancel_button = Button(
				text="Cancel", on_press=self.rule_func_editor_modal.dismiss
			)
			self.rule_func_editor_layout = BoxLayout(orientation="vertical")
			self.rule_func_editor_buttons_layout = BoxLayout(
				orientation="horizontal", size_hint_y=0.1
			)
			self.rule_func_editor_buttons_layout.add_widget(
				self.rule_func_editor_cancel_button
			)
			self.rule_func_editor_buttons_layout.add_widget(
				self.rule_func_editor_save_button
			)
			self.rule_func_editor_layout.add_widget(self.rule_func_editor)
			self.rule_func_editor_layout.add_widget(
				self.rule_func_editor_buttons_layout
			)
			self.rule_func_editor_modal.add_widget(
				self.rule_func_editor_layout
			)
		# get the source code of the function to edit
		store = self.rule_func_editor.store = getattr(self.engine, what_store)
		self.rule_func_editor.source = store.get_source(what_function)
		self.rule_func_editor.name_wid.hint_text = what_function
		self.rule_func_editor.name_wid.disabled = True
		# show the modal
		self.rule_func_editor_modal.open()

	@logwrap(section="RulesView")
	def get_functions_cards(self, what, allfuncs):
		"""Return a pair of lists of Card widgets for used and unused functions.

		:param what: a string: 'trigger', 'prereq', or 'action'
		:param allfuncs: a sequence of functions' (name, sourcecode, signature)

		"""
		if not self.rule:
			return [], []
		rulefuncnames = list(
			map(attrgetter("name"), getattr(self.rule, what + "s"))
		)
		unused = [
			Card(
				ud={"type": what, "funcname": name, "signature": sig},
				headline_text=name,
				show_art=False,
				midline_text=what.capitalize(),
				text=source,
				editable=True,
				edit_func=self._edit_something,
			)
			for (name, source, sig) in allfuncs
			if name not in rulefuncnames
		]
		used = [
			Card(
				ud={
					"type": what,
					"funcname": name,
				},
				headline_text=name,
				show_art=False,
				midline_text=what.capitalize(),
				text=str(getattr(getattr(self.engine, what), name)),
				editable=True,
				edit_func=self._edit_something,
			)
			for name in rulefuncnames
		]
		return used, unused

	@logwrap(section="RulesView")
	def set_functions(self, what, allfuncs):
		"""Set the cards in the ``what`` builder to ``allfuncs``

		:param what: a string, 'trigger', 'prereq', or 'action'
		:param allfuncs: a sequence of triples of (name, sourcecode, signature)
		                as taken by my ``get_function_cards`` method.

		"""
		setattr(
			getattr(self, "_{}_builder".format(what)),
			"decks",
			self.get_functions_cards(what, allfuncs),
		)

	@logwrap(section="RulesView")
	def _pull_functions(self, what, truth=True):
		it = map(self.inspect_func, getattr(self.engine, what)._cache.items())
		if not truth:
			it = filter(lambda x: x[0] != "truth", it)
		return self.get_functions_cards(what, list(it))

	@logwrap(section="RulesView")
	def pull_triggers(self, *args):
		"""Refresh the cards in the trigger builder"""
		binds = App.get_running_app()._bindings
		binds["RulesView", "trigger", "decks"].remove(
			self._trigger_push_triggers_uid
		)
		self._trigger_builder.unbind_uid(
			"decks", self._trigger_push_triggers_uid
		)
		self._trigger_builder.decks = self._pull_functions("trigger")
		self._trigger_push_triggers_uid = self._trigger_builder.fbind(
			"decks", self._trigger_push_triggers
		)
		binds["RulesView", "trigger", "decks"].add(
			self._trigger_push_triggers_uid
		)

	_trigger_pull_triggers = trigger(pull_triggers)

	@logwrap(section="RulesView")
	def pull_prereqs(self, *args):
		"""Refresh the cards in the prereq builder"""
		binds = App.get_running_app()._bindings
		binds["RulesView", "prereq", "decks"].remove(
			self._trigger_push_prereqs_uid
		)
		self._prereq_builder.unbind_uid(
			"decks", self._trigger_push_prereqs_uid
		)
		self._prereq_builder.decks = self._pull_functions(
			"prereq", truth=False
		)
		self._trigger_push_prereqs_uid = self._prereq_builder.fbind(
			"decks", self._trigger_push_prereqs
		)
		binds["RulesView", "prereq", "decks"].add(
			self._trigger_push_prereqs_uid
		)

	_trigger_pull_prereqs = trigger(pull_prereqs)

	@logwrap(section="RulesView")
	def pull_actions(self, *args):
		"""Refresh the cards in the action builder"""
		binds = App.get_running_app()._bindings
		binds["RulesView", "action", "decks"].remove(
			self._trigger_push_actions_uid
		)
		self._action_builder.unbind_uid(
			"decks", self._trigger_push_actions_uid
		)
		self._action_builder.decks = self._pull_functions(
			"action", truth=False
		)
		self._trigger_push_actions_uid = self._action_builder.fbind(
			"decks", self._trigger_push_actions
		)
		binds["RulesView", "action", "decks"].add(
			self._trigger_push_actions_uid
		)

	_trigger_pull_actions = trigger(pull_actions)

	@logwrap(section="RulesView")
	def inspect_func(self, namesrc):
		"""Take a function's (name, sourcecode) and return a triple of (name, sourcecode, signature)"""
		(name, src) = namesrc
		glbls = {}
		lcls = {}
		exec(src, glbls, lcls)
		assert name in lcls
		func = lcls[name]
		return name, src, signature(func)

	@logwrap(section="RulesView")
	def update_builders(self, *args):
		for attrn in "_trigger_builder", "_prereq_builder", "_action_builder":
			if not hasattr(self, attrn):
				dbg("RulesView: no {}".format(attrn))
				Clock.schedule_once(self.update_builders, 0)
				return
		self._trigger_builder.clear_widgets()
		self._prereq_builder.clear_widgets()
		self._action_builder.clear_widgets()
		if self.rule is None:
			dbg("RulesView: no rule")
			return
		if hasattr(self, "_list"):
			self._list.redata()
		self.pull_triggers()
		self.pull_prereqs()
		self.pull_actions()

	_trigger_update_builders = trigger(update_builders)

	@logwrap(section="RulesView")
	def _upd_unused(self, what):
		"""Make sure to have exactly one copy of every valid function in the
		"unused" pile on the right.

		Doesn't read from the database.

		:param what: a string, 'trigger', 'prereq', or 'action'

		"""
		binds = App.get_running_app()._bindings
		builder = getattr(self, "_{}_builder".format(what))
		updtrig = getattr(self, "_trigger_upd_unused_{}s".format(what))
		while binds["RulesView", "builder", what]:
			self.unbind_uid("decks", binds[RulesView, "builder", what].pop())
		funcs = OrderedDict()
		cards = list(self._action_builder.decks[1])
		cards.reverse()
		for card in cards:
			funcs[card.ud["funcname"]] = card
		for card in self._action_builder.decks[0]:
			if card.ud["funcname"] not in funcs:
				funcs[card.ud["funcname"]] = card.copy()
		unused = list(funcs.values())
		unused.reverse()
		builder.decks[1] = unused
		binds["RulesView", "builder", what].add(builder.bind("decks", updtrig))

	@logwrap(section="RulesView")
	def upd_unused_actions(self, *_):
		self._upd_unused("action")

	_trigger_upd_unused_actions = trigger(upd_unused_actions)

	@logwrap(section="RulesView")
	def upd_unused_triggers(self, *_):
		self._upd_unused("trigger")

	_trigger_upd_unused_triggers = trigger(upd_unused_triggers)

	@logwrap(section="RulesView")
	def upd_unused_prereqs(self, *_):
		self._upd_unused("prereq")

	_trigger_upd_unused_prereqs = trigger(upd_unused_prereqs)

	@logwrap(section="RulesView")
	def _push_funcs(self, what):
		if not self.rule:
			Logger.debug(
				"RulesView: not pushing {} for lack of rule".format(what)
			)
			return
		funcs = [
			getattr(getattr(self.engine, what), card.ud["funcname"])
			for card in getattr(self, "_{}_builder".format(what)).decks[0]
		]
		funlist = getattr(self.rule, what + "s")
		if funlist != funcs:
			setattr(self.rule, what + "s", funcs)

	@logwrap(section="RulesView")
	def push_actions(self, *_):
		self._push_funcs("action")

	_trigger_push_actions = trigger(push_actions)

	@logwrap(section="RulesView")
	def push_prereqs(self, *_):
		self._push_funcs("prereq")

	_trigger_push_prereqs = trigger(push_prereqs)

	@logwrap(section="RulesView")
	def push_triggers(self, att, *_):
		self._push_funcs("trigger")

	_trigger_push_triggers = trigger(push_triggers)


class RulesBox(BoxLayout):
	"""A BoxLayout containing a RulesList and a RulesView

	As well as an input for a new rule name; a button to add a new rule by that
	name; and a close button.

	Currently has no way to rename rules (2018-08-15)

	"""

	rulebook = ObjectProperty()
	rulebook_name = StringProperty()
	entity = ObjectProperty()
	entity_name = StringProperty()
	new_rule_name = StringProperty()
	toggle = ObjectProperty()
	ruleslist = ObjectProperty()
	rulesview = ObjectProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		App.get_running_app()._unbinders.append(self.unbind_all)

	@property
	def engine(self):
		return App.get_running_app().engine

	def unbind_all(self):
		binds = App.get_running_app()._bindings
		for uid in devour(binds["RulesBox", "ruleslist", "children"]):
			self.unbind_uid("children", uid)

	@logwrap(section="RulesBox")
	def on_ruleslist(self, *_):
		if not self.ruleslist.children:
			Clock.schedule_once(self.on_ruleslist, 0)
			return
		binds = App.get_running_app()._bindings[
			"RulesBox", "ruleslist", "children"
		]
		while binds:
			self.unbind_uid("children", binds.pop())
		binds.add(self.fbind("children", self._upd_ruleslist_selection))

	@logwrap(section="RulesBox")
	def new_rule(self, *_):
		if self.new_rule_name in self.engine.rule:
			# TODO: feedback to say you already have such a rule
			return
		self._new_rule_name = self.new_rule_name
		new_rule = self.engine.rule.new_empty(self.new_rule_name)
		assert new_rule is not None
		self.rulebook.append(new_rule)
		self.ruleslist.redata()
		self.ids.rulename.text = ""

	@logwrap(section="RulesBox")
	def _upd_ruleslist_selection(self, *_):
		if not hasattr(self, "_new_rule_name"):
			return
		for child in self.ruleslist.iter_rule_buttons():
			if child.text == self._new_rule_name:
				child.state = "down"
			else:
				child.state = "normal"


class RulesScreen(Screen):
	"""Screen containing a RulesBox for one rulebook"""

	entity = ObjectProperty()
	rulebook = ObjectProperty()
	toggle = ObjectProperty()
	rulesview = ObjectProperty()
	box = ObjectProperty()

	@property
	def engine(self):
		return App.get_running_app().engine

	@logwrap(section="RulesScreen")
	def new_rule(self, *_):
		self.children[0].new_rule()


class CharacterRulesScreen(Screen):
	"""Screen with TabbedPanel for all the character-rulebooks"""

	character = ObjectProperty()
	toggle = ObjectProperty()

	@logwrap(section="CharacterRulesScreen")
	def _get_rulebook(self, rb):
		return {
			"character": self.character.rulebook,
			"unit": self.character.unit.rulebook,
			"character_thing": self.character.thing.rulebook,
			"character_place": self.character.place.rulebook,
			"character_portal": self.character.portal.rulebook,
		}[rb]

	@logwrap(section="CharacterRulesScreen")
	def finalize(self, *args):
		if hasattr(self, "_finalized"):
			return
		if not (self.toggle and self.character):
			Clock.schedule_once(self.finalize, 0)
			return
		self._tabs = TabbedPanel(do_default_tab=False)
		for rb, txt in (
			("character", "character"),
			("unit", "unit"),
			("character_thing", "thing"),
			("character_place", "place"),
			("character_portal", "portal"),
		):
			tab = TabbedPanelItem(text=txt)
			setattr(self, "_{}_tab".format(rb), tab)
			box = RulesBox(
				rulebook=self._get_rulebook(rb),
				entity=self.character,
				toggle=self.toggle,
			)
			setattr(self, "_{}_box".format(rb), box)
			tab.add_widget(box)
			self._tabs.add_widget(tab)
		self.add_widget(self._tabs)
		self._finalized = True

	@logwrap(section="CharacterRulesScreen")
	def on_character(self, *_):
		if not hasattr(self, "_finalized"):
			self.finalize()
			return
		for rb in (
			"character",
			"unit",
			"character_thing",
			"character_place",
			"character_portal",
		):
			tab = getattr(self, "_{}_tab".format(rb))
			tab.content.entity = self.character
			tab.content.rulebook = self._get_rulebook(rb)
		# Currently there's no way to assign a new rulebook to an entity
		# in elide, so I don't need to account for that, but what if the
		# rulebook changes as a result of some code running in the lisien core?
		# 2018-08-13


store_kv(
	__name__,
	"""
<RuleButton>:
    text: self.rule.name if self.rule else ''
<RuleButtonBox>:
	orientation: 'horizontal'
	button: rule_button
	BoxLayout:
		orientation: 'vertical'
		size_hint_x: 0.2
		Button:
			id: up
			text: '↑'
			font_name: 'Symbola'
			on_release: root.move_rule_up()
		Button:
			id: down
			text: '↓'
			font_name: 'Symbola'
			on_release: root.move_rule_down()
	RuleButton:
		id: rule_button
		rulesview: root.rulesview
		ruleslist: root.ruleslist
		rule: root.rule
		size_hint_x: 0.8
<RulesList>:
    viewclass: 'RuleButtonBox'
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
            rulebook: root.rulebook
            entity: root.entity
            size_hint_x: 0.8
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.07
        TextInput:
            id: rulename
            hint_text: 'New rule name'
            write_tab: False
            multiline: False
        Button:
            text: '+'
            on_release: root.new_rule()
        Button:
            id: closebut
            text: 'Close'
            on_release: root.toggle()
<RulesScreen>:
    name: 'rules'
    rulesview: box.rulesview
    box: box
    RulesBox:
        id: box
        rulebook: root.rulebook
        entity: root.entity
        toggle: root.toggle
<CharacterRulesScreen>:
    name: 'charrules'
""",
)
