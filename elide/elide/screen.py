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
"""The big layout that you view all of elide through.

Handles touch, selection, and time control. Contains a graph, a stat
grid, the time control panel, and the menu.

"""

from ast import literal_eval
from functools import partial
from threading import Thread

from kivy.app import App
from kivy.clock import Clock, mainthread, triggered
from kivy.core.text import DEFAULT_FONT
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.properties import (
	BooleanProperty,
	BoundedNumericProperty,
	DictProperty,
	ListProperty,
	NumericProperty,
	ObjectProperty,
	ReferenceListProperty,
	StringProperty,
	VariableListProperty,
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from .calendar import Agenda
from .charmenu import CharMenu
from .graph.board import GraphBoardView
from .grid.board import GridBoardView
from .stepper import RuleStepper
from .util import devour, dummynum, load_kv, logwrap


def trigger(func):
	return triggered()(func)


Factory.register("CharMenu", cls=CharMenu)


def release_edit_lock(*_):
	app = App.get_running_app()
	app.edit_locked = app.simulate_button_down


class KvLayout(FloatLayout):
	pass


class StatListPanel(BoxLayout):
	"""A panel that displays a simple two-column grid showing the stats of
	the selected entity, defaulting to those of the character being
	viewed.

	Has a button on the bottom to open the StatWindow in which
	to add and delete stats, or to change the way they are displayed
	in the StatListPanel.

	"""

	selection_name = StringProperty()
	button_text = StringProperty("Configure stats")
	cfgstatbut = ObjectProperty()
	statlist = ObjectProperty()
	proxy = ObjectProperty()
	toggle_stat_cfg = ObjectProperty()

	@logwrap(section="StatListPanel")
	def on_proxy(self, *_):
		if hasattr(self.proxy, "name"):
			self.selection_name = str(self.proxy.name)

	@logwrap(section="StatListPanel")
	def set_value(self, k, v):
		if v is None:
			del self.proxy[k]
		else:
			try:
				vv = literal_eval(v)
			except (TypeError, ValueError):
				vv = v
			self.proxy[k] = vv


class SimulateButton(ToggleButton):
	@logwrap(section="SimulateButton")
	def on_state(self, *_):
		app = App.get_running_app()
		app.edit_locked = app.simulate_button_down = self.state == "down"


class OneTurnButton(Button):
	screen = ObjectProperty()

	@logwrap(section="OneTurnButton")
	def on_release(self):
		App.get_running_app().edit_locked = True
		self.screen.next_turn(cb=release_edit_lock)


class TimePanel(BoxLayout):
	"""A panel that starts and stop the game, or sets the time.

	There's a "simulate" button, which is toggleable. When toggled on, the
	simulation will continue to run until it's toggled off
	again. Next to this is a "1 turn" button, which will simulate
	exactly one turn and stop. And there are two text fields in which
	you can manually enter a Branch and Tick to go to. Moving through
	time this way doesn't simulate anything--you'll only see what
	happened as a result of "simulate," "1 turn," or some other way
	the lisien rules engine has been made to run.

	"""

	screen = ObjectProperty()
	buttons_font_size = NumericProperty(18)
	disable_one_turn = BooleanProperty()

	@logwrap(section="TimePanel")
	def set_branch(self, *_):
		branch = self.ids.branchfield.text
		self.ids.branchfield.text = ""
		self.screen.app.branch = branch
		self.screen.charmenu.switch_to_menu()

	@logwrap(section="TimePanel")
	def set_turn(self, *_):
		turn = int(self.ids.turnfield.text)
		self.ids.turnfield.text = ""
		self.screen.app.set_turn(turn)
		self.screen.charmenu.switch_to_menu()

	@logwrap(section="TimePanel")
	def set_tick(self, *_):
		tick = int(self.ids.tickfield.text)
		self.ids.tickfield.text = ""
		self.screen.app.set_tick(tick)
		self.screen.charmenu.switch_to_menu()

	@mainthread
	@logwrap(section="TimePanel")
	def _upd_branch_hint(self, app, *_):
		self.ids.branchfield.hint_text = app.branch

	@mainthread
	@logwrap(section="TimePanel")
	def _upd_turn_hint(self, app, *_):
		self.ids.turnfield.hint_text = str(app.turn)

	@mainthread
	@logwrap(section="TimePanel")
	def _upd_tick_hint(self, app, *_):
		self.ids.tickfield.hint_text = str(app.tick)

	@logwrap(section="TimePanel")
	def on_screen(self, *_):
		if not all(
			field in self.ids
			for field in ("branchfield", "turnfield", "tickfield")
		):
			Clock.schedule_once(self.on_screen, 0)
			return
		app = App.get_running_app()
		binds = app._bindings
		self.ids.branchfield.hint_text = self.screen.app.branch
		self.ids.turnfield.hint_text = str(self.screen.app.turn)
		self.ids.tickfield.hint_text = str(self.screen.app.tick)
		binds["ElideApp", "branch"].add(
			app.fbind("branch", self._upd_branch_hint)
		)
		binds["ElideApp", "turn"].add(app.fbind("turn", self._upd_turn_hint))
		binds["ElideApp", "tick"].add(app.fbind("tick", self._upd_tick_hint))


class MainScreen(Screen):
	"""A master layout that contains one graph and some menus.

	This contains three elements: a scrollview (containing the graph),
	a menu, and the time control panel. This class has some support methods
	for handling interactions with the menu and the character sheet,
	but if neither of those happen, the scrollview handles touches on its
	own.

	"""

	name = "mainscreen"

	graphboards = DictProperty()
	gridboards = DictProperty()
	boardview = ObjectProperty()
	mainview = ObjectProperty()
	charmenu = ObjectProperty()
	statlist = ObjectProperty()
	statpanel = ObjectProperty()
	timepanel = ObjectProperty()
	turnscroll = ObjectProperty()
	kv = StringProperty()
	use_kv = BooleanProperty()
	play_speed = NumericProperty(1)
	playbut = ObjectProperty()
	portaladdbut = ObjectProperty()
	portaldirbut = ObjectProperty()
	dummyplace = ObjectProperty()
	dummything = ObjectProperty()
	dummies = ReferenceListProperty(dummyplace, dummything)
	dialoglayout = ObjectProperty()
	visible = BooleanProperty()
	_touch = ObjectProperty(None, allownone=True)
	rules_per_frame = BoundedNumericProperty(10, min=1)
	tmp_block = BooleanProperty(False)

	@property
	def app(self):
		return App.get_running_app()

	def __init__(self, **kw):
		for screen_kv in [
			"screen",
			"dummy",
			"charmenu",
			"statcfg",
			"stepper",
		]:
			load_kv(screen_kv + ".kv")
		super().__init__(**kw)

	@logwrap(section="TimePanel")
	def _update_adding_portal(self, *_):
		self.boardview.adding_portal = (
			self.charmenu.portaladdbut.state == "down"
		)

	@logwrap(section="TimePanel")
	def _update_board(self, *_):
		self.boardview.board = self.graphboards[self.app.character_name]
		self.gridview.board = self.gridboards[self.app.character_name]

	@logwrap(section="TimePanel")
	def populate(self, *_):
		if hasattr(self, "_populated"):
			return
		if (
			None in (self.statpanel, self.charmenu)
			or None in (self.app.character_name, self.charmenu.portaladdbut)
			or self.app.character_name not in self.graphboards
		):
			if self.statpanel is None:
				Logger.warning("MainScreen: no statpanel")
			if self.charmenu is None:
				Logger.warning("MainScreen: no charmenu")
			elif self.charmenu.portaladdbut is None:
				Logger.warning("MainScreen: CharMenu has no portal button")
			if self.app.character_name is None:
				Logger.warning("MainScreen: no character chosen")
			elif self.app.character_name not in self.graphboards:
				Logger.warning(
					"MainScreen: no graphboard to represent "
					+ repr(self.app.character_name)
				)
			Clock.schedule_once(self.populate, 0)
			return
		app = App.get_running_app()
		binds = app._bindings
		self.boardview = GraphBoardView(
			scale_min=0.2,
			scale_max=4.0,
			size=self.mainview.size,
			pos=self.mainview.pos,
			board=self.graphboards[self.app.character_name],
			adding_portal=self.charmenu.portaladdbut.state == "down",
		)
		binds["MainScreen", "mainview", "size"].add(
			self.mainview.fbind("size", self.boardview.setter("size"))
		)
		binds["MainScreen", "mainview", "pos"].add(
			self.mainview.fbind("pos", self.boardview.setter("pos"))
		)
		binds["CharMenu", "portaladdbut", "state"].add(
			self.charmenu.portaladdbut.fbind(
				"state", self._update_adding_portal
			)
		)
		binds["ElideApp", "character_name"].add(
			app.fbind("character_name", self._update_board)
		)
		self.calendar = Agenda(update_mode="present")
		self.calendar_view = ScrollView(
			size=self.mainview.size, pos=self.mainview.pos
		)
		self.gridview = GridBoardView(
			scale_min=0.2,
			scale_max=4.0,
			size=self.mainview.size,
			pos=self.mainview.pos,
			board=self.gridboards[self.app.character_name],
		)
		binds["MainScreen", "mainview", "size"].add(
			self.mainview.fbind("size", self.calendar_view.setter("size"))
		)
		binds["MainScreen", "mainview", "pos"].add(
			self.mainview.fbind("pos", self.calendar_view.setter("pos"))
		)
		binds["MainScreen", "mainview", "size"].add(
			self.mainview.fbind("size", self.gridview.setter("size"))
		)
		binds["MainScreen", "mainview", "pos"].add(
			self.mainview.fbind("pos", self.gridview.setter("pos"))
		)
		self.calendar_view.add_widget(self.calendar)
		self.mainview.add_widget(self.boardview)
		app._unbinders.append(self.unbind_all)
		self._populated = True

	def unbind_all(self):
		app = App.get_running_app()
		binds = app._bindings
		for uid in devour(binds["MainScreen", "mainview", "size"]):
			self.mainview.unbind_uid("size", uid)
		for uid in devour(binds["MainScreen", "mainview", "pos"]):
			self.mainview.unbind_uid("pos", uid)
		for uid in devour(binds["CharMenu", "portaladdbut", "state"]):
			self.charmenu.portaladdbut.unbind_uid("state", uid)
		for uid in devour(binds["ElideApp", "character_name"]):
			app.unbind_uid("character_name", uid)
		for uid in devour(binds["MainScreen", "mainview", "size"]):
			self.mainview.unbind_uid("size", uid)
		for uid in devour(binds["MainScreen", "mainview", "pos"]):
			self.mainview.unbind_uid("pos", uid)

	@logwrap(section="TimePanel")
	def on_statpanel(self, *_):
		if not self.app:
			Clock.schedule_once(self.on_statpanel, 0)
			return
		app = App.get_running_app()
		binds = app._bindings
		for att in ("selected_proxy", "branch", "turn", "tick"):
			binds["ElideApp", att].add(app.fbind(att, self._update_statlist))

	@trigger
	@logwrap(section="TimePanel")
	def _update_statlist(self, *_):
		if not self.app or not hasattr(self.app, "engine"):
			return
		if not self.app.selected_proxy:
			self._update_statlist()
			return
		self.app.update_calendar(
			self.statpanel.statlist, past_turns=0, future_turns=0
		)

	@logwrap(section="TimePanel")
	def pull_visibility(self, *_):
		if not self.manager:
			self.visible = False
			return
		self.visible = self.manager.current == "main"

	@logwrap(section="TimePanel")
	def on_manager(self, *_):
		if not self.manager:
			return
		self.pull_visibility()
		self.manager.bind(current=self.pull_visibility)

	@logwrap(section="TimePanel")
	def on_playbut(self, *_):
		if hasattr(self, "_play_scheduled"):
			return
		self._play_scheduled = Clock.schedule_interval(
			self.play, 1.0 / self.play_speed
		)

	@logwrap(section="TimePanel")
	def on_play_speed(self, *_):
		"""Change the interval at which ``self.play`` is called to match my
		current ``play_speed``.

		"""
		if hasattr(self, "_play_scheduled"):
			Clock.unschedule(self._play_scheduled)
		self._play_scheduled = Clock.schedule_interval(
			self.play, 1.0 / self.play_speed
		)

	@logwrap(section="TimePanel")
	def on_touch_down(self, touch):
		if self.boardview is None:
			return
		if self.visible:
			touch.grab(self)
		for interceptor in (
			self.timepanel,
			self.turnscroll,
			self.charmenu,
			self.statpanel,
			self.dummyplace,
			self.dummything,
		):
			if interceptor.collide_point(*touch.pos):
				interceptor.dispatch("on_touch_down", touch)
				self.boardview.keep_selection = (
					self.gridview.keep_selection
				) = True
				return True
		if self.dialoglayout.dispatch("on_touch_down", touch):
			return True
		return self.mainview.dispatch("on_touch_down", touch)

	@logwrap(section="TimePanel")
	def on_touch_up(self, touch):
		if self.timepanel.collide_point(*touch.pos):
			return self.timepanel.dispatch("on_touch_up", touch)
		elif self.turnscroll.collide_point(*touch.pos):
			return self.turnscroll.dispatch("on_touch_up", touch)
		elif self.charmenu.collide_point(*touch.pos):
			return self.charmenu.dispatch("on_touch_up", touch)
		elif self.statpanel.collide_point(*touch.pos):
			return self.statpanel.dispatch("on_touch_up", touch)
		return self.mainview.dispatch("on_touch_up", touch)

	@logwrap(section="TimePanel")
	def on_dummies(self, *_):
		"""Give the dummies numbers such that, when appended to their names,
		they give a unique name for the resulting new
		:class:`graph.Pawn` or :class:`graph.Spot`.

		"""
		app = App.get_running_app()
		if not app.character:
			Clock.schedule_once(self.on_dummies, 0)
			return
		binds = app._bindings

		def renum_dummy(dummy, *_):
			dummy.num = dummynum(self.app.character, dummy.prefix) + 1

		for dummy in self.dummies:
			if dummy is None or hasattr(dummy, "_numbered"):
				continue
			if dummy == self.dummything:
				binds["ElideApp", "pawncfg", "imgpaths"].add(
					app.pawncfg.fbind("imgpaths", dummy.setter("paths"))
				)
			if dummy == self.dummyplace:
				binds["ElideApp", "spotcfg", "imgpaths"].add(
					app.spotcfg.fbind("imgpaths", dummy.setter("paths"))
				)
			dummy.num = dummynum(self.app.character, dummy.prefix) + 1
			Logger.debug("MainScreen: dummy #{}".format(dummy.num))
			dummy.bind(prefix=partial(renum_dummy, dummy))
			dummy._numbered = True

	@logwrap(section="TimePanel")
	def update_from_time_travel(
		self, command, branch, turn, tick, result, **kwargs
	):
		self._update_from_delta(command, branch, turn, tick, result[-1])

	@logwrap(section="TimePanel")
	def _update_from_delta(self, cmd, branch, turn, tick, delta, **kwargs):
		self.app.branch = branch
		self.app.turn = turn
		self.app.tick = tick
		self.statpanel.statlist.mirror = dict(self.app.selected_proxy)

	@logwrap(section="TimePanel")
	def play(self, *_):
		"""If the 'play' button is pressed, advance a turn.

		If you want to disable this, set ``engine.universal['block'] = True``

		"""
		if (
			self.playbut is None
			or self.playbut.state == "normal"
			or not hasattr(self.app, "engine")
			or self.app.engine is None
			or self.app.engine.closed
			or self.app.engine.universal.get("block")
			or not hasattr(self.app, "manager")
			or self.app.manager.current != "mainscreen"
		):
			return
		self.next_turn(cb=release_edit_lock)

	@logwrap(section="TimePanel")
	def _update_from_next_turn(
		self, command, branch, turn, tick, result, cb=None
	):
		todo, deltas = result
		if isinstance(todo, list):
			self.dialoglayout.todo = todo
			self.dialoglayout.idx = 0
		self._update_from_delta(command, branch, turn, tick, deltas)
		self.dialoglayout.advance_dialog()
		if cb is not None:
			cb(command, branch, turn, tick, result)
		self.tmp_block = False

	@logwrap(section="TimePanel")
	def next_turn(self, cb=None, *_):
		"""Advance time by one turn, if it's not blocked.

		Block time by setting ``engine.universal['block'] = True``"""
		if self.tmp_block:
			return
		eng = self.app.engine
		dial = self.dialoglayout
		if eng.universal.get("block"):
			Logger.info(
				"MainScreen: next_turn blocked, delete universal['block'] to unblock"
			)
			return
		if dial.todo:
			if not dial.children:
				Logger.info(
					"MainScreen: DialogLayout has todo but no children, advancing."
				)
				dial.advance_dialog()
			Clock.schedule_once(partial(self.next_turn, cb), self.play_speed)
			return
		self.tmp_block = True
		self._next_turn_thread = Thread(
			target=eng.next_turn,
			kwargs={"cb": partial(self._update_from_next_turn, cb=cb)},
		)
		self._next_turn_thread.start()
		self.ids.charmenu.switch_to_menu()

	@logwrap(section="TimePanel")
	def switch_to_calendar(self, *_):
		self.app.update_calendar(self.calendar)
		self.mainview.clear_widgets()
		self.mainview.add_widget(self.calendar_view)

	@logwrap(section="TimePanel")
	def switch_to_boardview(self, *_):
		self.mainview.clear_widgets()
		self.app.engine.handle(
			"apply_choices", choices=[self.calendar.get_track()]
		)
		self.mainview.add_widget(self.boardview)

	@logwrap(section="TimePanel")
	def toggle_gridview(self, *_):
		if self.gridview in self.mainview.children:
			self.mainview.clear_widgets()
			self.mainview.add_widget(self.boardview)
		else:
			self.mainview.clear_widgets()
			self.mainview.add_widget(self.gridview)

	@logwrap(section="TimePanel")
	def toggle_calendar(self, *_):
		# TODO decide how to handle switching between >2 view types
		if self.boardview in self.mainview.children:
			self.switch_to_calendar()
		else:
			self.switch_to_boardview()

	@trigger
	@logwrap(section="TimePanel")
	def toggle_timestream(self, *_):
		if self.manager.current != "timestream":
			self.manager.current = "timestream"
		else:
			self.manager.current = "mainscreen"

	@trigger
	@logwrap(section="TimePanel")
	def toggle_log(self, *_):
		if self.manager.current != "log":
			self.manager.current = "log"
		else:
			self.manager.current = "mainscreen"


class CharMenuContainer(BoxLayout):
	screen = ObjectProperty()
	dummyplace = ObjectProperty()
	dummything = ObjectProperty()
	portaladdbut = ObjectProperty()
	toggle_gridview = ObjectProperty()
	toggle_timestream = ObjectProperty()

	def __init__(self, **kwargs):
		super(CharMenuContainer, self).__init__(**kwargs)
		app = App.get_running_app()
		binds = app._bindings
		self.charmenu = CharMenu(screen=self.screen, size_hint_y=0.9)
		binds["CharMenuContainer", "screen"].add(
			self.fbind("screen", self.charmenu.setter("screen"))
		)
		self.dummyplace = self.charmenu.dummyplace
		binds["CharMenu", "dummyplace"].add(
			self.charmenu.fbind("dummyplace", self.setter("dummyplace"))
		)
		self.dummything = self.charmenu.dummything
		binds["CharMenu", "dummything"].add(
			self.charmenu.fbind("dummything", self.setter("dummything"))
		)
		self.portaladdbut = self.charmenu.portaladdbut
		binds["CharMenu", "portaladdbut"].add(
			self.charmenu.fbind("portaladdbut", self.setter("portaladdbut"))
		)
		if self.toggle_gridview:
			self.charmenu = self.toggle_gridview
		binds["CharMenuContainer", "toggle_gridview"].add(
			self.fbind(
				"toggle_gridview", self.charmenu.setter("toggle_gridview")
			)
		)
		binds["CharMenuContainer", "toggle_timestream"].add(
			self.fbind(
				"toggle_timestream", self.charmenu.setter("toggle_timestream")
			)
		)
		self.stepper = RuleStepper(size_hint_y=0.9)
		self.button = Button(
			on_release=self._toggle_stepper,
			text="Rule\nstepper",
			size_hint_y=0.1,
		)
		binds["ElideApp", "branch"].add(
			app.fbind("branch", self.switch_to_menu)
		)
		binds["ElideApp", "turn"].add(app.fbind("turn", self.switch_to_menu))
		binds["ElideApp", "edit_locked"].add(
			app.fbind("edit_locked", self.button.setter("disabled"))
		)
		app._unbinders.append(self.unbind_all)

	def unbind_all(self):
		app = App.get_running_app()
		binds = app._bindings
		for uid in devour(binds["CharMenuContainer", "screen"]):
			app.unbind_uid("screen", uid)
		for uid in devour(binds["CharMenu", "dummyplace"]):
			self.charmenu.unbind_uid("dummyplace", uid)
		for uid in devour(binds["CharMenu", "dummything"]):
			self.charmenu.unbind_uid("dummything", uid)
		for uid in devour(binds["CharMenu", "portaladdbut"]):
			self.charmenu.unbind_uid("portaladdbut", uid)
		for uid in devour(binds["CharMenuContainer", "toggle_gridview"]):
			self.charmenu.unbind_uid("toggle_gridview", uid)
		for uid in devour(binds["CharMenuContainer", "toggle_timestream"]):
			self.charmenu.unbind_uid("toggle_timestream", uid)
		for uid in devour(binds["ElideApp", "branch"]):
			app.unbind_uid("branch", uid)
		for uid in devour(binds["ElideApp", "turn"]):
			app.unbind_uid("turn", uid)
		for uid in devour(binds["ElideApp", "tick"]):
			app.unbind_uid("tick", uid)
		for uid in devour(binds["ElideApp", "edit_locked"]):
			app.unbind_uid("edit_locked", uid)
		for uid in devour(binds["ElideApp", "pawncfg", "imgpaths"]):
			app.pawncfg.unbind_uid("imgpaths", uid)
		for uid in devour(binds["ElideApp", "spotcfg", "imgpaths"]):
			app.spotcfg.unbind_uid("imgpaths", uid)

	@logwrap(section="TimePanel")
	def on_parent(self, *_):
		if (
			not self.screen
			or not hasattr(self, "charmenu")
			or not hasattr(self, "stepper")
			or not hasattr(self, "button")
		):
			Clock.schedule_once(self.on_parent, 0)
			return
		self.add_widget(self.charmenu)
		self.add_widget(self.button)

	@trigger
	@logwrap(section="TimePanel")
	def _toggle_stepper(self, *_):
		if self.charmenu in self.children:
			engine = self.screen.app.engine
			self.clear_widgets()
			self.stepper.from_rules_handled_turn(
				engine.handle("rules_handled_turn")
			)
			self.add_widget(self.stepper)
			self.button.text = "Menu"
		else:
			self.clear_widgets()
			self.add_widget(self.charmenu)
			self.button.text = "Rule stepper"
		self.add_widget(self.button)

	@trigger
	@logwrap(section="TimePanel")
	def switch_to_menu(self, *args):
		if self.charmenu not in self.children:
			self.clear_widgets()
			self.add_widget(self.charmenu)
			self.button.text = "Rule stepper"
			self.add_widget(self.button)


class TurnScroll(Slider):
	def __init__(self, **kwargs):
		kwargs["step"] = 1
		super().__init__(**kwargs)
		self._collect_engine()

	@logwrap(section="TimePanel")
	def _collect_engine(self, *args):
		app = App.get_running_app()
		if not hasattr(app, "engine"):
			Clock.schedule_once(self._collect_engine, 0)
			return
		engine = app.engine
		self.min = engine.branch_start_turn()
		self.max = engine.branch_end_turn()
		self.value = engine.turn
		engine.time.connect(self._receive_time)

	@logwrap(section="TimePanel")
	def _receive_time(self, engine, then, now):
		(_, turn, _) = now
		self.value = turn
		try:
			self.min = engine.branch_start_turn()
		except KeyError:
			self.min = turn
		try:
			self.max = engine.branch_end_turn()
		except KeyError:
			self.max = turn

	@logwrap(section="TimePanel")
	def on_touch_move(self, touch):
		if touch.grab_current == self:
			app = App.get_running_app()
			app.mainscreen.timepanel.ids.turnfield.hint_text = str(
				int(self.value)
			)
		return super().on_touch_move(touch)

	@logwrap(section="TimePanel")
	def on_touch_up(self, touch):
		if touch.grab_current == self:
			app = App.get_running_app()
			app.engine.time.disconnect(self._receive_time)
			app.time_travel(app.branch, int(self.value))
			app.engine.time.connect(self._receive_time)


class Box(Widget):
	padding = VariableListProperty(6)
	border = VariableListProperty(4)
	font_size = StringProperty("15sp")
	font_name = StringProperty(DEFAULT_FONT)
	background = StringProperty()
	background_color = VariableListProperty([1, 1, 1, 1])
	foreground_color = VariableListProperty([0, 0, 0, 1])


class ScrollableLabel(ScrollView):
	font_size = StringProperty("15sp")
	font_name = StringProperty(DEFAULT_FONT)
	color = VariableListProperty([0, 0, 0, 1])
	line_spacing = NumericProperty(0)
	text = StringProperty()


class MessageBox(Box):
	"""Looks like a TextInput but doesn't accept any input.

	Does support styled text with BBcode.

	"""

	line_spacing = NumericProperty(0)
	text = StringProperty()


class DialogMenu(Box):
	"""Some buttons that make the game do things.

	Set ``options`` to a list of pairs of ``(text, function)`` and the
	menu will be populated with buttons that say ``text`` that call
	``function`` when pressed.

	"""

	options = ListProperty()
	"""List of pairs of (button_text, callable)"""

	def _set_sv_size(self, *_):
		self._sv.width = self.width - self.padding[0] - self.padding[2]
		self._sv.height = self.height - self.padding[1] - self.padding[3]

	def _set_sv_pos(self, *_):
		self._sv.x = self.x + self.padding[0]
		self._sv.y = self.y + self.padding[3]

	@mainthread
	@partial(logwrap, section="DialogMenu")
	def on_options(self, *_):
		binds = App.get_running_app()._bindings
		if not hasattr(self, "_sv"):
			self._sv = ScrollView(size=self.size, pos=self.pos)
			assert not binds["DialogMenu", "size"]
			binds["DialogMenu", "size"].add(
				self.fbind("size", self._set_sv_size)
			)
			assert not binds["DialogMenu", "pos"]
			binds["DialogMenu", "pos"].add(self.fbind("pos", self._set_sv_pos))
			layout = BoxLayout(orientation="vertical")
			self._sv.add_widget(layout)
			self.add_widget(self._sv)
		else:
			layout = self._sv.children[0]
			layout.clear_widgets()
		for txt, part in self.options:
			if not callable(part):
				raise TypeError("Menu options must be callable")
			butn = Button(
				text=txt,
				on_release=part,
				font_name=self.font_name,
				font_size=self.font_size,
				valign="center",
				halign="center",
			)
			if not binds["DialogMenu", "Button", id(butn), "size"]:
				binds["DialogMenu", "Button", id(butn), "size"].add(
					butn.fbind("size", butn.setter("text_size"))
				)
			layout.add_widget(butn)


class Dialog(BoxLayout):
	"""MessageBox with a DialogMenu beneath it.

	Set the properties ``message_kwargs`` and ``menu_kwargs``,
	respectively, to control them -- but you probably want
	to do that by returning a pair of dicts from an action
	in lisien.

	"""

	message_kwargs = DictProperty({})
	menu_kwargs = DictProperty({})

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		app = App.get_running_app()
		binds = app._bindings
		assert not binds["Dialog", id(self), "message_kwargs"]
		binds["Dialog", id(self), "message_kwargs"].add(
			self.fbind("message_kwargs", self._propagate_msg_kwargs)
		)
		assert not binds["Dialog", id(self), "menu_kwargs"]
		binds["Dialog", id(self), "menu_kwargs"].add(
			self.fbind("menu_kwargs", self._propagate_menu_kwargs)
		)
		self._propagate_msg_kwargs()
		self._propagate_menu_kwargs()
		app._unbinders.append(self.unbind_all)

	def unbind_all(self):
		binds = App.get_running_app()._bindings
		for uid in devour(binds["Dialog", id(self), "message_kwargs"]):
			self.unbind_uid("message_kwargs", uid)
		for uid in devour(binds["Dialog", id(self), "menu_kwargs"]):
			self.unbind_uid("menu_kwargs", uid)

	@partial(logwrap, section="Dialog")
	def _propagate_msg_kwargs(self, *_):
		if "msg" not in self.ids:
			Clock.schedule_once(self._propagate_msg_kwargs, 0)
			return
		kw = dict(self.message_kwargs)
		kw.setdefault(
			"background", "atlas://data/images/defaulttheme/textinput"
		)
		for k, v in kw.items():
			setattr(self.ids.msg, k, v)

	@partial(logwrap, section="Dialog")
	def _propagate_menu_kwargs(self, *_):
		if "menu" not in self.ids:
			Clock.schedule_once(self._propagate_menu_kwargs, 0)
			return
		kw = dict(self.menu_kwargs)
		kw.setdefault(
			"background",
			"atlas://data/images/defaulttheme/vkeyboard_background",
		)
		for k, v in kw.items():
			setattr(self.ids.menu, k, v)


class DialogLayout(FloatLayout):
	"""A layout, normally empty, that can generate dialogs

	To make dialogs, set my ``todo`` property to a list. It may contain:

	* Strings, which will be displayed with an "OK" button to dismiss them
	* Lists of pairs of strings and callables, which generate buttons with the
	  string on them that, when clicked, call the callable
	* Lists of pairs of dictionaries, which are interpreted as keyword
	  arguments to :class:`MessageBox` and :class:`DialogMenu`

	In place of a callable you can use the name of a function in my
	``usermod``, a Python module given by name. I'll import it when I need it.

	Needs to be instantiated with a lisien ``engine`` -- probably an
	:class:`EngineProxy`.

	"""

	dialog = ObjectProperty()
	todo = ListProperty()
	usermod = StringProperty("user")
	userpkg = StringProperty(None, allownone=True)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.dialog = Dialog()
		self._finalize()

	@logwrap(section="DialogLayout")
	def _finalize(self, *_):
		app = App.get_running_app()
		if not hasattr(app, "engine"):
			Clock.schedule_once(self._finalize, 0)
			return
		engine = app.engine
		todo = engine.universal.get("last_result")
		if isinstance(todo, list):
			self.todo = todo
		else:
			self.todo = []
		self.idx = engine.universal.get("last_result_idx", 0)
		engine.universal.connect(self._pull)
		if self.todo:
			self.advance_dialog()

	@logwrap(section="DialogLayout")
	def _pull(self, *_, key, value):
		if key == "last_result":
			self.todo = value if value and isinstance(value, list) else []
		elif key == "last_result_idx":
			self.idx = value if value and isinstance(value, int) else 0

	@logwrap(section="DialogLayout")
	def on_idx(self, *_):
		lidx = self.engine.universal.get("last_result_idx")
		if lidx is not None and lidx != self.idx:
			self.engine.universal["last_result_idx"] = self.idx
		Logger.debug(f"DialogLayout.idx = {self.idx}")

	@mainthread
	@logwrap(section="DialogLayout")
	def advance_dialog(self, after_ok=None, *args):
		"""Try to display the next dialog described in my ``todo``.

		:param after_ok: An optional callable. Will be called after the user clicks
		a dialog option--as well as after any game-specific code for that option
		has run.

		"""
		self.clear_widgets()
		try:
			Logger.debug(f"About to update dialog: {self.todo[0]}")
			self._update_dialog(self.todo.pop(0), after_ok)
		except IndexError:
			if after_ok is not None:
				after_ok()

	@mainthread
	@logwrap(section="DialogLayout")
	def _update_dialog(self, diargs, after_ok, **kwargs):
		if diargs is None:
			Logger.debug("DialogLayout: null dialog")
			return
		if isinstance(diargs, tuple) and diargs[0] == "stop":
			if len(diargs) == 1:
				Logger.debug("DialogLayout: null dialog")
				return
			diargs = diargs[1]
		dia = self.dialog
		# Simple text dialogs just tell the player something and let them click OK
		if isinstance(diargs, str):
			dia.message_kwargs = {"text": diargs}
			dia.menu_kwargs = {
				"options": [("OK", partial(self.ok, cb=after_ok))]
			}
		# List dialogs are for when you need the player to make a choice and don't care much
		# about presentation
		elif isinstance(diargs, list):
			dia.message_kwargs = {"text": "Select from the following:"}
			dia.menu_kwargs = {
				"options": list(
					map(partial(self._munge_menu_option, after_ok), diargs)
				)
			}
		# For real control of the dialog, you need a pair of dicts --
		# the 0th describes the message shown to the player, the 1th
		# describes the menu below
		elif isinstance(diargs, tuple):
			if len(diargs) != 2:
				raise TypeError(
					"Need a tuple of (message, menu) where message is a string, "
					"and menu is a dict or list describing options"
				)
			msgkwargs, mnukwargs = diargs
			if isinstance(msgkwargs, dict):
				dia.message_kwargs = msgkwargs
			elif isinstance(msgkwargs, str):
				dia.message_kwargs["text"] = msgkwargs
			else:
				raise TypeError("Message must be dict or str")
			if isinstance(mnukwargs, dict):
				mnukwargs["options"] = list(
					map(
						partial(self._munge_menu_option, after_ok),
						mnukwargs["options"],
					)
				)
				dia.menu_kwargs = mnukwargs
			elif isinstance(mnukwargs, (list, tuple)):
				dia.menu_kwargs["options"] = list(
					map(partial(self._munge_menu_option, after_ok), mnukwargs)
				)
			else:
				raise TypeError("Menu must be dict or list")
		else:
			raise TypeError(
				"Don't know how to turn {} into a dialog".format(type(diargs))
			)
		if dia.parent != self:
			Logger.debug("DialogLayout: Adding the dialog to the layout")
			self.add_widget(dia)
		else:
			Logger.debug("DialogLayout: Dialog is already in the layout")

	@logwrap(section="DialogLayout")
	def ok(self, *_, cb=None, cb2=None):
		"""Clear dialog widgets, call ``cb`` if provided, and advance the dialog queue

		``cb2`` will be called after advancing the dialog."""
		self.clear_widgets()
		if cb:
			cb()
		self.advance_dialog(after_ok=cb2)

	@logwrap(section="DialogLayout")
	def _lookup_func(self, funcname):
		from importlib import import_module

		if not hasattr(self, "_usermod"):
			self._usermod = import_module(self.usermod, self.userpkg)
		return getattr(self.usermod, funcname)

	@logwrap(section="DialogLayout")
	def _munge_menu_option(self, after_ok, option):
		if not isinstance(option, tuple):
			raise TypeError
		name, func = option
		if func is None:
			return name, partial(self.ok, cb2=after_ok)
		if callable(func):
			return name, partial(self.ok, cb=func, cb2=after_ok)
		if isinstance(func, tuple):
			fun = func[0]
			if isinstance(fun, str):
				fun = self._lookup_func(fun)
			args = func[1]
			if len(func) == 3:
				kwargs = func[2]
				func = partial(fun, *args, **kwargs)
			else:
				func = partial(fun, *args)
		if isinstance(func, str):
			func = self._lookup_func(func)
		return name, partial(self.ok, cb=func, cb2=after_ok)
