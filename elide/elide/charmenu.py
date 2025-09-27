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
from functools import partial

from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.properties import (
	BooleanProperty,
	ObjectProperty,
	ReferenceListProperty,
)
from kivy.uix.boxlayout import BoxLayout

from lisien.proxy import CharStatProxy

from .graph.arrow import GraphArrowWidget
from .util import dummynum, logwrap, store_kv


def trigger(func):
	return triggered()(func)


class CharMenu(BoxLayout):
	screen = ObjectProperty()
	reciprocal_portal = BooleanProperty(True)
	revarrow = ObjectProperty(None, allownone=True)
	dummyplace = ObjectProperty()
	dummything = ObjectProperty()
	toggle_gridview = ObjectProperty()
	toggle_timestream = ObjectProperty()
	dummies = ReferenceListProperty(dummyplace, dummything)

	@property
	def engine(self):
		if not self.screen or not self.screen.app:
			raise AttributeError("Can't get engine from screen")
		return self.screen.app.engine

	@logwrap(section="CharMenu")
	def on_screen(self, *_):
		if not (
			self.screen
			and self.screen.boardview
			and self.screen.app
			and "emptyleft" in self.ids
			and "emptyright" in self.ids
		):
			Clock.schedule_once(self.on_screen, 0)
			return
		binds = App.get_running_app()._bindings
		self.forearrow = GraphArrowWidget(
			board=self.screen.boardview.board,
			origin=self.ids.emptyleft,
			destination=self.ids.emptyright,
		)
		self.ids.portaladdbut.add_widget(self.forearrow)
		assert not binds["CharMenu", "emptyleft", "fore"]
		binds["CharMenu", "emptyleft", "fore"].add(
			self.ids.emptyleft.fbind("pos", self.forearrow._trigger_repoint)
		)
		assert not binds["CharMenu", "emptyright", "fore"]
		binds["CharMenu", "emptyright", "fore"].add(
			self.ids.emptyright.fbind("pos", self.forearrow._trigger_repoint)
		)
		if self.reciprocal_portal:
			assert self.revarrow is None
			self.revarrow = GraphArrowWidget(
				board=self.screen.boardview.board,
				origin=self.ids.emptyright,
				destination=self.ids.emptyleft,
			)
			self.ids.portaladdbut.add_widget(self.revarrow)
			assert not binds["CharMenu", "emptyleft", "rev"]
			binds["CharMenu", "emptyleft", "rev"].add(
				self.ids.emptyleft.fbind("pos", self.revarrow._trigger_repoint)
			)
			assert not binds["CharMenu", "emptyright", "rev"]
			binds["CharMenu", "emptyright", "rev"].add(
				self.ids.emptyright.fbind(
					"pos", self.revarrow._trigger_repoint
				)
			)
		assert not binds["CharMenu", "reciprocal_portal"]
		binds["CharMenu", "reciprocal_portal"].add(
			self.fbind(
				"reciprocal_portal",
				self.screen.boardview.setter("reciprocal_portal"),
			)
		)

	@logwrap(section="CharMenu")
	def spot_from_dummy(self, dummy):
		if self.screen.boardview.parent != self.screen.mainview:
			return
		if dummy.collide_widget(self):
			return
		app = App.get_running_app()
		name = dummy.name
		self.screen.boardview.spot_from_dummy(dummy)
		graphboard = self.screen.graphboards[app.character_name]
		if name not in graphboard.spot:
			graphboard.add_spot(name)
		gridboard = self.screen.gridboards[app.character_name]
		if (
			name not in gridboard.spot
			and isinstance(name, tuple)
			and len(name) == 2
		):
			gridboard.add_spot(name)

	@logwrap(section="CharMenu")
	def pawn_from_dummy(self, dummy):
		name = dummy.name
		if not self.screen.mainview.children[0].pawn_from_dummy(dummy):
			return
		app = App.get_running_app()
		graphboard = self.screen.graphboards[app.character_name]
		if name not in graphboard.pawn:
			graphboard.add_pawn(name)
		gridboard = self.screen.gridboards[app.character_name]
		if (
			name not in gridboard.pawn
			and app.character.thing[name]["location"] in gridboard.spot
		):
			gridboard.add_pawn(name)

	@logwrap(section="CharMenu")
	def toggle_chars_screen(self, *_):
		"""Display or hide the list you use to switch between characters."""
		# TODO: update the list of chars
		App.get_running_app().chars.toggle()

	@logwrap(section="CharMenu")
	def toggle_rules(self, *_):
		"""Display or hide the view for constructing rules out of cards."""
		app = App.get_running_app()
		if app.manager.current != "rules" and not isinstance(
			app.selected_proxy, CharStatProxy
		):
			app.rules.entity = app.selected_proxy
			app.rules.rulebook = app.selected_proxy.rulebook
		if isinstance(app.selected_proxy, CharStatProxy):
			app.charrules.character = app.selected_proxy
			app.charrules.toggle()
		else:
			app.rules.toggle()

	@logwrap(section="CharMenu")
	def toggle_funcs_editor(self):
		"""Display or hide the text editing window for functions."""
		App.get_running_app().funcs.toggle()

	@logwrap(section="CharMenu")
	def toggle_strings_editor(self):
		App.get_running_app().strings.toggle()

	@logwrap(section="CharMenu")
	def toggle_spot_cfg(self):
		"""Show the dialog where you select graphics and a name for a place,
		or hide it if already showing.

		"""
		app = App.get_running_app()
		if app.manager.current == "spotcfg":
			dummyplace = self.screendummyplace
			self.ids.placetab.remove_widget(dummyplace)
			dummyplace.clear()
			if self.app.spotcfg.prefix:
				dummyplace.prefix = app.spotcfg.prefix
				dummyplace.num = dummynum(app.character, dummyplace.prefix) + 1
			if app.spotcfg.imgpaths:
				dummyplace.paths = app.spotcfg.imgpaths
			else:
				dummyplace.paths = ["atlas://rltiles/floor/floor-stone"]
			dummyplace.center = self.ids.placetab.center
			self.ids.placetab.add_widget(dummyplace)
		else:
			app.spotcfg.prefix = self.ids.dummyplace.prefix
		app.spotcfg.toggle()

	@logwrap(section="CharMenu")
	def toggle_pawn_cfg(self):
		"""Show or hide the pop-over where you can configure the dummy pawn"""
		app = App.get_running_app()
		if app.manager.current == "pawncfg":
			dummything = app.dummything
			self.ids.thingtab.remove_widget(dummything)
			dummything.clear()
			if app.pawncfg.prefix:
				dummything.prefix = app.pawncfg.prefix
				dummything.num = dummynum(app.character, dummything.prefix) + 1
			if app.pawncfg.imgpaths:
				dummything.paths = app.pawncfg.imgpaths
			else:
				dummything.paths = ["atlas://rltiles/base/unseen"]
			self.ids.thingtab.add_widget(dummything)
		else:
			app.pawncfg.prefix = self.ids.dummything.prefix
		app.pawncfg.toggle()

	@logwrap(section="CharMenu")
	def toggle_reciprocal(self):
		"""Flip my ``reciprocal_portal`` boolean, and draw (or stop drawing)
		an extra arrow on the appropriate button to indicate the
		fact.

		"""
		binds = App.get_running_app()._bindings
		self.reciprocal_portal = (
			self.screen.boardview.reciprocal_portal
		) = not self.screen.boardview.reciprocal_portal
		if self.screen.boardview.reciprocal_portal:
			assert self.revarrow is None
			self.revarrow = GraphArrowWidget(
				board=self.screen.boardview.board,
				origin=self.ids.emptyright,
				destination=self.ids.emptyleft,
			)
			self.ids.portaladdbut.add_widget(self.revarrow)
			if not binds["CharMenu", "emptyright", "rev"]:
				binds["CharMenu", "emptyright", "rev"].add(
					self.ids.emptyright.fbind(
						"pos", self.revarrow._trigger_repoint
					)
				)
			if not binds["CharMenu", "emptyleft", "rev"]:
				binds["CharMenu", "emptyleft", "rev"].add(
					self.ids.emptyleft.fbind(
						"pos", self.revarrow._trigger_repoint
					)
				)
		else:
			if hasattr(self, "revarrow"):
				self.ids.portaladdbut.remove_widget(self.revarrow)
				self.revarrow = None

	@logwrap(section="CharMenu")
	def new_character(self, but):
		app = App.get_running_app()
		name = app.chars.ids.newname.text
		try:
			charn = app.engine.unpack(name)
		except (TypeError, ValueError):
			charn = name
		app.select_character(self.app.engine.new_character(charn))
		app.chars.ids.newname.text = ""
		app.chars.charsview.adapter.data = list(self.engine.character.keys())
		Clock.schedule_once(self.toggle_chars_screen, 0.01)

	@logwrap(section="CharMenu")
	def on_dummyplace(self, *_):
		if not self.dummyplace.paths:
			self.dummyplace.paths = ["atlas://rltiles/floor.atlas/floor-stone"]

	@logwrap(section="CharMenu")
	def on_dummything(self, *_):
		if not self.dummything.paths:
			self.dummything.paths = ["atlas://rltiles/base.atlas/unseen"]

	@trigger
	@logwrap(section="CharMenu")
	def _trigger_deselect(self, *_):
		app = App.get_running_app()
		if hasattr(app.selection, "selected"):
			app.selection.selected = False
		app.selection = None

	def close_game(self, *_):
		App.get_running_app().close_game()


store_kv(
	__name__,
	"""
<CharMenu>:
	orientation: 'vertical'
	dummyplace: dummyplace
	dummything: dummything
	portaladdbut: portaladdbut
	portaldirbut: portaldirbut
	Button:
		text: 'Logs'
		on_release: app.log_screen.toggle()
	Button:
		text: 'Delete'
		disabled: app.edit_locked
		on_release: app.delete_selection()
	Button:
		id: timestreambut
		text: 'Timestream'
		disabled: app.edit_locked
		on_release: root.toggle_timestream()
	Button:
		id: gridviewbut
		text: 'Toggle grid'
		on_release: root.toggle_gridview()
	Button:
		text: 'Strings'
		disabled: app.edit_locked
		on_release: root.toggle_strings_editor()
	Button:
		text: 'Python'
		disabled: app.edit_locked
		on_release: root.toggle_funcs_editor()
	Button:
		text: 'Rules'
		id: rules_button
		disabled: app.edit_locked
		on_release: root.toggle_rules()
	Button:
		text: 'Characters'
		on_release: root.toggle_chars_screen()
	BoxLayout:
		Widget:
			id: placetab
			Dummy:
				id: dummyplace
				center: placetab.center
				prefix: 'place'
				disabled: app.edit_locked
				on_pos_up: root.spot_from_dummy(self)
		Button:
			text: 'cfg'
			disabled: app.edit_locked
			on_release: root.toggle_spot_cfg()
	BoxLayout:
		Widget:
			id: thingtab
			Dummy:
				id: dummything
				center: thingtab.center
				prefix: 'thing'
				disabled: app.edit_locked
				on_pos_up: root.pawn_from_dummy(self)
		Button:
			text: 'cfg'
			disabled: app.edit_locked
			on_release: root.toggle_pawn_cfg()
	BoxLayout:
		orientation: 'vertical'
		ToggleButton:
			id: portaladdbut
			disabled: app.edit_locked
			on_state: root._trigger_deselect()
			Widget:
				id: emptyleft
				center_x: portaladdbut.x + portaladdbut.width / 3
				center_y: portaladdbut.center_y
				size: (0, 0)
			Widget:
				id: emptyright
				center_x: portaladdbut.right - portaladdbut.width / 3
				center_y: portaladdbut.center_y
				size: (0, 0)
		Button:
			id: portaldirbut
			text: 'One-way' if root.reciprocal_portal else 'Two-way'
			disabled: app.edit_locked
			on_release: root.toggle_reciprocal()
	Button:
		text: 'Quit'
		id: quit_button
		on_release: root.close_game()
""",
)
