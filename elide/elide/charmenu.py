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
from kivy.clock import Clock, triggered
from kivy.properties import (
	BooleanProperty,
	ObjectProperty,
	ReferenceListProperty,
)
from kivy.uix.boxlayout import BoxLayout

from lisien.proxy import CharStatProxy

from .graph.arrow import GraphArrowWidget
from .util import dummynum, load_string_once


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
	def app(self):
		if not self.screen:
			raise AttributeError("No screen, therefore no app")
		return self.screen.app

	@property
	def engine(self):
		if not self.screen or not self.screen.app:
			raise AttributeError("Can't get engine from screen")
		return self.screen.app.engine

	def on_screen(self, *_):
		if not (self.screen and self.screen.boardview and self.screen.app):
			Clock.schedule_once(self.on_screen, 0)
			return
		self.forearrow = GraphArrowWidget(
			board=self.screen.boardview.board,
			origin=self.ids.emptyleft,
			destination=self.ids.emptyright,
		)
		self.ids.portaladdbut.add_widget(self.forearrow)
		self.ids.emptyleft.bind(pos=self.forearrow._trigger_repoint)
		self.ids.emptyright.bind(pos=self.forearrow._trigger_repoint)
		if self.reciprocal_portal:
			assert self.revarrow is None
			self.revarrow = GraphArrowWidget(
				board=self.screen.boardview.board,
				origin=self.ids.emptyright,
				destination=self.ids.emptyleft,
			)
			self.ids.portaladdbut.add_widget(self.revarrow)
			self.ids.emptyleft.bind(pos=self.revarrow._trigger_repoint)
			self.ids.emptyright.bind(pos=self.revarrow._trigger_repoint)
		self.bind(
			reciprocal_portal=self.screen.boardview.setter("reciprocal_portal")
		)

	def spot_from_dummy(self, dummy):
		if self.screen.boardview.parent != self.screen.mainview:
			return
		if dummy.collide_widget(self):
			return
		name = dummy.name
		self.screen.boardview.spot_from_dummy(dummy)
		graphboard = self.screen.graphboards[self.app.character_name]
		if name not in graphboard.spot:
			graphboard.add_spot(name)
		gridboard = self.screen.gridboards[self.app.character_name]
		if (
			name not in gridboard.spot
			and isinstance(name, tuple)
			and len(name) == 2
		):
			gridboard.add_spot(name)

	def pawn_from_dummy(self, dummy):
		name = dummy.name
		if not self.screen.mainview.children[0].pawn_from_dummy(dummy):
			return
		graphboard = self.screen.graphboards[self.app.character_name]
		if name not in graphboard.pawn:
			graphboard.add_pawn(name)
		gridboard = self.screen.gridboards[self.app.character_name]
		if (
			name not in gridboard.pawn
			and self.app.character.thing[name]["location"] in gridboard.spot
		):
			gridboard.add_pawn(name)

	def toggle_chars_screen(self, *_):
		"""Display or hide the list you use to switch between characters."""
		# TODO: update the list of chars
		self.app.chars.toggle()

	def toggle_rules(self, *_):
		"""Display or hide the view for constructing rules out of cards."""
		if self.app.manager.current != "rules" and not isinstance(
			self.app.selected_proxy, CharStatProxy
		):
			self.app.rules.entity = self.app.selected_proxy
			self.app.rules.rulebook = self.app.selected_proxy.rulebook
		if isinstance(self.app.selected_proxy, CharStatProxy):
			self.app.charrules.character = self.app.selected_proxy
			self.app.charrules.toggle()
		else:
			self.app.rules.toggle()

	def toggle_funcs_editor(self):
		"""Display or hide the text editing window for functions."""
		self.app.funcs.toggle()

	def toggle_strings_editor(self):
		self.app.strings.toggle()

	def toggle_spot_cfg(self):
		"""Show the dialog where you select graphics and a name for a place,
		or hide it if already showing.

		"""
		if self.app.manager.current == "spotcfg":
			dummyplace = self.screendummyplace
			self.ids.placetab.remove_widget(dummyplace)
			dummyplace.clear()
			if self.app.spotcfg.prefix:
				dummyplace.prefix = self.app.spotcfg.prefix
				dummyplace.num = (
					dummynum(self.app.character, dummyplace.prefix) + 1
				)
			if self.app.spotcfg.imgpaths:
				dummyplace.paths = self.app.spotcfg.imgpaths
			else:
				dummyplace.paths = ["atlas://rltiles/floor/floor-stone"]
			dummyplace.center = self.ids.placetab.center
			self.ids.placetab.add_widget(dummyplace)
		else:
			self.app.spotcfg.prefix = self.ids.dummyplace.prefix
		self.app.spotcfg.toggle()

	def toggle_pawn_cfg(self):
		"""Show or hide the pop-over where you can configure the dummy pawn"""
		if self.app.manager.current == "pawncfg":
			dummything = self.app.dummything
			self.ids.thingtab.remove_widget(dummything)
			dummything.clear()
			if self.app.pawncfg.prefix:
				dummything.prefix = self.app.pawncfg.prefix
				dummything.num = (
					dummynum(self.app.character, dummything.prefix) + 1
				)
			if self.app.pawncfg.imgpaths:
				dummything.paths = self.app.pawncfg.imgpaths
			else:
				dummything.paths = ["atlas://rltiles/base/unseen"]
			self.ids.thingtab.add_widget(dummything)
		else:
			self.app.pawncfg.prefix = self.ids.dummything.prefix
		self.app.pawncfg.toggle()

	def toggle_reciprocal(self):
		"""Flip my ``reciprocal_portal`` boolean, and draw (or stop drawing)
		an extra arrow on the appropriate button to indicate the
		fact.

		"""
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
			self.ids.emptyright.bind(pos=self.revarrow._trigger_repoint)
			self.ids.emptyleft.bind(pos=self.revarrow._trigger_repoint)
		else:
			if hasattr(self, "revarrow"):
				self.ids.portaladdbut.remove_widget(self.revarrow)
				self.revarrow = None

	def new_character(self, but):
		name = self.app.chars.ids.newname.text
		try:
			charn = self.app.engine.unpack(name)
		except (TypeError, ValueError):
			charn = name
		self.app.select_character(self.app.engine.new_character(charn))
		self.app.chars.ids.newname.text = ""
		self.app.chars.charsview.adapter.data = list(
			self.engine.character.keys()
		)
		Clock.schedule_once(self.toggle_chars_screen, 0.01)

	def on_dummyplace(self, *_):
		if not self.dummyplace.paths:
			self.dummyplace.paths = ["atlas://rltiles/floor.atlas/floor-stone"]

	def on_dummything(self, *_):
		if not self.dummything.paths:
			self.dummything.paths = ["atlas://rltiles/base.atlas/unseen"]

	@trigger
	def _trigger_deselect(self, *_):
		if hasattr(self.app.selection, "selected"):
			self.app.selection.selected = False
		self.app.selection = None


load_string_once("""
<CharMenu>:
	orientation: 'vertical'
	dummyplace: dummyplace
	dummything: dummything
	portaladdbut: portaladdbut
	portaldirbut: portaldirbut
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
""")
