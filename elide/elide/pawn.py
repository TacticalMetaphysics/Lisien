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
from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.properties import NumericProperty, ObjectProperty

from .util import logwrap


def trigger(func):
	return triggered()(func)


class PawnBehavior:
	"""Mix-in class for things in places represented graphically"""

	loc_name = ObjectProperty()
	default_image_paths = ["atlas://rltiles/base.atlas/unseen"]
	priority = NumericProperty()
	board = ObjectProperty()

	def __init__(self, **kwargs):
		if "thing" in kwargs:
			kwargs["proxy"] = kwargs["thing"]
			del kwargs["thing"]
		if "proxy" in kwargs:
			kwargs["loc_name"] = kwargs["proxy"]["location"]
		super().__init__(**kwargs)
		self.register_event_type("on_drop")

	@logwrap(section="PawnBehavior")
	def on_proxy(self, *args):
		self.loc_name = self.proxy["location"]

	@logwrap(section="PawnBehavior")
	def on_parent(self, *args):
		if not self.parent:
			Clock.schedule_once(self.on_parent, 0)
			return
		binds = App.get_running_app()._bindings
		self.board = self.parent.board
		self._relocate_binding = self.fbind("loc_name", self._trigger_relocate)
		binds[
			"Pawn",
			self.parent.board.character.name,
			self.proxy.name,
			"loc_name",
		].add(self._relocate_binding)
		if self.proxy:
			self._trigger_relocate()

	@logwrap(section="PawnBehavior")
	def finalize(self, initial=True):
		binds = App.get_running_app()._bindings
		if initial:
			self.loc_name = self.proxy["location"]
			self.priority = self.proxy.get("_priority", 0.0)
		self._push_loc_binding = self.fbind(
			"loc_name", self._trigger_push_location
		)
		binds[
			"Pawn",
			self.parent.board.character.name,
			self.proxy.name,
			"loc_name",
		].add(self._push_loc_binding)
		super().finalize(initial)

	@logwrap(section="PawnBehavior")
	def unfinalize(self):
		binds = App.get_running_app()._bindings
		binds[
			"Pawn",
			self.parent.board.character.name,
			self.proxy.name,
			"loc_name",
		].remove(self._push_loc_binding)
		self.unbind_uid("loc_name", self._push_loc_binding)
		del self._push_loc_binding
		super().unfinalize()

	@logwrap(section="PawnBehavior")
	def pull_from_proxy(self, *args):
		binds = App.get_running_app()._bindings
		super().pull_from_proxy(*args)
		relocate = False
		if self.loc_name != self.proxy["location"]:
			self.unfinalize()
			binds_here = binds[
				"Pawn",
				self.parent.board.character.name,
				self.proxy.name,
				"loc_name",
			]
			binds_here.remove(self._relocate_binding)
			self.unbind_uid("loc_name", self._relocate_binding)
			self.loc_name = self.proxy["location"]
			self._relocate_binding = self.fbind(
				"loc_name", self._trigger_relocate
			)
			binds_here.add(self._relocate_binding)
			self.finalize(initial=False)
			relocate = True
		if "_priority" in self.proxy:
			self.priority = self.proxy["_priority"]
		if relocate:
			self.relocate()

	@logwrap(section="PawnBehavior")
	def relocate(self, *args):
		if (
			not getattr(self, "_finalized", False)
			or not self.parent
			or not self.proxy
			or not self.proxy.exists
		):
			return
		try:
			location = self._get_location_wid()
		except KeyError:
			return
		if location != self.parent:
			if self.parent:
				self.parent.remove_widget(self)
			location.add_widget(self)

	_trigger_relocate = trigger(relocate)

	@logwrap(section="PawnBehavior")
	def on_priority(self, *args):
		if self.proxy["_priority"] != self.priority:
			self.proxy["_priority"] = self.priority
		self.parent.restack()

	@logwrap(section="PawnBehavior")
	def push_location(self, *args):
		if self.proxy["location"] != self.loc_name:
			self.proxy["location"] = self.loc_name

	_trigger_push_location = trigger(push_location)

	def _get_location_wid(self):
		return self.board.spot[self.loc_name]

	@logwrap(section="PawnBehavior")
	def on_touch_up(self, touch):
		if touch.grab_current is not self:
			return False
		for spot in self.board.spot.values():
			if self.collide_widget(spot) and spot.name != self.loc_name:
				new_spot = spot
				break
		else:
			new_spot = None

		self.dispatch("on_drop", new_spot)
		touch.ungrab(self)
		return True

	@logwrap(section="PawnBehavior")
	def on_drop(self, spot):
		parent = self.parent
		if spot:
			self.loc_name = self.proxy["location"] = spot.name
			parent.remove_widget(self)
			spot.add_widget(self)
		else:
			x, y = getattr(self, "rel_pos", (0, 0))
			self.pos = parent.x + x, parent.y + y
