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
import os
from functools import partial
from threading import Thread

from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.properties import (
	BooleanProperty,
	DictProperty,
	NumericProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager

import lisien.proxy

from .dialog import DialogLayout
from .graph.board import GraphBoard, GraphBoardView
from .grid.board import GridBoard, GridBoardView

Factory.register("GraphBoard", GraphBoard)
Factory.register("GridBoard", GridBoard)
Factory.register("GraphBoardView", GraphBoardView)
Factory.register("GridBoardView", GridBoardView)
Factory.register("DialogLayout", DialogLayout)


class GameScreen(Screen):
	switch_screen = ObjectProperty()
	"""Method to set the ``screen`` attribute of the main :class:`kivy.uix.screenmanager.ScreenManager`"""
	disabled = BooleanProperty(False)
	"""If you bind your widgets' ``disabled`` to this, they will be disabled when a game command is in mid-execution"""

	@property
	def app(self):
		return App.get_running_app()

	@property
	def engine(self):
		return App.get_running_app().engine

	def disable_input(self, cb=None):
		"""Set ``self.disabled`` to ``True``, then call ``cb`` if provided

		:param cb: callback function for after disabling
		:return: ``None``

		"""
		self.disabled = True
		if cb:
			cb()

	def enable_input(self, cb=None):
		"""Call ``cb`` if provided, then set ``self.disabled`` to ``False``

		:param cb: callback function for before enabling
		:return: ``None``

		"""
		if cb:
			cb()
		self.disabled = False

	def wait_travel(self, character, thing, dest, cb=None):
		"""Schedule a thing to travel someplace, then wait for it to finish.

		:param character: name of the character
		:param thing: name of the thing that will travel
		:param dest: name of the place it will travel to
		:param cb: callback function for when it's done, optional
		:return: ``None``

		"""
		self.disable_input()
		self.app.wait_travel(
			character, thing, dest, cb=partial(self.enable_input, cb)
		)

	def wait_turns(self, turns, cb=None):
		"""Call ``self.app.engine.next_turn()`` ``n`` times, waiting ``self.app.turn_length`` in between

		Disables input for the duration.

		:param turns: number of turns to wait
		:param cb: function to call when done waiting, optional
		:return: ``None``

		"""
		self.disable_input()
		self.app.wait_turns(turns, cb=partial(self.enable_input, cb))

	def wait_command(self, start_func, turns=1, end_func=None):
		"""Call ``start_func``, wait ``turns``, and then call ``end_func`` if provided

		Disables input for the duration.

		:param start_func: function to call just after disabling input
		:param turns: number of turns to wait
		:param end_func: function to call just before re-enabling input
		:return: ``None``

		"""
		self.disable_input()
		start_func()
		self.app.wait_turns(turns, cb=partial(self.enable_input, end_func))

	def wait_travel_command(
		self,
		character,
		thing,
		dest,
		start_func,
		turns=1,
		end_func=lambda: None,
	):
		"""Schedule a thing to travel someplace and do something, then wait for it to finish.

		Input will be disabled for the duration.

		:param character: name of the character
		:param thing: name of the thing
		:param dest: name of the destination (a place)
		:param start_func: function to call when the thing gets to dest
		:param turns: number of turns to wait after start_func before re-enabling input
		:param end_func: optional. Function to call after waiting ``turns`` after start_func
		:return: ``None``

		"""
		self.disable_input()
		self.app.wait_travel_command(
			character,
			thing,
			dest,
			start_func,
			turns,
			partial(self.enable_input, end_func),
		)


class GameApp(App):
	modules = []
	turn_length = NumericProperty(0.5)
	branch = StringProperty("trunk")
	turn = NumericProperty(0)
	tick = NumericProperty(0)
	prefix = StringProperty(".")
	selection = ObjectProperty(allownone=True)
	engine_kwargs = DictProperty({})

	def wait_turns(self, turns, *, cb=None):
		"""Call ``self.engine.next_turn()`` ``turns`` times, waiting ``self.turn_length`` in between

		If provided, call ``cb`` when done.

		:param turns: number of turns to wait
		:param dt: unused, just satisfies the clock
		:param cb: callback function to call when done, optional
		:return: ``None``

		"""
		if hasattr(self, "_next_turn_thread"):
			Clock.schedule_once(partial(self.wait_turns, turns, cb=cb), 0)
			return
		if turns == 0:
			if cb:
				cb()
			return
		self.next_turn()
		turns -= 1
		Clock.schedule_once(
			partial(self.wait_turns, turns, cb=cb), self.turn_length
		)

	def wait_travel(self, character, thing, dest, cb=None):
		"""Schedule a thing to travel someplace, then wait for it to finish, and call ``cb`` if provided

		:param character: name of the character
		:param thing: name of the thing
		:param dest: name of the destination (a place)
		:param cb: function to be called when I'm done
		:return: ``None``

		"""
		self.wait_turns(
			self.engine.character[character].thing[thing].travel_to(dest),
			cb=cb,
		)

	def wait_command(self, start_func, turns=1, end_func=None):
		"""Call ``start_func``, and wait to call ``end_func`` after simulating ``turns`` (default 1)

		:param start_func: function to call before waiting
		:param turns: number of turns to wait
		:param end_func: function to call after waiting
		:return: ``None``

		"""
		start_func()
		self.wait_turns(turns, cb=end_func)

	def wait_travel_command(
		self, character, thing, dest, start_func, turns=1, end_func=None
	):
		"""Schedule a thing to travel someplace and do something, then wait for it to finish.

		:param character: name of the character
		:param thing: name of the thing
		:param dest: name of the destination (a place)
		:param start_func: function to call when the thing gets to dest
		:param turns: number of turns to wait after start_func before re-enabling input
		:param end_func: optional. Function to call after waiting ``turns`` after start_func
		:return: ``None``
		"""
		self.wait_travel(
			character,
			thing,
			dest,
			cb=partial(self.wait_command, start_func, turns, end_func),
		)

	def _pull_time(self, *_, then, now):
		self.branch, self.turn, self.tick = now

	def build(self):
		self.procman = lisien.proxy.EngineProcessManager()
		self.engine = self.procman.start(
			self.prefix,
			logger=Logger,
			loglevel=getattr(self, "loglevel", "debug"),
			install_modules=self.modules,
			**self.engine_kwargs,
		)
		self.branch, self.turn, self.tick = self.engine._btt()
		self.engine.time.connect(self._pull_time, weak=False)
		self.screen_manager = ScreenManager(transition=NoTransition())
		if hasattr(self, "inspector"):
			from kivy.core.window import Window
			from kivy.modules import inspector

			inspector.create_inspector(Window, self.screen_manager)
		return self.screen_manager

	def on_pause(self):
		"""Sync the database with the current state of the game."""
		self.engine.commit()
		self.config.write()

	def on_stop(self, *_):
		"""Sync the database, wrap up the game, and halt."""
		self.procman.shutdown()
		self.config.write()

	def _del_next_turn_thread(self, *_, **__):
		del self._next_turn_thread

	def next_turn(self, *_):
		"""Smoothly advance to the next turn in the simulation

		This uses a subthread to wait for lisien to finish simulating
		the turn and report the changes. The interface will remain responsive.

		If you're wiring up the interface, consider binding user
		input to `trigger_next_turn` instead, so that the user doesn't
		mistakenly go two or three turns into the future.

		"""
		if hasattr(self, "_next_turn_thread"):
			return
		self._next_turn_thread = Thread(
			target=self.engine.next_turn,
			kwargs={"cb": self._del_next_turn_thread},
		)
		self._next_turn_thread.start()

	trigger_next_turn = triggered(next_turn)
