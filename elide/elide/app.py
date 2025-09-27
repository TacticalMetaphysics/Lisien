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
"""Object to configure, start, and stop elide."""

import json
import os
import shutil
from collections import defaultdict
from functools import cached_property, partial
from threading import Thread
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

from lisien.exc import OutOfTimelineError

if "KIVY_NO_ARGS" not in os.environ:
	os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.logger import Logger
from kivy.properties import (
	AliasProperty,
	BooleanProperty,
	DictProperty,
	NumericProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.resources import resource_find
from kivy.uix.screenmanager import NoTransition, Screen, ScreenManager

import elide
import elide.charsview
import elide.dialog
import elide.logview
import elide.menu
import elide.rulesview
import elide.screen
import elide.spritebuilder
import elide.statcfg
import elide.stores
import elide.timestream
from elide.graph.arrow import GraphArrow
from elide.graph.board import GraphBoard
from elide.grid.board import GridBoard
from elide.util import devour, load_kv, logwrap
from lisien.proxy import (
	CharacterProxy,
	CharStatProxy,
	EngineProcessManager,
	PlaceProxy,
	ThingProxy,
)


def trigger(func):
	return triggered()(func)


class ElideApp(App):
	"""Extensible lisien Development Environment."""

	title = "elide"

	branch = StringProperty("trunk")
	turn = NumericProperty(0)
	tick = NumericProperty(0)
	character = ObjectProperty()
	selection = ObjectProperty(None, allownone=True)
	selected_proxy = ObjectProperty()
	selected_proxy_name = StringProperty("")
	statcfg = ObjectProperty()
	edit_locked = BooleanProperty(False)
	simulate_button_down = BooleanProperty(False)
	prefix = StringProperty()
	games_dir = StringProperty("games")
	logs_dir = StringProperty(None, allownone=True)
	game_name = StringProperty("game0")
	use_thread = BooleanProperty(False)
	connect_string = StringProperty(None, allownone=True)
	workers = NumericProperty(None, allownone=True)
	immediate_start = BooleanProperty(False)
	character_name = ObjectProperty()
	closed = BooleanProperty(True)
	stopped = BooleanProperty(False)

	@cached_property
	def _bindings(self) -> dict[tuple, set[int]]:
		class OnceSet(set):
			def add(self, __element):
				if __element in self:
					raise RuntimeError("Already bound", __element)
				super().add(__element)

		return defaultdict(OnceSet)

	@cached_property
	def _unbinders(self) -> list[Callable[[], None]]:
		return [self.unbind_all]

	def unbind_all(self):
		Logger.debug("ElideApp: unbinding everything")
		for uid in devour(self._bindings["ElideApp", "character"]):
			self.unbind_uid("character", uid)
		for uid in devour(self._bindings["ElideApp", "character_name"]):
			self.unbind_uid("character_name", uid)
		for uid in devour(
			self._bindings["CharactersScreen", "character_name"]
		):
			self.chars.unbind_uid("character_name", uid)
		for uid in devour(self._bindings["ElideApp", "selected_proxy"]):
			self.unbind_uid("selected_proxy", uid)
		for uid in devour(self._bindings["MainScreen", "statlist"]):
			self.mainscreen.unbind_uid("statlist", uid)
		for uid in devour(self._bindings["ElideApp", "selection"]):
			self.unbind_uid("selection", uid)
		for graphboard in self.mainscreen.graphboards:
			for uid in devour(
				self._bindings["GraphBoard", graphboard, "selection"]
			):
				self.mainscreen.graphboards[graphboard].unbind_uid(
					"selection", uid
				)
		for gridboard in self.mainscreen.gridboards:
			for uid in devour(
				self._bindings["GridBoard", gridboard, "selection"]
			):
				self.mainscreen.gridboards[gridboard].unbind_uid(
					"selection", uid
				)

	@cached_property
	def _togglers(self):
		return {}

	def _get_games_path(self):
		return os.path.join(self.prefix, self.games_dir)

	def _set_games_path(self, str_or_pair: str | tuple[str, str]):
		if isinstance(str_or_pair, str):
			a, b = os.path.split(str_or_pair)
		else:
			a, b = str_or_pair
		self.prefix, self.games_dir = a, b

	games_path = AliasProperty(
		_get_games_path, _set_games_path, bind=("prefix", "games_dir")
	)

	def _get_play_dir(self):
		return os.path.join(self.prefix, self.game_name)

	def _set_play_dir(self, str_or_pair: str | tuple[str, str]):
		if isinstance(str_or_pair, str):
			*a, b = str.split(os.path.sep)
			a = os.path.sep.join(a)
		else:
			a, b = str_or_pair
		self.prefix, self.game_name = a, b

	play_path = AliasProperty(
		_get_play_dir, _set_play_dir, bind=("prefix", "game_name")
	)

	@logwrap
	def on_selection(self, *_):
		Logger.debug("App: {} selected".format(self.selection))

	@logwrap
	def on_selected_proxy(self, *_):
		if hasattr(self.selected_proxy, "name"):
			self.selected_proxy_name = str(self.selected_proxy.name)
			return
		selected_proxy = self.selected_proxy
		assert hasattr(selected_proxy, "origin"), "{} has no origin".format(
			type(selected_proxy)
		)
		assert hasattr(selected_proxy, "destination"), (
			"{} has no destination".format(type(selected_proxy))
		)
		origin = selected_proxy.origin
		destination = selected_proxy.destination
		self.selected_proxy_name = (
			str(origin.name) + "->" + str(destination.name)
		)

	def on_character_name(self, _, name):
		if (
			hasattr(self, "engine")
			and name in self.engine.character
			and (not self.character or self.character.name != name)
		):
			self.character = self.engine.character[name]

	@logwrap
	def _pull_time(self, *_):
		if not hasattr(self, "engine"):
			Clock.schedule_once(self._pull_time, 0)
			return
		branch, turn, tick = self.engine._btt()
		self.branch = branch
		self.turn = turn
		self.tick = tick

	pull_time = trigger(_pull_time)

	@logwrap
	def _really_time_travel(self, branch, turn, tick):
		try:
			self.engine._set_btt(
				branch, turn, tick, cb=self._update_from_time_travel
			)
		except OutOfTimelineError as ex:
			Logger.warning(
				f"App: couldn't time travel to {(branch, turn, tick)}: "
				+ ex.args[0],
				exc_info=ex,
			)
			(self.branch, self.turn, self.tick) = (
				ex.branch_from,
				ex.turn_from,
				ex.tick_from,
			)
		finally:
			self.edit_locked = False
			del self._time_travel_thread

	@logwrap
	def time_travel(self, branch, turn, tick=None):
		if hasattr(self, "_time_travel_thread"):
			return
		self.edit_locked = True
		self._time_travel_thread = Thread(
			target=self._really_time_travel, args=(branch, turn, tick)
		)
		self._time_travel_thread.start()

	@logwrap
	def _really_time_travel_to_tick(self, tick):
		try:
			self.engine._set_btt(
				self.branch, self.turn, tick, cb=self._update_from_time_travel
			)
		except OutOfTimelineError as ex:
			Logger.warning(
				f"App: couldn't time travel to {(self.branch, self.turn, tick)}: "
				+ ex.args[0],
				exc_info=ex,
			)
			(self.branch, self.turn, self.tick) = (
				ex.branch_from,
				ex.turn_from,
				ex.tick_from,
			)
		finally:
			self.edit_locked = False
			if hasattr(self, "_time_travel_thread"):
				del self._time_travel_thread

	@logwrap
	def time_travel_to_tick(self, tick):
		self._time_travel_thread = Thread(
			target=self._really_time_travel_to_tick, args=(tick,)
		)
		self._time_travel_thread.start()

	@logwrap
	def _update_from_time_travel(
		self, command, branch, turn, tick, result, **kwargs
	):
		(self.branch, self.turn, self.tick) = (branch, turn, tick)
		self.mainscreen.update_from_time_travel(
			command, branch, turn, tick, result, **kwargs
		)

	@logwrap
	def set_tick(self, t):
		"""Set my tick to the given value, cast to an integer."""
		self.tick = int(t)

	@logwrap
	def set_turn(self, t):
		"""Set the turn to the given value, cast to an integer"""
		self.turn = int(t)

	@logwrap
	def select_character(self, char: CharacterProxy):
		"""Change my ``character`` to the selected character object if they
		aren't the same.

		"""
		if char == self.character:
			return
		if char.name not in self.mainscreen.graphboards:
			load_kv("elide.graph.board")
			self.mainscreen.graphboards[char.name] = GraphBoard(character=char)
		if char.name not in self.mainscreen.gridboards:
			load_kv("elide.grid.board")
			self.mainscreen.gridboards[char.name] = GridBoard(character=char)
		self.character = char
		self.selected_proxy = self._get_selected_proxy()
		self.engine.eternal["boardchar"] = char.name

	@logwrap
	def build_config(self, config):
		"""Set config defaults"""
		Logger.debug("ElideApp: build_config")
		for sec in "lisien", "elide":
			config.adddefaultsection(sec)
		config.setdefaults(
			"lisien",
			{
				"language": "eng",
				"logfile": "lisien.log",
				"loglevel": "debug",
				"replayfile": "",
				"connect_str": "",
			},
		)
		config.setdefaults(
			"elide",
			{
				"debugger": "no",
				"inspector": "no",
				"user_kv": "yes",
				"play_speed": "1",
				"thing_graphics": json.dumps(
					[
						("Kenney: 1 bit", "kenney1bit.atlas"),
						("RLTiles: Body", "base.atlas"),
						("RLTiles: Basic clothes", "body.atlas"),
						("RLTiles: Armwear", "arm.atlas"),
						("RLTiles: Legwear", "leg.atlas"),
						("RLTiles: Right hand", "hand1.atlas"),
						("RLTiles: Left hand", "hand2.atlas"),
						("RLTiles: Boots", "boot.atlas"),
						("RLTiles: Hair", "hair.atlas"),
						("RLTiles: Beard", "beard.atlas"),
						("RLTiles: Headwear", "head.atlas"),
					]
				),
				"place_graphics": json.dumps(
					[
						("Kenney: 1 bit", "kenney1bit.atlas"),
						("RLTiles: Dungeon", "dungeon.atlas"),
						("RLTiles: Floor", "floor.atlas"),
					]
				),
			},
		)

	def build(self):
		Logger.debug("ElideApp: build")
		self.icon = "icon_24px.png"
		config = self.config

		if config["elide"]["debugger"] == "yes":
			import pdb

			pdb.set_trace()

		self.manager = ScreenManager(transition=NoTransition())
		print(f"created screen manager with id {id(self.manager)}")
		if config["elide"]["inspector"] == "yes":
			from kivy.core.window import Window
			from kivy.modules import inspector

			inspector.create_inspector(Window, self.manager)
		self.mainmenu = elide.menu.MainMenuScreen(toggle=self.toggler("main"))
		self.manager.add_widget(self.mainmenu)
		if self.immediate_start:
			self.start_game()
		else:
			Clock.schedule_once(self.update_root_viewport, 3)
		return self.manager

	def update_root_viewport(self, *_):
		if not self.root_window:
			Clock.schedule_once(self.update_root_viewport, 0)
			return
		self.root_window.update_viewport()
		Logger.debug("ElideApp: updated root viewport")

	def _pull_lang(self, *_, **kwargs):
		self.strings.language = kwargs["language"]

	def _pull_chars(self, *_, **__):
		self.chars.names = list(self.engine.character)

	def _pull_time_from_signal(self, *_, then, now):
		self.branch, self.turn, self.tick = now
		self.mainscreen.ids.turnscroll.value = self.turn

	def start_subprocess(self, path=None, archive_path=None, *_):
		"""Start the lisien core and get a proxy to it

		Must be called before ``init_board``

		"""
		Logger.debug(f"ElideApp: start_subprocess(path={path!r})")
		if hasattr(self, "procman") and hasattr(self.procman, "engine_proxy"):
			raise ChildProcessError("Subprocess already running")
		self.closed = False
		config = self.config
		enkw = {
			"do_game_start": getattr(self, "do_game_start", False),
		}
		if s := (
			config["lisien"].get("connect_string") or self.connect_string
		):
			s = str(s)
			if "{prefix}" in s:
				enkw["connect_string"] = s.format(prefix=path)
			else:
				Logger.warning("{prefix} not found in " + s)
				enkw["connect_string"] = s
		elif os.path.isfile(os.path.join(path, "world.sqlite3")):
			enkw["connect_string"] = "sqlite:///" + str(
				os.path.join(path, "world.sqlite3")
			)
		workers = config["lisien"].get("workers", "")
		if workers:
			enkw["workers"] = workers
		if config["lisien"].get("logfile"):
			enkw["logfile"] = config["lisien"]["logfile"]
		if config["lisien"].get("loglevel"):
			enkw["loglevel"] = config["lisien"]["loglevel"]
		if config["lisien"].get("replayfile"):
			self._replayfile = open(config["lisien"].get("replayfile"), "at")
			enkw["replay_file"] = self._replayfile
		if self.workers is not None:
			enkw["workers"] = int(self.workers)
		elif workers := config["lisien"].get("workers"):
			enkw["workers"] = int(workers)
		if path:
			os.makedirs(path, exist_ok=True)
		Logger.debug(f"About to start EngineProcessManager with kwargs={enkw}")
		self.procman = EngineProcessManager(
			use_thread=self.use_thread,
		)
		if archive_path is None:
			self.engine = engine = self.procman.start(path, **enkw)
		else:
			self.engine = engine = self.procman.load_archive(
				archive_path, path, **enkw
			)
		Logger.debug("Got EngineProxy")
		if "boardchar" in engine.eternal:
			self.character_name = engine.eternal["boardchar"]
		elif self.character_name is not None:
			if self.character_name in engine.character:
				self.character = engine.character[self.character_name]
			else:
				self.character = engine.new_character(self.character_name)
		elif engine.character:
			self.character = next(iter(engine.character.values()))
		else:
			self.character = engine.new_character("physical")
		Logger.debug("Pulled character")
		self.pull_time()
		Logger.debug("Pulled time")

		self.engine.time.connect(self._pull_time_from_signal, weak=False)
		self.engine.character.connect(self._pull_chars, weak=False)

		self.strings.store = self.engine.string
		Logger.debug("EngineProxy is ready")
		return engine

	trigger_start_subprocess = trigger(start_subprocess)

	@logwrap
	def init_board(self, *_):
		"""Get the board widgets initialized to display the game state

		Must be called after start_subprocess

		"""
		self.chars.names = char_names = list(self.engine.character)
		Logger.debug(f"ElideApp: making grid boards for: {char_names}")
		for name in char_names:
			if name not in self.mainscreen.graphboards:
				load_kv("elide.graph.board")
				self.mainscreen.graphboards[name] = GraphBoard(
					character=self.engine.character[name]
				)
			if name not in self.mainscreen.gridboards:
				load_kv("elide.grid.board")
				self.mainscreen.gridboards[name] = GridBoard(
					character=self.engine.character[name]
				)
		if "boardchar" in self.engine.eternal:
			self.select_character(
				self.engine.character[self.engine.eternal["boardchar"]]
			)

	def toggler(self, screenname):
		"""Return a function that shows or hides a named screen"""
		if screenname in self._togglers:
			return self._togglers[screenname]

		def tog(*_):
			if self.manager.current == screenname:
				Logger.debug("ElideApp: toggling back to mainscreen")
				self.manager.current = "mainscreen"
			else:
				Logger.debug(f"ElideApp: toggling to {screenname}")
				self.manager.current = screenname

		self._togglers[screenname] = tog

		return tog

	def _add_screens(self, *_):
		Logger.debug("ElideApp: _add_screens")
		toggler = self.toggler
		config = self.config

		pawndata = json.loads(config["elide"]["thing_graphics"])
		custom_pawns = resource_find("custom_pawn_imgs/custom.atlas")
		if custom_pawns:
			pawndata = [
				["Custom pawns", "custom_pawn_imgs/custom.atlas"]
			] + pawndata

		load_kv("elide.spritebuilder")
		self.pawncfg = elide.spritebuilder.PawnConfigScreen(
			toggle=toggler("pawncfg"),
			data=pawndata,
		)

		spotdata = json.loads(config["elide"]["place_graphics"])
		custom_spots = resource_find("custom_spot_imgs/custom.atlas")
		if custom_spots:
			spotdata = [
				["Custom spots", "custom_spot_imgs/custom.atlas"]
			] + spotdata
		self.spotcfg = elide.spritebuilder.SpotConfigScreen(
			toggle=toggler("spotcfg"),
			data=spotdata,
		)
		for builder in (
			self.pawncfg.ids.dialog.ids.builder.__ref__(),
			self.spotcfg.ids.dialog.ids.builder.__ref__(),
		):
			self._bindings["SpriteBuilder", id(builder), "data"].add(
				builder.fbind("data", builder._trigger_update)
			)
			self._unbinders.append(builder.unbind_all)
			builder.update()

		load_kv("elide.statcfg")
		self.statcfg = elide.statcfg.StatScreen(toggle=toggler("statcfg"))
		load_kv("elide.rulesview")
		load_kv("elide.card")
		self.rules = elide.rulesview.RulesScreen(toggle=toggler("rules"))

		self.charrules = elide.rulesview.CharacterRulesScreen(
			character=self.character, toggle=toggler("charrules")
		)
		self._bindings["ElideApp", "character"].add(
			self.fbind("character", self.charrules.setter("character"))
		)
		Logger.debug("ElideApp: bound charrules setter")
		load_kv("elide.charsview")
		self.chars = elide.charsview.CharactersScreen(
			toggle=toggler("chars"), new_board=self.new_board
		)
		self._bindings["ElideApp", "character_name"].add(
			self.fbind("character_name", self.chars.setter("character_name"))
		)

		load_kv("elide.stores")
		self.strings = elide.stores.StringsEdScreen(toggle=toggler("strings"))

		self.funcs = elide.stores.FuncsEdScreen(
			name="funcs", toggle=toggler("funcs")
		)
		self._bindings["ElideApp", "selected_proxy"].add(
			self.fbind("selected_proxy", self.statcfg.setter("proxy"))
		)
		load_kv("elide.timestream")
		self.timestream = elide.timestream.TimestreamScreen(
			name="timestream", toggle=toggler("timestream")
		)
		load_kv("elide.logview")
		self.log_screen = elide.logview.LogScreen(
			name="log", toggle=toggler("log")
		)
		load_kv("elide.screen")
		load_kv("elide.charmenu")
		load_kv("elide.statcfg")
		load_kv("elide.stepper")
		load_kv("elide.dialog")
		self.mainscreen = elide.screen.MainScreen(
			use_kv=config["elide"]["user_kv"] == "yes",
			play_speed=int(config["elide"]["play_speed"]),
		)
		if self.mainscreen.statlist:
			self.statcfg.statlist = self.mainscreen.statlist
		self._bindings["MainScreen", "statlist"].add(
			self.mainscreen.fbind("statlist", self.statcfg.setter("statlist"))
		)
		self._bindings["ElideApp", "selection"].add(
			self.fbind("selection", self.refresh_selected_proxy)
		)
		self._bindings["ElideApp", "character"].add(
			self.fbind("character", self.refresh_selected_proxy)
		)
		for wid in (
			self.mainscreen,
			self.pawncfg,
			self.spotcfg,
			self.statcfg,
			self.rules,
			self.charrules,
			self.chars,
			self.strings,
			self.funcs,
			self.timestream,
			self.log_screen,
		):
			self.manager.add_widget(wid)

	def _remove_screens(self):
		Logger.debug("ElideApp: _remove_screens")
		if hasattr(self, "mainscreen"):
			Clock.unschedule(self.mainscreen.play)
		if not hasattr(self, "manager"):
			return
		for widname in (
			"mainscreen",
			"pawncfg",
			"spotcfg",
			"statcfg",
			"rules",
			"charrules",
			"chars",
			"strings",
			"funcs",
			"timestream",
		):
			if not hasattr(self, widname):
				continue
			wid = getattr(self, widname)
			if not isinstance(wid, Screen):
				Logger.info(
					f"ElideApp: not removing {widname} because it's just a mock"
				)
				continue
			self.manager.remove_widget(wid)
		self.manager.current = "main"

	def start_game(self, *_, name=None, archive_path=None, cb=None):
		Logger.debug(f"ElideApp: start_game(name={name!r}, cb={cb!r})")
		if hasattr(self, "engine"):
			Logger.error("Already started the game")
			raise RuntimeError("Already started the game")
		game_name = name or self.game_name
		if self.game_name != game_name:
			self.game_name = game_name
		os.makedirs(
			self.play_path,
			exist_ok=True,
		)
		self._add_screens()
		engine = self.engine = self.start_subprocess(
			self.play_path, archive_path
		)
		self.init_board()
		self.mainscreen.populate()
		if cb:
			cb()
		self.manager.current = "mainscreen"
		return engine

	def close_game(self, *_, cb=None):
		Logger.debug(f"ElideApp: close_game(cb={cb!r})")
		self.mainmenu.invalidate_popovers()
		if hasattr(self, "manager") and "main" in self.manager.screen_names:
			self.manager.current = "main"
		if hasattr(self, "procman"):
			self.procman.shutdown()
		if hasattr(self, "engine"):
			del self.engine
		else:
			Logger.debug("ElideApp: already closed")
			return  # already closed

		self._copy_log_files()
		pycache = os.path.join(self.play_path, "__pycache__")
		if os.path.exists(pycache):
			shutil.rmtree(pycache)
		archived_base = self.game_name + ".zip"
		os.makedirs(self.games_path, exist_ok=True)
		archived_abs = str(os.path.join(self.games_path, archived_base))
		if os.path.exists(archived_abs):
			os.remove(archived_abs)
		with ZipFile(archived_abs, "x", ZIP_DEFLATED) as zf:
			for fn in os.listdir(self.play_path):
				if os.path.isdir(os.path.join(self.play_path, fn)):
					for fnn in os.listdir(os.path.join(self.play_path, fn)):
						if os.path.isdir(
							os.path.join(self.play_path, fn, fnn)
						):
							for pqfn in os.listdir(
								os.path.join(self.play_path, fn, fnn)
							):
								zf.write(
									os.path.join(
										self.play_path, fn, fnn, pqfn
									),
									os.path.join(fn, fnn, pqfn),
								)
						else:
							zf.write(
								os.path.join(self.play_path, fn, fnn),
								os.path.join(fn, fnn),
							)
				else:
					zf.write(os.path.join(self.play_path, fn), fn)
		if not hasattr(self, "leave_game"):
			shutil.rmtree(self.play_path)
		self._remove_screens()
		if cb:
			cb()
		self.closed = True

	def update_calendar(self, calendar, past_turns=1, future_turns=5):
		"""Fill in a calendar widget with actual simulation data"""
		Logger.debug(
			f"ElideApp: update_calendar({calendar!r}, "
			f"past_turns={past_turns!r}, "
			f"future_turns={future_turns!r})"
		)
		startturn = self.turn - past_turns
		endturn = self.turn + future_turns
		stats = [
			stat
			for stat in self.selected_proxy
			if isinstance(stat, str)
			and not stat.startswith("_")
			and stat not in ("character", "name", "units", "wallpaper")
		]
		if "_config" in self.selected_proxy:
			stats.append("_config")
		if isinstance(self.selected_proxy, CharStatProxy):
			sched_entity = self.engine.character[self.selected_proxy.name]
		else:
			sched_entity = self.selected_proxy
		calendar.entity = sched_entity
		if startturn == endturn == self.turn:
			# It's the "calendar" that's actually just the current stats
			# of the selected entity, on the left side of elide
			schedule = {stat: [self.selected_proxy[stat]] for stat in stats}
		else:
			schedule = (
				self.engine.handle(
					"get_schedule",
					entity=sched_entity,
					stats=stats,
					beginning=startturn,
					end=endturn,
				),
			)
		calendar.from_schedule(schedule, start_turn=startturn)

	def _set_language(self, lang):
		self.engine.string.language = lang

	def _get_selected_proxy(self):
		Logger.debug("ElideApp: _get_selected_proxy")
		if self.selection is None:
			return self.character.stat
		elif hasattr(self.selection, "proxy"):
			return self.selection.proxy
		elif hasattr(self.selection, "origin") and hasattr(
			self.selection, "destination"
		):
			return self.character.portal[self.selection.origin.name][
				self.selection.destination.name
			]
		else:
			raise ValueError("Invalid selection: {}".format(self.selection))

	def refresh_selected_proxy(self, *_):
		self.selected_proxy = self._get_selected_proxy()

	def on_character(self, *_):
		if not hasattr(self, "mainscreen"):
			Logger.debug("ElideApp: got character before mainscreen")
			Clock.schedule_once(self.on_character, 0)
			return
		if (
			self.character.name not in self.mainscreen.graphboards
			or self.character.name not in self.mainscreen.gridboards
		):
			Logger.debug("ElideApp: got character before boards made")
			Clock.schedule_once(self.on_character, 0)
			return
		Logger.debug("ElideApp: changed character, deselecting")
		if self.character_name != self.character.name:
			self.character_name = self.character.name
		if hasattr(self, "_oldchar"):
			for uid in devour(
				self._bindings["GraphBoard", self._oldchar.name, "selection"]
			):
				self.mainscreen.graphboards[self._oldchar.name].unbind_uid(
					"selection", uid
				)
			for uid in devour(
				self._bindings["GridBoard", self._oldchar.name, "selection"]
			):
				self.mainscreen.gridboards[self._oldchar.name].unbind_uid(
					"selection", uid
				)
		self.selection = None
		self._bindings["GraphBoard", self.character.name, "selection"].add(
			self.mainscreen.graphboards[self.character.name].fbind(
				"selection", self.setter("selection")
			)
		)
		self._bindings["GridBoard", self.character.name, "selection"].add(
			self.mainscreen.gridboards[self.character.name].fbind(
				"selection", self.setter("selection")
			)
		)

	def copy_to_shared_storage(
		self,
		filename: str,
		mimetype: str | None = None,
	) -> None:
		Logger.debug(
			f"ElideApp: copy_to_shared_storage({filename!r}, {mimetype!r})"
		)
		try:
			from androidstorage4kivy import SharedStorage
		except ModuleNotFoundError:
			# "shared storage" is just the working directory
			try:
				shutil.copy(
					filename,
					os.path.join(os.path.curdir, os.path.basename(filename)),
				)
			except shutil.SameFileError:
				pass
			return
		if not hasattr(self, "_ss"):
			self._ss = SharedStorage()
		storage = self._ss
		storage.copy_to_shared(filename)

	def _copy_log_files(self):
		Logger.debug("ElideApp: _copy_log_files")
		try:
			from android.storage import app_storage_path
			from jnius import JavaException

			log_dirs = {
				os.path.join(app_storage_path(), "app", ".kivy", "logs")
			}
		except ModuleNotFoundError:
			log_dirs = set()
			JavaException = Exception
		for handler in Logger.handlers:
			if hasattr(handler, "log_dir"):
				log_dir = handler.log_dir
			elif hasattr(handler, "filename"):
				log_dir = os.path.dirname(handler.filename)
			elif hasattr(handler, "baseFilename"):
				log_dir = handler.baseFilename
				if not os.path.isdir(log_dir):
					log_dir = os.path.dirname(log_dir)
			else:
				Logger.error(
					f"ElideApp: handler {handler} (of type {type(handler)})"
					f"has neither log_dir nor filename nor baseFilename"
				)
				continue
			log_dirs.add(log_dir)
		os.makedirs(
			os.path.join(self.prefix, self.game_name, "logs"),
			exist_ok=True,
		)
		failed = []
		for log_dir in log_dirs:
			for logfile in os.listdir(log_dir):
				if logfile.endswith(".log") or (
					logfile.startswith("kivy_") and logfile.endswith(".txt")
				):
					shutil.copy(
						os.path.join(log_dir, logfile),
						str(
							os.path.join(
								self.prefix,
								self.game_name,
								"logs",
								logfile,
							)
						),
					)
					try:
						self.copy_to_shared_storage(
							os.path.join(log_dir, logfile),
							mimetype="text/plain",
						)
					except JavaException:
						failed.append(os.path.join(log_dir, logfile))
			if failed:
				Logger.error("Failed to copy log files: " + ", ".join(failed))
			else:
				Logger.info(
					f"ElideApp: copied log files from {log_dir}"
					f" to {os.path.join(self.prefix, self.game_name, 'logs')} and shared storage"
				)

	@triggered()
	def copy_log_files(self):
		self._copy_log_files()

	def on_pause(self):
		"""Sync the database with the current state of the game."""
		Logger.debug("ElideApp: pausing")
		if hasattr(self, "engine"):
			self.engine.commit()
			Logger.debug("ElideApp: committed")
		if hasattr(self, "strings"):
			self.strings.save()
			Logger.debug("ElideApp: saved strings")
		if hasattr(self, "funcs"):
			self.funcs.save()
			Logger.debug("ElideApp: saved funcs")
		self._copy_log_files()
		Logger.debug("ElideApp: paused")
		return True

	def on_resume(self):
		Logger.debug("ElideApp: resuming")
		self.update_root_viewport()
		return True

	def on_stop(self, *largs):
		"""Sync the database, wrap up the game, and halt."""
		if self.stopped:
			return
		Logger.debug("ElideApp: stopping")
		if hasattr(self, "funcs"):
			self.funcs.save()
		if hasattr(self, "engine"):
			self.close_game(cb=partial(self.setter("stopped"), 0.0, True))
		else:
			self.stopped = True
		for unbinder in self._unbinders:
			unbinder()
		for k, v in self._bindings.items():
			if v:
				raise RuntimeError("Still bound", k, v)
		return True

	def on_stopped(self, *_):
		if self.stopped:
			Logger.debug("ElideApp: stopped")

	def delete_selection(self):
		"""Delete both the selected widget and whatever it represents."""
		Logger.debug("ElideApp: delete_selection")
		selection = self.selection
		if selection is None:
			return
		if isinstance(selection, GraphArrow):
			self.mainscreen.boardview.board.rm_arrow(
				selection.origin.name, selection.destination.name
			)
			selection.character.portal[selection.origin.name][
				selection.destination.name
			].delete()
		elif isinstance(selection.proxy, PlaceProxy):
			charn = selection.board.character.name
			self.mainscreen.graphboards[charn].rm_spot(selection.name)
			gridb = self.mainscreen.gridboards[charn]
			if selection.name in gridb.spot:
				gridb.rm_spot(selection.name)
			selection.proxy.delete()
		else:
			assert isinstance(selection.proxy, ThingProxy)
			charn = selection.board.character.name
			self.mainscreen.graphboards[charn].rm_pawn(selection.name)
			self.mainscreen.gridboards[charn].rm_pawn(selection.name)
			selection.proxy.delete()
		self.selection = None
		Logger.debug("ElideApp: selection deleted")

	def new_board(self, name):
		"""Make a graph for a character name, and switch to it."""
		Logger.debug(f"ElideApp: new_board({name!r})")
		char = self.engine.character[name]
		self.mainscreen.graphboards[name] = GraphBoard(character=char)
		self.mainscreen.gridboards[name] = GridBoard(character=char)
		self.character = char
		Logger.debug("ElideApp: made new board for %s", name)

	def on_edit_locked(self, *_):
		Logger.debug(
			"ELiDEApp: "
			+ ("edit locked" if self.edit_locked else "edit unlocked")
		)
