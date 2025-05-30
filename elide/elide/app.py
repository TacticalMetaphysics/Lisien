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
from threading import Thread

from lisien.exc import OutOfTimelineError

if "KIVY_NO_ARGS" not in os.environ:
	os.environ["KIVY_NO_ARGS"] = "1"

from kivy.app import App
from kivy.clock import Clock, triggered
from kivy.logger import Logger
from kivy.properties import (
	AliasProperty,
	BooleanProperty,
	NumericProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.resources import resource_add_path, resource_find
from kivy.uix.screenmanager import NoTransition, ScreenManager

import elide
import elide.charsview
import elide.dialog
import elide.rulesview
import elide.screen
import elide.spritebuilder
import elide.statcfg
import elide.stores
import elide.timestream
from elide.graph.arrow import GraphArrow
from elide.graph.board import GraphBoard
from elide.grid.board import GridBoard
from lisien.proxy import (
	CharStatProxy,
	EngineProcessManager,
	PlaceProxy,
	ThingProxy,
	CharacterProxy,
)

resource_add_path(elide.__path__[0] + "/assets")
resource_add_path(elide.__path__[0] + "/assets/rltiles")
resource_add_path(elide.__path__[0] + "/assets/kenney1bit")


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

	def on_selection(self, *_):
		Logger.debug("App: {} selected".format(self.selection))

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

	def _get_character_name(self, *_):
		if self.character is None:
			return
		return self.character.name

	def _set_character_name(self, name):
		if self.character.name != name:
			self.character = self.engine.character[name]

	character_name = AliasProperty(
		_get_character_name, _set_character_name, bind=("character",)
	)

	def _pull_time(self, *_):
		if not hasattr(self, "engine"):
			Clock.schedule_once(self._pull_time, 0)
			return
		branch, turn, tick = self.engine._btt()
		self.branch = branch
		self.turn = turn
		self.tick = tick

	pull_time = trigger(_pull_time)

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

	def time_travel(self, branch, turn, tick=None):
		if hasattr(self, "_time_travel_thread"):
			return
		self.edit_locked = True
		self._time_travel_thread = Thread(
			target=self._really_time_travel, args=(branch, turn, tick)
		)
		self._time_travel_thread.start()

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
			del self._time_travel_thread

	def time_travel_to_tick(self, tick):
		self._time_travel_thread = Thread(
			target=self._really_time_travel_to_tick, args=(tick,)
		)
		self._time_travel_thread.start()

	def _update_from_time_travel(
		self, command, branch, turn, tick, result, **kwargs
	):
		(self.branch, self.turn, self.tick) = (branch, turn, tick)
		self.mainscreen.update_from_time_travel(
			command, branch, turn, tick, result, **kwargs
		)

	def set_tick(self, t):
		"""Set my tick to the given value, cast to an integer."""
		self.tick = int(t)

	def set_turn(self, t):
		"""Set the turn to the given value, cast to an integer"""
		self.turn = int(t)

	def select_character(self, char: CharacterProxy):
		"""Change my ``character`` to the selected character object if they
		aren't the same.

		"""
		if char == self.character:
			return
		if char.name not in self.mainscreen.graphboards:
			self.mainscreen.graphboards[char.name] = GraphBoard(character=char)
		if char.name not in self.mainscreen.gridboards:
			self.mainscreen.gridboards[char.name] = GridBoard(character=char)
		self.character = char
		self.selected_proxy = self._get_selected_proxy()
		self.engine.eternal["boardchar"] = char.name

	def build_config(self, config):
		"""Set config defaults"""
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
		config.write()

	def build(self):
		self.icon = "icon_24px.png"
		config = self.config

		if config["elide"]["debugger"] == "yes":
			import pdb

			pdb.set_trace()

		self.manager = ScreenManager(transition=NoTransition())
		if config["elide"]["inspector"] == "yes":
			from kivy.core.window import Window
			from kivy.modules import inspector

			inspector.create_inspector(Window, self.manager)

		self.mainmenu = elide.menu.MainMenuScreen(toggle=self._toggler("main"))
		self.manager.add_widget(self.mainmenu)
		if self.immediate_start:
			self._add_screens()
			self.manager.current = "mainscreen"
			Clock.schedule_once(self.start_game, 0)
		else:
			self.update_root_viewport()
		return self.manager

	@triggered(0.1)
	def update_root_viewport(self, *_):
		self.root_window.update_viewport()

	def _pull_lang(self, *_, **kwargs):
		self.strings.language = kwargs["language"]

	def _pull_chars(self, *_, **__):
		self.chars.names = list(self.engine.character)

	def _pull_time_from_signal(self, *_, then, now):
		self.branch, self.turn, self.tick = now
		self.mainscreen.ids.turnscroll.value = self.turn

	def start_subprocess(self, path=None, *_):
		"""Start the lisien core and get a proxy to it

		Must be called before ``init_board``

		"""
		if hasattr(self, "procman") and hasattr(self.procman, "engine_proxy"):
			raise ChildProcessError("Subprocess already running")
		config = self.config
		enkw = {
			"logger": Logger,
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
		self.engine = engine = self.procman.start(path, **enkw)
		Logger.debug("Got EngineProxy")
		self.pull_time()
		Logger.debug("Pulled time")

		self.engine.time.connect(self._pull_time_from_signal, weak=False)
		self.engine.character.connect(self._pull_chars, weak=False)

		self.strings.store = self.engine.string
		Logger.debug("EngineProxy is ready")
		return engine

	trigger_start_subprocess = trigger(start_subprocess)

	def init_board(self, *_):
		"""Get the board widgets initialized to display the game state

		Must be called after start_subprocess

		"""
		self.chars.names = char_names = list(self.engine.character)
		for name in char_names:
			if name not in self.mainscreen.graphboards:
				self.mainscreen.graphboards[name] = GraphBoard(
					character=self.engine.character[name]
				)
			if name not in self.mainscreen.gridboards:
				self.mainscreen.gridboards[name] = GridBoard(
					character=self.engine.character[name]
				)
		if "boardchar" in self.engine.eternal:
			self.select_character(
				self.engine.character[self.engine.eternal["boardchar"]]
			)

	def _toggler(self, screenname):
		def tog(*_):
			if self.manager.current == screenname:
				self.manager.current = "mainscreen"
			else:
				self.manager.current = screenname

		return tog

	def _add_screens(self, *_):
		toggler = self._toggler
		config = self.config

		pawndata = json.loads(config["elide"]["thing_graphics"])
		custom_pawns = resource_find("custom_pawn_imgs/custom.atlas")
		if custom_pawns:
			pawndata = [
				["Custom pawns", "custom_pawn_imgs/custom.atlas"]
			] + pawndata

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

		self.statcfg = elide.statcfg.StatScreen(toggle=toggler("statcfg"))

		self.rules = elide.rulesview.RulesScreen(toggle=toggler("rules"))

		self.charrules = elide.rulesview.CharacterRulesScreen(
			character=self.character, toggle=toggler("charrules")
		)
		self.bind(character=self.charrules.setter("character"))

		self.chars = elide.charsview.CharactersScreen(
			toggle=toggler("chars"), new_board=self.new_board
		)
		self.bind(character_name=self.chars.setter("character_name"))

		def chars_push_character_name(*_):
			self.unbind(character_name=self.chars.setter("character_name"))
			self.character_name = self.chars.character_name
			self.bind(character_name=self.chars.setter("character_name"))

		self.chars.push_character_name = chars_push_character_name

		self.strings = elide.stores.StringsEdScreen(toggle=toggler("strings"))

		self.funcs = elide.stores.FuncsEdScreen(
			name="funcs", toggle=toggler("funcs")
		)

		self.bind(selected_proxy=self.statcfg.setter("proxy"))

		self.timestream = elide.timestream.TimestreamScreen(
			name="timestream", toggle=toggler("timestream")
		)

		self.mainscreen = elide.screen.MainScreen(
			use_kv=config["elide"]["user_kv"] == "yes",
			play_speed=int(config["elide"]["play_speed"]),
		)
		if self.mainscreen.statlist:
			self.statcfg.statlist = self.mainscreen.statlist
		self.mainscreen.bind(statlist=self.statcfg.setter("statlist"))
		self.bind(
			selection=self.refresh_selected_proxy,
			character=self.refresh_selected_proxy,
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
		):
			self.manager.add_widget(wid)

	def _remove_screens(self):
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
			wid = getattr(self, widname)
			self.manager.remove_widget(wid)

	def start_game(self, *_, name=None, cb=None):
		if hasattr(self, "engine"):
			Logger.error("Already started the game")
			raise RuntimeError("Already started the game")
		game_name = name or self.game_name
		if self.game_name != game_name:
			self.game_name = game_name
		gamedir = os.path.join(self.prefix, game_name)
		self._add_screens()
		self.manager.current = "mainscreen"
		engine = self.engine = self.start_subprocess(gamedir)
		if "boardchar" in engine.eternal:
			self.select_character(
				engine.character[engine.eternal["boardchar"]]
			)
		elif "physical" in engine.character:
			self.select_character(engine.character["physical"])
		elif engine.character:
			self.select_character(next(iter(engine.character.values())))
		else:
			Logger.warning("No characters yet")
		self.mainscreen.populate()
		self.init_board()
		if cb:
			cb()
		return engine

	def close_game(self, *_, cb=None):
		self.mainmenu.invalidate_popovers()
		self.manager.current = "main"
		if hasattr(self, "procman"):
			self.procman.shutdown()
		if hasattr(self, "engine"):
			del self.engine
		else:
			Logger.debug("ElideApp: already closed")
			return  # already closed

		self._copy_log_files()
		game_wd = os.path.join(self.prefix, self.game_name)
		archived = shutil.make_archive(
			self.game_name, "zip", str(game_wd), logger=Logger
		)
		archived_base = os.path.basename(archived)
		if os.path.exists(os.path.join(self.games_dir, archived_base)):
			os.remove(os.path.join(self.games_dir, archived_base))
		os.rename(archived, os.path.join(self.games_dir, archived_base))
		shutil.rmtree(game_wd)
		self._remove_screens()
		if cb:
			cb()

	def update_calendar(self, calendar, past_turns=1, future_turns=5):
		"""Fill in a calendar widget with actual simulation data"""
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

	def on_character_name(self, *_):
		if not hasattr(self, "engine"):
			Clock.schedule_once(self.on_character_name, 0)
			return
		self.engine.eternal["boardchar"] = self.engine.character[
			self.character_name
		]

	def on_character(self, *_):
		if not hasattr(self, "mainscreen"):
			Clock.schedule_once(self.on_character, 0)
			return
		if hasattr(self, "_oldchar"):
			self.mainscreen.graphboards[self._oldchar.name].unbind(
				selection=self.setter("selection")
			)
			self.mainscreen.gridboards[self._oldchar.name].unbind(
				selection=self.setter("selection")
			)
		self.selection = None
		self.mainscreen.graphboards[self.character.name].bind(
			selection=self.setter("selection")
		)
		self.mainscreen.gridboards[self.character.name].bind(
			selection=self.setter("selection")
		)

	@staticmethod
	def copy_to_shared_storage(
		filename: str,
		mimetype: str | None = None,
	) -> None:
		try:
			from android import autoclass, mActivity
			from android.permissions import request_permissions, Permission
		except ModuleNotFoundError:
			# "shared storage" is just the working directory
			shutil.copy(
				filename,
				os.path.join(os.path.curdir, os.path.basename(filename)),
			)
			return
		if mimetype is None:
			import mimetypes

			mimetype = mimetypes.guess_type(filename)[0]
		request_permissions([Permission.WRITE_EXTERNAL_STORAGE])
		collection = autoclass("android.os.Environment").DIRECTORY_DOCUMENTS
		root_uri = autoclass(
			"android.provider.MediaStore$Files"
		).getContentUri("external")
		context = mActivity.getApplicationContext()
		appinfo = context.getApplicationInfo()
		if appinfo.labelRes:
			name = context.getString(appinfo.labelRes)
		else:
			name = appinfo.nonLocalizedLabel.toString()
		basefn = os.path.basename(filename)
		dest_uri = (
			root_uri.buildUpon()
			.appendPath(collection)
			.appendPath(name)
			.appendPath(basefn)
			.build()
		)
		resolver = context.getContentResolver()
		try:
			writer = resolver.openOutputStream(dest_uri, "wt")
		except:
			cv = autoclass("android.content.ContentValues")()
			MediaStoreMediaColumns = autoclass(
				"android.provider.MediaStore$MediaColumns"
			)
			cv.put(MediaStoreMediaColumns.DISPLAY_NAME, basefn)
			cv.put(MediaStoreMediaColumns.MIME_TYPE, mimetype)
			cv.put(
				MediaStoreMediaColumns.RELATIVE_PATH,
				os.path.join(collection, name),
			)
			uri = resolver.insert(root_uri, cv)
			writer = resolver.openOutputStream(uri, "wt")
		reader = autoclass("java.io.FileInputStream")(filename)
		autoclass("android.os.FileUtils").copy(reader, writer)
		writer.flush()
		writer.close()
		reader.close()

	def _copy_log_files(self):
		try:
			from android.storage import app_storage_path

			log_dirs = {
				os.path.join(app_storage_path(), "app", ".kivy", "logs")
			}
		except ModuleNotFoundError:
			log_dirs = set()
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
					self.copy_to_shared_storage(
						os.path.join(log_dir, logfile),
						mimetype="text/plain",
					)
			Logger.info(
				f"ElideApp: copied log files from {log_dir}"
				f" to {os.path.join(self.prefix, self.game_name, 'logs')} and shared storage"
			)

	@triggered()
	def copy_log_files(self):
		self._copy_log_files()

	def on_pause(self):
		"""Sync the database with the current state of the game."""
		if hasattr(self, "engine"):
			self.engine.commit()
		if hasattr(self, "strings"):
			self.strings.save()
		if hasattr(self, "funcs"):
			self.funcs.save()
		self._copy_log_files()
		return True

	def on_stop(self, *largs):
		"""Sync the database, wrap up the game, and halt."""
		if hasattr(self, "stopped"):
			return
		self.stopped = True
		self.close_game()
		return True

	def delete_selection(self):
		"""Delete both the selected widget and whatever it represents."""
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

	def new_board(self, name):
		"""Make a graph for a character name, and switch to it."""
		char = self.engine.character[name]
		self.mainscreen.graphboards[name] = GraphBoard(character=char)
		self.mainscreen.gridboards[name] = GridBoard(character=char)
		self.character = char

	def on_edit_locked(self, *_):
		Logger.debug(
			"ELiDEApp: "
			+ ("edit locked" if self.edit_locked else "edit unlocked")
		)
