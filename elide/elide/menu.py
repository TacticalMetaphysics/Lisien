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
import shutil
import zipfile
from functools import partial

from kivy import Logger
from kivy.app import App
from kivy.clock import Clock, mainthread, triggered
from kivy.properties import ObjectProperty, OptionProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput

from .gen import GridGeneratorDialog
from .util import devour, load_kv, logwrap, store_kv


class MenuTextInput(TextInput):
	"""Special text input for setting the branch"""

	set_value = ObjectProperty()
	name = StringProperty()

	def __init__(self, **kwargs):
		"""Disable multiline, and bind ``on_text_validate`` to ``on_enter``"""
		kwargs["multiline"] = False
		super().__init__(**kwargs)

	def on_name(self, *_):
		app = App.get_running_app()
		app._bindings[type(self).__name__, self.name, "on_text_validate"].add(
			self.fbind("on_text_validate", self.on_enter)
		)
		app._unbinders.append(self.unbind_all)

	def unbind_all(self):
		for uid in devour(
			App.get_running_app()._bindings[
				type(self).__name__, self.name, "on_text_validate"
			]
		):
			self.unbind_uid("on_text_validate", uid)

	@logwrap(section="MenuTextInput")
	def on_enter(self, *_):
		"""Call the setter and blank myself out so that my hint text shows
		up. It will be the same you just entered if everything's
		working.

		"""
		if self.text == "":
			return
		self.set_value(Clock.get_time(), self.text)
		self.text = ""
		self.focus = False

	@logwrap(section="MenuTextInput")
	def on_focus(self, *args):
		"""If I've lost focus, treat it as if the user hit Enter."""
		if not self.focus:
			self.on_enter(*args)

	@logwrap(section="MenuTextInput")
	def on_text_validate(self, *_):
		"""Equivalent to hitting Enter."""
		self.on_enter()


class MenuIntInput(MenuTextInput):
	"""Special text input for setting the turn or tick"""

	@logwrap(section="MenuIntInput")
	def insert_text(self, s, from_undo=False):
		"""Natural numbers only."""
		return super().insert_text(
			"".join(c for c in s if c in "0123456789"), from_undo
		)


class GeneratorButton(Button):
	pass


class WorldStartConfigurator(BoxLayout):
	"""Give options for how to initialize the world state"""

	generator_type = OptionProperty("none", options=["none", "grid"])
	dismiss = ObjectProperty()
	init_board = ObjectProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		binds = App.get_running_app()._bindings
		self.grid_config = GridGeneratorDialog()
		self.generator_dropdown = DropDown()

		def select_txt(btn):
			self.generator_dropdown.select(btn.text)

		for opt in ["None", "Grid"]:
			self.generator_dropdown.add_widget(
				GeneratorButton(text=opt, on_release=select_txt)
			)
		binds_on_select = binds[
			"WorldStartConfigurator", "generator_dropdown", "on_select"
		]
		while binds_on_select:
			self.unbind_uid("on_select", binds_on_select.pop())
		binds["WorldStartConfigurator", "generator_dropdown", "on_select"].add(
			self.fbind("on_select", self.select_generator_type)
		)

	@logwrap(section="WorldStartConfigurator")
	def select_generator_type(self, instance, value):
		self.ids.drop.text = value
		if value == "none":
			self.ids.controls.clear_widgets()
			self.generator_type = "none"
		elif value == "Grid":
			self.ids.controls.clear_widgets()
			self.ids.controls.add_widget(self.grid_config)
			self.grid_config.size = self.ids.controls.size
			self.grid_config.pos = self.ids.controls.pos
			self.generator_type = "grid"


class GamePickerModal(ModalView):
	headline = StringProperty()

	def __init__(self, **kwargs):
		load_kv("elide.menu")
		super().__init__(**kwargs)

	@logwrap(section="GamePickerModal")
	def _decompress_and_start(self, game_file_path, game, *_):
		app = App.get_running_app()
		game_name = (
			os.path.basename(game_file_path)
			.removesuffix(".lisien")
			.removesuffix(".zip")
		)
		app.game_name = game_name
		play_dir = str(app.play_path)
		if os.path.exists(play_dir):
			# Likely left over from a failed run of Elide
			shutil.rmtree(play_dir)
		if hasattr(self, "_decompress"):
			Logger.debug(
				f"GamePickerModal: unpacking {game_file_path} to {play_dir}"
			)
			self._decompress(game_file_path, play_dir)
			archive_path = None
		else:
			archive_path = game_file_path
		Clock.schedule_once(
			partial(
				app.start_game,
				name=game,
				archive_path=archive_path,
			),
			0.001,
		)
		self.dismiss(force=True)


class GameExporterModal(GamePickerModal):
	path = StringProperty()

	def on_open(self):
		app = App.get_running_app()
		self.path = str(app.games_path)

	@triggered()
	@logwrap(section="GameExporterModal")
	def pick(self, game, *_):
		import shutil
		from tempfile import TemporaryDirectory

		from lisien.engine import Engine

		app = App.get_running_app()
		with TemporaryDirectory() as td:
			shutil.unpack_archive(
				str(os.path.join(app.games_path, game + ".zip")), td
			)
			connect_string = None
			if "world.sqlite3" in os.listdir(td):
				connect_string = f"sqlite:///{td}/world.sqlite3"
			Logger.debug(f"GameExporterModal: about to export {game}")
			with Engine(
				td,
				workers=0,
				keyframe_on_close=False,
				connect_string=connect_string,
			) as eng:
				if hasattr(app, "_ss"):
					eng._shared_storage = app._ss
				eng.export(game)
		self.dismiss()

	@logwrap(section="GameExporterModal")
	def regen(self):
		if "game_list" not in self.ids:
			return
		self.ids.game_list.regen()


class GameImporterModal(GamePickerModal):
	@triggered()
	@logwrap(section="GameImporterModal")
	def pick(self, selection, *_):
		if not selection:
			return
		if len(selection) > 1:
			raise RuntimeError(
				"That file picker is supposed to be single select"
			)
		uri = selection[0]
		if isinstance(uri, str):
			uri_s = uri
		else:
			uri_s = uri.toString()
		if os.path.isdir(uri_s):
			return
		try:
			game_name = os.path.basename(uri).removesuffix(".lisien")
			self._decompress_and_start(uri, game_name)
		except (
			NotADirectoryError,
			FileNotFoundError,
			FileExistsError,
			zipfile.error,
		) as err:
			Logger.error(repr(err))
			modal = ModalView()
			error_box = BoxLayout(orientation="vertical")
			error_box.add_widget(Label(text=repr(err), font_size=80))
			error_box.add_widget(Button(text="OK", on_release=modal.dismiss))
			modal.add_widget(error_box)
			modal.open()

	@logwrap(section="GameImporterModal")
	def on_pre_open(self, *_):
		try:
			from android.storage import primary_external_storage_path

			Logger.error(
				"GameImporterModal: running on Android, where it won't work"
			)
			path = primary_external_storage_path()
			self._android = True
		except ImportError:
			path = App.get_running_app().prefix
			self._android = False
		if not hasattr(self, "_file_chooser"):
			self._file_chooser = FileChooserIconView(path=path)
			self.ids.chooser_goes_here.add_widget(self._file_chooser)


class GameLoaderModal(GamePickerModal):
	path = StringProperty()

	def on_open(self, *_):
		app = App.get_running_app()
		self.path = str(app.games_path)

	@triggered()
	@logwrap(section="GameLoaderModal")
	def pick(self, game, *_):
		app = App.get_running_app()
		games_path = str(app.games_path)
		if os.path.isfile(games_path):
			raise RuntimeError(
				"You put a file where I want to keep the games directory",
				app.games_dir,
			)
		if not os.path.exists(games_path):
			os.makedirs(app.games_path)
		if game + ".zip" in os.listdir(games_path):
			game_file_path = str(os.path.join(games_path, game + ".zip"))
			if not zipfile.is_zipfile(game_file_path):
				raise RuntimeError("Game format invalid", game_file_path)
		else:
			raise RuntimeError("Invalid game name", game)
		self.clear_widgets()
		self.add_widget(Label(text="Please wait...", font_size=80))
		Clock.schedule_once(
			partial(self._decompress_and_start, game_file_path, game), 0.05
		)

	_decompress = staticmethod(shutil.unpack_archive)


class GameList(RecycleView):
	picker = ObjectProperty()
	path = StringProperty()
	name = StringProperty()

	def on_open(self, *_):
		app = App.get_running_app()
		self.path = str(app.games_path)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		app = App.get_running_app()
		binds = app._bindings
		self._trigger_regen = Clock.create_trigger(self.regen)
		for att in ("picker", "path"):
			assert not binds["GameList", id(self), att]
			binds["GameList", id(self), att].add(
				self.fbind(att, self._trigger_regen)
			)
		app._unbinders.append(self.unbind_all)

	def unbind_all(self):
		binds = App.get_running_app()._bindings
		for att in ("picker", "path"):
			for uid in devour(binds["GameList", id(self), att]):
				self.unbind_uid(att, uid)

	@logwrap(section="GameList")
	def regen(self, *_):
		if not self.picker:
			Logger.debug("GameList: awaiting picker")
			Clock.schedule_once(self.regen, 0)
			return
		if not os.path.isdir(self.path):
			Logger.error(
				f"GameList: Can't list games at non-directory {self.path}"
			)
			return
		Logger.debug(f"GameList: listing games in {self.path}")
		self.data = [
			{
				"text": game[:-4],
				"on_release": partial(self.picker.pick, game[:-4]),
			}
			for game in filter(
				lambda game: game[-4:] == ".zip" and not game.startswith("."),
				os.listdir(self.path or "."),
			)
		]
		Logger.debug(f"GameList: generated {len(self.data)} entries")


class NewGameModal(ModalView):
	path = StringProperty()

	@triggered()
	@logwrap(section="NewGameModal")
	def validate_and_start(self, *_):
		game_name = self.ids.game_name.text
		self.ids.game_name.text = ""
		if not game_name:
			self.ids.game_name.hint_text = "Must be nonempty"
			return
		app = App.get_running_app()
		if os.path.isdir(app.games_dir):
			games = [
				fn.removesuffix(".zip") for fn in os.listdir(app.games_dir)
			]
		else:
			os.makedirs(app.games_dir)
			games = []
		if game_name in games:
			self.ids.game_name.hint_text = "Name already taken"
			return
		game_archive_path = os.path.join(app.games_dir, game_name + ".zip")
		game_dir_path = os.path.join(app.prefix, game_name)
		can_start = False
		try:
			zipfile.ZipFile(game_archive_path, "w").close()
			os.makedirs(game_dir_path)
			can_start = True
		except Exception as ex:
			self.ids.game_name.hint_text = repr(ex)
		finally:
			if os.path.isfile(game_archive_path):
				os.remove(game_archive_path)
		if can_start and (
			self.ids.worldstart.generator_type.lower() == "none"
			or self.ids.worldstart.grid_config.validate()
		):
			self.clear_widgets()
			self.add_widget(Label(text="Please wait...", font_size=80))
			self.canvas.ask_update()
			if os.path.exists(app.prefix) and any(
				fn not in {".", ".."} for fn in os.listdir(app.prefix)
			):
				app.close_game()
			Clock.schedule_once(partial(self._really_start, game_name), 0.05)

	def on_dismiss(self, *_):
		binds = App.get_running_app()._bindings
		world_on_select = binds[
			"WorldStartConfigurator", "generator_dropdown", "on_select"
		]
		while world_on_select:
			self.ids.worldstart.unbind_uid("on_select", world_on_select.pop())

	@logwrap(section="NewGameModal")
	def _really_start(self, game_name, *_):
		app = App.get_running_app()
		worldstart = self.ids.worldstart
		if worldstart.generator_type == "grid":
			app.start_game(
				name=game_name,
				cb=lambda: worldstart.grid_config.generate(app.engine),
			)
		else:
			app.start_game(name=game_name)
		app.select_character(app.engine.character["physical"])
		self.dismiss()


def trigger(func: callable) -> callable:
	return triggered()(func)


class MainMenuScreen(Screen):
	toggle = ObjectProperty()

	@trigger
	@logwrap(section="MainMenuScreen")
	def new_game(self, *_):
		if not hasattr(self, "_popover_new_game"):
			self._popover_new_game = NewGameModal()
		self._popover_new_game.open()

	@trigger
	@logwrap(section="MainMenuScreen")
	def load_game(self, *_):
		if not hasattr(self, "_popover_load_game"):
			self._popover_load_game = GameLoaderModal(
				headline="Pick game to load"
			)
		self._popover_load_game.open()

	@trigger
	@logwrap(section="MainMenuScreen")
	def import_game(self, *_):
		try:
			import android
			from androidstorage4kivy import Chooser, SharedStorage

			Logger.debug("Using Android system file chooser")

			if not hasattr(self, "_system_file_chooser"):
				self._ss = SharedStorage()
				self._system_file_chooser = Chooser(
					self._copy_from_shared_and_start_game
				)
			self._system_file_chooser.choose_content(
				"application/octet-stream"
			)
		except ImportError as err:
			Logger.debug(repr(err))
			Logger.debug("Using Kivy file chooser")
			if not hasattr(self, "_popover_import_game"):
				self._popover_import_game = GameImporterModal(
					headline="Pick .lisien game to import"
				)
			self._popover_import_game.open()

	@mainthread
	@logwrap(section="MainMenuScreen")
	def _copy_from_shared_and_start_game(self, files):
		game_file_path = self._ss.copy_from_shared(files[0])
		if not game_file_path.endswith(".lisien"):
			return
		game = str(os.path.basename(game_file_path).removesuffix(".lisien"))
		self._please_wait = ModalView()
		self._please_wait.add_widget(
			Label(text="Please wait...", font_size=80)
		)
		self._please_wait.open()
		Clock.schedule_once(
			partial(self._unpack_and_open, game_file_path, game),
			0.05,
		)

	def _unpack_and_open(self, game_file_path, game, *_):
		app = App.get_running_app()
		app.game_name = game
		app.start_game(
			name=game,
			archive_path=game_file_path,
			cb=self._please_wait.dismiss,
		)

	@trigger
	@logwrap(section="MainMenuScreen")
	def export_game(self, *_):
		if not hasattr(self, "_popover_export_game"):
			self._popover_export_game = GameExporterModal(
				headline="Pick game to export"
			)
		self._popover_export_game.regen()
		self._popover_export_game.open()

	@trigger
	@logwrap(section="MainMenuScreen")
	def invalidate_popovers(self, *_):
		if hasattr(self, "_popover_new_game"):
			del self._popover_new_game
		if hasattr(self, "_popover_load_game"):
			del self._popover_load_game
		if hasattr(self, "_popover_export_game"):
			del self._popover_export_game


kv = """
<GameList>:
	viewclass: 'Button'
	RecycleBoxLayout:
		default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        size_hint_y: None
        orientation: 'vertical'
"""
store_kv(__name__, kv)
