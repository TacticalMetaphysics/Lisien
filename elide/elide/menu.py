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
import shutil
from functools import partial
import os
import zipfile

from kivy import Logger
from kivy.app import App
from kivy.clock import Clock, triggered
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
from sqlalchemy import and_, column, bindparam

from .gen import GridGeneratorDialog
from .util import load_string_once


class MenuTextInput(TextInput):
	"""Special text input for setting the branch"""

	set_value = ObjectProperty()

	def __init__(self, **kwargs):
		"""Disable multiline, and bind ``on_text_validate`` to ``on_enter``"""
		kwargs["multiline"] = False
		super().__init__(**kwargs)
		self.bind(on_text_validate=self.on_enter)

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

	def on_focus(self, *args):
		"""If I've lost focus, treat it as if the user hit Enter."""
		if not self.focus:
			self.on_enter(*args)

	def on_text_validate(self, *_):
		"""Equivalent to hitting Enter."""
		self.on_enter()


class MenuIntInput(MenuTextInput):
	"""Special text input for setting the turn or tick"""

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
		self.grid_config = GridGeneratorDialog()
		self.generator_dropdown = DropDown()

		def select_txt(btn):
			self.generator_dropdown.select(btn.text)

		for opt in ["None", "Grid"]:
			self.generator_dropdown.add_widget(
				GeneratorButton(text=opt, on_release=select_txt)
			)
		self.generator_dropdown.bind(on_select=self.select_generator_type)

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

	def _decompress_and_start(self, game_file_path, game, *_):
		app = App.get_running_app()
		game_dir = str(os.path.join(app.prefix, game))
		if os.path.exists(game_dir):
			# Likely left over from a failed run of Elide
			shutil.rmtree(game_dir)
		shutil.unpack_archive(game_file_path, game_dir)
		Clock.schedule_once(partial(app.start_game, name=game), 0.001)
		self.dismiss(force=True)


class GameExporterModal(GamePickerModal):
	@triggered()
	def pick(self, game, *_):
		app = App.get_running_app()
		app.copy_to_shared_storage(
			os.path.join(app.games_dir, game + ".zip"), "application/zip"
		)
		self.dismiss()


class GameImporterModal(GamePickerModal):
	def _pick(self, uri):
		app = App.get_running_app()
		try:
			from android import mActivity, autoclass, cast
			from android.storage import primary_external_storage_path
		except ImportError:
			game_name = os.path.basename(uri).removesuffix(".zip")
			Clock.schedule_once(
				partial(
					self._decompress_and_start,
					uri,
					game_name,
				)
			)
			return

		def copyfile(fn):
			root_dir = str(
				autoclass("android.os.Environment")
				.getExternalStorageDirectory()
				.getAbsolutePath()
			)
			if not os.path.isdir(root_dir):
				raise NotADirectoryError(root_dir)
			abspath = os.path.join(root_dir, fn)
			if not os.path.isfile(abspath):
				raise FileNotFoundError(abspath)
			game_name = os.path.basename(abspath).removesuffix(".zip")
			dest = os.path.join(app.prefix, game_name)
			if os.path.exists(dest):
				shutil.rmtree(dest)
			shutil.unpack_archive(abspath, dest)
			return

		if uri.startswith("file://"):
			return copyfile(uri.removeprefix("file://"))
		MediaStoreFiles = autoclass("android.provider.MediaStore$Files")
		MediaStoreMediaColumns = autoclass(
			"android.provider.MediaStore$MediaColumns"
		)
		root_uri = MediaStoreFiles.getContentUri("external")
		context = mActivity.getApplicationContext()
		select_stmt = and_(
			column(MediaStoreMediaColumns.DISPLAY_NAME) == bindparam("a"),
			column(MediaStoreMediaColumns.RELATIVE_PATH) == bindparam("b"),
		)
		select_stmt.stringify_dialect = "sqlite"
		select_s = str(select_stmt)
		Logger.debug(
			"GameImporterModal: looking for URI using the query: %s", select_s
		)
		args = [
			os.path.basename(uri),
			os.path.dirname(uri)
			.replace(primary_external_storage_path(), "")
			.strip("/")
			+ "/",
		]
		Logger.debug(
			"GameImporterModal: with the arguments: %s",
			", ".join(map(repr, args)),
		)
		resolver = context.getContentResolver()
		cursor = resolver.query(root_uri, None, select_s, args, None)
		if not cursor:
			raise FileNotFoundError(uri)
		while cursor.moveToNext():
			idx = cursor.getColumnIndex(MediaStoreMediaColumns.DISPLAY_NAME)
			file_name = cursor.getString(idx)
			Logger.debug("GameImporterModal: file %d. %s", idx, file_name)
			if file_name == os.path.basename(uri):
				id_ = cursor.getLong(
					cursor.getColumnIndex(MediaStoreMediaColumns._ID)
				)
				uri = autoclass("android.content.ContentUris").withAppendedId(
					root_uri, id_
				)
				break
		else:
			cursor.close()
			raise FileNotFoundError(uri)
		if uri.getScheme().lower() == "file":
			return copyfile(str(uri.getEncodedPath()))
		display_name = MediaStoreMediaColumns.DISPLAY_NAME
		name_idx = cursor.getColumnIndex(display_name)
		cursor.moveToFirst()
		file_name = cursor.getString(name_idx)
		if not file_name.endswith(".zip"):
			raise zipfile.error("not a zip file", file_name)
		os.makedirs(app.games_dir, exist_ok=True)
		dest = os.path.join(app.games_dir, file_name)
		if os.path.exists(dest):
			raise FileExistsError(
				"Already have a game named " + file_name.removesuffix(".zip")
			)
		try:
			reader = resolver.openInputStream(uri)
			writer = autoclass("java.io.FileOutputStream")(dest)
			autoclass("android.os.FileUtils").copy(reader, writer)
			writer.flush()
			writer.close()
			reader.close()
		except Exception as ex:
			Logger.error("GameImporterModal: %s", ex)
			raise
		cursor.close()
		Clock.schedule_once(
			partial(
				self._decompress_and_start,
				dest,
				file_name.removesuffix(".zip"),
			)
		)

	@triggered()
	def pick(self, selection, *_):
		if not selection:
			return
		if len(selection) > 1:
			raise RuntimeError(
				"That file picker is supposed to be single select"
			)
		uri = selection[0]
		if os.path.isdir(uri):
			return
		try:
			self._pick(uri)
		except (
			NotADirectoryError,
			FileNotFoundError,
			FileExistsError,
			zipfile.error,
		) as err:
			modal = ModalView()
			error_box = BoxLayout(orientation="vertical")
			error_box.add_widget(Label(text=repr(err), font_size=80))
			error_box.add_widget(Button(text="OK", on_release=modal.dismiss))
			modal.add_widget(error_box)
			modal.open()

	def on_pre_open(self, *_):
		try:
			from android.storage import primary_external_storage_path

			path = primary_external_storage_path()
			self._android = True
		except ImportError:
			path = os.getcwd()
			self._android = False
		if not hasattr(self, "_file_chooser"):
			self._file_chooser = FileChooserIconView(path=path)
			self.ids.chooser_goes_here.add_widget(self._file_chooser)


class GameLoaderModal(GamePickerModal):
	@triggered()
	def pick(self, game, *_):
		app = App.get_running_app()
		if os.path.isfile(app.games_dir):
			raise RuntimeError(
				"You put a file where I want to keep the games directory",
				app.games_dir,
			)
		if not os.path.exists(app.games_dir):
			os.makedirs(app.games_dir)
		if game + ".zip" in os.listdir(app.games_dir):
			game_file_path = str(os.path.join(app.games_dir, game + ".zip"))
			if not zipfile.is_zipfile(game_file_path):
				raise RuntimeError("Game format invalid", game_file_path)
		else:
			raise RuntimeError("Invalid game name", game)
		self.clear_widgets()
		self.add_widget(Label(text="Please wait...", font_size=80))
		Clock.schedule_once(
			partial(self._decompress_and_start, game_file_path, game), 0
		)


class GameList(RecycleView):
	picker = ObjectProperty()
	path = StringProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._trigger_regen = Clock.create_trigger(self.regen)
		self.bind(picker=self._trigger_regen, path=self._trigger_regen)

	def regen(self, *_):
		if not os.path.isdir(self.path):
			return
		if not self.picker:
			Logger.debug("GameList: awaiting picker")
			Clock.schedule_once(self.on_path, 0)
			return
		self.data = [
			{
				"text": game.removesuffix(".zip"),
				"on_release": partial(
					self.picker.pick, game.removesuffix(".zip")
				),
			}
			for game in filter(
				lambda game: not game.startswith("."), os.listdir(self.path)
			)
		]


class NewGameModal(ModalView):
	@triggered()
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
			self._really_start(game_name)

	def _really_start(self, game_name, *_):
		app = App.get_running_app()
		worldstart = self.ids.worldstart
		if worldstart.generator_type == "grid":
			app.start_game(
				name=game_name,
				cb=lambda: worldstart.grid_config.generate(app.engine),
			)
		else:
			app.start_game(
				name=game_name, cb=lambda: app.engine.add_character("physical")
			)
		app.select_character(app.engine.character["physical"])
		self.dismiss()


def trigger(func: callable) -> callable:
	return triggered()(func)


class MainMenuScreen(Screen):
	toggle = ObjectProperty()

	@trigger
	def new_game(self, *_):
		if not hasattr(self, "_popover_new_game"):
			self._popover_new_game = NewGameModal()
		self._popover_new_game.open()

	@trigger
	def load_game(self, *_):
		if not hasattr(self, "_popover_load_game"):
			self._popover_load_game = GameLoaderModal(
				headline="Pick game to load"
			)
		self._popover_load_game.open()

	@trigger
	def import_game(self, *_):
		if not hasattr(self, "_popover_import_game"):
			self._popover_import_game = GameImporterModal(
				headline="Pick zipped game to import"
			)
		self._popover_import_game.open()

	@trigger
	def export_game(self, *_):
		if not hasattr(self, "_popover_export_game"):
			self._popover_export_game = GameExporterModal(
				headline="Pick game to export"
			)
		self._popover_export_game.open()

	@trigger
	def invalidate_popovers(self, *_):
		if hasattr(self, "_popover_new_game"):
			del self._popover_new_game
		if hasattr(self, "_popover_load_game"):
			del self._popover_load_game
		if hasattr(self, "_popover_export_game"):
			del self._popover_export_game


load_string_once("""
#: import os os
<GeneratorButton>:
	size_hint_y: None
	font_size: 50
	height: self.texture_size[1] + 10
<WorldStartConfigurator>:
	orientation: 'vertical'
	init_board: app.init_board
	starter: app.start_subprocess
	Label:
		text: 'Generate an initial map?'
	Button:
		id: drop
		text: 'None'
		on_release: root.generator_dropdown.open(drop)
	Widget:
		id: controls
<NewGameModal>:
	size_hint_x: 0.6
	BoxLayout:
		orientation: 'vertical'
		Label:
			text: 'Generate an initial map?'
			font_size: 50
			size_hint_y: None
		WorldStartConfigurator:
			id: worldstart
			dismiss: root.dismiss
		Label:
			text: 'Please name your game'
			font_size: 50
			size_hint_y: None
		BoxLayout:
			orientation: 'horizontal'
			size_hint_y: 0.2
			TextInput:
				id: game_name
				multiline: False
				size_hint_x: 0.8
			Button:
				text: 'Start'
				on_release: root.validate_and_start()
		
<MainMenuScreen>:
	name: 'main'
	BoxLayout:
		orientation: 'horizontal'
		Widget:
			size_hint_x: 0.2
		BoxLayout:
			orientation: 'vertical'
			Label:
				text: 'Elide'
				font_size: 80
				size_hint_y: None
				height: self.texture_size[1]
			Button:
				text: 'New game'
				font_size: 50
				on_release: root.new_game()
			Button:
				text: 'Load game'
				font_size: 50
				on_release: root.load_game()
			Button:
				text: 'Import game'
				font_size: 50
				on_release: root.import_game()
			Button:
				text: 'Export game'
				font_size: 50
				on_release: root.export_game()
		Widget:
			size_hint_x: 0.2
<GameList>:
	viewclass: 'Button'
	RecycleBoxLayout:
		default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        size_hint_y: None
        orientation: 'vertical'
<GameLoaderModal>:
	size_hint_x: 0.6
	BoxLayout:
		orientation: 'vertical'
		Label:
			text: root.headline
			size_hint_y: 0.1
			font_size: self.height
		GameList:
			path: app.games_dir
			picker: root
			size_hint_y: 0.8
		Button:
			text: 'Cancel'
			on_release: root.dismiss()
			size_hint_y: 0.1
			font_size: self.height
<GameImporterModal>:
	size_hint_x: 0.6
	BoxLayout:
		orientation: 'vertical'
		Label:
			text: root.headline
			size_hint_y: 0.1
			font_size: self.height
		RelativeLayout:
			id: chooser_goes_here
		BoxLayout:
			orientation: 'horizontal'
			size_hint_y: 0.1
			Button:
				text: 'Cancel'
				on_release: root.dismiss()
			Button:
				text: 'OK'
				on_release: root.pick(root._file_chooser.selection)
""")
