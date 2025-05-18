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
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.uix.textinput import TextInput

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


class GameExporterModal(GamePickerModal):
	@triggered()
	def pick(self, game, *_):
		try:
			from android import autoclass, mActivity
			from android.permissions import request_permissions, Permission
		except ModuleNotFoundError:
			app = App.get_running_app()
			shutil.copytree(
				str(os.path.join(app.games_dir, game + ".zip")),
				os.path.join(os.getcwd(), game + ".zip"),
			)
			return
		app = App.get_running_app()
		request_permissions([Permission.WRITE_EXTERNAL_STORAGE])
		root_uri = str(
			autoclass("android.provider.MediaStore$Files")
			.getContentUri("external")
			.toString()
		)
		context = mActivity.getApplicationContext()
		resolver = context.getContentResolver()
		writer = resolver.openOutputStream(
			os.path.join(root_uri, game + ".zip")
		)
		reader = autoclass("java.io.FileInputStream")(
			str(os.path.join(app.games_dir, game + ".zip"))
		)
		autoclass("android.os.FileUtils").copy(reader, writer)
		writer.flush()
		writer.close()
		reader.close()


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

	def _decompress_and_start(self, game_file_path, game, *_):
		app = App.get_running_app()
		with zipfile.ZipFile(
			game_file_path, "r", compression=zipfile.ZIP_DEFLATED
		) as zipf:
			# should validate that it has what we expect...
			zipf.extractall(app.prefix)
		if any(d not in {".", ".."} for d in os.listdir(app.prefix)):
			app.close_game()
		app.game_name = game
		app.start_game(cb=partial(self.dismiss, force=True))


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
		game_path = os.path.join(app.games_dir, game_name + ".zip")
		can_start = False
		try:
			zipfile.ZipFile(game_path, "w").close()
			can_start = True
		except Exception as ex:
			self.ids.game_name.hint_text = repr(ex)
		finally:
			if os.path.isfile(game_path):
				os.remove(game_path)
		if can_start and (
			self.ids.worldstart.generator_type is None
			or self.ids.worldstart.grid_config.validate()
		):
			self.clear_widgets()
			self.add_widget(Label(text="Please wait...", font_size=80))
			self.canvas.ask_update()
			if os.path.exists(app.prefix) and any(
				not fn.startswith(".") for fn in os.listdir(app.prefix)
			):
				Clock.schedule_once(
					partial(
						app.close_game,
						partial(self._really_start, game_name),
					),
					0,
				)
			else:
				Clock.schedule_once(partial(self._really_start, game_name), 0)

	def _really_start(self, game_name, *_):
		app = App.get_running_app()
		app.game_name = game_name
		worldstart = self.ids.worldstart
		if worldstart.generator_type == "grid":
			engine = app.start_subprocess()
			worldstart.grid_config.generate(engine)
			app.init_board()
			app.manager.current = "mainscreen"
		else:
			app.start_game()
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
<GamePickerModal>:
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
""")
