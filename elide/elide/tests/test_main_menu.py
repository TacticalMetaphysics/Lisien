import os
import shutil

import pytest
from kivy.tests.common import UnitTestTouch
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.textinput import TextInput

from ..app import ElideApp
from .util import advance_frames, idle_until


def test_new_game(elide_app_main_menu):
	app: ElideApp = elide_app_main_menu
	manager = app.manager
	new_game_button: Button = manager.current_screen.ids.new_game_button
	x, y = new_game_button.center
	touch = UnitTestTouch(x=x, y=y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: hasattr(app.mainmenu, "_popover_new_game"),
		100,
		"Never created new game popover modal",
	)
	modal = app.mainmenu._popover_new_game
	idle_until(lambda: modal._is_open, 100, "Never opened game popover modal")
	game_name_input: TextInput = modal.ids.game_name
	game_name_input.text = "not a real game"
	start_new_game_button: Button = modal.ids.start_new_game_button
	x, y = start_new_game_button.center
	touch = UnitTestTouch(x=x, y=y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: manager.current == "mainscreen",
		1000,
		"Never switched to 'mainscreen' screen",
	)


@pytest.fixture
def zipped_kobold(prefix, kobold_sim):
	yield shutil.make_archive("kobold", "zip", prefix, prefix)


@pytest.fixture
def zipped_kobold_in_games_dir(prefix, zipped_kobold):
	archive_name = os.path.basename(zipped_kobold)
	games_dir = os.path.join(prefix, "games")
	shutil.move(
		zipped_kobold,
		os.path.join(games_dir, archive_name),
	)
	assert archive_name in os.listdir(games_dir)


def test_load_game(zipped_kobold_in_games_dir, elide_app_main_menu):
	app = elide_app_main_menu
	manager = app.manager
	load_game_button: Button = manager.current_screen.ids.load_game_button
	x, y = load_game_button.center
	touch = UnitTestTouch(x=x, y=y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: hasattr(manager.current_screen, "_popover_load_game"),
		100,
		"Never created game selection popover",
	)
	modal = manager.current_screen._popover_load_game
	idle_until(
		lambda: modal._is_open, 100, "Never opened game selection modal"
	)
	idle_until(lambda: "game_list" in modal.ids, 100, "Never built game list")
	game_list = modal.ids.game_list
	idle_until(lambda: game_list.data, 100, "Never got saved game data")
	idle_until(
		lambda: game_list._viewport, 100, "Never got game list viewport"
	)
	button = game_list._viewport.children[0]
	assert button.text == "kobold"
	x, y = game_list.to_parent(*button.center)
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: manager.current == "mainscreen",
		100,
		"Never switched to mainscreen",
	)


def test_import_game(kobold_sim_exported, elide_app_main_menu):
	app = elide_app_main_menu
	manager = app.manager
	import_game_button: Button = manager.current_screen.ids.import_game_button
	x, y = import_game_button.center
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: hasattr(manager.current_screen, "_popover_import_game"),
		100,
		"Never created game import popover",
	)
	modal = manager.current_screen._popover_import_game
	idle_until(lambda: modal._is_open, 100, "Never opened game import modal")
	idle_until(
		lambda: hasattr(modal, "_file_chooser"), 100, "Never made file chooser"
	)
	chooser: FileChooserIconView = modal._file_chooser
	idle_until(lambda: chooser.layout, 100, "Never filled file chooser")
	scrollview = chooser.layout.children[0]
	scatter = scrollview._viewport
	stacklayout = scatter.children[0]
	idle_until(
		lambda: stacklayout.children,
		100,
		"Never filled the stack of file icons",
	)
	for file_icon in stacklayout.children:
		if file_icon.path == kobold_sim_exported:
			x, y = file_icon.center
			break
	else:
		raise RuntimeError(
			"File isn't visible to the chooser: " + kobold_sim_exported
		)
	x, y = chooser.parent.to_parent(*scrollview.to_parent(x, y))
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	advance_frames(5)
	idle_until(
		lambda: chooser.selection,
		100,
		"Chooser never got the selection we pressed",
	)
	ok_button: Button = modal.ids.ok_button
	x, y = modal.to_parent(*ok_button.center)
	assert modal.collide_point(x, y)
	assert ok_button.collide_point(x, y)
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	advance_frames(5)
	idle_until(
		lambda: manager.current == "mainscreen",
		100,
		"Never loaded into the imported game",
	)


def test_export_game(zipped_kobold_in_games_dir, elide_app_main_menu):
	app = elide_app_main_menu
	manager = app.manager
	export_game_button: Button = manager.current_screen.ids.export_game_button
	x, y = export_game_button.center
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: hasattr(manager.current_screen, "_popover_export_game"),
		100,
		"Never created game export modal",
	)
	modal = manager.current_screen._popover_export_game
	idle_until(lambda: modal._is_open, 100, "Never opened game export modal")
	idle_until(lambda: "game_list" in modal.ids, 100, "Never built game list")
	game_list = modal.ids.game_list
	idle_until(lambda: game_list.data, 100, "Never got saved game data")
	button = game_list._viewport.children[0]
	assert button.text == "kobold"
	x, y = game_list.to_parent(*button.center)
	touch = UnitTestTouch(x, y)
	touch.touch_down()
	advance_frames(5)
	touch.touch_up()
	idle_until(
		lambda: not modal._is_open, 100, "Never closed game export modal"
	)
	assert "kobold.lisien" in os.listdir(".")
