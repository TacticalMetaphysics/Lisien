import os
import shutil

import pytest
from kivy.tests.common import UnitTestTouch
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from ..app import ElideApp
from .util import idle_until, advance_frames


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
		100,
		"Never switched to 'mainscreen' screen",
	)


@pytest.fixture
def zipped_kobold(play_dir, kobold_sim):
	made = shutil.make_archive("kobold", "zip", play_dir, play_dir)
	archive_name = os.path.basename(made)
	shutil.move(made, os.path.join(os.path.dirname(play_dir), archive_name))
	assert archive_name in os.listdir(os.path.dirname(play_dir))


def test_load_game(zipped_kobold, elide_app_main_menu):
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
