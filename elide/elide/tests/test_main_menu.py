from kivy.core.window import Window
from kivy.tests.common import UnitTestTouch
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from ..app import ElideApp
from .util import idle_until, advance_frames


def test_new_game(elide_app_main_menu):
	app: ElideApp = elide_app_main_menu
	manager = app.manager
	idle_until(
		lambda: manager.current == "main",
		100,
		"Never switched to 'main' screen",
	)
	new_game_button: Button = manager.current_screen.ids.new_game_button
	idle_until(
		lambda: new_game_button.pos != [0, 0],
		100,
		"Never laid out the buttons",
	)
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
