from kivy.tests.common import UnitTestTouch

from .util import advance_frames, idle_until, transition_over


def test_character_switcher(polygons_sim, elide_app):
	app = elide_app
	idle_until(
		lambda: app.manager.current == "mainscreen",
		100,
		"Never switched to mainscreen",
	)
	idle_until(lambda: app.mainscreen.boardview, 100, "never got boardview")
	idle_until(
		lambda: app.mainscreen.boardview.board.spot, 100, "never got spots"
	)
	physspots = len(app.mainscreen.boardview.board.spot)
	app.mainscreen.charmenu.charmenu.toggle_chars_screen()
	idle_until(
		lambda: app.manager.current == "chars",
		100,
		"Never switched to chars",
	)
	boxl = app.chars.ids.charsview.ids.boxl
	idle_until(
		lambda: len(boxl.children) == 3,
		100,
		"Didn't get all three characters",
	)
	idle_until(transition_over)
	for charb in boxl.children:
		if charb.text == "triangle":
			touch = UnitTestTouch(
				*charb.parent.parent.to_parent(*charb.center)
			)
			touch.touch_down()
			advance_frames(5)
			idle_until(
				lambda: charb.state == "down",
				100,
				"Button press did not work",
			)
			touch.touch_up()
			break
	else:
		assert False, 'No button for "triangle" character'
	idle_until(
		lambda: app.chars.ids.charsview.character_name == "triangle",
		100,
		"Never propagated character_name",
	)
	app.chars.toggle()
	idle_until(
		lambda: app.manager.current == "mainscreen",
		100,
		"Didn't switch back to mainscreen",
	)
	idle_until(
		lambda: not app.mainscreen.boardview.board.spot,
		100,
		"Didn't clear out spots, {} left".format(
			len(app.mainscreen.boardview.board.spot)
		),
	)
	app.mainscreen.charmenu.charmenu.toggle_chars_screen()
	idle_until(
		lambda: app.manager.current == "chars",
		100,
		"Never switched to chars",
	)
	idle_until(transition_over)
	for charb in boxl.children:
		if charb.text == "physical":
			pos = charb.parent.parent.to_parent(
				*charb.parent.to_parent(*charb.to_parent(*charb.center))
			)
			touch = UnitTestTouch(*pos)
			touch.touch_down()
			idle_until(
				lambda: charb.state == "down",
				100,
				"Button press did not work",
			)
			advance_frames(5)
			touch.touch_up()
			break
	else:
		assert False, 'No button for "physical" character'
	idle_until(
		lambda: app.chars.ids.charsview.character_name == "physical",
		100,
		"Never chose character_name",
	)
	idle_until(
		lambda: app.character_name == "physical",
		100,
		"Never propagated 'physical' back to app.character_name",
	)
	app.chars.toggle()
	idle_until(transition_over)
	idle_until(
		lambda: len(app.mainscreen.boardview.board.spot) == physspots,
		100,
		"Never got physical back",
	)
