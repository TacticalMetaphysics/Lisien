from kivy.tests import UnitTestTouch
from .util import idle_until, advance_frames


def test_spot_from_dummy(elide_app):
	@idle_until(timeout=100)
	def charmenu_present():
		return (
			hasattr(elide_app, "mainscreen")
			and hasattr(elide_app.mainscreen, "charmenu")
			and elide_app.mainscreen.charmenu.charmenu
		)

	charmenu = elide_app.mainscreen.charmenu.charmenu

	@idle_until(timeout=100)
	def charmenu_has_parent():
		return charmenu.parent is not None

	@idle_until(timeout=100)
	def charmenu_has_screen():
		return charmenu.screen is not None

	@idle_until(timeout=100)
	def dummy_place_created():
		return "dummyplace" in charmenu.ids

	dummy_place = charmenu.ids.dummyplace
	x0, y0 = charmenu.to_parent(*dummy_place.center)
	touch = UnitTestTouch(x0, y0)
	touch.touch_down()

	@idle_until(timeout=100)
	def dummy_got_touch():
		return getattr(dummy_place, "_touch") is touch

	x1, y1 = elide_app.mainscreen.mainview.center

	xdist = x1 - x0
	ydist = y1 - y0
	dummy_name = dummy_place.name
	for i in range(15, 0, -1):
		touch.touch_move(x0 + (xdist / i), y0 + (ydist / i))
		advance_frames(1)
	advance_frames(3)
	touch.touch_up()

	boardview = elide_app.mainscreen.boardview
	idle_until(
		lambda: dummy_name in elide_app.character.place,
		100,
		"Didn't create first new place from dummy",
	)
	idle_until(
		lambda: dummy_name in boardview.board.spot,
		100,
		"Didn't create first new spot from dummy",
	)
	assert boardview.plane.to_parent(
		*boardview.board.spot[dummy_name].center
	) == (
		x1,
		y1,
	)

	@idle_until(timeout=100)
	def dummy_returned():
		return charmenu.to_parent(*dummy_place.center) == (x0, y0)

	idle_until(
		lambda: dummy_name != dummy_place.name, 100, "Never renamed dummy"
	)
	dummy_name = dummy_place.name
	x2 = x1 - 50
	y2 = y1 - 50
	xdist = x2 - x0
	ydist = y2 - y0
	touch = UnitTestTouch(x0, y0)
	touch.touch_down()

	idle_until(dummy_got_touch, timeout=100)
	for i in range(15, 0, -1):
		touch.touch_move(x0 + xdist / i, y0 + ydist / i)
		advance_frames(1)
	touch.touch_up()

	idle_until(
		lambda: dummy_name in elide_app.character.place,
		100,
		"Didn't create second new place from dummy",
	)
	assert boardview.plane.to_parent(
		*boardview.board.spot[dummy_name].center
	) == (x2, y2)
