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
	print(f"xdist={xdist}, ydist={ydist}")
	for i in range(15, 0, -1):
		touch.touch_move(x0 + (xdist / i), y0 + (ydist / i))
		print(touch.pos)
		advance_frames(1)
	advance_frames(3)
	former_place_len = len(elide_app.character.place)
	touch.touch_up()

	@idle_until(timeout=100)
	def place_created():
		return len(elide_app.character.place) > former_place_len
