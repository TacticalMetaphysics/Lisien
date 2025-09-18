import os

from kivy.base import EventLoop
from kivy.tests import UnitTestTouch

from lisien import Engine

from .util import advance_frames, idle_until


def test_strings_editor(prefix, elide_app):
	assert "lisien" in elide_app.config
	app = elide_app
	idle_until(
		lambda: hasattr(app, "mainscreen"), 100, "app never got mainscreen"
	)
	idle_until(
		lambda: app.manager.has_screen("timestream"),
		100,
		"timestream never added to manager",
	)

	def app_has_engine():
		return hasattr(app, "engine")

	idle_until(app_has_engine, 600, "app never got engine")
	idle_until(lambda: app.strings.children, 100, "strings never got children")
	idle_until(lambda: app.strings.edbox, 100, "strings never got edbox")
	idle_until(
		lambda: "physical" in app.mainscreen.graphboards,
		100,
		"never got physical in graphboards",
	)
	edbox = app.strings.edbox
	strings_list = edbox.ids.strings_list
	idle_until(lambda: strings_list.store, 100, "strings_list never got store")
	strings_ed = edbox.ids.strings_ed
	app.strings.toggle()
	advance_frames(10)
	touchy = UnitTestTouch(*strings_ed.ids.stringname.center)
	touchy.touch_down()
	EventLoop.idle()
	touchy.touch_up()
	EventLoop.idle()
	strings_ed.ids.stringname.text = "a string"
	idle_until(lambda: strings_ed.name == "a string", 100, "name never set")
	touchier = UnitTestTouch(*strings_ed.ids.string.center)
	touchier.touch_down()
	EventLoop.idle()
	touchier.touch_up()
	advance_frames(10)
	strings_ed.ids.string.text = "its value"
	idle_until(
		lambda: strings_ed.source == "its value", 100, "source never set"
	)
	advance_frames(10)
	edbox.dismiss()
	app.stop()
	with Engine(os.path.join(prefix, "test"), workers=0) as eng:
		assert "a string" in eng.string
		assert eng.string["a string"] == "its value"
