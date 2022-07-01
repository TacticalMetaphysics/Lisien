from kivy.base import EventLoop
from kivy.tests import UnitTestTouch

from .util import ELiDEAppTest, idle_until, window_with_widget
from LiSE import Engine


class StringsEditorTest(ELiDEAppTest):

	def test_strings_editor(self):
		assert 'LiSE' in self.app.config
		app = self.app
		print('app', id(app))
		win = window_with_widget(app.build())
		idle_until(lambda: hasattr(app, 'mainscreen'), 100,
					"app never got mainscreen")
		idle_until(lambda: app.manager.has_screen('timestream'), 100,
					"timestream never added to manager")
		idle_until(lambda: hasattr(app.mainmenu, 'configurator'), 100,
					'DirPicker never got configurator')
		app.mainmenu.configurator.start()  # start with blank world
		idle_until(lambda: app.engine, 100, "app never got engine")
		idle_until(lambda: app.strings.children, 100,
					"strings never got children")
		idle_until(lambda: app.strings.edbox, 100, 'strings never got edbox')
		idle_until(lambda: 'physical' in app.mainscreen.graphboards, 100,
					'never got physical in graphboards')
		edbox = app.strings.edbox
		strings_list = edbox.ids.strings_list
		idle_until(lambda: strings_list.store, 100,
					"strings_list never got store")
		strings_ed = edbox.ids.strings_ed
		app.strings.toggle()
		idle_until(lambda: strings_list.data, 100,
					"strings_list never got data")
		self.advance_frames(10)
		touchy = UnitTestTouch(*strings_ed.ids.stringname.center)
		touchy.touch_down()
		EventLoop.idle()
		touchy.touch_up()
		EventLoop.idle()
		strings_ed.ids.stringname.text = 'a string'
		idle_until(lambda: strings_ed.name == 'a string', 100,
					"name never set")
		touchier = UnitTestTouch(*strings_ed.ids.string.center)
		touchier.touch_down()
		EventLoop.idle()
		touchier.touch_up()
		self.advance_frames(10)
		strings_ed.ids.string.text = 'its value'
		idle_until(lambda: strings_ed.source == 'its value', 100,
					'source never set')
		self.advance_frames(10)
		edbox.dismiss()
		app.stop()
		with Engine(self.prefix) as eng:
			assert 'a string' in eng.string
			assert eng.string['a string'] == 'its value'
