from types import SimpleNamespace

from kivy.base import EventLoop
from kivy.config import ConfigParser
from kivy.tests.common import UnitTestTouch

from LiSE.character import Facade
from ELiDE.menu import DirPicker
from ELiDE.screen import MainScreen
from ELiDE.spritebuilder import PawnConfigScreen, SpotConfigScreen
from ELiDE.statcfg import StatScreen
from ELiDE.graph.board import GraphBoard
from ELiDE.grid.board import GridBoard
from .util import ELiDEAppTest, ListenableDict, MockEngine, idle_until, \
 window_with_widget


class MockStore:

	def save(self, *args):
		pass


class ScreenTest(ELiDEAppTest):

	def test_advance_time(self):
		app = self.app
		app.mainmenu = DirPicker()
		app.spotcfg = SpotConfigScreen()
		app.pawncfg = PawnConfigScreen()
		app.statcfg = StatScreen()
		char = Facade()
		char.name = 'physical'
		app.character = char
		app.engine = MockEngine()
		app.strings = MockStore()
		app.funcs = MockStore()
		char.character = SimpleNamespace(engine=app.engine)
		app.engine.character['physical'] = char
		entity = ListenableDict()
		entity.engine = app.engine
		entity.name = 'name'
		app.selected_proxy = app.proxy = app.statcfg.proxy = entity
		screen = MainScreen(
			graphboards={'physical': GraphBoard(character=char)},
			gridboards={'physical': GridBoard(character=char)})
		win = window_with_widget(screen)
		idle_until(lambda: 'timepanel' in screen.ids, 100,
					"timepanel never got id")
		timepanel = screen.ids['timepanel']
		idle_until(lambda: timepanel.size != [100, 100], 100,
					"timepanel never resized")
		turnfield = timepanel.ids['turnfield']
		turn_before = int(turnfield.hint_text)
		stepbut = timepanel.ids['stepbut']
		motion = UnitTestTouch(*stepbut.center)
		motion.touch_down()
		motion.touch_up()
		EventLoop.idle()
		assert int(turnfield.hint_text) == turn_before + 1

	def test_play(self):
		app = self.app
		app.spotcfg = SpotConfigScreen()
		app.pawncfg = PawnConfigScreen()
		app.statcfg = StatScreen()
		app.mainmenu = DirPicker()
		app.strings = MockStore()
		app.funcs = MockStore()
		char = Facade()
		char.name = 'foo'
		app.character = char
		app.engine = MockEngine()
		char.character = SimpleNamespace(engine=app.engine)
		app.engine.character['foo'] = char
		entity = ListenableDict()
		entity.engine = app.engine
		entity.name = 'name'
		app.selected_proxy = app.proxy = app.statcfg.proxy = entity
		screen = MainScreen(graphboards={'foo': GraphBoard(character=char)},
							gridboards={'foo': GridBoard(character=char)},
							play_speed=1.0)
		win = window_with_widget(screen)
		idle_until(lambda: 'timepanel' in screen.ids, 100,
					"timepanel never got id")
		timepanel = screen.ids['timepanel']
		idle_until(lambda: timepanel.size != [100, 100], 100,
					"timepanel never resized")
		turnfield = timepanel.ids['turnfield']
		turn_before = int(turnfield.hint_text)
		playbut = timepanel.ids['playbut']
		motion = UnitTestTouch(*playbut.center)
		motion.touch_down()
		motion.touch_up()
		idle_until(lambda: int(turnfield.hint_text) == 3, 400,
					"Time didn't advance fast enough")
