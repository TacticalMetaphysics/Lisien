from functools import partial
from types import SimpleNamespace

from kivy.base import EventLoop
from kivy.tests.common import UnitTestTouch
from kivy.uix.screenmanager import ScreenManager

from elide.graph.board import GraphBoard
from elide.grid.board import GridBoard
from elide.menu import DirPicker
from elide.screen import MainScreen
from elide.spritebuilder import PawnConfigScreen, SpotConfigScreen
from elide.statcfg import StatScreen
from lisien import Engine
from lisien.facade import CharacterFacade

from .util import ELiDEAppTest, ListenableDict, MockEngine, idle_until


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
		char = CharacterFacade("physical")
		app.character = char
		app.engine = MockEngine()
		app.strings = MockStore()
		app.funcs = MockStore()
		char.character = SimpleNamespace(engine=app.engine)
		app.engine.character["physical"] = char
		entity = ListenableDict()
		entity.engine = app.engine
		entity.name = "name"
		app.selected_proxy = app.proxy = app.statcfg.proxy = entity
		screen = MainScreen(
			graphboards={"physical": GraphBoard(character=char)},
			gridboards={"physical": GridBoard(character=char)},
		)
		self.Window.add_widget(screen)
		idle_until(
			lambda: "timepanel" in screen.ids, 100, "timepanel never got id"
		)
		timepanel = screen.ids["timepanel"]
		idle_until(
			lambda: timepanel.size != [100, 100],
			100,
			"timepanel never resized",
		)
		turnfield = timepanel.ids["turnfield"]
		turn_before = int(turnfield.hint_text)
		stepbut = timepanel.ids["stepbut"]
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
		char = CharacterFacade("foo")
		app.character = char
		app.engine = MockEngine()
		app.manager = ScreenManager()
		char.character = SimpleNamespace(engine=app.engine)
		app.engine.character["foo"] = char
		entity = ListenableDict()
		entity.engine = app.engine
		entity.name = "name"
		app.selected_proxy = app.proxy = app.statcfg.proxy = entity
		screen = MainScreen(
			graphboards={"foo": GraphBoard(character=char)},
			gridboards={"foo": GridBoard(character=char)},
			play_speed=1.0,
		)
		app.manager.add_widget(screen)
		self.Window.add_widget(app.manager)
		idle_until(
			lambda: "timepanel" in screen.ids, 100, "timepanel never got id"
		)
		timepanel = screen.ids["timepanel"]
		idle_until(
			lambda: timepanel.size != [100, 100],
			100,
			"timepanel never resized",
		)
		turnfield = timepanel.ids["turnfield"]
		playbut = screen.playbut = timepanel.ids["playbut"]
		motion = UnitTestTouch(*playbut.center)
		motion.touch_down()
		motion.touch_up()
		idle_until(
			lambda: int(turnfield.hint_text) == 3,
			400,
			"Time didn't advance fast enough",
		)
		playbut.state = "normal"

	def test_update(self):
		def almost(a, b):
			if isinstance(a, tuple) and isinstance(b, tuple):
				return all(almost(aa, bb) for (aa, bb) in zip(a, b))
			return abs(a - b) < 1

		with Engine(self.prefix) as eng:
			phys = eng.new_character("physical")
			here = phys.new_place((0, 0))
			phys.add_place((1, 1))
			phys.add_place(9)  # test that gridboard can handle this
			this = here.new_thing(2)

			@this.rule(always=True)
			def go(me):
				if me["location"] == (1, 1):
					me["location"] = 9
				elif me["location"] == 9:
					me["location"] = (0, 0)
				else:
					me["location"] = (1, 1)

		app = self.app
		app.starting_dir = self.prefix
		app.build()
		idle_until(
			lambda: hasattr(app, "engine"), 100, "Never got engine proxy"
		)
		assert app.engine.character["physical"].thing[2]["location"] == (0, 0)
		graphboard = app.mainscreen.graphboards["physical"]
		gridboard = app.mainscreen.gridboards["physical"]
		idle_until(
			lambda: graphboard.size != [100, 100],
			100,
			"Never resized graphboard",
		)
		idle_until(
			lambda: gridboard.size != [100, 100],
			100,
			"Never resized gridboard",
		)
		idle_until(
			lambda: (0, 0) in graphboard.spot,
			100,
			"Never made spot for location 0",
		)
		idle_until(
			lambda: 2 in graphboard.pawn, 100, "Never made pawn for thing 2"
		)
		locspot0 = graphboard.spot[0, 0]
		gridspot0 = gridboard.spot[0, 0]
		locspot1 = graphboard.spot[1, 1]
		gridspot1 = gridboard.spot[1, 1]
		graphpawn = graphboard.pawn[2]
		gridpawn = gridboard.pawn[2]
		idle_until(
			lambda: almost(graphpawn.x, locspot0.right),
			100,
			f"Never positioned pawn to 0's right (it's at {graphpawn.x}"
			f", not {locspot0.right})",
		)
		idle_until(
			lambda: almost(graphpawn.y, locspot0.top),
			100,
			"Never positioned pawn to 0's top",
		)
		idle_until(
			lambda: almost(gridpawn.pos, gridspot0.pos),
			100,
			"Never positioned pawn to grid 0, 0",
		)
		app.mainscreen.next_turn()
		idle_until(lambda: not app.edit_locked, 100, "Never unlocked")

		loc = app.engine.character["physical"].thing[2]["location"]

		def relocated_to(dest):
			nonlocal loc
			loc = app.engine.character["physical"].thing[2]["location"]
			return loc == dest

		idle_until(
			partial(relocated_to, (1, 1)),
			1000,
			f"Thing 2 didn't go to location (1, 1); instead, it's at {loc}",
		)
		idle_until(
			lambda: almost(graphpawn.x, locspot1.right),
			100,
			"Never positioned pawn to 1's right "
			f"(pawn is at {graphpawn.x} not {locspot1.right})",
		)
		idle_until(
			lambda: almost(graphpawn.y, locspot1.top),
			100,
			"Never positioned pawn to 1's top "
			f"(it's at {graphpawn.y}, not {locspot1.top})",
		)
		idle_until(
			lambda: almost(gridpawn.pos, gridspot1.pos),
			100,
			"Never positioned pawn to grid 1, 1",
		)
		locspot9 = graphboard.spot[9]
		app.mainscreen.next_turn()
		idle_until(lambda: not app.edit_locked, 100, "Never unlocked")
		loc = app.engine.character["physical"].thing[2]["location"]
		idle_until(
			partial(relocated_to, 9),
			1000,
			f"Thing 2 didn't relocate to 9; it's at {loc}",
		)
		idle_until(
			lambda: 2 not in gridboard.pawn,
			100,
			"pawn never removed from grid",
		)
		idle_until(
			lambda: almost(graphpawn.x, locspot9.right)
			and almost(graphpawn.y, locspot9.top),
			100,
			f"Never positioned pawn to 9's top-right, "
			f"it's at {graphpawn.pos} not {locspot9.right, locspot9.top}",
		)
		app.mainscreen.next_turn()
		idle_until(lambda: not app.edit_locked, 100, "Never unlocked")
		loc = app.engine.character["physical"].thing[2]["location"]
		idle_until(
			partial(relocated_to, (0, 0)),
			1000,
			f"Thing 2 didn't relocate to (0, 0); it's at {loc}",
		)
		idle_until(
			lambda: 2 in gridboard.pawn, 100, "pawn never returned to grid"
		)
		idle_until(
			lambda: almost(graphpawn.x, locspot0.right)
			and almost(graphpawn.y, locspot0.top),
			100,
			f"Never returned to 0's top-right "
			f"(stuck at {graphpawn.pos}, should be "
			f"{locspot0.right, locspot0.top})",
		)
		idle_until(
			lambda: almost(gridpawn.pos, gridspot0.pos),
			100,
			"Never returned to grid 0, 0",
		)
