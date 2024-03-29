from tempfile import mkdtemp
from multiprocessing import freeze_support
from inspect import getsource

from kivy.clock import Clock
from kivy.lang.builder import Builder
from kivy.properties import BooleanProperty

from ELiDE.game import GameApp, GameScreen, GridBoard


def game_start(engine) -> None:
	import networkx as nx

	# ensure we're on a fresh branch
	if engine.turn != 0 or engine.tick != 0:
		if engine.branch == 'trunk':
			new_branch_name = 'trunk0'
		else:
			new_branch_num = int(engine.branch[5:])
			new_branch_name = 'trunk' + str(new_branch_num)
		engine.turn = 0
		engine.tick = 0
		engine.switch_main_branch(new_branch_name)

	wide = engine.eternal.setdefault("max-pxcor", 36)
	high = engine.eternal.setdefault("max-pycor", 37)
	initworld = nx.grid_2d_graph(wide, high)
	# world wraps vertically
	for x in range(wide):
		initworld.add_edge((x, high - 1), (x, 0))
	# world wraps horizontally
	for y in range(high):
		initworld.add_edge((wide - 1, y), (0, y))

	locs = set(initworld.nodes.keys())

	for turtle in range(engine.eternal.setdefault("people", 60)):
		initworld.add_node("turtle" + str(turtle), location=locs.pop())

	for center in range(engine.eternal.setdefault("centers", 20)):
		initworld.add_node(
			"center" + str(center),
			location=locs.pop(),
			_image_paths=["atlas://rltiles/dungeon/dngn_altar_xom"])

	phys = engine.new_character("physical", initworld)


class MainGame(GameScreen):

	def on_parent(self, *args):
		if 'game' not in self.ids:
			Clock.schedule_once(self.on_parent, 0)
			return
		AwarenessApp.get_running_app().set_up()
		self.ids.game.board = GridBoard(
			character=self.engine.character['physical'])


class AwarenessApp(GameApp):
	placing_centers = BooleanProperty(False)
	inspector = True

	def set_up(self):
		"""Regenerate the whole map"""
		self.engine.game_start()


kv = """
<ScreenManager>:
	MainGame:
		name: 'play'
<MainGame>:
	BoxLayout:
		orientation: 'horizontal'
		BoxLayout:
			id: sidebar
			orientation: 'vertical'
			width: 333
			size_hint_x: None
			BoxLayout:
				id: controls
				orientation: 'vertical'
				size_hint_y: 0.3
				BoxLayout:
					id: toprow
					Button:
						id: setup
						text: 'setup'
						on_release: root.set_up()
					Slider:
						id: people
						min: 0
						max: 300
						step: 1
						Label:
							text: 'people'
							x: people.x
							y: people.y
							size: self.texture_size
							size_hint: None, None
						Label:
							text: str(int(people.value))
							x: people.right - self.texture_size[0]
							y: people.y
							size: self.texture_size
							size_hint: None, None
					ToggleButton:
						text: 'place centers'
						on_release: root.placing_centers = self.state
				BoxLayout:
					id: midrow
					Slider:
						id: nonusage
						min: 0
						max: 500
						step: 1
						Label:
							text: 'non-usage limit'
							size: self.texture_size
							pos: nonusage.pos
						Label:
							text: '{:d} ticks'.format(int(nonusage.value))
							size: self.texture_size
							x: nonusage.right - self.texture_size[0]
							y: nonusage.y
			Widget:
				id: filler
		Widget:
			id: gamebox
			size_hint_x: 0.7
			GridBoardView:
				id: game
				character: root.engine.character['physical'] if root.engine and 'physical' in root.engine.character else None		
				pos: gamebox.pos
				size: gamebox.size
"""

Builder.load_string(kv)

if __name__ == "__main__":
	freeze_support()
	d = mkdtemp()
	with open(d + '/game_start.py', 'w', encoding='utf-8') as outf:
		outf.write(getsource(game_start))
	AwarenessApp(prefix=d).run()
	print("Files are in " + d)
