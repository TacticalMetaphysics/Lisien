from tempfile import mkdtemp
from multiprocessing import freeze_support

from kivy.lang.builder import Builder
from kivy.properties import BooleanProperty

from ELiDE.game import GameApp, GameScreen


class MainGame(GameScreen):
	pass


class AwarenessApp(GameApp):
	placing_centers = BooleanProperty(False)
	inspector = True

	def set_up(self):
		"""Regenerate the whole map"""
		raise NotImplementedError


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
			size_hint_x: 0.5
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
			id: game
			size_hint_x: 0.7
"""

Builder.load_string(kv)


if __name__ == "__main__":
	freeze_support()
	d = mkdtemp()
	AwarenessApp(prefix=d).run()
	print("Files are in " + d)
