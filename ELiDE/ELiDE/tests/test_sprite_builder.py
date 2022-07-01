from pprint import pprint

from ..pallet import Pallet
from .util import ELiDEAppTest, idle_until, window_with_widget


class TestSpriteBuilder(ELiDEAppTest):

	def test_build_pawn(self):
		app = self.app
		win = window_with_widget(app.build())
		app.manager.current = 'pawncfg'
		idle_until(lambda: 'dialog' in app.pawncfg.ids, 100,
					"Never made dialog for pawncfg")
		pawn_cfg_dialog = app.pawncfg.ids.dialog
		idle_until(lambda: 'builder' in pawn_cfg_dialog.ids, 100,
					"Never made pawn builder")
		builder = pawn_cfg_dialog.ids.builder
		idle_until(lambda: builder.labels, 100, "Never got any builder labels")
		idle_until(lambda: builder.pallets, 100,
					"Never got any builder pallets")
		idle_until(lambda: len(builder.labels) == len(builder.pallets), 100,
					"Never updated pawn builder")
		palbox = builder._palbox
		for child in palbox.children:
			if not isinstance(child, Pallet):
				continue
			idle_until(lambda: child.swatches, 100,
						"Never got swatches for " + child.filename)
			if 'draconian_m' in child.swatches:
				child.swatches['draconian_m'].state = 'down'
				idle_until(
					lambda: child.swatches['draconian_m'] in child.selection,
					100, "Selection never updated")
			if 'robe_red' in child.swatches:
				child.swatches['robe_red'].state = 'down'
				idle_until(
					lambda: child.swatches['robe_red'] in child.selection, 100,
					"Selection never updated")
		idle_until(lambda: pawn_cfg_dialog.ids.selector.imgpaths, 100,
					"Never got imgpaths")
		pawn_cfg_dialog.pressed()
		idle_until(lambda: pawn_cfg_dialog.imgpaths, 100,
					"Never propagated imgpaths")
		assert pawn_cfg_dialog.imgpaths == [
			'atlas://base.atlas/draconian_m', 'atlas://body.atlas/robe_red'
		]
