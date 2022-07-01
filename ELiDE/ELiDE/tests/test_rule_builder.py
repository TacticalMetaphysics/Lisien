from kivy.tests.common import UnitTestTouch

from abc import abstractmethod
from LiSE import Engine
from LiSE.examples import kobold
from LiSE.examples import polygons
from .util import idle_until, window_with_widget, ELiDEAppTest
from ..card import Card


class RuleBuilderTest(ELiDEAppTest):

	@abstractmethod
	def install(self, engine: Engine):
		raise NotImplementedError()

	@abstractmethod
	def get_selection(self):
		raise NotImplementedError()

	def setUp(self):
		super(RuleBuilderTest, self).setUp()
		with Engine(self.prefix) as eng:
			self.install(eng)
		app = self.app
		mgr = app.build()
		self.win = window_with_widget(mgr)
		screen = app.mainscreen
		idle_until(lambda: app.rules.rulesview, 100, 'Never made rules view')
		idle_until(lambda: 'physical' in screen.graphboards, 100,
					'Never made physical board')
		self.board = screen.graphboards['physical']
		idle_until(lambda: 'kobold' in self.board.pawn, 100,
					'never got the pawn for the kobold')
		app.selection = self.get_selection()
		screen.charmenu.charmenu.toggle_rules()
		rules = app.rules
		rules_box = rules.children[0]
		idle_until(lambda: 'ruleslist' in rules_box.ids, 100,
					'Never made rules list')
		self.rules_list = rules_list = rules_box.ids.ruleslist
		self.rules_view = rules_box.ids.rulesview
		idle_until(lambda: rules_list.children[0].children, 100,
					'Never filled rules list')


class TestRuleBuilderKobold(RuleBuilderTest):

	def install(self, engine: Engine):
		kobold.inittest(engine)

	def get_selection(self):
		return self.board.pawn['kobold']

	def test_rule_builder_display_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: 'shrubsprint' in
			{rulebut.text
				for rulebut in rules_list.children[0].children}, 100,
			'Never made shrubsprint button')
		for rulebut in rules_list.children[0].children:
			if rulebut.text == 'shrubsprint':
				rulebut.state = 'down'
				break
		idle_until(lambda: rules_view.children)
		idle_until(lambda: hasattr(rules_view, '_trigger_tab'), 100,
					'Never made trigger tab')
		builder = rules_view._trigger_builder
		idle_until(
			lambda:
			[child for child in builder.children if isinstance(child, Card)],
			100, 'Never filled trigger builder')
		card_names = {
			card.headline_text
			for card in builder.children if isinstance(card, Card)
		}
		assert card_names == {
			'standing_still', 'aware', 'uncovered', 'sametile', 'breakcover',
			'kobold_alive'
		}

	def test_rule_builder_remove_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: 'shrubsprint' in
			{rulebut.text
				for rulebut in rules_list.children[0].children}, 100,
			'Never made shrubsprint button')
		for rulebut in rules_list.children[0].children:
			if rulebut.text == 'shrubsprint':
				rulebut.state = 'down'
				break
		idle_until(lambda: rules_view.children)
		idle_until(lambda: hasattr(rules_view, '_trigger_tab'), 100,
					'Never made trigger tab')
		builder = rules_view._trigger_builder
		idle_until(
			lambda:
			[child for child in builder.children if isinstance(child, Card)],
			100, 'Never filled trigger builder')
		for card in builder.children:
			if not isinstance(card, Card):
				continue
			if card.headline_text == 'breakcover':
				break
		else:
			assert False, "No breakcover card"
		for foundation in builder.children:
			if isinstance(foundation, Card):
				continue
			if foundation.x > card.right:
				break
		else:
			assert False, "No right foundation"
		mov = UnitTestTouch(*card.center)
		mov.touch_down()
		dist_x = foundation.center_x - card.center_x
		dist_y = foundation.y - card.center_y
		for i in range(1, 11):
			coef = 1 / i
			x = foundation.center_x - coef * dist_x
			y = foundation.y - coef * dist_y
			mov.touch_move(x, y)
			self.advance_frames(1)
		mov.touch_up(foundation.center_x, foundation.y)
		idle_until(lambda: card.x == foundation.x, 100, "card didn't move")
		idle_until(
			lambda: 'breakcover' not in self.app.engine.rule['shrubsprint'].
			triggers, 100, 'breakcover never removed from rulebook')

	def test_rule_builder_add_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: 'shrubsprint' in
			{rulebut.text
				for rulebut in rules_list.children[0].children}, 100,
			'Never made shrubsprint button')
		for rulebut in rules_list.children[0].children:
			if rulebut.text == 'shrubsprint':
				rulebut.state = 'down'
				break
		idle_until(lambda: rules_view.children)
		idle_until(lambda: hasattr(rules_view, '_trigger_tab'), 100,
					'Never made trigger tab')
		builder = rules_view._trigger_builder
		idle_until(
			lambda:
			[child for child in builder.children if isinstance(child, Card)],
			100, 'Never filled trigger builder')
		aware = breakcover = None
		for card in builder.children:
			if not isinstance(card, Card):
				continue
			if card.headline_text == 'aware':
				aware = card
			elif card.headline_text == 'breakcover':
				breakcover = card
		assert None not in (
			aware, breakcover), "Didn't get 'aware' and 'breakcover' cards"
		start_x = aware.center_x
		start_y = aware.top - 10
		mov = UnitTestTouch(start_x, start_y)
		mov.touch_down()
		dist_x = start_x - breakcover.center_x
		dist_y = start_y - breakcover.center_y
		decr_x = dist_x / 10
		decr_y = dist_y / 10
		x = start_x
		y = start_y
		for i in range(1, 11):
			x -= decr_x
			y -= decr_y
			mov.touch_move(x, y)
			self.advance_frames(1)
		mov.touch_up(*breakcover.center)
		idle_until(lambda: aware.x == breakcover.x, 100,
					"aware didn't move to its new place")
		idle_until(
			lambda: 'aware' in self.app.engine.rule['shrubsprint'].triggers,
			100, 'aware never added to rulebook')


class TestCharRuleBuilder(ELiDEAppTest):

	def setUp(self):
		super(TestCharRuleBuilder, self).setUp()
		with Engine(self.prefix) as eng:
			polygons.install(eng)
			assert list(
				eng.character['triangle'].unit.rule['relocate'].triggers) == [
					eng.trigger.similar_neighbors,
					eng.trigger.dissimilar_neighbors
				]
		app = self.app
		mgr = app.build()
		self.win = window_with_widget(mgr)
		idle_until(lambda: getattr(app, 'engine'), 100,
					'App never made engine')
		idle_until(lambda: 'triangle' in app.engine.character, 100,
					'Engine proxy never made triangle character proxy')
		app.select_character(app.engine.character['triangle'])
		idle_until(lambda: app.character_name == 'triangle', 100,
					'Never changed character')
		app.mainscreen.charmenu.charmenu.toggle_rules()
		idle_until(lambda: getattr(app.charrules, '_finalized', False), 100,
					'Never finalized')

	def test_char_rule_builder_remove_unit_trigger(self):
		app = self.app
		idle_until(lambda: getattr(app.charrules, '_finalized', False), 100,
					"Never finalized charrules")
		tabitem = app.charrules._unit_tab
		idle_until(lambda: tabitem.content, 100, 'unit tab never got content')
		tabitem.on_press()
		self.advance_frames(1)
		tabitem.on_release()
		idle_until(lambda: app.charrules._tabs.current_tab == tabitem, 100,
					'Never switched tab')
		rules_box = app.charrules._unit_box
		idle_until(lambda: rules_box.parent, 100, 'unit box never got parent')
		idle_until(lambda: getattr(rules_box.rulesview, '_finalized', False),
					100, "Never finalized unit rules view")
		idle_until(lambda: rules_box.children, 100,
					'_unit_box never got children')
		idle_until(lambda: rules_box.rulesview.children, 100,
					'Never filled rules view')
		rules_list = rules_box.ruleslist
		idle_until(lambda: rules_list.children[0].children, 100,
					'Never filled rules list')
		idle_until(
			lambda: 'relocate' in
			{rulebut.text
				for rulebut in rules_list.children[0].children}, 100,
			'Never made relocate button')
		for rulebut in rules_list.children[0].children:
			if rulebut.text == 'relocate':
				rulebut.state = 'down'
				break
		builder = rules_box.rulesview._trigger_builder
		assert rules_box.rulesview._tabs.current_tab == rules_box.rulesview._trigger_tab
		idle_until(lambda: builder.children, 100,
					'trigger builder never got children')

		def builder_foundation():
			for child in builder.children:
				if not isinstance(child, Card):
					return True
			return False

		idle_until(builder_foundation, 100, 'Never filled trigger builder')
		idle_until(lambda: builder.parent, 100,
					"trigger builder never got parent")
		card_names = {
			card.headline_text
			for card in builder.children if isinstance(card, Card)
		}
		assert card_names == {'similar_neighbors', 'dissimilar_neighbors'}
		for card in builder.children:
			if not isinstance(card, Card):
				continue
			if card.headline_text == 'similar_neighbors':
				break
		else:
			assert False, "Didn't get similar_neighbors"
		startx = card.center_x
		starty = card.top - 1
		assert card.collide_point(startx, starty), "card didn't collide itself"
		for cardother in builder.children:
			if not isinstance(cardother, Card) or cardother == card:
				continue
			assert not cardother.collide_point(
				startx, starty), "other card will grab the touch"
		touch = UnitTestTouch(startx, starty)
		for target in builder.children:
			if isinstance(target, Card):
				continue
			if target.x > card.right:
				break
		else:
			assert False, "Didn't get target foundation"
		targx, targy = target.center
		distx = targx - startx
		disty = targy - starty
		x, y = startx, starty
		touch.touch_down()
		for i in range(1, 11):
			x += distx / 10
			y += disty / 10
			touch.touch_move(x, y)
			self.advance_frames(1)
		touch.touch_up()
		self.advance_frames(5)
		rules_box.ids.closebut.on_release()
		self.advance_frames(5)
		app.stop()
		with Engine(self.prefix) as eng:
			assert list(
				eng.character['triangle'].unit.rule['relocate'].triggers) == [
					eng.trigger.dissimilar_neighbors
				]
