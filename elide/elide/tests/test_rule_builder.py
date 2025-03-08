from abc import abstractmethod
from functools import partial

from kivy.tests.common import UnitTestTouch

from lisien import Engine
from lisien.examples import kobold, polygons

from ..card import Card, Foundation
from .util import ELiDEAppTest, idle_until


def builder_foundation(builder):
	for child in builder.children:
		if isinstance(child, Foundation):
			return True
	return False


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
		self.Window.add_widget(mgr)
		screen = app.mainscreen
		idle_until(lambda: app.rules.rulesview, 100, "Never made rules view")
		idle_until(
			lambda: "physical" in screen.graphboards,
			100,
			"Never made physical board",
		)
		self.board = screen.graphboards["physical"]
		idle_until(
			lambda: "kobold" in self.board.pawn,
			100,
			"never got the pawn for the kobold",
		)
		app.selection = self.get_selection()
		screen.charmenu.charmenu.toggle_rules()
		rules = app.rules
		rules_box = rules.children[0]
		idle_until(
			lambda: "ruleslist" in rules_box.ids, 100, "Never made rules list"
		)
		self.rules_list = rules_list = rules_box.ids.ruleslist
		self.rules_view = rules_box.ids.rulesview
		idle_until(
			lambda: rules_list.children[0].children,
			100,
			"Never filled rules list",
		)


class TestRuleBuilderKobold(RuleBuilderTest):
	def install(self, engine: Engine):
		kobold.inittest(engine)

	def get_selection(self):
		return self.board.pawn["kobold"]

	def test_rule_builder_display_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: "shrubsprint"
			in {rulebut.text for rulebut in rules_list.children[0].children},
			100,
			"Never made shrubsprint button",
		)
		for rulebut in rules_list.children[0].children:
			if rulebut.text == "shrubsprint":
				rulebut.state = "down"
				break
		idle_until(lambda: rules_view.children)
		idle_until(
			lambda: hasattr(rules_view, "_trigger_tab"),
			100,
			"Never made trigger tab",
		)
		builder = rules_view._trigger_builder
		idle_until(
			lambda: [
				child for child in builder.children if isinstance(child, Card)
			],
			100,
			"Never filled trigger builder",
		)
		card_names = {
			card.headline_text
			for card in builder.children
			if isinstance(card, Card)
		}
		assert card_names == {
			"standing_still",
			"aware",
			"uncovered",
			"sametile",
			"kobold_alive",
		}

	def test_rule_builder_remove_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: "shrubsprint"
			in {rulebut.text for rulebut in rules_list.children[0].children},
			100,
			"Never made shrubsprint button",
		)
		for rulebut in rules_list.children[0].children:
			if rulebut.text == "shrubsprint":
				rulebut.state = "down"
				break
		idle_until(lambda: rules_view.children)
		idle_until(
			lambda: hasattr(rules_view, "_trigger_tab"),
			100,
			"Never made trigger tab",
		)
		builder = rules_view._trigger_builder
		idle_until(
			lambda: [
				child for child in builder.children if isinstance(child, Card)
			],
			100,
			"Never filled trigger builder",
		)

		uncovered = None

		def have_uncovered():
			nonlocal uncovered
			for card in builder.children:
				if not isinstance(card, Card):
					continue
				if card.headline_text == "uncovered":
					uncovered = card
					return True
			return False

		idle_until(have_uncovered, 100, "Never got 'uncovered' card")

		right_foundation = None

		def have_right_foundation():
			nonlocal right_foundation
			for foundation in builder.children:
				if not isinstance(foundation, Foundation):
					continue
				if foundation.x > uncovered.right:
					right_foundation = foundation
					return True
			return False

		idle_until(have_right_foundation, 100, "Never built right foundation")

		assert uncovered is not None
		assert right_foundation is not None

		def uncovered_is_flush_with_right_foundation():
			for card in builder.children:
				if not isinstance(card, Card):
					continue
				if card.headline_text == "uncovered":
					breakcover = card
					right_foundation = None
					for foundation in builder.children:
						if isinstance(foundation, Card):
							continue
						if (
							right_foundation is None
							or foundation.x > right_foundation.x
						):
							right_foundation = foundation
					assert right_foundation is not None, "No foundations??"
					return breakcover.x == right_foundation.x
			return False

		card = uncovered
		foundation = right_foundation
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
		idle_until(
			partial(builder_foundation, builder),
			100,
			"didn't replace foundations",
		)
		idle_until(
			uncovered_is_flush_with_right_foundation, 100, "card didn't move"
		)
		idle_until(
			lambda: not any(
				func.name == "breakcover"
				for func in self.app.engine.rule["shrubsprint"].triggers
			),
			100,
			"breakcover never removed from rulebook",
		)

	def test_rule_builder_add_trigger(self):
		rules_list = self.rules_list
		rules_view = self.rules_view
		idle_until(
			lambda: "shrubsprint"
			in {rulebut.text for rulebut in rules_list.children[0].children},
			100,
			"Never made shrubsprint button",
		)
		for rulebut in rules_list.children[0].children:
			if rulebut.text == "shrubsprint":
				rulebut.state = "down"
				break
		idle_until(lambda: rules_view.children)
		idle_until(
			lambda: hasattr(rules_view, "_trigger_tab"),
			100,
			"Never made trigger tab",
		)
		builder = rules_view._trigger_builder
		idle_until(
			lambda: [
				child for child in builder.children if isinstance(child, Card)
			],
			100,
			"Never filled trigger builder",
		)
		idle_until(
			lambda: [child.x for child in builder.children if child.x > 0],
			100,
			"Never positioned trigger builder's children",
		)
		aware = None
		for card in builder.children:
			if isinstance(card, Foundation):
				continue
			assert isinstance(card, Card)
			if card.headline_text == "aware":
				aware = card
				break
		assert aware is not None, "Didn't get 'aware' card"
		uncovered = None
		for card in builder.children:
			if isinstance(card, Foundation):
				continue
			assert isinstance(card, Card)
			if card.headline_text == "uncovered":
				uncovered = card
				break
		assert uncovered is not None, "Didn't get 'uncovered' card"
		start_x = aware.center_x
		start_y = aware.top - 10
		assert aware.collide_point(start_x, start_y)
		mov = UnitTestTouch(start_x, start_y)
		mov.touch_down()
		dist_x = start_x - uncovered.center_x
		dist_y = start_y - uncovered.center_y
		decr_x = dist_x / 10
		decr_y = dist_y / 10
		x = start_x
		y = start_y
		for i in range(1, 11):
			x -= decr_x
			y -= decr_y
			mov.touch_move(x, y)
			self.advance_frames(1)
		mov.touch_up(*uncovered.center)
		idle_until(
			lambda: abs(aware.x - uncovered.x) < 2,
			100,
			"aware didn't move to its new place",
		)
		idle_until(
			lambda: any(
				func.name == "aware"
				for func in self.app.engine.rule["shrubsprint"].triggers
			),
			100,
			"aware never added to rulebook",
		)


class TestCharRuleBuilder(ELiDEAppTest):
	def setUp(self):
		with Engine(self.prefix) as eng:
			polygons.install(eng)
			assert list(
				eng.character["triangle"].unit.rule["relocate"].triggers
			) == [
				eng.trigger.similar_neighbors,
				eng.trigger.dissimilar_neighbors,
			]
		super(TestCharRuleBuilder, self).setUp()
		app = self.app
		mgr = app.build()
		self.Window.add_widget(mgr)
		idle_until(
			lambda: hasattr(app, "engine"), 100, "App never made engine"
		)
		idle_until(
			lambda: "triangle" in app.engine.character,
			100,
			"Engine proxy never made triangle character proxy",
		)
		app.select_character(app.engine.character["triangle"])
		idle_until(
			lambda: app.character_name == "triangle",
			100,
			"Never changed character",
		)
		app.mainscreen.charmenu.charmenu.toggle_rules()
		idle_until(
			lambda: getattr(app.charrules, "_finalized", False),
			100,
			"Never finalized",
		)

	def test_char_rule_builder_remove_unit_trigger(self):
		app = self.app
		idle_until(
			lambda: getattr(app.charrules, "_finalized", False),
			100,
			"Never finalized charrules",
		)
		tabitem = app.charrules._unit_tab
		idle_until(lambda: tabitem.content, 100, "unit tab never got content")
		tabitem.on_press()
		self.advance_frames(1)
		tabitem.on_release()
		idle_until(
			lambda: app.charrules._tabs.current_tab == tabitem,
			100,
			"Never switched tab",
		)
		rules_box = app.charrules._unit_box
		idle_until(lambda: rules_box.parent, 100, "unit box never got parent")
		idle_until(
			lambda: getattr(rules_box.rulesview, "_finalized", False),
			100,
			"Never finalized unit rules view",
		)
		idle_until(
			lambda: rules_box.children, 100, "_unit_box never got children"
		)
		idle_until(
			lambda: rules_box.rulesview.children,
			100,
			"Never filled rules view",
		)
		rules_list = rules_box.ruleslist
		idle_until(
			lambda: rules_list.children[0].children,
			1000,
			"Never filled rules list",
		)
		idle_until(
			lambda: "relocate"
			in {rulebut.text for rulebut in rules_list.children[0].children},
			1000,
			"Never made relocate button",
		)
		for rulebut in rules_list.children[0].children:
			if rulebut.text == "relocate":
				rulebut.state = "down"
				break
		builder = rules_box.rulesview._trigger_builder
		assert (
			rules_box.rulesview._tabs.current_tab
			== rules_box.rulesview._trigger_tab
		)
		idle_until(
			lambda: builder.children,
			1000,
			"trigger builder never got children",
		)
		idle_until(
			partial(builder_foundation, builder),
			100,
			"Never filled trigger builder",
		)
		idle_until(
			lambda: builder.parent, 1000, "trigger builder never got parent"
		)
		card_names = {
			card.headline_text
			for card in builder.children
			if isinstance(card, Card)
		}
		assert card_names == {
			"similar_neighbors",
			"dissimilar_neighbors",
		}
		for card in builder.children:
			if not isinstance(card, Card):
				continue
			if card.headline_text == "similar_neighbors":
				break
		else:
			assert False, "Didn't get similar_neighbors"
		startx = card.center_x
		starty = card.top - 1
		assert card.collide_point(startx, starty), "card didn't collide itself"
		for cardother in builder.children:
			if not isinstance(cardother, Card) or cardother == card:
				continue
			assert not cardother.collide_point(startx, starty), (
				"other card will grab the touch"
			)
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
		self.advance_frames(1)
		for i in range(1, 11):
			x += distx / 10
			y += disty / 10
			touch.touch_move(x, y)
			self.advance_frames(1)
		touch.touch_up()
		self.advance_frames(5)
		rules_box.ids.closebut.on_release()
		idle_until(
			lambda: all(
				card.headline_text != "similar_neighbors"
				for card in builder.decks[0]
			),
			100,
			"similar_neighbors still in used pile",
		)
		idle_until(
			lambda: not any(
				trig.name == "similar_neighbors"
				for trig in app.charrules.character.unit.rulebook[0].triggers
			),
			100,
			"similar_neighbors still in proxy triggers list",
		)
		app.stop()
		with Engine(self.prefix) as eng:
			assert list(
				eng.character["triangle"].unit.rule["relocate"].triggers
			) == [eng.trigger.dissimilar_neighbors]
