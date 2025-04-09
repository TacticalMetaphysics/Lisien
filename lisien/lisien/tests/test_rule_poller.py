def test_character_rule_poll(proxyless_engine):
	phys = proxyless_engine.new_character("physical")
	notphys = proxyless_engine.new_character("ethereal")

	@phys.rule(always=True)
	def hello(char):
		char.stat["run"] = True

	@notphys.rule
	def goodbye(char):
		char.stat["run"] = True

	proxyless_engine.next_turn()

	assert "run" in phys.stat
	assert "run" not in notphys.stat


def test_unit_rule_poll(proxyless_engine):
	phys = proxyless_engine.new_character("physical")
	notphys = proxyless_engine.new_character("ethereal")

	unit = phys.new_place("unit")
	notunit1 = notphys.new_place("notunit")
	notunit2 = phys.new_place("notunit")
	notphys.add_unit(unit)

	@notphys.unit.rule(always=True)
	def rule1(unit):
		unit["run"] = True

	@phys.unit.rule
	def rule2(unit):
		unit["run"] = True

	proxyless_engine.next_turn()

	assert unit["run"]
	assert "run" not in notunit1
	assert "run" not in notunit2


def test_character_thing_rule_poll(proxyless_engine):
	phys = proxyless_engine.new_character("physical")
	notphys = proxyless_engine.new_character("ethereal")

	there = phys.new_place("there")
	this = there.new_thing("this")
	that = there.new_thing("that")

	yonder = notphys.new_place("yonder")
	thother = yonder.new_thing("thother")

	@phys.thing.rule(always=True)
	def rule1(thing):
		thing["run"] = True

	@phys.thing.rule
	def rule2(thing):
		thing["notrun"] = False

	proxyless_engine.next_turn()

	assert this["run"]
	assert that["run"]
	assert "notrun" not in that
	assert "run" not in thother
	assert "run" not in there
	assert "run" not in yonder


def test_character_place_rule_poll(proxyless_engine):
	phys = proxyless_engine.new_character("physical")
	notphys = proxyless_engine.new_character("ethereal")

	here = phys.new_place("here")
	there = phys.new_place("there")

	nowhere = notphys.new_place("nowhere")

	@phys.place.rule(always=True)
	def rule1(place):
		place["run"] = True

	@notphys.place.rule
	def rule2(place):
		place["notrun"] = False

	proxyless_engine.next_turn()

	assert here["run"]
	assert there["run"]
	assert "run" not in nowhere
	assert "notrun" not in here
	assert "notrun" not in there


def test_character_portal_rule_poll(proxyless_engine):
	phys = proxyless_engine.new_character("physical")
	nonphys = proxyless_engine.new_character("ethereal")

	place0 = phys.new_place(0)
	place1 = phys.new_place(1)
	portl0 = place0.new_portal(place1)
	place2 = phys.new_place(2)
	portl1 = place1.new_portal(place2)

	nonplace0 = nonphys.new_place(0)
	nonplace1 = nonphys.new_place(1)
	nonportl = nonplace0.new_portal(nonplace1)

	@phys.portal.rule(always=True)
	def rule0(portal):
		portal["run"] = True

	@phys.portal.rule
	def rule1(portal):
		portal["notrun"] = False

	proxyless_engine.next_turn()

	assert portl0["run"]
	assert portl1["run"]
	assert "run" not in nonportl
	assert "notrun" not in portl0
	assert "notrun" not in portl1
	assert "notrun" not in nonportl
