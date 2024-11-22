def test_character_rule_poll(engy):
	phys = engy.new_character("physical")
	notphys = engy.new_character("ethereal")

	@phys.rule(always=True)
	def hello(char):
		char.stat["run"] = True

	@notphys.rule
	def goodbye(char):
		char.stat["run"] = True

	engy.next_turn()

	assert "run" in phys.stat
	assert "run" not in notphys.stat


def test_unit_rule_poll(engy):
	phys = engy.new_character("physical")
	notphys = engy.new_character("ethereal")

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

	engy.next_turn()

	assert unit["run"]
	assert "run" not in notunit1
	assert "run" not in notunit2
