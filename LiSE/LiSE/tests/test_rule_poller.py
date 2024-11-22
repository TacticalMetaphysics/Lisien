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
