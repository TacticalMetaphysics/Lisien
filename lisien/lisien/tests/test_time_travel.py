from collections import defaultdict


def test_build_keyframe_window(null_engine):
	null_engine._branch_parents = defaultdict(
		set, {"lol": {"trunk"}, "omg": {"trunk", "lol"}, "trunk": {None}}
	)
	null_engine._keyframes_loaded = {
		("lol", 5, 241),
		("lol", 9, 37),
		("lol", 10, 163),
		("lol", 10, 2187),
		("trunk", 0, 0),
		("trunk", 0, 2),
		("trunk", 0, 3),
		("trunk", 5, 240),
		("trunk", 8, 566),
		("trunk", 10, 578),
		("trunk", 10, 3139),
	}
	for b, r, t in null_engine._keyframes_loaded:
		if b in null_engine._keyframes_dict:
			if r in null_engine._keyframes_dict[b]:
				null_engine._keyframes_dict[b][r].add(t)
			else:
				null_engine._keyframes_dict[b][r] = {t}
		else:
			null_engine._keyframes_dict[b] = {r: {t}}
	null_engine._branches_d.update(
		{
			"lol": ("trunk", 5, 240, 10, 3877),
			"omg": ("lol", 5, 241, 5, 241),
			"trunk": (None, 0, 0, 10, 3284),
		}
	)
	assert null_engine._build_keyframe_window("lol", 5, 241) == (
		("lol", 5, 241),
		("lol", 9, 37),
	)
	assert null_engine._build_keyframe_window("omg", 5, 241) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("lol", 5, 242) == (
		("lol", 5, 241),
		("lol", 9, 37),
	)
	assert null_engine._build_keyframe_window("omg", 5, 241) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("omg", 6, 0) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("lol", 5, 240) == (
		("trunk", 5, 240),
		("lol", 5, 241),
	)
	assert null_engine._build_keyframe_window("trunk", 0, 2) == (
		("trunk", 0, 2),
		("trunk", 0, 3),
	)


def test_bookmark(engine):
	engine.bookmark("a")
	engine.next_turn()
	engine.bookmark("b")
	engine.branch = "branch"
	engine.bookmark("c")
	assert engine.time == ("branch", 1, 0)
	engine.bookmark("b")
	assert engine.time == ("trunk", 1, 0)
	engine.bookmark("a")
	assert engine.time == ("trunk", 0, 0)
