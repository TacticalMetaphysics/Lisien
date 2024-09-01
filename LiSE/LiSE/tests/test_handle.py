import pytest


def test_btt(handle_initialized):
	assert handle_initialized.get_btt() == ("trunk", 0, 156)


def test_language(handle_initialized):
	assert handle_initialized.get_language() == "eng"
	handle_initialized.set_string("foo", "bar")
	assert handle_initialized.get_string_lang_items("eng") == [("foo", "bar")]
	handle_initialized.set_language("esp")
	assert handle_initialized.get_language() == "esp"
	assert handle_initialized.get_string_lang_items("esp") == []
	assert handle_initialized.get_string_lang_items() == []
	handle_initialized.set_language("eng")
	assert handle_initialized.get_string_lang_items() == [("foo", "bar")]
	assert handle_initialized.strings_copy() == {"foo": "bar"}
	handle_initialized.del_string("foo")
	handle_initialized.set_language("esp")
	assert handle_initialized.strings_copy("eng") == {}


def test_eternal(handle_initialized):
	unpack = handle_initialized.unpack
	assert unpack(handle_initialized.get_eternal("_lise_schema_version")) == 0
	assert unpack(handle_initialized.get_eternal("main_branch")) == "trunk"
	assert unpack(handle_initialized.get_eternal("language")) == "eng"
	handle_initialized.set_eternal("haha", "lol")
	assert unpack(handle_initialized.get_eternal("haha")) == "lol"
	handle_initialized.del_eternal("branch")
	with pytest.raises(KeyError):
		handle_initialized.get_eternal("branch")
	assert handle_initialized.eternal_copy() == {
		b"\xb4_lise_schema_version": b"\x00",
		b"\xabmain_branch": b"\xa5trunk",
		b"\xa4turn": b"\x00",
		b"\xa4tick": b"\x00",
		b"\xa8language": b"\xa3eng",
		b"\xa4haha": b"\xa3lol",
	}


def test_universal(handle_initialized):
	handle_initialized.set_universal("foo", "bar")
	handle_initialized.set_universal("spam", "tasty")
	univ = handle_initialized.snap_keyframe()["universal"]
	assert univ["foo"] == "bar"
	assert univ["spam"] == "tasty"
	handle_initialized.del_universal("foo")
	univ = handle_initialized.snap_keyframe()["universal"]
	assert "foo" not in univ
	assert univ["spam"] == "tasty"


def test_character(handle_initialized):
	handle_initialized.add_character(
		"hello",
		{
			"node": {
				"hi": {"yes": "very yes"},
				"hello": {"you": "smart"},
				"morning": {"good": 100},
				"salutations": {},
			},
			"thing": {"me": {"location": "hi"}},
			"edge": {"hi": {"hello": {"good": "morning"}}},
		},
		{"stat": "also"},
	)
	assert handle_initialized.node_exists("hello", "hi")
	handle_initialized.set_character_stat("hello", "stoat", "bitter")
	handle_initialized.del_character_stat("hello", "stat")
	handle_initialized.set_node_stat("hello", "hi", "no", "very no")
	handle_initialized.del_node_stat("hello", "hi", "yes")
	handle_initialized.del_character("physical")
	handle_initialized.del_node("hello", "salutations")
	handle_initialized.update_nodes(
		"hello",
		{"hi": {"tainted": True}, "bye": {"toodles": False}, "morning": None},
	)
	handle_initialized.character_set_node_predecessors(
		"hello", "bye", {"hi": {"is-an-edge": True}}
	)
	kf = handle_initialized.snap_keyframe()
	del kf["universal"]
	assert kf == {
		"graph_val": {("hello",): {"stat": None, "stoat": "bitter"}},
		"nodes": {
			("hello",): {"hi": True, "hello": True, "me": True, "bye": True}
		},
		"node_val": {
			("hello", "hi"): {"name": "hi", "no": "very no", "tainted": True},
			("hello", "hello"): {"name": "hello", "you": "smart"},
			("hello", "me"): {"name": "me", "location": "hi"},
			("hello", "bye"): {"name": "bye", "toodles": False},
		},
		"edges": {
			("hello", "hi", "hello"): {0: True},
			("hello", "bye", "hi"): {0: True},
		},
		"edge_val": {
			("hello", "hi", "hello", 0): {"good": "morning"},
			("hello", "bye", "hi", 0): {},
		},
		"triggers": {
			"shrubsprint": ("uncovered", "breakcover"),
			"fight": ("sametile",),
			"kill_kobold": ("kobold_alive",),
			"go2kobold": ("aware",),
			"wander": ("standing_still",),
		},
		"prereqs": {
			"shrubsprint": ("not_traveling",),
			"fight": ("kobold_alive", "aware"),
			"kill_kobold": ("unmerciful",),
			"go2kobold": ("kobold_alive", "kobold_not_here"),
			"wander": (),
		},
		"actions": {
			"shrubsprint": ("shrubsprint",),
			"fight": ("fight",),
			"kill_kobold": ("kill_kobold",),
			"go2kobold": ("go2kobold",),
			"wander": ("wander",),
		},
		"rulebook": {
			("physical", "kobold"): (["shrubsprint"], 0.0),
			("physical", "dwarf"): (
				["fight", "kill_kobold", "go2kobold", "wander"],
				0.0,
			),
		},
	}
