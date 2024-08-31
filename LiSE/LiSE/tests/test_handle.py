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
			"node": {"hi": {"yes": "very yes"}, "hello": {"you": "smart"}},
			"edge": {"hi": {"hello": {"good": "morning"}}},
		},
		{"stat": "also"},
	)
	handle_initialized.set_character_stat("hello", "stoat", "bitter")
	handle_initialized.del_character_stat("hello", "stat")
	handle_initialized.del_character("physical")
	kf = handle_initialized.snap_keyframe()
	del kf["universal"]
	assert kf == {
		"graph_val": {("hello",): {"stat": None, "stoat": "bitter"}},
		"nodes": {("hello",): {"hi": True, "hello": True}},
		"node_val": {
			("hello", "hi"): {"name": "hi", "yes": "very yes"},
			("hello", "hello"): {"name": "hello", "you": "smart"},
		},
		"edges": {("hello", "hi", "hello"): {0: True}},
		"edge_val": {("hello", "hi", "hello", 0): {"good": "morning"}},
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
