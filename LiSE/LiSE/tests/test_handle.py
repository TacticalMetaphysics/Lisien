import pytest

from LiSE.tests import data


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
	origtime = handle_initialized.get_btt()
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
	handle_initialized.set_thing(
		"hello", "evening", {"location": "bye", "moon": 1.0}
	)
	handle_initialized.add_thing(
		"hello", "moon", "evening", {"phase": "waxing gibbous"}
	)
	handle_initialized.character_set_node_predecessors(
		"hello", "bye", {"hi": {"is-an-edge": True}}
	)
	handle_initialized.add_thing("hello", "neal", "hi", {})
	handle_initialized.add_character("astronauts", {}, {})
	handle_initialized.add_unit("astronauts", "hello", "neal")
	handle_initialized.set_character_rulebook("astronauts", "nasa")
	handle_initialized.set_thing_location("hello", "neal", "moon")
	handle_initialized.set_place("hello", "earth", {})
	handle_initialized.add_portal("hello", "moon", "earth", {})
	assert handle_initialized.thing_travel_to("hello", "neal", "earth") == 1
	kf0 = handle_initialized.snap_keyframe()
	del kf0["universal"]
	assert kf0 == data.KEYFRAME0
	desttime = handle_initialized.get_btt()
	handle_initialized.time_travel(*origtime)
	kf1 = handle_initialized.snap_keyframe()
	del kf1["universal"]
	assert kf1 == data.KEYFRAME1
	handle_initialized.time_travel(*desttime)
	kf2 = handle_initialized.snap_keyframe()
	del kf2["universal"]
	assert kf2 == kf0
