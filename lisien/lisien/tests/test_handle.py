import pytest

from lisien import Engine
from lisien.proxy.handle import EngineHandle


@pytest.fixture
def handle_empty(tmp_path, database):
	Engine(
		tmp_path if database == "parquetdb" else None,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	).close()
	handle = EngineHandle(
		tmp_path if database == "parquetdb" else None,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	)
	yield handle
	handle.close()


def test_language(handle_empty):
	assert handle_empty.get_language() == "eng"
	handle_empty.set_string("foo", "bar")
	assert handle_empty.get_string_lang_items("eng") == [("foo", "bar")]
	handle_empty.set_language("esp")
	assert handle_empty.get_language() == "esp"
	assert handle_empty.get_string_lang_items("esp") == []
	assert handle_empty.get_string_lang_items() == []
	handle_empty.set_language("eng")
	assert handle_empty.get_string_lang_items() == [("foo", "bar")]
	assert handle_empty.strings_copy() == {"foo": "bar"}
	handle_empty.del_string("foo")
	handle_empty.set_language("esp")
	assert handle_empty.strings_copy("eng") == {}


def test_eternal(handle_empty, database):
	unpack = handle_empty.unpack
	assert unpack(handle_empty.get_eternal("_lisien_schema_version")) == 0
	assert unpack(handle_empty.get_eternal("main_branch")) == "trunk"
	assert unpack(handle_empty.get_eternal("language")) == "eng"
	handle_empty.set_eternal("haha", "lol")
	assert unpack(handle_empty.get_eternal("haha")) == "lol"
	handle_empty.del_eternal("branch")
	with pytest.raises(KeyError):
		handle_empty.get_eternal("branch")
	assert handle_empty.eternal_copy() == {
		b"\xb6_lisien_schema_version": b"\x00",
		b"\xabmain_branch": b"\xa5trunk",
		b"\xa4turn": b"\x00",
		b"\xa4tick": b"\x00",
		b"\xa8language": b"\xa3eng",
		b"\xa4haha": b"\xa3lol",
	}


def test_universal(handle_empty):
	handle_empty.set_universal("foo", "bar")
	handle_empty.set_universal("spam", "tasty")
	univ = handle_empty.snap_keyframe()["universal"]
	assert univ["foo"] == "bar"
	assert univ["spam"] == "tasty"
	handle_empty.del_universal("foo")
	univ = handle_empty.snap_keyframe()["universal"]
	assert "foo" not in univ
	assert univ["spam"] == "tasty"


def test_character(handle_initialized):
	origtime = handle_initialized.get_btt()
	handle_initialized.next_turn()
	handle_initialized.add_character(
		"hello",
		node={
			"hi": {"yes": "very yes"},
			"hello": {"you": "smart"},
			"morning": {"good": 100},
			"salutations": {},
			"me": {"location": "hi"},
		},
		edge={"hi": {"hello": {"good": "morning"}}},
		stat="also",
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
	assert kf0 == handle_initialized.keyframe[0]
	desttime = handle_initialized.get_btt()
	handle_initialized.time_travel(*origtime)
	kf1 = handle_initialized.snap_keyframe()
	del kf1["universal"]
	assert kf1 == handle_initialized.keyframe[1]
	handle_initialized.time_travel(*desttime)
	kf2 = handle_initialized.snap_keyframe()
	del kf2["universal"]
	assert kf2 == kf0
