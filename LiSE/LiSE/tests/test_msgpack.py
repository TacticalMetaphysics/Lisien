from LiSE import Engine
from LiSE.proxy import EngineProcessManager


def test_serialize_character(engy):
	char = engy.new_character("physical")
	assert engy.unpack(engy.pack(char)) == char


def test_serialize_thing(engy):
	char = engy.new_character("physical")
	place = char.new_place("here")
	thing = place.new_thing("that")
	assert engy.unpack(engy.pack(thing)) == thing


def test_serialize_place(engy):
	char = engy.new_character("physical")
	place = char.new_place("here")
	assert engy.unpack(engy.pack(place)) == place


def test_serialize_portal(engy):
	char = engy.new_character("physical")
	a = char.new_place("a")
	b = char.new_place("b")
	port = a.new_portal(b)
	assert engy.unpack(engy.pack(port)) == port


def test_serialize_function(tmp_path):
	with Engine(
		tmp_path, random_seed=69105, enforce_end_of_time=False, workers=0
	) as eng:

		@eng.function
		def foo(bar: str, bas: str) -> str:
			return bar + bas + " is correct"

	procm = EngineProcessManager()
	engprox = procm.start(tmp_path)
	funcprox = engprox.function.foo
	assert funcprox("foo", "bar") == "foobar is correct"
	procm.shutdown()


def test_serialize_method(tmp_path):
	with Engine(
		tmp_path, random_seed=69105, enforce_end_of_time=False, workers=0
	) as eng:

		@eng.method
		def foo(self, bar: str, bas: str) -> str:
			return bar + bas + " is correct"

	procm = EngineProcessManager()
	engprox = procm.start(tmp_path)
	assert engprox.foo("bar", "bas") == "barbas is correct"
	procm.shutdown()
