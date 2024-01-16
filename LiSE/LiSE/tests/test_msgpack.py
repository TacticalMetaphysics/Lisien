def test_serialize_character(engy):
	char = engy.new_character("physical")
	assert engy.unpack(engy.pack(char)) == char


def test_serialize_thing(engy):
	char = engy.new_character("physical")
	place = char.new_place('here')
	thing = place.new_thing("that")
	assert engy.unpack(engy.pack(thing)) == thing


def test_serialize_place(engy):
	char = engy.new_character("physical")
	place = char.new_place("here")
	assert engy.unpack(engy.pack(place)) == place


def test_serialize_portal(engy):
	char = engy.new_character('physical')
	a = char.new_place('a')
	b = char.new_place('b')
	port = a.new_portal(b)
	assert engy.unpack(engy.pack(port)) == port
