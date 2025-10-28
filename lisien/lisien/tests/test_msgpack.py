# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from lisien import Engine
from lisien.proxy.manager import EngineProxyManager


def test_serialize_character(sqleng):
	char = sqleng.new_character("physical")
	assert sqleng.unpack(sqleng.pack(char)) == char


def test_serialize_thing(sqleng):
	char = sqleng.new_character("physical")
	place = char.new_place("here")
	thing = place.new_thing("that")
	assert sqleng.unpack(sqleng.pack(thing)) == thing


def test_serialize_place(sqleng):
	char = sqleng.new_character("physical")
	place = char.new_place("here")
	assert sqleng.unpack(sqleng.pack(place)) == place


def test_serialize_portal(sqleng):
	char = sqleng.new_character("physical")
	a = char.new_place("a")
	b = char.new_place("b")
	port = a.new_portal(b)
	assert sqleng.unpack(sqleng.pack(port)) == port


def test_serialize_function(tmp_path):
	with Engine(
		tmp_path, random_seed=69105, enforce_end_of_time=False, workers=0
	) as eng:

		@eng.function
		def foo(bar: str, bas: str) -> str:
			return bar + bas + " is correct"

	procm = EngineProxyManager(workers=0)
	try:
		engprox = procm.start(tmp_path)
		funcprox = engprox.function.foo
		assert funcprox("foo", "bar") == "foobar is correct"
	finally:
		procm.shutdown()


def test_serialize_method(tmp_path):
	with Engine(
		tmp_path, random_seed=69105, enforce_end_of_time=False, workers=0
	) as eng:

		@eng.method
		def foo(self, bar: str, bas: str) -> str:
			return bar + bas + " is correct"

	procm = EngineProxyManager()
	try:
		engprox = procm.start(tmp_path, workers=0)
		assert engprox.foo("bar", "bas") == "barbas is correct"
	finally:
		procm.shutdown()
