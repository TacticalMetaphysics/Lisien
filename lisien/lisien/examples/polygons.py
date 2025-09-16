# Parable of the Polygons is public domain.
# This implementation is part of lisien, a framework for life simulation games.
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
"""Implementation of Parable of the Polygons http://ncase.me/polygons/"""

from operator import attrgetter

from lisien.character import grid_2d_8graph


def install(eng):
	@eng.function
	def cmp_neighbor_shapes(poly, cmp, stat):
		"""Compare the proportion of neighboring polys with the same shape as this one

		Count the neighboring polys that are the same shape as this one, and return how that compares with
		some stat on the poly's user.

		"""
		from operator import attrgetter

		home = poly.location
		similar = 0
		n = 0
		for neighbor_home in sorted(home.neighbors(), key=attrgetter("name")):
			# assume only 1 poly per home for now; this is faithful to the original
			try:
				neighbor = next(
					iter(
						sorted(
							neighbor_home.contents(), key=attrgetter("name")
						)
					)
				)
			except StopIteration:
				continue
			if neighbor.leader is poly.leader:
				similar += 1
		if n == 0:
			# You'd always want to move if you had *no* neighbors, I guess
			return True
		return cmp(poly.character.stat[stat], similar / n)

	@eng.rule(neighborhood=1)
	def relocate(poly):
		"""Move to a random unoccupied place"""
		if "unoccupied" not in poly.engine.universal:
			# .values() sets, like sets generally, are unordered.
			# You have to sort them yourself if you want determinism.
			poly.engine.universal["unoccupied"] = sorted(
				[
					place
					for place in poly.character.place.values()
					if not place.content
				],
				key=attrgetter("name"),
			)
		unoccupied = poly.engine.universal["unoccupied"]
		newloc = unoccupied.pop(poly.engine.randrange(0, len(unoccupied)))
		while not newloc:  # the unoccupied location may have been deleted
			newloc = unoccupied.pop(poly.engine.randrange(0, len(unoccupied)))
		unoccupied.append(poly.location)
		poly.location = newloc

	@relocate.trigger
	def similar_neighbors(poly):
		"""Trigger when my neighborhood fails to be enough like me"""
		from operator import ge

		return poly.engine.function.cmp_neighbor_shapes(
			poly, ge, "min_sameness"
		)

	@relocate.trigger
	def dissimilar_neighbors(poly):
		"""Trigger when my neighborhood gets too much like me"""
		from operator import lt

		return poly.engine.function.cmp_neighbor_shapes(
			poly, lt, "max_sameness"
		)

	eng.rulebook["parable"] = [relocate]

	physical = eng.new_character(
		"physical",
		min_sameness=0.1,
		max_sameness=0.9,
		_config={
			"min_sameness": {"control": "slider", "min": 0.0, "max": 1.0},
			"max_sameness": {"control": "slider", "min": 0.0, "max": 1.0},
		},
		data=grid_2d_8graph(20, 20),
	)
	square = eng.new_character("square")
	triangle = eng.new_character("triangle")
	square.unit.rulebook = triangle.unit.rulebook = "parable"

	empty = sorted(physical.place.values(), key=attrgetter("name"))
	eng.shuffle(empty)
	# distribute 30 of each shape randomly among the empty places
	for i in range(1, 31):
		square.add_unit(
			empty.pop().new_thing(
				"square%i" % i, _image_paths=["atlas://polygons/meh_square"]
			)
		)
	for i in range(1, 31):
		triangle.add_unit(
			empty.pop().new_thing(
				"triangle%i" % i,
				_image_paths=["atlas://polygons/meh_triangle"],
			)
		)


if __name__ == "__main__":
	import shutil
	from tempfile import TemporaryDirectory

	from lisien import Engine

	with TemporaryDirectory() as td:
		with Engine(
			td,
			random_seed=69105,
			connect_string=f"sqlite:///{td}/world.sqlite3",
			workers=0,
		) as eng:
			install(eng)
		archive_filename = shutil.make_archive("polygons", "zip", td)
		print("Exported to " + str(archive_filename))
