Lisien is a tool for developing life simulation games.

[Download on Itch](https://clayote.itch.io/Lisien)

[Discuss on Gamemaking Tools Forum](https://gamemaking.tools/forum/categories/lisien)

[Read Documentation](https://clayote.codeberg.page/lisien/manual.html)


<a rel="me" href="https://peoplemaking.games/@clayote">Follow on Mastodon</a>

# What is a life simulation game?

Life simulation games simulate the world in relatively high detail,
but not in the sense that physics engines are concerned with --
rather, each game in the genre has a different take on the mundane
events that constitute everyday life for its purpose. Logistics and
RPG elements tend to feature heavily in these games, but a lot of the
appeal is in the parts that are not directly under player control.
The game world feels like it continues to exist when you're not playing
it, because so much of it seems to operate independently of you.

Existing games that Lisien seeks to imitate include:

* The Sims
* SimLife (1992)
* Redshirt
* Rimworld
* Princess Maker
* Monster Rancher
* Dwarf Fortress
* Democracy
* Crusader Kings
* The King of Dragon Pass
* [Galimulator](https://snoddasmannen.itch.io/galimulator)
* [Vilmonic](https://bludgeonsoft.itch.io/)

# Why should I use Lisien for this purpose?

Lisien assumes
that there are certain problems any designer of life simulators will
have, and provides powerful tools specialized to those
problems. Though you will still need to write some Python code for
your game, it should only be the code that describes how your game's
world works. If you don't want to worry about the data structure that
represents the world, Lisien gives you one that will work. And if you
don't want to write a user interface, you can play the game in the
IDE.

The Lisien data model has been designed from the ground up to support
debugging of complex simulations. It remembers everything that ever
happened in the simulation, so that when something strange happens
in a playtester's game, and they send you their save file, you can
track down the cause, even if it happened long before the
tester knew to look for it.

# Features

## Core

* *Infinite time travel*, rendering traditional save files obsolete.
* Integration with [NetworkX](http://networkx.github.io) for
  convenient access to various *graph algorithms*, particularly
  pathfinding.
* *Rules engine* for game logic. Rules are written in plain
  Python. They are composable, and can be disabled or reassigned to
  different entities mid-game.
* Can be run as a *web server*, so that you can control Lisien and query
  its world state from any other game engine you please.

## IDE

* *Instant replay*: go back to previous world states whenever you feel
  like it, with minimal loading.
* *Rule constructor*: Build rules out of short functions representing
  triggers and actions.
* *Rule stepper*: view the world state in the middle of a turn, just
  after one rule's run and before another.
* Edit state in graph or grid view
* Edit rule functions with syntax highlighting

# Setup

Lisien is available [on PyPI](https://pypi.org/project/Lisien/), so
`pip install lisien elide` will work, but won't always have the latest experimental
code. If you want that, then in a command line, with [Python](https://python.org)
(preferably [version 3.12](https://www.python.org/downloads/release/python-3129/)) already installed, run:

```
python -m pip install --user --upgrade https://codeberg.org/clayote/Lisien/archive/main.zip
```

Run it again whenever you want the latest Lisien code.

# Getting started

You could now start the graphical frontend with ``python -m elide``,
but this might not be very useful, as you don't have any world state
to edit yet. You could laboriously assemble a gameworld by hand, but
instead let's generate one, Parable of the Polygons by Nicky Case.

Make a new Python script, let's say 'polygons.py', and write the
following in it (or use the [example
version](https://codeberg.org/clayote/Lisien/src/branch/main/Lisien/Lisien/examples/polygons.py)):

```python
from lisien import Engine
import networkx as nx

with Engine(clear=True) as eng:
  phys = eng.new_character('physical', nx.grid_2d_graph(20, 20))
  tri = eng.new_character('triangle')
  sq = eng.new_character('square')
```

This starts a new game with its world state stored in the file
'world.db'. Because of ``clear`` being ``True``, it will delete any
existing world state and game code each time it's run, which is often
useful when you're getting started. It creates three characters, one
of which, named 'physical', has a 20x20 grid in it.  The others are
empty, and in fact we don't intend to put any graph in them; they're
just for keeping track of things in ``physical``. Add the following
inside the `with` block of `polygons.py`:

```python
	empty = list(phys.place.values())
	eng.shuffle(empty)
	# distribute 30 of each shape randomly among the empty places
	for i in range(1, 31):
		place = empty.pop()
		square = place.new_thing('square%i' % i, _image_paths=['atlas://polygons/meh_square'])
		sq.add_unit(square)
	for i in range(1, 31):
		place = empty.pop()
		triangle = place.new_thing('triangle%i' % i, _image_paths=['atlas://polygons/meh_triangle'])
		tri.add_unit(triangle)
```

Now there are thirty each of squares and triangles in the world. They
are things, rather than places, which just means they have locations
-- each square and triangle is located in a place in the graph.

The new_thing method of a place object creates a new thing and puts it
there. You have to give the thing a name as its first argument. You
can supply further keyword arguments to customize the thing's stats;
in this case, I've given the things graphics representing what shape
they are. If you wanted, you could set the `_image_paths` to a list of
paths to whatever graphics. The 'atlas://' in the front is only
necessary if you're using graphics packed in the way that the default
ones are; [read about
atlases](https://kivy.org/doc/stable/api-kivy.atlas.html) if you like,
or just use some .png files you have lying around.

The add_unit method of a character object marks a thing or place so
that it's considered part of a character whose graph it is not
in. This doesn't do anything yet, but we'll be using it to write our
rules in a little while.

Now we have our world, but nothing ever happens in it. Let's add the
rules of the simulation:

```python
	@eng.function
	def cmp_neighbor_shapes(poly, cmp, stat):
		"""Compare the proportion of neighboring polys with the same shape as this one
	
		Count the neighboring polys that are the same shape as this one, and return how that compares with
		some stat on the poly's user.
	
		"""
		home = poly.location
		similar = 0
		n = 0
		# iterate over portals leading outward from home
		for neighbor_home in home.neighbors():
			n += 1
			# there's really only 1 polygon per home right now, but this will still work if there are more
			for neighbor in neighbor_home.contents():
				if neighbor.user is poly.user:
					similar += 1
		return cmp(poly.user.stat[stat], similar / n)
	
	
	@phys.thing.rule(neighborhood=1)
	def relocate(poly):
		"""Move to a random unoccupied place"""
		unoccupied = [place for place in poly.character.place.values() if not place.content]
		poly.location = poly.engine.choice(unoccupied)
	
	
	@relocate.trigger
	def similar_neighbors(poly):
		"""Trigger when my neighborhood fails to be enough like me"""
		from operator import ge
		return poly.engine.function.cmp_neighbor_shapes(poly, ge, 'min_sameness')
	
	
	@relocate.trigger
	def dissimilar_neighbors(poly):
		"""Trigger when my neighborhood gets too much like me"""
		from operator import lt
		return poly.engine.function.cmp_neighbor_shapes(poly, lt, 'max_sameness')
```

The core of this ruleset is the ``cmp_neighbor_shapes`` function,
which is a plain Python function that I've chosen to store in the
engine because that makes it easier for the rules to get at.
Functions decorated with ``@engine.function`` become accessible as
attributes of ``engine.function``.  Every Lisien entity has an attribute
``engine`` that you can use to get at that function store and lots of
other utilities.

If you didn't want to use the function store, you could just import
``cmp_neighbor_shapes`` in every rule that uses it, like I've done
with the operators ``ge`` and ``lt`` here.

``cmp_neighbor_shapes`` looks over the places that are directly
connected to the one a given shape is in, counts the number that
contain the same shape, and compares the result to a stat of the
``user``--the character of which this thing is a unit. When called
in ``similar_neighbors`` and ``dissimilar_neighbors``, the stats in
question are 'min_sameness' and 'max_sameness' respectively, so let's
set those:

```python
	sq.stat['min_sameness'] = 0.1
	sq.stat['max_sameness'] = 0.9
	tri.stat['min_sameness'] = 0.2
	tri.stat['max_sameness'] = 0.8
```

Here we diverge from the original simulation a bit by setting these
values differently for the different shapes, demonstrating an
advantage of units.

The argument `neighborhood=1` to `@phys.thing.rule` tells it that it only
needs to check its triggers if something changed in either the location of
the thing in question, or its neighbor places. `neighborhood=2` would include
neighbors of those neighbors as well, and so on. You never really *need* this,
but it makes this simulation go fast.

Run ``python3 polygons.py`` to generate the simulation. To view it,
run ``python3 -m elide`` in the same directory.  Just click the big
&gt; button and watch it for a little while. There's a control panel
on the bottom of the screen that lets you go back in time, if you
wish, and you can use that to browse different runs of the simulation
with different starting conditions, or even stats and rules
arbitrarily changing in the middle of a run.

If you'd prefer to run the simulation without Elide, though, you can add this to your script:
```python
	for i in range(10):
		eng.next_turn()
```

Every change to the world will be saved in the database so that you
can browse it in Elide at your leisure.  If you want to travel through
time programmatically, set the properties ``eng.branch`` (to a
string), ``eng.turn``, and ``eng.tick`` (to integers).

To prevent locking when running `next_turn()`, you might want to run Lisien in a subprocess. This is done by
instantiating `Lisien.proxy.EngineProcessManager()`,
getting a proxy to the engine from its `start()` method, and treating that proxy much as you would an actual Lisien
engine, except that you can call `next_turn()` in a thread and then do something else in parallel. Call
`EngineProcessManager.shutdown()` when it's time to quit the game.

What next? If you wanted, you could set rules to be followed by only
some of the shapes, like so:

```python
	# this needs to replace any existing rule code you've written,
	# it won't work so well together with eg. @phys.thing.rule
	@tri.unit.rule(neighborhood=1)
	def tri_relocate(poly):
		"""Move to a random unoccupied place"""
		unoccupied = [place for place in poly.character.place.values() if not place.content]
		poly.location = poly.engine.choice(unoccupied)
	
	
	@tri_relocate.trigger
	def similar_neighbors(poly):
		"""Trigger when my neighborhood fails to be enough like me"""
		from operator import ge
		return poly.engine.function.cmp_neighbor_shapes(poly, ge, 'min_sameness')
	
	
	@sq.unit.rule(neighborhood=1)
	def sq_relocate(poly):
		"""Move to a random unoccupied place"""
		unoccupied = [place for place in poly.character.place.values() if not place.content]
		poly.location = poly.engine.choice(unoccupied)
	
	
	@sq_relocate.trigger
	def dissimilar_neighbors(poly):
		"""Trigger when my neighborhood gets too much like me"""
		from operator import lt
		return poly.engine.function.cmp_neighbor_shapes(poly, lt, 'max_sameness')
```

Now the triangles only relocate whenever their neighborhood looks too
much like them, whereas squares only relocate when they have too many
triangle neighbors.

When you have a set of rules that needs to apply to many
entities, and you can't just make them all units, you can have the
entities share a rulebook. This works:

```python
	sq.unit.rulebook = tri.unit.rulebook
```

And would result in pretty much the same simulation as in the first
place, with all the shapes following the same rules, but now you could
have other things in ``phys``, and they wouldn't necessarily follow
those rules.

Or you could build a rulebook ahead-of-time and assign it to many
entities:

```python
	# this needs to replace any existing rule code you've written,
	# it won't work so well together with eg. @phys.thing.rule
	@eng.rule(neighborhood=1)
	def relocate(poly):
		"""Move to a random unoccupied place"""
		unoccupied = [place for place in poly.character.place.values() if not place.content]
		poly.location = poly.engine.choice(unoccupied)
	
	
	@relocate.trigger
	def similar_neighbors(poly):
		"""Trigger when my neighborhood fails to be enough like me"""
		from operator import ge
		return poly.engine.function.cmp_neighbor_shapes(poly, ge, 'min_sameness')
	
	
	@relocate.trigger
	def dissimilar_neighbors(poly):
		"""Trigger when my neighborhood gets too much like me"""
		from operator import lt
		return poly.engine.function.cmp_neighbor_shapes(poly, lt, 'max_sameness')
	
	
	# rulebooks need names too, so you have to make it like this
	eng.rulebook['parable'] = [relocate]
	sq.rulebook = tri.rulebook = 'parable'
```

# Making a game

Elide is meant to support repurposing its widgets to build a rudimentary graphical
interface for a game. For an example of what that might look like, see
[the Awareness sim](https://codeberg.org/clayote/Lisien/src/branch/main/elide/elide/examples/awareness.py).
You may prefer to work with some other Python-based game engine, such as
[Pyglet](http://pyglet.org/) or [Ursina](https://www.ursinaengine.org/), in which case you don't really need Elide--
though you may find it useful to open Elide in your game folder when you're
trying to track down a bug.


# License Information

Elide uses third-party graphics sets:

* The [RLTiles](http://rltiles.sourceforge.net/), available under
  [CC0](http://creativecommons.org/publicdomain/zero/1.0/), being in
  the public domain where it exists.
* The Elide icon is by Robin Hill, used with permission.
* Everything else is by [Kenney](https://kenney.nl), available under
  [CC0](http://creativecommons.org/publicdomain/zero/1.0/).

The allegedb, Lisien, and Elide source files are licensed under the
terms of the GNU Affero Public License version 3 (and no later).
If you make a game with it, you have to release any modifications you
make to Elide, allegedb, or Lisien itself under the AGPL, but 
this doesn't apply to your game code.

Game code is that which is loaded into the engine at
launch time, either from a file named ``game_start.py`` in the game prefix,
or from modules specified by the following parameters
to the Lisien engine:
* ``trigger``
* ``prereq``
* ``action``
* ``function``
* ``method``

Or stored in files by those names (plus extensions) inside the game's prefix. 
Game code must not alter the function of Lisien itself (no "hot patching"). If 
it does, then it is part of Lisien.

If you write another application (not using any allegedb, Lisien, or
Elide code) that accesses a Lisien server via HTTP(S), it is separate
from Lisien and not subject to its license. If you run Lisien in a
Python interpreter embedded into your application, the Lisien
license only covers Lisien itself, and not any code run outside
of that Python interpreter. You must still release any modifications
you make to Lisien, but the embedding application remains your own.
