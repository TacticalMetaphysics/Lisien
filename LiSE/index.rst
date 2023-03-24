.. LiSE documentation master file, created by
	sphinx-quickstart on Mon Feb 19 10:28:00 2018.
	You can adapt this file completely to your liking, but it should at least
	contain the root `toctree` directive.

LiSE
====
The only LiSE class that you should ever instantiate yourself is
``Engine``. All the other simulation objects should be
created and accessed through it. By default, it keeps the simulation
code and world state in the working directory, but you can pass in another
directory if you prefer. Either use it with a context manager
(``with Engine() as eng:``) or call its ``.close()`` method when you're done
changing things.

The top-level data structure within LiSE is the character. Most
data within the world model is kept in some character or other;
these will quite frequently represent people, but can be readily
adapted to represent any kind of data that can be comfortably
described as a graph or a JSON object. Every change to a character
will be written to the database.

LiSE tracks history as a series of turns. In each turn, each
simulation rule is evaluated once for each of the simulated
entities it's been applied to. World changes in a given turn are
remembered together, such that the whole world state can be
rewound: simply set the properties ``branch`` and ``turn`` back to
what they were just before the change you want to undo.

World Modelling
---------------

Start by calling the engine's ``new_character`` method with a string
``name``.  This will return a character object with the name you
provided. Now draw a map by calling the method ``new_place`` with many
different ``name`` s, then linking them together with the
method ``new_portal(origin, destination)``.  To store data pertaining
to some particular place, retrieve the place from the ``place``
mapping of the character: if the character is ``world`` and the place
name is ``'home'``, you might do it like
``home = world.place['home']``. Portals are retrieved from the ``portal``
mapping, where you'll need the origin and the destination: if there's
a portal from ``'home'`` to ``'narnia'``, you can get it like
``wardrobe = world.portal['home']['narnia']``, but if you haven't also
made another portal going the other way,
``world.portal['narnia']['home']`` will raise ``KeyError``.

Things, usually being located in places (but possibly in other things),
are most conveniently created by the ``new_thing`` method of Place objects:
``alice = home.new_thing('alice')`` gets you a new Thing object
located in ``home``. Things can be retrieved like ``alice = world.thing['alice']``.
Ultimately, things and places are both just nodes, and both can be
retrieved in a character's ``node`` mapping, but only things have
methods like ``travel_to``, which finds a path to a destination
and schedules movement along it.

You can store data in things, places, and portals by treating them
like dictionaries.  If you want to store data in a character, use its
``stat`` property as a dictionary instead. Data stored in these
objects, and in the ``universal`` property of the engine, can vary
over time. The engine's ``eternal`` property is not time-sensitive,
and is mainly for storing settings, not simulation data.

Rule Creation
-------------

To create a rule, first decide what objects the rule should apply
to. You can put a rule on a character, thing, place, or portal; and
you can put a rule on a character's ``thing``, ``place``, and
``portal`` mappings, meaning the rule will be applied to *every* such
entity within the character, even if it didn't exist when the rule was
declared.

All these items have a property ``rule`` that can be used as a
decorator. Use this to decorate a function that performs the rule's
action by making some change to the world state.  Functions decorated
this way always get passed the engine as the first argument and the
character as the second; if the function is more specific than that, a
particular thing, place, or portal will be the third argument. This
will get you a rule object of the same name as your action function.

At first, the rule object will not have any triggers, meaning the action
will never happen. If you want it to run on *every* tick, call its
``always`` method and think no more of it. But if you want to be
more selective, use the rule's ``trigger`` decorator on another
function with the same signature, and have it return ``True`` if the
world is in such a state that the rule ought to run. There is nothing
really stopping you from modifying the rule from inside a trigger, but
it's not recommended.

If you like, you can also add prerequisites. These are like triggers,
but use the ``prereq`` decorator, and should return ``True`` *unless*
the action should *not* happen; if a single prerequisite returns
``False``, the action is cancelled.

Time Control
------------

The current time is always accessible from the engine's ``branch`` and
``turn`` properties. In the common case where time is advancing
forward one tick at a time, it should be done with the engine's
``next_turn`` method, which polls all the game rules before going to
the next tick; but you can also change the time whenever you want, as
long as ``branch`` is a string and ``turn`` is an integer. The rules
will never be followed in response to your changing the time "by
hand".

It is possible--indeed, expected--to change the time as part of the
action of a rule. This is how you would make something happen after a
delay. Say you want a rule that puts the character ``alice`` to sleep,
then wakes her up after eight turns (presumably hour-long).::

	alice = engine.character['alice']

	@alice.rule
	def sleep(character):
		character.stat['awake'] = False
		start_turn = character.engine.turn
		with character.engine.plan():
			character.engine.turn += 8
			character.stat['awake'] = True

At the end of a ``plan():`` block, the game-time will be reset to its
position at the start of that block.

Input Prompts
-------------

LiSE itself doesn't know what a player is or how to accept input from them,
but does use some conventions for communicating with a user interface
such as ELiDE.

To ask the player to make a decision, first define a method for them to
call, then return a menu description like this one.::

	@engine.method
	def wake_alice(self):
		self.character['alice'].stat['awake'] = True

	alice = engine.character['alice']

	@alice.rule
	def wakeup(character):
		return "Wake up?", [("Yes", character.engine.wake_alice), ("No", None)]

Only methods defined with the ``@engine.method`` function store may be used in a menu.

engine
------
.. automodule:: LiSE.engine

	.. autoclass:: LiSE.Engine

		.. autoproperty:: LiSE.Engine.branch

		.. automethod:: is_parent_of

		.. autoproperty:: LiSE.Engine.turn

		.. py:property:: Engine.time

			Acts like a tuple of (branch, turn) for the most part.

			This wraps a :class:`blinker.Signal`. To set a function to be called whenever the
			branch or turn changes, pass it to the ``Engine.time.connect`` method.

		.. py:property:: Engine.rule

			A mapping of all rules that have been made.

		.. py:property:: Engine.rulebook

			A mapping of lists of rules.

			They are followed in their order. A whole rulebook full of rules may be
			assigned to an entity at once.

		.. py:property:: Engine.eternal

			A mapping of arbitrary data, not sensitive to sim-time.

			It's stored in the database. A good place to keep your game's settings.

		.. py:property:: Engine.universal

			A mapping of arbitrary data that changes over sim-time.

			Each turn, the state of the randomizer is saved here under the key ``'rando_state'``.

		.. py:property:: Engine.trigger

			A mapping of, and decorator for, functions that might trigger a rule.

			Decorated functions get stored in the mapping as well as a file, so they can be
			loaded back in when the game is resumed.

		.. py:property:: Engine.prereq

			A mapping of, and decorator for, functions a rule might require to return True for it to run.

		.. py:property:: Engine.action

			A mapping of, and decorator for, functions that might manipulate the world state as a result of a rule running.

		.. py:property:: Engine.method

			A mapping of, and decorator for, extension methods to be added to the engine object.

		.. py:property:: Engine.function

			A mapping of, and decorator for, generic functions.

		.. py:property:: rule

			A mapping of :class:`LiSE.rule.Rule` objects, whether applied to an entity or not.

			Can also be used as a decorator on functions to make them into new rules, with the decorated function as
			their initial action.

		.. py:method:: Engine.next_turn

			Make time move forward in the simulation.

			Stops when the turn has ended, or a rule returns something non-``None``.

			This is also a :class:`blinker.Signal`, so you can register functions to be
			called when the simulation runs. Pass them to ``Engine.next_turn.connect(..)``.

			:return: a pair, of which item 0 is the returned value from a rule if applicable (default: ``[]``),
				and item 1 is a delta describing changes to the simulation resulting from this call.
				See the following method, :method:`get_delta`, for a description of the delta format.

		.. automethod:: advancing

		.. automethod:: batch

		.. automethod:: plan

		.. automethod:: get_delta

		.. automethod:: snap_keyframe

		.. automethod:: delete_plan

		.. automethod:: new_character

		.. automethod:: add_character

		.. automethod:: del_character

		.. automethod:: turns_when

		.. automethod:: apply_choices

		.. automethod:: flush

		.. automethod:: commit

		.. automethod:: close

		.. automethod:: unload

character
---------
.. automodule:: LiSE.character

	.. autoclass:: Character

		.. py:property:: stat

			A mapping of game-time-sensitive data.

		.. py:property:: place

			A mapping of :class:`LiSE.node.Place` objects in this :class:`Character`.

			Has a ``rule`` method for applying new rules to every :class:`Place` here, and a ``rulebook`` property for
			assigning premade rulebooks.

		.. py:property:: thing

			A mapping of :class:`LiSE.node.Thing` objects in this :class:`Character`.

			Has a ``rule`` method for applying new rules to every :class:`Thing` here, and a ``rulebook`` property for
			assigning premade rulebooks.

		.. py:property:: node

			A mapping of :class:`LiSE.node.Thing` and :class:`LiSE.node.Place` objects in this :class:`Character`.

			Has a ``rule`` method for applying new rules to every :class:`Node` here, and a ``rulebook`` property for
			assigning premade rulebooks.

		.. py:property:: portal

			A two-layer mapping of :class:`LiSE.portal.Portal` objects in this :class:`Character`, by origin and destination

			Has a ``rule`` method for applying new rules to every :class:`Portal` here, and a ``rulebook`` property for
			assigning premade rulebooks.

			Aliases:  ``adj``, ``edge``, ``succ``

		.. py:property:: preportal

			A two-layer mapping of :class:`LiSE.portal.Portal` objects in this :class:`Character`, by destination and origin

			Has a ``rule`` method for applying new rules to every :class:`Portal` here, and a ``rulebook`` property for
			assigning premade rulebooks.

			Alias: ``pred``

		.. py:property:: unit

			A mapping of this character's units in other characters.

			Units are nodes in other characters that are in some sense part of this one. A common example in strategy
			games is when a general leads an army: the general is one :class:`Character`, with a graph representing the
			state of their AI; the battle map is another :class:`Character`; and the general's units, though not in the
			general's :class:`Character`, are still under their command, and therefore follow rules defined on the
			general's ``unit.rule`` subproperty.

		.. automethod:: add_portal

		.. automethod:: new_portal

		.. automethod:: add_portals_from

		.. automethod:: add_thing

		.. automethod:: new_thing

		.. automethod:: add_things_from

		.. automethod:: add_place

		.. automethod:: add_places_from

		.. automethod:: new_place

		.. automethod:: historical

		.. automethod:: place2thing

		.. automethod:: portals

		.. automethod:: remove_portal

		.. automethod:: remove_unit

		.. automethod:: thing2place

		.. automethod:: units

		.. automethod:: facade

node
----
.. automodule:: LiSE.node

	.. autoclass:: LiSE.node.Node

		.. py:property:: user

			A mapping of the characters that have this node as an avatar.

			When there's only one user, you can use the special sub-property
			``Node.user.only`` to get it.

		.. autoproperty:: portal

		.. autoproperty:: preportal

		.. autoproperty:: content

		.. automethod:: contents

		.. automethod:: successors

		.. automethod:: predecessors

		.. automethod:: shortest_path

		.. automethod:: shortest_path_length

		.. automethod:: path_exists

		.. automethod:: new_portal

		.. automethod:: new_thing

		.. automethod:: historical

		.. automethod:: delete

	.. autoclass:: Place
		:members:

	.. autoclass:: Thing
		:members:

portal
------
.. automodule:: LiSE.portal

	.. autoclass:: Portal

		.. py:attribute:: origin

			The :class:`LiSE.node.Place` or :class:`LiSE.node.Thing` that this leads out from

		.. py:attribute:: destination

			The :class:`LiSE.node.Place` or :class:`LiSE.node.Thing` that this leads into

		.. py:property:: character

			The :class:`LiSE.character.Character` that this is in

		.. py:property:: engine

			The :class:`LiSE.engine.Engine` that this is in

		.. autoproperty:: reciprocal

		.. automethod:: historical

		.. automethod:: delete

rule
----
.. automodule:: LiSE.rule

	.. autoclass:: Rule
		:members:

query
-----
.. automodule:: LiSE.query

	.. autoclass:: QueryResult

xcollections
------------
.. automodule:: LiSE.xcollections

	.. autoclass:: StringStore

	.. autoclass:: FunctionStore
