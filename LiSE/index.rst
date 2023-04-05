.. LiSE documentation master file, created by
	sphinx-quickstart on Mon Feb 19 10:28:00 2018.
	You can adapt this file completely to your liking, but it should at least
	contain the root `toctree` directive.

LiSE
====

engine
------
.. automodule:: LiSE.engine

	.. autoclass:: LiSE.Engine

		.. autoproperty:: LiSE.Engine.branch

		.. automethod:: is_ancestor_of

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

			The state of the randomizer is saved here under the key ``'rando_state'``.

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
				See the following method, :meth:`get_delta`, for a description of the delta format.

		.. automethod:: get_delta

		.. automethod:: advancing

		.. automethod:: batch

		.. automethod:: plan

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

		.. autoproperty:: user

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
