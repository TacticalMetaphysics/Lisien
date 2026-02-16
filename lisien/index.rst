=======
  API
=======

Here is the portion of Lisien's codebase that game developers should be
familiar with. Generally, you should access everything through an
:class:`lisien.engine.Engine` object. The exception is when Lisien is not
running in the same process, in which case, you'll need
:class:`lisien.proxy.manager.EngineProxyManager` to make you a proxy to it.
But :class:`lisien.proxy.engine.EngineProxy` works just like
:class:`lisien.engine.Engine`, for the most part.

########
 engine
########

.. automodule:: lisien.engine

   .. autoclass:: Engine

    .. autoproperty:: branch

    .. autoproperty:: trunk

    .. automethod:: is_ancestor_of

    .. autoproperty:: turn

    .. autoproperty:: tick

    .. py:property:: time

        A tuple-like object holding the *current* ``branch, turn, tick`` --
        which may not be the same as when last you accessed ``time``.

        To register a function ``time_passes`` to be called when the time changes, pass
        it to ``time.connect``::

            engine.time.connect(time_passes)

        Or decorate it::

            @engine.time.connect
            def time_passes(
                time,
                then: tuple[str, int, int],
                now: tuple[str, int, int]
            ):
                ...

        It will be passed the time object itself, as well as tuples of the
        previous time ``then`` and the current time ``now``.
    .. py:property:: rule

        A mapping of all :class:`lisien.rule.Rule` objects that have been made.

        It's possible to make new rules with this. It works the same way as the
        ``rule`` decorator that all Lisien entities have, but doesn't result
        in the rule being assigned to anything::

            @engine.rule
            def do_something(obj):
                ...

            print(type(do_something))

        This prints ``<class 'lisien.rule.Rule'>``. You'll need to put the rule
        in a rulebook yourself.

    .. py:property:: rulebook

        A mapping of :class:`lisien.rule.Rulebook` objects that exist.

        Rulebooks must be assigned to entities' ``rulebook`` property before
        they will run, but each entity gets its own rulebook by default.
        Those rulebooks appear here when the entity is created.

        Rules created with an entity's ``@rule`` decorator go on the end of
        the entity's rulebook. You can reorder that rulebook the same way as
        a Python list, or put :class:`lisien.rule.Rule` objects in it yourself.
        For convenience, :class:`lisien.rule.Rulebook` will interpret
        the names of rules as if they were the :class:`lisien.rule.Rule`
        by that name.

    .. py:property:: eternal

        A mapping of arbitrary data, not sensitive to changes in :attr:`time`.

        Data kept here will be stored in the same database as the game itself,
        and will be exported with all the rest of the game's data if you call
        :meth:`export`. This makes :attr:`eternal` a good place to keep
        the settings of the game.

        There are a few special keys here you shouldn't set directly:

        * ``"branch"``, ``"turn"``, and ``"tick"`` hold the last :attr:`time` that
            the game was at. If you close and reopen the same game, it will
            resume at that time.
        * ``"language"`` is the three-letter code of the language the game
            displays to the player. To change it, for instance, to French::

                engine.string.language = "fra"

            Each language's strings are kept in a JSON file in the game's
            ``strings/`` directory. These may be edited while the game is not
            running, so long as all the expected strings are present when the
            game either starts with a given language, or switches to it.
        * ``"trunk"`` is the name of the branch that the game started from.
            To start a new game, which shares no history with the current game,
            go to tick 0 of turn 0 and set the engine's :attr:`trunk` attribute
            to the name of your new trunk.
        * ``"_lisien_schema_version"`` is an integer used to check compatibility.
    .. py:property:: universal

         A mapping of arbitrary data that changes over :attr:`time`.

         The state of the randomizer is saved here under the key ``"rando_state"``.
         It's a bad idea to set that directly; instead, set :attr:`random_seed`.

    .. py:property:: trigger

        A module of, and decorator for, functions that might trigger a rule.

        Decorated functions get stored in the module, so they can be
        loaded back in when the game is resumed.

    .. py:property:: prereq

        A module of, and decorator for, functions a rule might require to return
        True for it to run.

    .. py:property:: action

        A module of, and decorator for, functions that might manipulate the
        world state as a result of a rule running.

    .. py:property:: method

        A module of, and decorator for, extension methods to be added to an
        :class:`Engine` object.

    .. py:property:: function

        A module of, and decorator for, generic functions.

        Only functions stored in one of these attributes can be used for
        parallel processing. :attr:`function` will do, if none of the others
        make sense.

    .. py:property:: rule

         A mapping of :class:`lisien.rule.Rule` objects, whether applied to an entity or not.

         Can also be used as a decorator on functions to make them into new rules, with the decorated function as
         their initial action.

    .. py:method:: next_turn() -> tuple[list, DeltaDict]

         Make time move forward in the simulation.

         Stops when the turn has ended, or a rule returns something non-``None``.

         This is also a :class:`blinker.Signal`, so you can register functions to be
         called when the simulation runs. Pass them in like::

            from lisien import Engine

            def it_is_time(engine, branch, turn, tick):
                print((branch, turn, tick))

            with Engine() as engine:
                engine.next_turn.connect(it_is_time)

         Or decorate them::

            from lisien import Engine

            with Engine() as engine:
                @engine.next_turn.connect
                def it_is_time(engine, branch, turn, tick):
                    print((branch, turn, tick))

         :return: a pair, of which item 0 is the returned value from a rule if applicable (default: ``[]``),
                 and item 1 is a delta describing changes to the simulation resulting from this call.
                 See the following method, :meth:`get_delta`, for a description of the delta format.

    .. automethod:: get_delta

    .. automethod:: advancing

    .. automethod:: batch

    .. automethod:: plan

    .. automethod:: delete_plan

    .. automethod:: snap_keyframe

    .. automethod:: new_character

    .. automethod:: add_character

    .. automethod:: del_character

    .. automethod:: turns_when

    .. automethod:: apply_choices

    .. automethod:: flush

    .. automethod:: commit

    .. automethod:: close

    .. automethod:: unload

    .. automethod:: export

    .. automethod:: from_archive

###########
 character
###########

.. automodule:: lisien.character

   .. autoclass:: Character

      .. py:property:: stat

         A mapping of game-time-sensitive data.

      .. py:property:: place

         A mapping of :class:`lisien.node.Place` objects in this :class:`Character`.
         :class:`lisien.node.Thing` objects may be located in these.

         This mapping has a ``rule`` method for applying new rules to every
         :class:`Place` here, and a ``rulebook`` property for assigning premade
         rulebooks.

      .. py:property:: thing

         A mapping of :class:`lisien.node.Thing` objects in this :class:`Character`,
         representing the type of node that can be located in another node--either
         a :class:`lisien.node.Place` or a :class:`lisien.node.Thing`.

         This mapping a ``rule`` method for applying new rules to every
         :class:`Thing` here, and a ``rulebook`` property for
         assigning premade rulebooks.

      .. py:property:: node

         A mapping of :class:`lisien.node.Thing` and :class:`lisien.node.Place`
         objects in this :class:`Character`.

         This mapping has a ``rule`` method for applying new rules to every
         :class:`Node` here, and a ``rulebook`` property for
         assigning premade rulebooks.

      .. py:property:: unit

         A two-layer mapping of this character's units in other characters.

         Units are nodes in other characters that are in some sense part of this one. A common example in strategy
         games is when a general leads an army: the general is one :class:`Character`, with a graph representing the
         state of their AI; the battle map is another :class:`Character`; and the general's units, though not in the
         general's :class:`Character`, are still under their command, and therefore follow rules defined on the
         general's ``unit.rule`` subproperty.

         The outer layer is the name of the character. The inner layer is the name
         of the unit, which may be a :class:`lisien.node.Place` or a
         :class:`lisien.node.Thing`.

      .. py:property:: portal

         A two-layer mapping of :class:`lisien.portal.Portal` objects in this :class:`Character`, by origin and destination

         Has a ``rule`` method for applying new rules to every :class:`Portal` here, and a ``rulebook`` property for
         assigning premade rulebooks.

         Aliases:  ``adj``, ``edge``, ``succ``

      .. py:property:: preportal

         A two-layer mapping of :class:`lisien.portal.Portal` objects in this :class:`Character`, by destination and origin

         Has a ``rule`` method for applying new rules to every :class:`Portal` here, and a ``rulebook`` property for
         assigning premade rulebooks.

         Alias: ``pred``

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

######
 node
######

.. automodule:: lisien.node

   .. autoclass:: lisien.node.Node

      .. autoproperty:: leader

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

########
 portal
########

.. automodule:: lisien.portal

   .. autoclass:: Portal

      .. py:attribute:: origin

         The :class:`lisien.node.Place` or :class:`lisien.node.Thing` that this leads out from

      .. py:attribute:: destination

         The :class:`lisien.node.Place` or :class:`lisien.node.Thing` that this leads into

      .. py:property:: character

         The :class:`lisien.character.Character` that this is in

      .. py:property:: engine

         The :class:`lisien.engine.Engine` that this is in

      .. autoproperty:: reciprocal

      .. automethod:: historical

      .. automethod:: delete

######
 rule
######

.. automodule:: lisien.rule

   .. autoclass:: Rule
      :members:

#######
 query
#######

.. automodule:: lisien.query

########
 facade
########

.. automodule:: lisien.facade

===========
 Internals
===========

These modules are used by :mod:`lisien.engine` or :mod:`lisien.proxy` somehow.
You shouldn't need to know about them unless they're broken, or you want to
extend Lisien's capabilities.

#######
 cache
#######

.. automodule:: lisien.cache

    .. autoclass:: Cache

#############
 collections
#############

.. automodule:: lisien.collections

    .. autoclass:: StringStore

    .. autoclass:: FunctionStore

####
 db
####

.. automodule:: lisien.db

    .. autoclass:: AbstractDatabaseConnector

        .. automethod:: load_xml

        .. automethod:: to_xml

        .. automethod:: write_xml

    .. autoclass:: ThreadedDatabaseConnector

    .. autoclass:: PythonDatabaseConnector

    .. autoclass:: NullDatabaseConnector

-----
 sql
-----

.. automodule:: lisien.db.sql

    .. autoclass:: SQLAlchemyDatabaseConnector

------
 pqdb
------

.. automodule:: lisien.db.pqdb

    .. autoclass:: ParquetDatabaseConnector

#########
 futures
#########

.. automodule:: lisien.futures

    .. autoclass:: Executor

    .. autoclass:: Worker

#######
 types
#######

.. automodule:: lisien.types
