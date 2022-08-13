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

World Modelling
---------------

Start by calling the engine's ``new_character`` method with a string
``name``.  This will return a character object with the name you
provided. Now draw a map by calling the method ``new_place`` with many
different string ``name`` s, then linking them together with the
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

Only methods defined with the ``@engine.method`` decorator may be used in a menu.

engine
------
.. automodule:: LiSE.engine
   :members:

node
----
.. automodule:: LiSE.node
   :members:

place
-----
.. automodule:: LiSE.place
   :members:

thing
-----
.. automodule:: LiSE.thing
   :members:

portal
------
.. automodule:: LiSE.portal
   :members:

rule
----
.. automodule:: LiSE.rule
   :members:

query
-----
.. automodule:: LiSE.query
   :members:


allegedb
--------
State container and object-relational mapper for versioned graphs.

allegedb serves its own special variant on the networkx
DiGraph class (with more to come). Every change to them is
stored in an SQL database.

This means you can keep multiple versions of one set of graphs and
switch between them without the need to save, load, or run git-checkout.
Just point the ORM at the correct branch and turn, and all of the
graphs in the program will change. All the different branches and
revisions remain in the database to be brought back when needed.


usage
_____

::

    >>> from LiSE.allegedb import ORM
    >>> orm = ORM('sqlite:///test.db')
    >>> g = orm.new_digraph('test')
    >>> g.add_nodes_from(['spam', 'eggs', 'ham'])
    >>> g.add_edge('spam', 'eggs')
    >>> g.adj
    <LiSE.allegedb.graph.DiGraphSuccessorsMapping object containing {'ham': {}, 'eggs': {}, 'spam': {'eggs': {}}}>
    >>> del g
    >>> orm.close()
    >>> del orm
    >>> orm = ORM('sqlite:///test.db')
    >>> g = orm.graph['test']
    >>> g.adj
    <LiSE.allegedb.graph.DiGraphSuccessorsMapping object containing {'ham': {}, 'eggs': {}, 'spam': {'eggs': {}}}>
    >>> import networkx as nx
    >>> red = nx.random_lobster(10, 0.9, 0.9)
    >>> blue = orm.new_digraph('blue', red)  # initialize with data from the given graph
    >>> red.adj == blue.adj
    True
    >>> orm.turn = 1
    >>> blue.add_edge(17, 15)
    >>> red.adj == blue.adj
    False
    >>> orm.turn = 0  # undoing what I did when turn=1
    >>> red.adj == blue.adj
    True
    >>> orm.branch = 'test'    # navigating to a branch for the first time creates that branch
    >>> orm.turn = 1
    >>> red.adj == blue.adj
    True
    >>> orm.branch = 'trunk'
    >>> red.adj == blue.adj
    False

ORM
___
.. automodule:: LiSE.allegedb
   :members:

cache
_____
.. automodule:: LiSE.allegedb.cache
   :members:

graph
_____
.. automodule:: LiSE.allegedb.graph
   :members:

query
_____
.. automodule:: LiSE.allegedb.query
   :members:

wrap
____
.. automodule:: LiSE.allegedb.wrap
   :members:

