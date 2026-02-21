##############
 Introduction
##############

Life sims all seem to have two problems in common:

**********************
 Too much world state
**********************

The number of variables the game is tracking -- just for game logic, not
graphics or physics or anything -- is very large. Like how The Sims
tracks sims' opinions of one another, their likes and dislikes and so
forth, even for the ones you never talk to and have shown no interest
in. If you streamline a life sim to where it doesn't have extraneous
detail, you lose a huge part of what makes it lifelike. This causes
trouble for developers when even *they* don't understand why sims hate
each other.

To address all those problems, Lisien provides a persistent state container.
Everything that ever happens in a game gets recorded, so that you can
pick through the whole history and find out exactly when the butterfly
flapped its wings to cause the cyclone. Time travel is aggressively optimized,
so that the experience of browsing a playthrough's history is as smooth
as if you were browsing a video. All of that history gets saved
in a database, which is used in place of traditional save files. This
means that if your testers discover something strange and want you to
know about it, they can send you their database, and you'll know
everything they did and everything that happened in their game.

****************
 Too many rules
****************

Fans of life sims tend to appreciate complexity. Developers are best
served by reducing complexity as much as possible. So Lisien makes it
easy to compartmentalize complexity and choose what of it you want to
deal with and when.

It is a rules engine, an old concept from business software that lets
you determine what conditions cause what effects. Though you'll still
need to write some Python code for the conditions and the effects that
they have, the connections between them and the objects that follow the
rule are defined in data. Once you've written a short Python function
that works in the rules engine, it's easy to reuse it for another rule
on another object. Changing the rules while the game is running is just
as easy, so you can experiment with ideas for rules as soon as you have
them.

**********
 Concepts
**********

Lisien is a tool for constructing turn-based simulations following rules
in a directed graph-based world model. It has special affordances for
the kinds of things you might need to simulate in the life simulation
genre.

Rules are things the game should do in certain conditions. In Lisien,
the "things to do" are called Actions, and are functions that can run
arbitrary Python code. The conditions are divided into Triggers and
Prereqs, of which only Triggers are truly necessary: they are also
functions, but one of a rule's Triggers must return True for the Action
to proceed.

A directed graph is made of nodes and edges. The nodes are points
without fixed locations--when drawing a graph, you may arrange the nodes
however you like, as long as the edges connect them the same way. Edges
in a directed graph connect one node to another node, but not
vice-versa, so you can have nodes A and B where A is connected to B, but
B is not connected to A. But you can have edges going in both directions
between A and B. They're usually drawn as arrows.

In Lisien, edges are called Portals, and nodes may be Places or Things.
You can use these to represent whatever you want, but they have special
properties to make it easier to model physical space: in particular,
each Thing is located in exactly one node at a time (usually a Place).
Regardless, you can keep any data you like in a Thing, Place, or Portal
by treating it like a dictionary.

Lisien's directed graphs are called Characters. Every time something
about a Character changes, Lisien remembers when it happened -- that is,
which turn of the simulation. This allows the developer to look up the
state of the world at some point in the past. This time travel is nearly
real-time in most cases, to make it convenient to flip back and forth
between a correct world state and an incorrect one and use your
intuition to spot exactly what went wrong.

See :doc:`lisien/design` for details.

*******
 Usage
*******

The only Lisien class that you should ever instantiate yourself is
:class:`.Engine`. All simulation objects should be created
and accessed through it. By default, it keeps the simulation code and
world state in the working directory, but you can pass in another
directory if you prefer. Either use it with a context manager (``with
Engine() as eng:``) or call its ``.close()`` method when you're done.

World Modelling
===============

Start by calling the engine's ``new_character`` method with a string
``name`` to get a character object. Draw a graph by calling the method
``new_place`` with many different ``name`` s, then linking them together
with the method ``new_portal(origin, destination)``.

To store data pertaining to some specific place, retrieve the place from
the ``place`` mapping of the character: if the character is ``world``
and the place name is ``'home'``, you might do it like ``home =
world.place['home']``. Portals are retrieved from the ``portal``
mapping, where you'll need the origin and the destination: if there's a
portal from ``'home'`` to ``'narnia'``, you can get it like ``wardrobe =
world.portal['home']['narnia']``, but if you haven't also made another
portal going the other way, ``world.portal['narnia']['home']`` will
raise ``KeyError``.

Things, usually being located in places (but possibly in other things),
are most conveniently created by the ``new_thing`` method of Place
objects: ``alice = home.new_thing('alice')`` gets you a new Thing object
located in ``home``. Things can be retrieved like ``alice =
world.thing['alice']``. Ultimately, things and places are both just
nodes, and both can be retrieved in a character's ``node`` mapping, but
only things have methods like ``travel_to``, which finds a path to a
destination and schedules movement along it.

You can store data in things, places, and portals by treating them like
dictionaries. If you want to store data in a character, use its ``stat``
property as a dictionary instead. Data stored in these objects, and in
the ``universal`` property of the engine, can vary over time, and be
rewound by setting ``turn`` to some time before. The engine's
``eternal`` property is not time-sensitive, and is mainly for storing
settings, not simulation data.

Rule Creation
=============

To create a rule, first decide what objects the rule should apply to.
You can put a rule on a character, thing, place, or portal; and you can
put a rule on a character's ``thing``, ``place``, and ``portal``
mappings, meaning the rule will be applied to *every* such entity within
the character, even if it didn't exist when the rule was declared.

All these items have a property ``rule`` that can be used as a
decorator. Use this to decorate a function that performs the rule's
action by making some change to the world state. The function should
take only one argument, the item itself.

At first, the rule object will not have any triggers, meaning the action
will never happen. If you want it to run on *every* tick, pass the
decorator ``always=True`` and think no more of it. But if you want to be
more selective, use the rule's ``trigger`` decorator on another function
with the same signature, and have it return ``True`` if the world is in
such a state that the rule ought to run. Triggers must never mutate the
world or use any randomness.

If you like, you can also add prerequisites. These are like triggers,
but use the ``prereq`` decorator, and should return ``True`` *unless*
the action should *not* happen; if a single prerequisite returns
``False``, the action is cancelled. Prereqs may involve random elements.
Use the ``engine`` property of any Lisien entity to get the engine, then
use methods such as ``percent_chance`` and ``dice_check``.

Time Control
============

The current time is always accessible from the engine's ``branch`` and
``turn`` properties. In the common case where time is advancing forward
one tick at a time, it should be done with the engine's ``next_turn``
method, which polls all the game rules before going to the next turn;
but you can also change the time whenever you want, as long as
``branch`` is a string and ``turn`` is an integer. The rules will never
be followed in response to your changing the time "by hand".

It is possible to change the time as part of the action of a rule. This
is how you would make something happen after a delay. Say you want a
rule that puts the character ``alice`` to sleep, then wakes her up after
eight turns (presumably hour-long)::

   alice = engine.character['alice']

   @alice.rule
   def sleep(character):
           character.stat['awake'] = False
           start_turn = character.engine.turn
           with character.engine.plan() as plan_num:
                   character.engine.turn += 8
                   character.stat['awake'] = True
           character.stat['wake_plan'] = plan_num

At the end of a ``plan():`` block, the game-time will be reset to its
position at the start of that block. You can use the plan's ID number,
``plan_num`` in the above, to cancel it yourself -- some other rule
could call
``engine.delete_plan(engine.character['alice'].stat['wake_plan'])``.

Input Prompts
=============

Lisien itself doesn't know what a player is or how to accept input from
them, but does use some conventions for communicating with a user
interface such as Elide.

To ask the player to make a decision, first define a method for them to
call, then return a menu description like this one::

   @engine.method
   def wake_alice(self):
           self.character['alice'].stat['awake'] = True

   alice = engine.character['alice']

   @alice.rule
   def wakeup(character):
           return "Wake up?", [("Yes", character.engine.wake_alice), ("No", None)]

Only methods defined with the ``@engine.method`` function store may be
used in a menu. In Elide, that means you have to define them in the Method
tab of the Python editor.

*********
 Proxies
*********

Lisien may be run in a separate process from Elide, or any other frontend you may write for it.
To ease the process of writing such frontends in Python, Lisien provides proxy objects that reflect and control
their corresponding objects in the Lisien core.

Use :class:`lisien.proxy.EngineProxyManager` to start Lisien in a subprocess and get a
proxy to the engine::

    from lisien.proxy import EngineProxyManager


    manager = EngineProxyManager('gamedir/')
    engine_proxy = manager.start(workers=4)

    # do stuff here

    manager.shutdown()


You can pass :class:`lisien.Engine` arguments to the``start`` method.

The proxy objects are mostly the same as what they represent, with affordances for when you
have to do some work in the user interface while waiting for the core to finish something. Generally, you
can pass a callback function to the relevant object's ``connect`` method, and Lisien will call the callback
at the relevant time. Here's how you'd run some code whenever ``next_turn`` finishes running the rules engine::

    from threading import Thread

    from lisien.proxy import EngineProxyManager

    from my_excellent_game import display_menu, apply_delta

    manager = EngineProxyManager()
    engine_proxy = manager.start()


    def update_from_next_turn(engine, menu_info, delta):
        display_menu(*menu_info)
        apply_delta(delta)

    engine_proxy.next_turn.connect(update_from_next_turn)

    subthread = Thread(target=engine_proxy.next_turn)
    subthread.start()

    # do some UI work here

    subthread.join()
    manager.shutdown()
