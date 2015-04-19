Introduction
============

LiSE is a tool for constructing tick-based simulations following rules
in a directed graph-based world model. It has special affordances for
the kinds of things you might need to simulate in the life simulation
genre.

Tick-based simulations are those in which time is broken into discrete
units of uniform size, called "ticks". Each change to the simulation
happens in a single tick, and nothing can happen in any amount of time
shorter than the tick. Just how much time a tick represents will vary
depending on the game--it could be seconds, days, years, whatever you
want.

Rules are things the game should do in certain conditions. In LiSE,
the "things to do" are called Actions, and are functions that can run
arbitrary Python code. The conditions are divided into Triggers and
Prereqs, of which only Triggers are truly necessary: they are also
functions, but one of a rule's Triggers must return True for the
Action to proceed.

A directed graph is made of nodes and edges. The nodes are points
without fixed locations--when drawing a graph, you may arrange the
nodes however you like, as long as the edges connect them the same
way. Edges in a directed graph connect one node to another node, but
not vice-versa, so you can have nodes A and B where A is connected to
B, but B is not connected to A. But you can have edges going in both
directions between A and B. They're usually drawn as arrows.

In LiSE, edges are called Portals, and nodes may be Places or
Things. You can use these to represent whatever you want, but they
have special properties to make it easier to model physical space: in
particular, each Thing is located in exactly one node at a time
(usually a Place), and may be travelling through one of the Portals
leading out from there. Regardless, you can keep any data you like in
a Thing, Place, or Portal by treating it like a dictionary.

LiSE's directed graphs are called Characters. Every time something
about a Character changes, LiSE remembers when it happened -- that is,
which tick of the simulation. This allows the developer to look up the
state of the world at some point in the past.

When time moves forward in LiSE, it checks all its rules and allows
them to change the state of the world. Then, LiSE sets its clock to
the next tick, and is ready for time to move forward another
tick. LiSE can keep track of multiple timelines, called "branches,"
which can split off from one another. Otherwise, events in one branch
don't affect those in another, unless you write a rule that sets the
branch (and perhaps, the tick) before making a change to the world
state, or merely looking up some information to use. Either way, this
is safe--when the rule has been evaluated, the branch and tick will be
put back where they were.

Programming Interface
=====================

The only LiSE class that you should ever instantiate yourself is
Engine. All the other simulation objects should be
created and accessed through it. Engine is instantiated
with two arguments, which are file names of SQLite databases that will
be created if needed; the first will hold the state of the simulation,
including history, while the second will hold rules, including copies
of the functions used in the rules.

Start by calling the engine's ``new_character`` method with a string
``name``.  This will return a character object with the name you
provided. Now draw a map by calling the method ``add_place`` with many
different string ``name`` s, then linking them together with the method
``add_portal(origin, destination)``.  To store data pertaining to some
particular place, retrieve the place from the ``place`` mapping of the
character: if the character is ``world`` and the place name is
``'home'``, you might do it like ``home =
world.place['home']``. Portals are retrieved from the ``portal``
mapping, where you'll need the origin and the destination: if there's
a portal from ``'home'`` to ``'narnia'``, you can get it like
``wardrobe = world.portal['home']['narnia']``, but if you haven't also
made another portal going the other way,
``world.portal['narnia']['home']`` will raise ``KeyError``. Things are
created with the method ``add_thing(name, location)``, where
``location`` must be the name of a place you've already
created. Retrieve things from the ``thing`` mapping, which works much
like the ``place`` mapping.

You can store data in things, places, and portals by treating them
like dictionaries.  If you want to store data in a character, use its
``stat`` property as a dictionary instead. Data stored in these
objects, and in the ``universal`` property of the engine, can vary
over time. The engine's ``eternal`` property is not time-sensitive,
and is mainly for storing settings, not simulation data.

The current time is always accessible from the engine's ``branch`` and
``tick`` properties. In the common case where time is advancing
forward one tick at a time, it should be done with the engine's
``next_tick`` method, which polls all the game rules before going to
the next tick; but you can also change the time whenever you want, as
long as ``branch`` is a string and ``tick`` is an integer. The rules
will never be followed in response to your changing the time "by
hand".

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

If you need to access a character that you created previously, get it
from the engine's ``character`` mapping, eg. ``world =
engine.character['world']``.



IDE
===

The graphical interface, ELiDE, lets the developer change whatever
they want about the world. A game made with ELiDE will be more
restrictive about what the player is allowed to change, but all of the
player's input will be turned into changes to the world, which the
rules may respond to however they need. Thus you never have to write
any input handling code to make a functional game in ELiDE.

ELiDE has three columns. On the right are a lot of buttons to access
the parts of ELiDE that aren't visible right now, plus a couple of
icons that you can drag into the middle. In the middle, you have a
graphical display of the Character under consideraction; dragging
those icons here will make a new Place or Thing. To connect Places
with Portals, press the button with the arrow on it, then drag from
one Place to another. Press the button again when you're done. On the
left is the stat editor: it displays data that is stored in whatever
entity is presently selected. You can select Places, Things, and
Portals by clicking them--and once you've selected them, you can drag
them elsewhere. If no Place, Thing, or Portal is selected, then the
Character you are viewing is selected. There's a button in the
top-right to view another Character.

Below all this are some bits to let you manipulate time, mainly the
Play and Next Tick buttons. Play will start moving time forward when
you press it, and stop when you press it again. Next Tick will only
move time forward by one tick. There are also text fields with which
you can enter the Branch and Tick by hand. Note that rules are only
run when you advance time using Play or Next Tick.

Stat editor
-----------

This two-column table displays the keys and values in the selected
entity. By default, they are all rendered as plain text.
