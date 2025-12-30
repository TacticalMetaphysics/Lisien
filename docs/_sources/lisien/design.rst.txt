.. _design:

########
 Design
########

This document explains what Lisien does under the hood, and how it is
structured to accomplish this. It may be useful if you wish to modify
Lisien, and are having difficulty understanding why huge parts of its
codebase exist.

**************
 Requirements
**************

Lisien assumes that game worlds are directed graphs, and game rules are
collected snippets of Python code that operate on those graphs.

The world model needs to be streamed in and out of memory as the user
travels through time. Each change to the model needs to be indexed
monotonically--only one change can happen at a time, and they all occur
in order (within their branch). This is so it's easy to identify what to
load and unload, as well as associate changes with the rule that caused
them, for the benefit of debugging tools like Elide's rule stepper.

To support use from other processes--potentially in other engines, or on
other computers--Lisien needs to report changes to its world as a result
of time travel. This includes the most mundane form of time travel:
playing the game at normal speed.

*********************
 Caching world state
*********************

Lisien games start with keyframes and proceed with facts.

A keyframe is, conceptually, not much different from a traditional save
file; it describes the complete state of the game world at a given time.
Only the very first keyframe in a given playthrough is truly necessary.
The remainder exist only to make time travel performant. It is safe to
delete them if they become inconvenient.

Every time something happens in the simulation, it creates a fact at a
given time. These are the ground truth of what happened during this
playthrough. Any keyframe, apart from the first, can only reflect facts.

Time in Lisien is a tree, or several of them--there can be multiple
"trunk" branches in the same database. The game is simulated in a series
of turns, each of which contains new facts in a series of ticks. Facts
also get stored in a big global list, mostly to make it convenient to
construct deltas describing the difference between two moments in the
same branch. When looking up data for use in simulation code, a
different data structure is used.

:class:`lisien.window.TurnDict` holds a variable's value for
each turn in a pair of stacks, which in turn hold the basic
:class:`lisien.window.WindowDict`, a pair of stacks kept in
order, tracking changes in a variable each tick. Popping from one stack
and appending to the other is the default way to look up the value at a
given time. As values are stored in pairs, with their tick as the
initial item, little mutation is needed to get the stacks in a state
where the most recent value is on top of the one holding past values.

So, the algorithm for finding the present effective value of some
variable is as follows:

#. Find the relevant ``TurnDict`` for the desired branch and variable
   (generally a couple of plain dictionary lookups)

#. Pop/append that ``TurnDict`` until the "past" stack's top entry is
   before or equal to the desired turn, and the "future" stack is either
   empty, or has a top entry for after the desired turn. If the turn of
   the pair on top of the "past" stack is at or after the previous
   keyframe:

#. Take the ``WindowDict`` from the top of the ``TurnDict``'s "past"
   stack, and pop/append the "past" and "future" stacks as in step 2. If
   the tick of the pair on top of the "past" stack is strictly after the
   previous keyframe, return that value. Otherwise, return the value
   from the keyframe, raising ``KeyError`` if there is none.

#. If neither a fact nor a keyframe value can be found in the current
   branch, look up the branch's parent and the time at which the branches
   diverged, and try looking up the value at that time, in that branch. Recurse
   as long as the branch has a parent and we haven't found a value yet.

#. If the branch has no parent -- that is, if it's a "main" branch --
   the value was never set, and a ``KeyError`` should be raised.

Keycaches
=========

Users may want to iterate over all entities in a graph. If we had to go
through the above process for every entity that had *ever* existed, to
check if it still does, this would be uselessly slow. So, every time an
entity is created or deleted in Lisien, it's added to, or removed from,
a set corresponding to the relevant entity collection. These sets are
versioned like any other game variable, but not persisted to disk.

A special keycache is the contents cache, for the case where one node is
inside another. Nodes may be turned into "things" that are located in
other nodes. Whenever a thing's location changes, it is removed from the
contents keycache of its former location, and added to that of its new
one.

***********************
 Loading and Unloading
***********************

For convenience, Lisien breaks the timestream into blocks between two
keyframes; the exception being when the last fact in a branch of time
occurs after a keyframe, in which case the block includes the final
keyframe and everything after it.

Upon startup, Lisien looks for the "current time," which is a triple of
a branch name, a turn number, and a tick number, identifying a point
within a turn. If present, it's stored in the ``engine.eternal`` keys
``"branch"``, ``"turn"``, and ``"tick"``, defaulting to the main branch
(``"trunk"``, unless the user has switched to a different main branch),
``0``, and ``0`` respectively. To decide what to load, Lisien looks for
the closest keyframe at or before the current time, and the keyframe
after that, if any. Lisien then loads the earlier keyframe and all facts
up to the later keyframe--or the end of time, if it must.

Whenever Lisien commits its facts to disk, it unloads everything it
*can* unload, which does not include the block of time containing the
current time.

*******
 Plans
*******

Keeping the whole history of the game means Lisien is *almost* an
append-only data store, but there is an exception: plans for future
changes. For example, in a game like The Sims, when a Sim gets hungry,
they need to go to the kitchen to prepare a meal. The path to the
kitchen differs depending on where the Sim is, and what the player has
done to their house lately, but even if the player has disabled free
will, the command "go to the kitchen" is still just one order. The user
should not need to specify the next step the Sim takes every tick of the
clock.

For this purpose, Lisien keeps track of what spans of time have "really
happened," but allows the user to manipulate the world state however
they like after that span--provided they do so within a ``with
engine.plan():`` block, and thereby accept that things might not go
according to the plan. Changes within the block get assigned an ID so
that the user can cancel it if they like, and Lisien will
cancel remaining changes automatically if it detects that they can no
longer happen. Lisien's understanding of causality is quite limited,
though. Currently, the only type of paradox it can detect is if
something's planned to follow a path, but isn't in the right place for
the next step.

New branches of time cannot be created while planning. Otherwise, the
only difference between planned changes and "real" ones is that planned
changes happen outside of the "real" window of time, which will only
grow as a result of calling ``engine.next_turn()``, thus running the
rules engine.

**************
 Rules engine
**************

Rules engines run specific code in specific conditions. In an Enterprise
Resource Planning app, a rules engine would be expected to have some
configuration language--possibly an entire logic language--for
specifying the rules, enabling algorithms such as Rete to efficiently
evaluate the conditions. `Lisien may have such a feature some day`_,
but, for maximum flexibility and minimum barrier to entry, rules may be
specified as collections of Python functions.

Every rule needs at least one action function, which is what the rule
does, and at least one trigger function, a Boolean function that returns
``True`` when the rule should run. It may be the truth function,
provided standard in ``engine.trigger.truth``, in which case the rule
runs every turn. You can add as many of either as you like, and the
actions will all be run when *any* of the triggers return ``True``. For
finer control over the conditions the rule runs in, you may also add any
number of prereq functions, which must *all* return ``True``, or the
rule will not run. All of these types of functions will be called with
only one argument: the Lisien entity that the rule is applied to.

By default, trigger functions will be evaluated in parallel. Lisien has
a process pool, in which worker processes keep copies of the current
world state for trigger functions to work with. You can run arbitrary
code in those processes, too, if you like; :class:`lisien.Engine` is an
implementation of the standard Python
:class:`concurrent.futures.Executor`. See
:keyword:`lisien.examples.pathfind` for a demonstration of using the
process pool to find many paths at once, then having things follow them.

Prereq functions, however, are always evaluated serially in the core
Lisien process. This enables them to change the state of the world,
which normally isn't recommended, but is necessary if a rule is to have
a random chance of running; the state of the randomizer is part of the
world, tracked like any other variable. It's called ``"rando_state"``,
and you'll find it in ``engine.universal``, a dictionary-like object
meant for game data that's not associated with any particular game
object. (The game's *configuration* is not tracked that way, and is held
in ``engine.eternal`` instead, which is a simple key-value store,
with no change tracking, persisted to the database.)

If any trigger function returned ``True``, and all prereq functions
returned ``True``, then the action functions will run. Ordinarily, they
will simply run whatever code you've written in them, on whatever Lisien
entity you've assigned them to, but some rules are too big for normal
execution. If you find that a certain rule is taking too long to run,
you can speed it up by setting the rule's ``big`` property to ``True``.
In that case, the rules engine will replace the Lisien entity with a
"facade," which presents the same interface, but records the changes
made to it, instead of putting them straight into the world model. The
changes will be applied to the world model only after all of the actions
have run. Doing them all at once lets Lisien use a batch processing mode
that's faster for big batches.

``big`` is a fact about the world, and your rule code may change it,
though if the rule in question is currently running, it won't apply
until the next turn. If you want that optimization on rare occasion, you
can access it within rule code using the ``with engine.batch():``
context manager.

********
 Deltas
********

Lisien has two delta algorithms for computing differences between world
states. The "slow" delta assumes no knowledge of how the states relate
to each other, and is therefore used when traveling from one branch of
time to another. The "fast" delta assumes that one state turned into the
other, and uses the facts Lisien stores about how that happened.

Slow
====

The slow delta operates on two key-value mappings representing two world
states. First, Python's basic set-difference operations are employed to
get the keys that were added or deleted. Then, the shared keys are put
into a list, and their values, kept in order, are put into numpy arrays.
Actually, the values' memory addresses are put into arrays--serializing
the values is too slow. So, once we've compared the numpy arrays in
parallel to find the addresses that differ, we do a normal, serial
inequality check on the values of differing address before putting them
in the delta.

Fast
====

The fast delta is a collection of facts that were set between two times.
To make it convenient to iterate over *all* facts, they are copied into
one big global ``TurnDict`` when they are set or loaded. Then, to make
the delta, we take a slice of that global ``TurnDict`` and compile the
facts in it into a dictionary.

Despite the name, the fast delta is only faster than the slow one when
the number of facts it needs to use is relatively small. Lisien will
switch to the slow delta if the number of facts it would need for the
fast one is larger than the gap between keyframes.

.. _lisien may have such a feature some day: <https://codeberg.org/clayote/Lisien/issues/28>
