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

Lisien needs a standard data structure for every game world and every
game rule. As this is impossible to do for *all* cases, it assumes that
game worlds are directed graphs, and game rules are made from snippets
of Python code that operate on those graphs.

The world model needs to be streamed in and out of memory as the user
travels through time. Each change to the model needs to be indexed
monotonically--only one change can happen at a time, and they all occur
in order (within their branch). This is so that it's easy to identify
what to load and unload, as well as to associate changes with the rule
that caused them, for the benefit of debugging tools like ELiDE's rule
stepper.

To support use from other processes, potentially in other engines or on
other computers, Lisien needs to report changes to its world as a result
of time travel. This includes the most mundane form of time travel, of
playing the game at normal speed.

*********************
 Caching world state
*********************

Lisien games start with keyframes and proceed with facts.

A keyframe is, conceptually, not much different from a traditional save
file; it describes the complete state of the game world at a given time.
Only the very first keyframe in a given playthrough is truly necessary.
The remainder exist only to make time travel performant; they are
completely redundant, and can be deleted if they become inconvenient.

Every time something happens in the simulation, it creates a fact at a
given time. These are the ground truth of what happened during this
playthrough. Any keyframe, apart from the first, can only reflect facts.

Time in Lisien is a tree, or several of them--there can be multiple
"trunk" branches in the same database. The game is simulated in a series
of turns, each of which contains new facts in a series of ticks. Facts
do get stored in a big list, mostly to make it convenient to construct
deltas describing the difference between two moments in the same branch.
When looking up data for use in simulation code, a different data
structure is used.

:class:`lisien.allegedb.window.TurnDict` holds a variable's value for
each turn in a pair of stacks, which in turn hold the basic
:class:`lisien.allegedb.window.WindowDict`, a pair of stacks kept in
order, used to track the values held by some simulated variable over
time. Popping from one stack, and appending to the other, is the default
way to look up the value at a given time; as values are stored in pairs
with their tick as the initial item, little mutation is needed to get
the stacks in a state where the most recent value is on top of the one
holding past values. Every combination of a branch and a variable has
its own ``TurnDict``.

To support use in games, Lisien needs a convenient interface for
expressing things moving around. The user needs to be able to tell an
entity to find a path to another place, and *follow* it, and Lisien
needs to take care of that.

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
   previous keyframe, return the value.

When a keyframe in this branch is more recent than the value in the
``TurnDict``, but not after the present time, return the value given by
the keyframe instead; if absent from the keyframe, the value is unset,
and a ``KeyError`` should be raised. If neither a fact nor a keyframe
value can be found in the current branch, look up the branch's parent
and the time at which the branches diverged, and try looking up the
value at that time, in that branch. If the branch has no parent -- that
is, if it's a "trunk" branch -- the value was never set, and a
``KeyError`` should be raised.

This is implemented in
:keyword:`lisien.allegedb.cache.Cache._base_retrieve`.

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
that the user can cancel that whole plan, if they like, and Lisien will
cancel remaining changes automatically if it detects that they can no
longer happen. Lisien's understanding of causality is quite limited,
though. Currently, the only type of paradox it can detect is if
something's planned to follow a path, but isn't in the right place for
the next step.

New branches of time cannot be created while planning. Otherwise, the
only difference between planned changes and "real" ones is that planned
changes happen outside of the "real" window of time, which will only
grow as a result of calling ``Engine.next_turn()``, thus running the
rules engine.

**************
 Rules engine
**************
