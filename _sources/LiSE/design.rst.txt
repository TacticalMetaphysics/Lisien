.. _design :

Design
------

This document explains what LiSE does under the hood, and how it is structured to accomplish this.
It may be useful if you wish to modify LiSE, and are having difficulty understanding why huge parts of its
codebase exist.

Requirements
============

LiSE needs a standard data structure for every game world and every game rule. As this is impossible to do for *all*
cases, it assumes that game worlds are directed graphs, and game rules are made from snippets of Python code that
operate on those graphs.

The world model needs to be streamed in and out of memory as the user travels through time. Each change to the
model needs to be indexed monotonically--only one change can happen at a time, and they all occur in order (within
their branch). This is so that it's easy to identify what to load and unload, as well as to associate changes with
the rule that caused them, for the benefit of debugging tools like ELiDE's rule stepper.

To support use from other processes, potentially in other engines or on other computers, LiSE needs to report
changes to its world as a result of time travel. This includes the most mundane form of time travel, of playing
the game at normal speed.

Caching world state
===================

LiSE games start with keyframes and proceed with facts.

A keyframe is, conceptually, not much different from a traditional save file; it
describes the complete state of the game world at a given time. Only the very first
keyframe in a given playthrough is truly necessary. The remainder exist only to make time
travel performant; they are completely redundant, and can be deleted if they become
inconvenient.

Every time something happens in the simulation, it creates a fact at a given time. These are the
ground truth of what happened during this playthrough. Any keyframe, apart from the first,
can only reflect facts.

Time in LiSE is a tree, or several of them--there can be multiple "trunk" branches in the same database.
The game is simulated in a series of turns, each of which contains new facts in a series of ticks.
Facts do get stored in a big list, mostly to make it convenient to construct deltas
describing the difference between two moments in the same branch. When looking up data for
use in simulation code, a different data structure is used.

:class:`LiSE.allegedb.window.TurnDict` holds a variable's value for each turn in a pair of stacks, which in turn hold
the basic :class:`LiSE.allegedb.window.WindowDict`, a pair of stacks kept in order, used to track the values held by
some simulated variable over time. Popping from one stack, and appending to the other, is the default way to look up
the value at a given time; as values are stored in pairs with their tick as the initial item, little mutation is
needed to get the stacks in a state where the most recent value is on top of the one holding past values. Every
combination of a branch and a variable has its own ``TurnDict``.

So, the algorithm for finding the present effective value of some variable is as follows:

1. Find the relevant ``TurnDict`` for the desired branch and variable (generally a couple of plain dictionary lookups)
2. Pop/append that ``TurnDict`` until the "past" stack's top entry is before or equal to the desired turn,
   and the "future" stack is either empty, or has a top entry for after the desired turn. If the turn
   of the pair on top of the "past" stack is at or after the previous keyframe:
3. Take the ``WindowDict`` from the top of the ``TurnDict``'s "past" stack, and pop/append the "past"
   and "future" stacks as in step 2. If the tick of the pair on top of the "past" stack is strictly
   after the previous keyframe, return the value.

When a keyframe in this branch is more recent than the value in the ``TurnDict``,
but not after the present time, return the value given by the keyframe instead; if absent from the
keyframe, the value is unset, and a ``KeyError`` should be raised. If neither a fact nor a keyframe
value can be found in the current branch, look up the branch's parent and the time at which
the branches diverged, and try looking up the value at that time, in that branch. If the branch has
no parent -- that is, if it's a "trunk" branch -- the value was never set, and a ``KeyError`` should be
raised.

This is implemented in :keyword:`LiSE.allegedb.cache.Cache._base_retrieve`.

Deltas
======

Rules engine
============
