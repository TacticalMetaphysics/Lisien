Introduction
============

Life sims all seem to have two problems in common:

Too much world state
--------------------

The number of variables the game is tracking -- just for game logic,
not graphics or physics or anything -- is very large. Like how The
Sims tracks sims' opinions of one another, their likes and dislikes
and so forth, even for the ones you never talk to and have shown no
interest in. If you streamline a life sim to where it doesn't have
extraneous detail complexity you lose a huge part of what makes it
lifelike.

This causes trouble for developers when even *they* don't really
understand why sims hate each other, and even if they do, failures of
bookkeeping can cause technical issues like how damn long it takes to
save or load your game in The Sims 3.

To address all those problems, LiSE provides a state container.
Everything that ever happens in a game gets recorded, so that you can
pick through the whole history and find out exactly when the butterfly
flapped its wings to cause the cyclone. All of that history gets saved
in a database, too, which is used in place of traditional save files.
This means that if your testers discover something strange and want
you to know about it, they can send you their database, and you'll
know everything they did and everything that happened in their game.

Too many rules
--------------

Fans of life sims tend to appreciate complexity. Developers are best
served by reducing complexity as much as possible. So LiSE makes it
easy to compartmentalize complexity and choose what of it you want to
deal with and when.

It is a rules engine, an old concept from business software that lets
you determine what conditions cause what effects. Here, conditions are
Triggers and effects are Actions, and they're both lists of Python
functions. Actions make some change to the state of the world, while
Triggers look at the world once-per-turn and return a Boolean to show
whether their Actions should happen.

The connection between Trigger and Action is arbitrary; you can mix
and match when you want. If you're doing it in the graphical
interface, they look sort of like trading cards, so constructing a
rule is like deckbuilding.  Triggers and Actions exist independent of
the game world, and can therefore be moved from one game to another
without much fuss. I intend to include a fair number of them with the
release version of LiSE, so that you can throw together a toy sim
without really writing any code.

Architecture
------------

LiSE is a tool for constructing turn-based simulations following rules
in a directed graph-based world model. It has special affordances for
the kinds of things you might need to simulate in the life simulation
genre.

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
which turn of the simulation. This allows the developer to look up the
state of the world at some point in the past.

When time moves forward in LiSE, it checks all its rules and allows
them to change the state of the world. Then, LiSE sets its clock to
the next turn, and is ready for time to move forward another
turn. LiSE remembers the entire history of the game, so that you can
travel back to previous turns and try things a different way.  This is
also convenient for debugging simulation rules.

LiSE can keep track of multiple timelines, called "branches," which
can split off from one another. Branches normally don't affect one
another, though it's possible to write rules that change one branch
when they are run in another.
