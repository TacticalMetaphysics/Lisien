Introduction
============

Life sims all seem to have two problems in common:

Too much world state
--------------------

The number of variables the game is tracking -- just for game logic, not
graphics or physics or anything -- is very large. Like how The Sims
tracks sims' opinions of one another, their likes and dislikes and so forth,
even for the ones you never talk to and have shown no interest in. If you
streamline a life sim to where it doesn't have extraneous detail
complexity you lose a huge part of what makes it lifelike.

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

It is a rules engine, an old concept from business software that lets you
determine what conditions cause what effects. Here, conditions are
Triggers and effects are Actions, and they're both lists of Python
functions. Actions make some change to the state of the world, while
Triggers look at the world once-per-turn and return a Boolean to show
whether their Actions should happen.

The connection between Trigger and Action is arbitrary; you can mix and
match when you want. If you're doing it in the graphical interface, they
look sort of like trading cards, so constructing a rule is like deckbuilding.
Triggers and Actions exist independent of the game world, and can
therefore be moved from one game to another without much fuss. I
intend to include a fair number of them with the release version of LiSE,
so that you can throw together a toy sim without really writing any code.

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
travel back to previous turns and try things a different way.
This is also convenient for debugging simulation rules.

LiSE can keep track of multiple timelines, called "branches,"
which can split off from one another. Branches normally don't affect
one another, though it's possible to write rules that change one
branch when they are run in another.

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
graphical display of the Character under consideration; dragging
those icons here will make a new Place or Thing. To connect Places
with Portals, press the button with the arrow on it, then drag from
one Place to another. Press the button again when you're done. On the
left is the stat editor: it displays data that is stored in whatever
entity is presently selected. You can select Places, Things, and
Portals by clicking them--and once you've selected them, you can drag
them elsewhere. If no Place, Thing, or Portal is selected, then the
Character you are viewing is selected. There's a button in the
top-right to view another Character.

On the bottom left are some bits to let you manipulate time, mainly the
Simulate and 1 Turn buttons. Simulate will start moving time forward when
you press it, and stop when you press it again.
There are also text fields with which you can enter the time by hand.
Note that rules are only run when you advance time using Simulate or 1 Turn.
The Tick field indicates how many changes have occurred in the current turn.

It's possible to view turns that haven't been simulated yet.
This is deliberate, but it's not a good idea at the moment,
because ELiDE doesn't know how to make plans yet.

Stat Editor
-----------

This two-column table displays the keys and values in the selected
entity. By default, they are all shown as Readouts, which is to say,
plain text.

The "cfg" button at the bottom of the stat editor opens a window in
which you can add new stats and customize the appearance of the
existing ones. If you pick an appearance other than "Readout," you
will be able to edit the stat when you're not in this
window.

"TextInput" is the most flexible appearance: you just type the
value that the stat should have. It will try to interpret your value
as a number, list, or dictionary if it can; by default, it will be
taken as a string.

"Toggle" is a button that, when pressed, changes the value from True to False
or vice versa. You can enter text to display instead of True or False, but
the actual value will still be True or False.

"Slider" is for numeric values that vary within a range. It needs a
minimum, a maximum, and a step size determining the smallest possible
change you can make with it.

You can use this to build a primitive interface to your game, or just monitor
the state of the world.

Python Editor
-------------

Click the Python button to edit your game code in the IDE if you like.
In this case, you can't use any of the decorators. Choose the appropriate tab
from Trigger, Prereq, or Action at the top, and the function you write will
show up in the appropriate part of the rules editor.

Rules Editor
------------

Here you can assemble rules out of prewritten functions. First pick which rule
to edit from the menu on the left, using the box at the bottom to add one if needed.
Then go through the trigger, prereq, and action tabs, and drag the functions from
the right pile to the left to include them in the rule. You may also reorder them
within the left pile.

Rules made here will apply to the entity currently selected in the main screen.
There is currently no graphical way to apply the same rulebook to many entities.
You can, however, select nothing, in which case you get the option to edit
rulebooks that apply to the current character overall,

Strings Editor
--------------

The LiSE engine has an attribute ``string`` that is accessed like a dictionary and
used to store arbitrary strings, such as might be shown in a menu. You can edit those
here. You can store strings for multiple languages, and switch between them
programmatically by setting ``engine.string.language``.