LiSE is an application for developing life simulation games.

# What is a life simulation game?

For the purposes of LiSE, it is any game where you are primarily concerned with "who you are" rather than "what you're doing". Everything your character in a life simulator can do should be represented somehow, but mostly in non-interactive form. This frees the player to attend to the high-level tasks of "living" in the game world: choosing what to do and when, rather than how.

In concrete terms, LiSE-style life simulators are games of time and resource management. The primary method for interacting with the game is to arrange events on a schedule, and then wait for your character(s) to carry them out. These games may be turn-based or real-time, but in the latter case, they allow the player to control the passage of game-time.

Existing games that LiSE seeks to imitate include:

* The Sims
* Kudos
* Redshirt
* Animal Crossing
* Monster Rancher
* Dwarf Fortress

# Why should I use LiSE for this purpose?

Most game engines--the ones that are called "game engines"
anyhow--provide graphics, networking, file system access, scripting,
and perhaps physics, since most games need these things. They usually
do not include anything you'd call a "game mechanic", so if you want
your game to include a common mechanic like a day/night cycle or a
crafting system, you need to script those things yourself.

There are exceptions. RPG Maker provides a battle framework, inventory
management, and other things that designers of Japanese-style computer
roleplaying games are likely to use. Adventure Game Studio helps
designers turn two-dimensional background art into something the
player character can walk around in, just by painting in the parts
that are walkable. RenPy assumes that the player will be reading a lot
and picking options from menus a lot, and so automates the
construction of menus and dialogs--but with an enormous variety of
options for stylizing them.

LiSE is a game engine in the latter sense. It assumes that there are
certain problems any designer of life simulators will have, and
provides powerful tools specialized to those problems.

Many designers of this sort of game will not care very much how it
looks. For them, there is a default interface style that will let
people play their game, and requires no further customization.

Life simulators, being life-like, tend to have a lot of data in their
world model, and should therefore be expected to have more problems
structuring that data. So instead of the traditional flat save-files,
LiSE stores its world-state in a relational database. Developers can
use SQL to get reports on all kinds of things in the game world, and
can edit the game world using any compatible database application, if
the LiSE interface doesn't suit.

Time control is handled by the core engine. This includes the ability
to control the speed of the simulation, as is traditional in this
genre, but also the ability to *rewind* time. Normally this would be
accomplished by keeping a lot of save files, but the database features
render multiple saves unnecessary.

# How does LiSE simulate life?

The engine provides a simple user interface
for both playing and developing games. The distinction is only in how
much control the user has over the world model--developers can
arbitrarily create and delete anything, players generally can't.

The interface metaphor is that of a board game. Developers put places
on the board--these, and the other visual elements, could be made to
look like anything, but are assumed to be dots or squares or other
simple geometric shapes, the likes of which you might see used in a
board game to distinguish the spots where you can put game
pieces. They draw lines between the places, called "portals," which
normally indicate that you can travel between two places. Then they
drag and drop a variety of things into the places, many of which
represent physical objects, such as people.

Actually, the engine distinguishes people from their bodies--more on
that later.

There is an event handler for the purpose of managing changes to the
world that occur at particular game-times. It watches particular parts
of the world for particular states to trigger an event, and resolves
the event into a set of changes to the world. The triggers, event
types, and changes are all wired together following rules stored in
the database. This is similar to the concept of "reactions" that Dwarf
Fortress uses.

One possible trigger for an event is that the player chose to trigger
it. The usual ways of doing this are by dragging their character to a
new place, thus triggering a movement event, or playing a card from
their hand. Cards may represent anything the player can do.

Having scheduled a variety of events, the player starts time. Much as
in The Sims, they can pause whenever they want, and they can decide
how fast time should go. They may also be allowed to rewind! By
default, every game made with LiSE generates a random seed when first
started, and whenever a random choice is needed, the same seed is used
to make it. This makes it possible to try several approaches to a
given situation and see how *all of them* will turn out. It is also
convenient for debugging those random events.

Events may be triggered by other events, as well. The triggered event
does not need to take place at the same time as the triggering event,
and indeed may take place in the past--events can rewrite history. For
a mundane example: when a movement event is triggered, and discovers
that the character to be moved is not where it is to be moved *from*,
it triggers a pathfinding event that will in turn trigger various
movement events at various times, resulting in the character moving to
the intended destination in a series of steps.

Most games will have more than one board in them. Events on all of
these can happen at the same time. Generally, only one of those boards
should represent "physical reality" in the form that video games
use--that is, only one board should manage the business of determining
where people are standing, how they get from here to there, and so
forth. The rest represent other game mechanics. You might use one to
implement a tech tree, as in Civilization, where instead of moving
about on it, the player has limited options for where to place
counters indicating they've acquired a given tech. You might use
another as a social graph, where the "places" are actually people, and
the "portals" are actually the relationships between them, connecting,
disconnecting, and changing color in response to drama. Templates for
several such boards will be provided.

Various places, things, and portals could represent the same person
for the purposes of different game mechanics. They are grouped
together in a "Character," along with whatever other information the
game needs to have about the person. It's similar to the character
sheets that are used in tabletop roleplaying games. The information in
a character may be used to resolve the effects of any given event, and
to decide how to schedule any event. The character is what determines
how fast someone can move, whether in the physical world, the tech
tree, or elsewhere.

It's all very abstract. You could implement physics in this engine, if
you wanted, but if that's your main concern, you might be happier with
OpenSimulator.

# Requirements

* [Python 2.7](http://python.org/download/releases/2.7.6/)
* [igraph 0.6.5](http://igraph.sourceforge.net/download.html)
* [python-igraph 0.6.5](http://python.org/pypi/python-igraph)
* [Kivy 1.7.2](http://kivy.org/#download)

# License Information
wallpape.jpg is copyright [Fantastic Maps](http://www.fantasticmaps.com/free-stuff/), freely available under the terms of [Creative Commons BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/).

igraph and python-igraph are freely available under the [MIT license](http://opensource.org/licenses/MIT).

The icons are [Entypo](http://entypo.com/), in the file Entypo.ttf, freely available at under the terms of [Creative Commons BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).

The LiSE source files themselves are licensed under the terms of the GNU General Public License version 3. See the text of the license in the file gpl-3.0.txt

Copyright (C) 2013 Zachary Spector. Email: zacharyspector@gmail.com
