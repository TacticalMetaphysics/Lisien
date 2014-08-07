LiSE is an application for developing life simulation games.

# What is a life simulation game?

For the purposes of LiSE, it is any game where you are primarily
concerned with "who you are" rather than "what you're
doing". Everything your character in a life simulator can do should be
represented somehow, but mostly in non-interactive form. This frees
the player to attend to the high-level tasks of "living" in the game
world: choosing what to do and when, rather than how.

In concrete terms, LiSE-style life simulators are games of time and
resource management. The primary method for interacting with the game
is to arrange events on a schedule, and then wait for your
character(s) to carry them out. These games may be turn-based or
real-time, but in the latter case, they allow the player to control
the passage of game-time. There may be time pressure on the character(s),
but the player experiences this only as a strategic concern.

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
construction of menus and dialogs, with an enormous variety of
options for stylizing them.

LiSE is a game engine in the latter sense. It assumes that there are
certain problems any designer of life simulators will have, and
provides powerful tools specialized to those problems.

## Examples

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
represent physical objects, such as the bodies of the characters in
the game

Characters may exist on several of these boards at once--more on that
later.

# How does LiSE simulate life?

There is an event handler for the purpose of managing changes to the
world that occur at particular game-times. Events are resolved into a
list of changes to the world. The triggers, event types, and changes
are all wired together following rules stored in the database. This is
similar to the concept of "reactions" that Dwarf Fortress uses.

One possible trigger for an event is that the player chose to trigger
it. The usual ways of doing this are by dragging their character to a
new place, thus triggering a movement event; picking an option from a
menu; or playing a card.

Various places, things, and portals could represent the same person
for the purposes of different game mechanics. They are grouped
together in a "Character," along with whatever other information the
game needs to have about the person. It's similar to the character
sheets that are used in tabletop roleplaying games. The information in
a character may be used to resolve the effects of any given event, and
to decide how to schedule any event. The character is what determines
how fast someone can move, whether in the physical world, the tech
tree, or elsewhere.

Each of those ways of looking at the game world gets its own
board. They all use the same graph data structure, and events on all
of them occur on the same timeline(s), but they may follow different
rules in any other respect.

It's all very abstract. You could implement physics in this engine, if
you wanted, but if that's your main concern, you might be happier with
OpenSimulator.

# Status

The graphical frontend, now called ELiDE, isn't usable yet. The core
engine, in the ``LiSE`` module, has demonstrated some of its
functionality as an ORM and rules engine, and you can see what that
functionality is by viewing the file ``test.py``. They might be
useful, but haven't been tested much, so don't expect a polished
experience. Please report bugs in the Issues tab.

# License Information
wallpape.jpg is copyright [Fantastic Maps](http://www.fantasticmaps.com/free-stuff/), freely available under the terms of [Creative Commons BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/).

LiSE currently has two graphics sets, the [RLTiles](http://rltiles.sourceforge.net/) and [Pixel City](http://opengameart.org/content/pixel-city), both available under [CC0](http://creativecommons.org/publicdomain/zero/1.0/), being in the public domain where it exists.

networkx, which forms the basis of LiSE's data model, is available under [BSD](http://networkx.github.io/documentation/latest/reference/legal.html). My versions of the networkx graph classes are kept in a separate package, [gorm](https://github.com/LogicalDash/gorm), and released under the same license.

The icons are [Entypo](http://entypo.com/), in the file Entypo.ttf, freely available at under the terms of [Creative Commons BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).

The LiSE source files themselves are licensed under the terms of the GNU General Public License version 3. See the text of the license in the file gpl-3.0.txt

Copyright (C) 2013-2014 Zachary Spector. Email: zacharyspector@gmail.com
