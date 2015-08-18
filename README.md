LiSE is an application for developing life simulation games.

[Development blog](http://forums.tigsource.com/index.php?topic=35227.0)

# What is a life simulation game?

For the purposes of LiSE, it is any game where you are primarily
concerned with "who you are" rather than "what you're
doing". Nearly everything your character can do should be
represented in the game somehow, but mostly in non-interactive form. This frees
the player to attend to the high-level tasks of "living" in the game
world--which, in concrete terms, boil down to time and resource
management.

Existing games that LiSE seeks to imitate include:

* The Sims
* Redshirt
* Animal Crossing
* Monster Rancher
* Dwarf Fortress
* Democracy
* Crusader Kings

To demonstrate the capabilities of the engine, I will develop the game
[Dungeon University](http://forums.tigsource.com/index.php?topic=43022).

# Why should I use LiSE for this purpose?

Most programs billed as "game engines" provide graphics, networking,
file system access, scripting, and perhaps physics, since most games
need these things. For the actual game logic, they tend to provide
little more than a scripting interface.

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

# Features

## Core

* *Object relational mapper* for graph based world models.
* *Journaling* to allow world state changes to be rewound and replayed.
* Integration with [NetworkX](http://networkx.github.io) for convenient access to various *graph algorithms*, particularly pathfinding.
* *Rules engine*: define your game's behavior in terms of actions that are performed in response to triggers. Change the connection from trigger to action without effort. Copy triggers and actions between games easily.

## IDE

* View and edit state graphs in a *drag-and-drop interface*.
* *Rewind time* and the interface will show you the state of the world back then.
* Code editor with syntax highlighting.
* *Rule constructor*: Build rules out of functions represented as cards. Looks like deckbuilding in a CCG.
* *Autosave*. Actually, anything you do gets put in a transaction that gets committed when you quit. In any case you never need to save

# License Information

wallpape.jpg is copyright [Fantastic
Maps](http://www.fantasticmaps.com/free-stuff/), freely available
under the terms of [Creative Commons
BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/).

ELiDE currently has two graphics sets, the
[RLTiles](http://rltiles.sourceforge.net/) and [Pixel
City](http://opengameart.org/content/pixel-city), both available under
[CC0](http://creativecommons.org/publicdomain/zero/1.0/), being in the
public domain where it exists.

[networkx](http://networkx.github.io/), which forms the basis of
LiSE's data model, is available under
[BSD](http://networkx.github.io/documentation/latest/reference/legal.html). My
versions of the networkx graph classes are kept in a separate package,
[gorm](https://github.com/LogicalDash/gorm), and released under the
same license.

The icons are [Symbola](http://users.teilar.gr/~g1951d/), by George
Douros, in the public domain.

The LiSE source files themselves are licensed under the terms of the
GNU General Public License version 3. See the text of the license in
the file gpl-3.0.txt

Copyright (C) 2013-2014 Zachary Spector. Email: zacharyspector@gmail.com
