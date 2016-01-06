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
* The King of Dragon Pass

To demonstrate the capabilities of the engine, I will develop the game
[Dungeon University](http://forums.tigsource.com/index.php?topic=43022).

# Why should I use LiSE for this purpose?

LiSE is a game engine in the sense of RPG Maker or Ren'Py. It assumes
that there are certain problems any designer of life simulators will
have, and provides powerful tools specialized to those
problems. Though you will still need to write some Python code for
your game, it should only be the code that describes how your game's
world works. If you don't want to worry about the data structure that
represents the world, LiSE gives you one that will work. If you don't
want to write a user interface, you can play the game with the same
program you drew its map in. And then again, if your game is similar
enough to one that's been freely released, you can import rules from
that one, and not write that code yourself, after all.

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
* *Rule constructor*: Build rules out of functions represented as cards. Looks like deckbuilding in a card game like Magic.
* *Autosave*. Actually, anything you do gets put in a transaction that gets committed when you quit. In any case you never need to save

# Testing

If you want to run the latest code, and not one of the devkits under the
Releases tab, please use Linux, and preferably Ubuntu.

Download and run ``ELiDE_bootstrap.sh``. It should set everything up for you.

# License Information

ELiDE uses third-party graphics sets:

* The [RLTiles](http://rltiles.sourceforge.net/), available under [CC0](http://creativecommons.org/publicdomain/zero/1.0/), being in the public domain where it exists.
* "Crypt" and "Island" from the [PROCJAM 2015 Art Pack](http://www.procjam.com/2015/09/procjam-art-pack-now-available/), by Marsh Davies, available under the terms of [Creative Commons BY-NC](http://creativecommons.org/licenses/by-nc/4.0/)
* The default wallpaper, wallpape.jpg, is copyright [Fantastic Maps](http://www.fantasticmaps.com/free-stuff/), freely available under the terms of [Creative Commons BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/).


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
