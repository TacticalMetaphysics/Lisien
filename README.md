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

# Programming interface

LiSE itself is a Python library. It has few external dependencies, and
those it has are pure-Python libraries, so it can be used in any game
engine that supports Python as a scripting language.

The LiSE world model is a collection of graph structures called
"Characters" that are customized so that some nodes are considered
"Things" that can move about between "Places" over time. Characters
may have "avatars" in other characters, so that eg. the same person may be
represented by a skill tree, a pawn on a tactical grid, and a member
of a party.

You can store arbitrary JSON-serializable data in these
graphs, and it will be kept in a database with version
control. "Loading a saved game" in LiSE just means setting the
engine's clock back a while, so that it looks up data from earlier in
the game.

Game rules are ordinary Python functions that are attached to
Characters using decorators:

```
# If the kobold is not in a shrubbery, it will try to get to one.
# If it is, there's a chance it will try to get to another one, anyway.
@kobold.avatar.rule
def shrubsprint(engine, character, avatar):
    """Sprint to a location, other than the one I'm in already, which has
    a shrub in it.

    """
    # pregenerated list of places with shrubs in
    shrub_places = character.stat['shrub_places']
    if avatar['location'] in shrub_places:
        shrub_places.remove(avatar['location'])
    avatar.travel_to(engine.choice(shrub_places))

@shrubsprint.trigger
def uncovered(engine, character, avatar):
    """Return True when I'm *not* in a place with a shrub in it."""
    for shrub_candidate in avatar.location.contents():
        if shrub_candidate.name[:5] == "shrub":
            return False
    return True

@shrubsprint.trigger
def breakcover(engine, character, avatar):
    """Return True when I *am* in a place with a shrub in it, but elect to
    sprint anyway.

    """
    # This is checked after uncovered so I don't need to re-check
    # for shrubbery.
    return engine.random() < character.stat['sprint_chance']

@shrubsprint.prereq
def notsprinting(engine, character, avatar):
    """Only start a new sprint when not already sprinting."""
    return avatar['next_arrival_time'] is None
```

The rules are likewise stored in the database, and can be activated,
deactivated, and modified at particular times in the game, perhaps as
an effect of some other rule.

# Graphical development environment

ELiDE will be a graphical tool for developing this sort of game. It
will provide an interface that lets you make arbitrary changes to the
world state, then watch it run for a while to see what will happen
before turning the clock back and trying something different. It will
be scriptable in Python, hopefully to the point that you could build
the actual interface to your game in ELiDE.

# License Information
wallpape.jpg is copyright [Fantastic Maps](http://www.fantasticmaps.com/free-stuff/), freely available under the terms of [Creative Commons BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/).

LiSE currently has two graphics sets, the [RLTiles](http://rltiles.sourceforge.net/) and [Pixel City](http://opengameart.org/content/pixel-city), both available under [CC0](http://creativecommons.org/publicdomain/zero/1.0/), being in the public domain where it exists.

networkx, which forms the basis of LiSE's data model, is available under [BSD](http://networkx.github.io/documentation/latest/reference/legal.html). My versions of the networkx graph classes are kept in a separate package, [gorm](https://github.com/LogicalDash/gorm), and released under the same license.

The icons are [Entypo](http://entypo.com/), in the file Entypo.ttf, freely available at under the terms of [Creative Commons BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).

The LiSE source files themselves are licensed under the terms of the GNU General Public License version 3. See the text of the license in the file gpl-3.0.txt

Copyright (C) 2013-2014 Zachary Spector. Email: zacharyspector@gmail.com
