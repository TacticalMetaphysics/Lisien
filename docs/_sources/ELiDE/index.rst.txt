.. ELiDE documentation master file, created by
sphinx-quickstart on Mon Feb 19 11:33:00 2018.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

ELiDE
=====

The graphical interface, ELiDE, lets the developer change whatever
they want about the world. A game made with ELiDE will be more
restrictive about what the player is allowed to change, but all of the
player's input will be turned into changes to the world, which the
rules may respond to however they need.

ELiDE has three columns. On the right are a lot of buttons to access
the parts of ELiDE that aren't visible right now, plus a couple of
icons that you can drag into the middle. In the middle, you have a
graphical display of the Character under consideration; dragging those
icons here will make a new Place or Thing. To connect Places with
Portals, press the button with the arrow on it, then drag from one
Place to another. Press the button again when you're done. On the left
is the stat editor: it displays data that is stored in whatever entity
is presently selected. You can select Places, Things, and Portals by
clicking them--and once you've selected them, you can drag them
elsewhere. If no Place, Thing, or Portal is selected, then the
Character you are viewing is selected. There's a button in the
top-right to view another Character.

On the bottom left are some bits to let you manipulate time, mainly
the Simulate and 1 Turn buttons. Simulate will start moving time
forward when you press it, and stop when you press it again.  There
are also text fields with which you can enter the time by hand.  Note
that rules are only run when you advance time using Simulate or 1
Turn.  The Tick field indicates how many changes have occurred in the
current turn.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

game
----
Tools to make it easier to script your own game using the ELiDE widgets.

.. automodule:: ELiDE.game
    :members:

graph
-----
The default view on the state of the world.

board
`````
.. automodule:: ELiDE.graph.board
    :members:

pawn
````
Representations of Things

.. automodule:: ELiDE.graph.pawn
    :members:

spot
````
Representations of Places

.. automodule:: ELiDE.graph.spot
    :members:

arrow
`````
Representations of directed edges

.. automodule:: ELiDE.graph.arrow
    :members:

grid
----
Alternate board view for graphs that are shaped like grids

.. automodule:: ELiDE.grid.board
    :members:

screen
------
.. automodule:: ELiDE.screen
    :members:

card
----
The widget used to represent functions in the rules editor

.. automodule:: ELiDE.card
    :members:

charmenu
--------
The menu along the right side of the screen, where you can add things to the character

.. automodule:: ELiDE.charmenu
    :members:

charsview
---------
Menu for selecting which Character to work on

.. automodule:: ELiDE.charsview
    :members:

dialog
------
Simple data-driven UI interactions

.. automodule:: ELiDE.dialog
    :members:

menu
----
The menu along the left side of the screen, containing time controls and the stat editor

.. automodule:: ELiDE.menu
    :members:

rulesview
---------
Here you can assemble rules out of prewritten functions. First pick
which rule to edit from the menu on the left, using the box at the
bottom to add one if needed.  Then go through the trigger, prereq, and
action tabs, and drag the functions from the right pile to the left to
include them in the rule. You may also reorder them within the left
pile.

Rules made here will apply to the entity currently selected in the
main screen.  There is currently no graphical way to apply the same
rulebook to many entities.  You can, however, select nothing, in which
case you get the option to edit rulebooks that apply to the current
character overall.

.. automodule:: ELiDE.rulesview
    :members:

spritebuilder
-------------
A screen to put together a graphic from premade parts for a Place or Thing.

.. automodule:: ELiDE.spritebuilder
    :members:


dummy
`````
The pawn and spot that you can drag to place into the world.

.. automodule:: ELiDE.dummy
    :members:

pallet
``````
Individual menus of parts for the sprites.

.. automodule:: ELiDE.pallet
    :members:

statlist
--------
A two-column table of an entity's stats and their values. You can use this to build a primitive interface to your
game, or just monitor the state of the world. By default, they are all shown as Readouts, which is to say, plain text.

By default, stats' values are displayed as read-only text, but an entity
with a dictionary stat named ``"_config"`` may display them other ways by
setting a key with the same name as the stat to a dictionary value,
with its key ``"control"`` set to one of:

* ``"readout"`` for the default read-only text display.
* ``"textinput"`` for editable text, to be parsed as a Python dictionary,
   list, tuple, or string. If the content cannot be parsed, it will be
   treated as a string. Surround the content with quotation marks if you
   want to be sure it is a string.
* ``"slider"`` for picking a number within a range. Set the keys ``"min"``
   and ``"max"`` to specify the range.
* ``"togglebutton"`` for switching between ``True`` and ``False``. To display
   a different string for each, set the keys ``"true_text"`` and
   ``"false_text"``.

.. automodule:: ELiDE.statlist
    :members:

statcfg
```````
Configurator to change stat display modes within ELiDE.

.. automodule:: ELiDE.statcfg
    :members:

stores
------
Editor widgets for strings and Python code.

.. automodule:: ELiDE.stores
    :members:

Python Editor
`````````````

Click the Python button to edit your game code in the IDE if you like.
In this case, you can't use any of the decorators. Choose the
appropriate tab from Trigger, Prereq, or Action at the top, and the
function you write will show up in the appropriate part of the rules
editor.

Strings Editor
``````````````

The LiSE engine has an attribute ``string`` that is accessed like a
dictionary and used to store arbitrary strings, such as might be shown
in a menu. You can edit those here. You can store strings for multiple
languages, and switch between them programmatically by setting
``engine.string.language``.

util
----
Miscellaneous helpful things

.. automodule:: ELiDE.util
    :members:


app
---
Entry point to ELiDE

.. automodule:: ELiDE.app
    :members:

