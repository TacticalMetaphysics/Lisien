.. ELiDE documentation master file, created by
   sphinx-quickstart on Mon Feb 19 11:33:00 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ELiDE
=====

.. toctree::
   :maxdepth: 2
   :caption: Contents:

game
----
.. automodule:: ELiDE.game
    :members:

board
-----
.. automodule:: ELiDE.graph.board
    :members:

pawn
````
.. automodule:: ELiDE.graph.pawn
    :members:

spot
````
.. automodule:: ELiDE.graph.spot
    :members:

arrow
`````
.. automodule:: ELiDE.graph.arrow
    :members:

screen
------
.. automodule:: ELiDE.screen
    :members:

card
----
.. automodule:: ELiDE.card
    :members:

charmenu
--------
.. automodule:: ELiDE.charmenu
    :members:

charsview
---------
.. automodule:: ELiDE.charsview
    :members:

dialog
------
.. automodule:: ELiDE.dialog
    :members:

dummy
-----
.. automodule:: ELiDE.dummy
    :members:

menu
----
.. automodule:: ELiDE.menu
    :members:

pallet
------
.. automodule:: ELiDE.pallet
    :members:

rulesview
---------
.. automodule:: ELiDE.rulesview
    :members:

spritebuilder
-------------
.. automodule:: ELiDE.spritebuilder
    :members:

statcfg
-------
.. automodule:: ELiDE.statcfg
    :members:

statlist
--------
A two-column table of an entity's stats and their values.

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

stores
------
.. automodule:: ELiDE.stores
    :members:

util
----
.. automodule:: ELiDE.util
    :members:


app
---
.. automodule:: ELiDE.app
    :members:

