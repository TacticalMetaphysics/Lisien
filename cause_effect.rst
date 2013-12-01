===============================
The LiSE Cause and Effect Model
===============================
:Info: Documentation for the cause-and-effect model in LiSE, a development toolkit for life simulators
:Copyright: 2013 by Zachary Spector. Available under the terms of the GNU Free Documentation License 1.3

Purpose
=======

Causes and Effects provide a simple programming interface for
describing simulation rules in a form that can be easily reused from
one simulation to another. They resemble plugins, but with a narrow
purpose, and appropriately restricted access to the rest of the API.

LiSE users who are not programmers should be able to import Causes and
Effects and associate them with one another, thereby constructing a
rich simulation without much knowledge about the underlying code. This
will most often be done through a graphical interface. As much as
possible, Causes and Effects will be presented in the same way as
assets, such as images, sound, and video.

The process of associating causes with effects might be regarded as a
kind of application scripting, but if you really want to extend LiSE,
rather than implement simulations and components thereof, you should
do it using the plugin interface proper.


Overview
========

An Effect is an encapsulated callback function. It takes a Character
object, though that Character may contain as much of the world model
as it likes. It returns a tuple describing a change to exactly one
fact about that character. A fact is like a database record; an exact
definition is below.

Despite their name, Effects must never have side effects. That would
cause problems tracking the history of the world, particularly the
question of when (in diegetic time) a particular change occurred--LiSE
often simulates parts of the timestream that the user has not seen
yet. But you can generally assume that the change that an Effect
returns will be enacted on the world model on the same tick when the
event is called, and that the world will reflect it in future ticks.

A Cause also takes a Character object, but it must return a Boolean
value. When it returns True, any Effects associated with it will be
fired--though possibly in any of several orderings, possibly after
some delays, and the change returned by the Effect might go through
some other tests before it is applied to the world.

Causes may pass extra keyword arguments to their Effects. These may
contain world data that is not kept in Characters, such as the branch
and tick, and perhaps other variables that are "global" in the world
under discussion. Causes cannot take any variables apart from their
Character.


Facts about Characters
======================

An Effect that changes a stat returns a 3-tuple of the string
"``character_stat``", stat's name and its new value. You can remove a
stat from a character by setting it to None. Python recognizes a
difference between not having a variable and having a variable
containing None, but LiSE doesn't.

Skills may be removed from Characters in the same way with the string
"``character_skill``". You add a skill or alter an existing one by
assigning a list of Causes to it. These Causes will be called with
this Character when it uses this skill. Make sure the list contains
actual Cause objects rather than, for instance, strings.

Components may be added to or removed from Characters by the use of
the strings "``character_thing``", "``character_place``", and
"``character_portal``", according to the type of the component. Things
and Places are triples in which the 1th item is the name of the
dimension they are in, and the 2th item is the name of the Thing or
Place proper. For "``character_portal``" you actually want a 4-tuple,
where the 1th item is the dimension name, the 2th is the name of the
Place whence the Portal originates, and the 3th is its destination.
