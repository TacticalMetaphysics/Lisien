Report any problems or feature requests for Lisien on
[Codeberg](https://codeberg.org/clayote/Lisien/issues).

The best way for new users to help with Lisien is to try to make a game with it. Try
modifying [the Awareness sim](https://github.com/TacticalMetaphysics/LiSE/blob/main/ELiDE/ELiDE/examples/awareness.py).
This template shows you how to use
Elide's widgets as the frontend to your game, which will likely be the easiest route in the long run, but if you have
trouble with it, or you need 3D, other Python game engines are available; consider [Pyglet](http://pyglet.org/)
or [Ursina](https://www.ursinaengine.org/). If you're using such an engine, you don't need Elide, though it might be helpful as a
way to browse the world state if it's not readily apparent from looking at your game.

In the likely case that something about the engine doesn't behave how you expect, or you need a feature, file a ticket
against the Lisien repository. Please link me to your source code or at least paste a runnable snippet in the ticket if at
all possible.

Lisien does not follow the formatting conventions of Python's standard `black`
autoformatter, due to ideological differences. Instead, use [ruff](https://github.com/astral-sh/ruff) with the included
`ruff.toml`.

If you need more help, send email to public@zacharyspector.com
