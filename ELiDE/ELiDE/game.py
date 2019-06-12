# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty
)
from kivy.resources import resource_find
from .app import ELiDEApp
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import LiSE.proxy
from .util import trigger
from functools import partial


class GameScreen(Screen):
    """A version of :class:`kivy.uix.screenmanager.Screen` that is easier to set up and use with ELiDE

    Should be a child of the :class:`ELiDE.game.Screens` widget, which will never itself be displayed.
    ``GameScreen`` instances in it will be added to the screen manager, so that you can switch
    to them with the ``switch_screen`` method.

    Every ``GameScreen`` needs a ``name``, just like regular ``Screen``.

    """
    switch_screen = ObjectProperty()
    """Method to set the ``screen`` attribute of the main :class:`kivy.uix.screenmanager.ScreenManager`"""
    app = ObjectProperty()
    """The running ``GameApp`` object"""
    engine = ObjectProperty()
    """An instance of ``EngineProxy``"""
    shutdown = ObjectProperty()
    """Call this to quit the game"""
    disabled = BooleanProperty(False)
    """If you bind your widgets' ``disabled`` to this, they will be disabled when a game command is in mid-execution"""

    def disable_input(self, cb=None):
        """Set ``self.disabled`` to ``True``, then call ``cb`` if provided

        :param cb: callback function for after disabling
        :return: ``None``

        """
        self.disabled = True
        if cb:
            cb()

    def enable_input(self, cb=None):
        """Call ``cb`` if provided, then set ``self.disabled`` to ``False``

        :param cb: callback function for before enabling
        :return: ``None``

        """
        if cb:
            cb()
        self.disabled = False

    def wait_travel(self, character, thing, dest, cb=None):
        """Schedule a thing to travel someplace, then wait for it to finish.

        :param character: name of the character
        :param thing: name of the thing that will travel
        :param dest: name of the place it will travel to
        :param cb: callback function for when it's done, optional
        :return: ``None``

        """
        self.disable_input()
        self.app.wait_travel(character, thing, dest, cb=partial(self.enable_input, cb))

    def wait_turns(self, turns, cb=None):
        """Call ``self.app.engine.next_turn()`` ``n`` times, waiting ``self.app.turn_length`` in between

        Disables input for the duration.

        :param turns: number of turns to wait
        :param cb: function to call when done waiting, optional
        :return: ``None``

        """
        self.disable_input()
        self.app.wait_turns(turns, cb=partial(self.enable_input, cb))

    def wait_command(self, start_func, turns=1, end_func=None):
        """Call ``start_func``, wait ``turns``, and then call ``end_func`` if provided

        Disables input for the duration.

        :param start_func: function to call just after disabling input
        :param turns: number of turns to wait
        :param end_func: function to call just before re-enabling input
        :return: ``None``

        """
        self.disable_input()
        start_func()
        self.app.wait_turns(turns, cb=partial(self.enable_input, end_func))

    def wait_travel_command(self, character, thing, dest, start_func, turns=1, end_func=lambda: None):
        """Schedule a thing to travel someplace and do something, then wait for it to finish.

        Input will be disabled for the duration.

        :param character: name of the character
        :param thing: name of the thing
        :param dest: name of the destination (a place)
        :param start_func: function to call when the thing gets to dest
        :param turns: number of turns to wait after start_func before re-enabling input
        :param end_func: optional. Function to call after waiting ``turns`` after start_func
        :return: ``None``

        """
        self.disable_input()
        self.app.wait_travel_command(character, thing, dest, start_func, turns, partial(self.enable_input, end_func))


class Screens(Widget):
    """Children of this widget will be added to the ``GameApp``'s :class:`kivy.uix.screenmanager.ScreenManager`"""
    app = ObjectProperty()

    def add_widget(self, wid, index=0, canvas=None):
        wid.app = self.app
        wid.engine = self.app.engine
        wid.switch_screen = self.app.screen_manager.setter('screen')
        wid.shutdown = self.app.stop
        super().add_widget(wid, index, canvas)


class GameApp(ELiDEApp):
    modules = []
    engine = ObjectProperty()
    world_file = None
    branch = StringProperty('trunk')
    turn = NumericProperty(0)
    tick = NumericProperty(0)
    turn_length = NumericProperty(0.5)

    def wait_turns(self, turns, dt=None, *, cb=None):
        """Call ``self.engine.next_turn()`` ``n`` times, waiting ``self.turn_length`` in between

        If provided, call ``cb`` when done.

        :param turns: number of turns to wait
        :param dt: unused, just satisfies the clock
        :param cb: callback function to call when done, optional
        :return: ``None``

        """
        if turns == 0:
            if cb:
                cb()
            return
        self.engine.next_turn()
        turns -= 1
        Clock.schedule_once(partial(self.wait_turns, turns, cb=cb), self.turn_length)

    def wait_travel(self, character, thing, dest, cb=None):
        """Schedule a thing to travel someplace, then wait for it to finish, and call ``cb`` if provided

        :param character: name of the character
        :param thing: name of the thing
        :param dest: name of the destination (a place)
        :param cb: function to be called when I'm done
        :return: ``None``
        
        """
        self.wait_turns(self.engine.character[character].thing[thing].travel_to(dest), cb=cb)

    def wait_command(self, start_func, turns=1, end_func=None):
        """Call ``start_func``, and wait to call ``end_func`` after simulating ``turns`` (default 1)

        :param start_func: function to call before waiting
        :param turns: number of turns to wait
        :param end_func: function to call after waiting
        :return: ``None``

        """
        start_func()
        self.wait_turns(turns, cb=end_func)

    def wait_travel_command(self, character, thing, dest, start_func, turns=1, end_func=None):
        """Schedule a thing to travel someplace and do something, then wait for it to finish.

        :param character: name of the character
        :param thing: name of the thing
        :param dest: name of the destination (a place)
        :param start_func: function to call when the thing gets to dest
        :param turns: number of turns to wait after start_func before re-enabling input
        :param end_func: optional. Function to call after waiting ``turns`` after start_func
        :return: ``None``
        """
        self.wait_travel(character, thing, dest, cb=partial(
            self.wait_command, start_func, turns, end_func)
        )

    def on_engine(self, *args):
        self.branch, self.turn, self.tick = self.engine._btt()
        self.engine.time.connect(self._pull_time, weak=False)

    def _pull_time(self, *args, branch, turn, tick):
        self.branch, self.turn, self.tick = branch, turn, tick

    def _get_worlddb(self):
        filen = self.world_file or \
                self.name + 'World.db' if self.name \
                else 'LiSEWorld.db'
        return resource_find(filen) or filen
    worlddb = AliasProperty(
        _get_worlddb,
        lambda self, v: None
    )

    def build(self):
        have_world = False
        try:
            os.stat(self.worlddb)
            have_world = True
        except FileNotFoundError:
            pass
        self.procman = LiSE.proxy.EngineProcessManager()
        self.engine = self.procman.start(
            self.worlddb,
            logger=Logger, loglevel=getattr(self, 'loglevel', 'debug'),
            do_game_start=not have_world,
            install_modules=self.modules
        )
        self.screen_manager = ScreenManager()
        self.screens = Screens(app=self)
        self.screens.bind(children=self._pull_screens)
        self._pull_screens()
        if hasattr(self, 'inspector'):
            from kivy.core.window import Window
            from kivy.modules import inspector
            inspector.create_inspector(Window, self.screen_manager)
        return self.screen_manager

    @trigger
    def _pull_screens(self, *args):
        for screen in reversed(self.screens.children):
            print('pulling screen ' + screen.name)
            self.screens.remove_widget(screen)
            self.screen_manager.add_widget(screen)

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.config.write()

    def on_stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.procman.shutdown()
        self.config.write()
