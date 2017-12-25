import os
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty
)
from kivy.resources import resource_find
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import LiSE.proxy
from .util import trigger
from functools import partial


class GameScreen(Screen):
    switch_screen = ObjectProperty()
    app = ObjectProperty()
    engine = ObjectProperty()
    shutdown = ObjectProperty()


class Screens(Widget):
    app = ObjectProperty()

    def add_widget(self, wid, index=0, canvas=None):
        wid.app = self.app
        wid.engine = self.app.engine
        wid.switch_screen = self.app.screen_manager.setter('screen')
        wid.shutdown = self.app.stop
        super().add_widget(wid, index, canvas)


class GameApp(App):
    modules = []
    engine = ObjectProperty()
    world_file = None
    branch = StringProperty('trunk')
    turn = NumericProperty(0)
    tick = NumericProperty(0)
    turn_length = 0.5

    def wait_turns(self, n, dt=None, *, cb=None):
        """Call ``self.engine.next_turn()`` ``n`` times, waiting ``self.turn_length`` in between

        If provided, call ``cb`` when done.

        """
        print(dt)
        self.engine.next_turn()
        n -= 1
        if n == 0:
            if cb:
                cb()
        else:
            Clock.schedule_once(partial(self.wait_turns, n, cb=cb), self.turn_length)

    def on_engine(self, *args):
        self.branch, self.turn, self.tick = self.engine.btt()
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
