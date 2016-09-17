import os
from kivy.logger import Logger
from kivy.properties import (
    AliasProperty,
    ObjectProperty
)
from kivy.resources import resource_find
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
import LiSE.proxy
from .util import trigger


class GameScreen(Screen):
    switch_screen = ObjectProperty()
    engine = ObjectProperty()
    shutdown = ObjectProperty()


class Screens(Widget):
    app = ObjectProperty()

    def add_widget(self, wid, index=0):
        wid.engine = self.app.engine
        wid.switch_screen = self.app.screen_manager.setter('screen')
        wid.shutdown = self.app.stop
        super().add_widget(wid, index)


class GameApp(App):
    modules = []
    engine = ObjectProperty()
    world_file = None
    code_file = None

    def _get_worlddb(self):
        filen = self.world_file or self.name + 'World.db' if self.name else 'LiSEWorld.db'
        return resource_find(filen) or filen
    worlddb = AliasProperty(
        _get_worlddb,
        lambda self, v: None
    )

    def _get_codedb(self):
        filen = self.code_file or self.name + 'Code.db' if self.name else 'LiSECode.db'
        return resource_find(filen) or filen
    codedb = AliasProperty(
        _get_codedb,
        lambda self, v: None
    )
    screens = ObjectProperty()

    def build(self):
        have_world = have_code = False
        try:
            os.stat(self.worlddb)
            have_world = True
        except FileNotFoundError:
            pass
        try:
            os.stat(self.codedb)
            have_code = True
        except FileNotFoundError:
            pass
        self.procman = LiSE.proxy.EngineProcessManager()
        self.engine = self.procman.start(self.worlddb, self.codedb, logger=Logger)
        if not have_code:
            for module in self.modules:
                self.engine.handle(command='install_module', module=module)
        if not have_world:
            self.engine.handle(command='do_game_start')
        if not (have_world and have_code):
            self.engine.pull()
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
