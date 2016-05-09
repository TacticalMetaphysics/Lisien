import os
from importlib import import_module
from kivy.properties import (
    AliasProperty,
    ObjectProperty
)
from kivy.resources import resource_find
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from LiSE.engine import Engine
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

    def _get_worlddb(self):
        filen = self.name + 'World.db' if self.name else 'LiSEWorld.db'
        return resource_find(filen) or filen
    worlddb = AliasProperty(
        _get_worlddb,
        lambda self, v: None
    )

    def _get_codedb(self):
        filen = self.name + 'Code.db' if self.name else 'LiSECode.db'
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
        if not (have_world and have_code):
            engine = Engine(self.worlddb, self.codedb)
            if not have_code:
                for module in self.modules:
                    import_module(module).install(engine)
            if not have_world:
                engine.function['__init__'](engine)
            engine.close()
        self.procman = LiSE.proxy.EngineProcessManager()
        self.engine = self.procman.start(self.worlddb, self.codedb)
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
