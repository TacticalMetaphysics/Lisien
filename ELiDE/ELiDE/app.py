# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com
"""Object to configure, start, and stop ELiDE."""

import json

from kivy.logger import Logger
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.resources import resource_add_path

from kivy.uix.screenmanager import ScreenManager

from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty,
    StringProperty
)
import LiSE
from LiSE.proxy import EngineProcessManager
import ELiDE
import ELiDE.screen
import ELiDE.stringsed
import ELiDE.funcsed
import ELiDE.statcfg
import ELiDE.spritebuilder
import ELiDE.rulesview
import ELiDE.charsview
from .util import trigger

resource_add_path(ELiDE.__path__[0] + "/assets")
resource_add_path(ELiDE.__path__[0] + "/assets/rltiles")


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    """
    engine = ObjectProperty()
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ListProperty(['master', 0])
    character = ObjectProperty()
    selection = ObjectProperty(None, allownone=True)
    selected_remote = ObjectProperty()

    def _get_character_name(self, *args):
        if self.character is None:
            return
        return self.character.name

    def _set_character_name(self, name):
        self.character = self.engine.character[name]

    character_name = AliasProperty(
        _get_character_name,
        _set_character_name,
        bind=('character',)
    )

    def _pull_time(self, *args):
        if not self.engine:
            Clock.schedule_once(self._pull_time, 0)
            return
        (self.branch, self.tick) = self.engine.time
    pull_time = trigger(_pull_time)

    def on_time(self, *args):
        local_time = (branch, tick) = tuple(self.time)
        if local_time != self.engine.time:
            self.engine.time = local_time
        if self.branch != branch:
            self.branch = branch
        if self.tick != tick:
            self.tick = tick

    def set_branch(self, b):
        """Set my branch to the given value."""
        self.branch = b
        if self.time[0] != b:
            self.time[0] = b
        if self.engine.time != self.time:
            self.engine.time_travel(
                *self.time,
                char=self.character.name,
                cb=self.mainscreen._update_from_chardiff
            )

    def set_tick(self, t):
        """Set my tick to the given value, cast to an integer."""
        self.tick = int(t)
        if self.time[1] != self.tick:
            self.time[1] = self.tick
        if self.engine.time != self.time:
            self.engine.time_travel(
                *self.time,
                char=self.character.name,
                cb=self.mainscreen._update_from_chardiff
            )

    def set_time(self, b, t=None):
        if t is None:
            (b, t) = b
        t = int(t)
        (self.branch, self.tick) = self.time = (b, t)

    def select_character(self, char):
        """Change my ``character`` to the selected character object if they
        aren't the same.

        """
        if char == self.character:
            return
        self.character = char

    def build_config(self, config):
        """Set config defaults"""
        for sec in 'LiSE', 'ELiDE':
            config.adddefaultsection(sec)
        config.setdefaults(
            'LiSE',
            {
                'world': 'sqlite:///LiSEworld.db',
                'code': 'LiSEcode.db',
                'language': 'en',
                'logfile': '',
                'loglevel': 'info'
            }
        )
        config.setdefaults(
            'ELiDE',
            {
                'boardchar': 'physical',
                'debugger': 'no',
                'inspector': 'no',
                'user_kv': 'yes',
                'play_speed': '1',
                'thing_graphics': json.dumps([
                    ("Marsh Davies' Island", 'marsh_davies_island_fg.atlas'),
                    ('RLTiles: Body', 'base.atlas'),
                    ('RLTiles: Basic clothes', 'body.atlas'),
                    ('RLTiles: Armwear', 'arm.atlas'),
                    ('RLTiles: Legwear', 'leg.atlas'),
                    ('RLTiles: Right hand', 'hand1.atlas'),
                    ('RLTiles: Left hand', 'hand2.atlas'),
                    ('RLTiles: Boots', 'boot.atlas'),
                    ('RLTiles: Hair', 'hair.atlas'),
                    ('RLTiles: Beard', 'beard.atlas'),
                    ('RLTiles: Headwear', 'head.atlas')
                ]),
                'place_graphics': json.dumps([
                    ("Marsh Davies' Island", 'marsh_davies_island_bg.atlas'),
                    ("Marsh Davies' Crypt", 'marsh_davies_crypt.atlas'),
                    ('RLTiles: Dungeon', 'dungeon.atlas')
                ])
            }
        )
        config.write()

    def build(self):
        """Make sure I can use the database, create the tables as needed, and
        return the root widget.

        """
        config = self.config
        Logger.debug(
            "ELiDEApp: starting with world {}, code {}, path {}".format(
                config['LiSE']['world'],
                config['LiSE']['code'],
                LiSE.__path__[-1]
            )
        )

        if config['ELiDE']['debugger'] == 'yes':
            import pdb
            pdb.set_trace()
        self.procman = EngineProcessManager()
        enkw = {}
        if 'logfile' in config['LiSE']:
            enkw['logfile'] = config['LiSE']['logfile']
        if 'loglevel' in config['LiSE']:
            enkw['loglevel'] = config['LiSE']['loglevel']
        self.engine = self.procman.start(
            config['LiSE']['world'],
            config['LiSE']['code'],
            **enkw
        )
        self.pull_time()

        #Clock.schedule_interval(self.procman.sync_log, 0.01)

        char = config['ELiDE']['boardchar']
        if char not in self.engine.character:
            self.engine.add_character(char)

        self.manager = ScreenManager()

        def toggler(screenname):
            def tog(*args):
                if self.manager.current == screenname:
                    self.manager.current = 'main'
                else:
                    self.manager.current = screenname
            return tog

        self.pawncfg = ELiDE.spritebuilder.PawnConfigScreen(
            toggle=toggler('pawncfg'),
            data=json.loads(config['ELiDE']['thing_graphics'])
        )

        self.spotcfg = ELiDE.spritebuilder.SpotConfigScreen(
            toggle=toggler('spotcfg'),
            data=json.loads(config['ELiDE']['place_graphics'])
        )

        self.rules = ELiDE.rulesview.RulesScreen(
            engine=self.engine,
            toggle=toggler('rules')
        )

        self.chars = ELiDE.charsview.CharactersScreen(
            engine=self.engine,
            toggle=toggler('chars')
        )

        self.strings = ELiDE.stringsed.StringsEdScreen(
            engine=self.engine,
            toggle=toggler('strings')
        )

        self.funcs = ELiDE.funcsed.FuncsEdScreen(
            app=self,
            name='funcs',
            toggle=toggler('funcs')
        )

        self.select_character(
            self.engine.character[
                config['ELiDE']['boardchar']
            ]
        )

        self.statcfg = ELiDE.statcfg.StatScreen(
            app=self,
            toggle=toggler('statcfg')
        )
        self.bind(time=self.statcfg.setter('time'))

        self.mainscreen = ELiDE.screen.MainScreen(
            app=self,
            use_kv=config['ELiDE']['user_kv'] == 'yes',
            play_speed=int(config['ELiDE']['play_speed'])
        )
        self.bind(selection=self.reremote)
        self.selected_remote = self._get_selected_remote()
        for wid in (
                self.mainscreen,
                self.pawncfg,
                self.spotcfg,
                self.statcfg,
                self.rules,
                self.chars,
                self.strings,
                self.funcs
        ):
            self.manager.add_widget(wid)
        if config['ELiDE']['inspector'] == 'yes':
            from kivy.core.window import Window
            from kivy.modules import inspector
            inspector.create_inspector(Window, self.mainscreen)

        return self.manager

    def _get_selected_remote(self):
        if self.selection is None:
            return self.character.stat
        elif hasattr(self.selection, 'remote'):
            return self.selection.remote
        elif (
                hasattr(self.selection, 'portal') and
                self.selection.portal is not None
        ):
            return self.selection.portal
        else:
            raise ValueError("Invalid selection: {}".format(self.selection))

    @trigger
    def reremote(self, *args):
        self.selected_remote = self._get_selected_remote()

    def on_character_name(self, *args):
        if self.config['ELiDE']['boardchar'] != self.character_name:
            self.config['ELiDE']['boardchar'] = self.character_name

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.config.write()

    def on_stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.procman.shutdown()
        self.config.write()

    def on_selection(self, *args):
        Logger.debug("ELiDEApp: selection {}".format(self.selection))


kv = """
<App>:
    time: [root.branch, root.tick]
    selected_remote: root.selection.remote if root.selection else None
<SymbolLabel@Label>:
    font_name: "Symbola.ttf"
    font_size: 50
<SymbolButton@Button>:
    font_name: "Symbola.ttf"
    font_size: 50
"""
Builder.load_string(kv)
