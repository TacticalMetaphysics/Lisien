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
    StringProperty
)
import LiSE
from LiSE.proxy import EngineProcessManager
import ELiDE
import ELiDE.screen
import ELiDE.stores
import ELiDE.statcfg
import ELiDE.spritebuilder
import ELiDE.rulesview
import ELiDE.charsview
from ELiDE.board.arrow import ArrowWidget
from ELiDE.board.spot import Spot
from ELiDE.board.pawn import Pawn
from .util import trigger

resource_add_path(ELiDE.__path__[0] + "/assets")


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    """
    engine = ObjectProperty()
    branch = StringProperty('trunk')
    tick = NumericProperty(0)
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

    @trigger
    def _push_time(self, *args):
        if self.engine.time != (self.branch, self.tick):
            self.engine.time_travel(
                self.branch, self.tick,
                chars=[self.character.name],
                cb=self.mainscreen._update_from_chardiff
            )

    def set_tick(self, t):
        """Set my tick to the given value, cast to an integer."""
        self.tick = int(t)

    def set_time(self, b, t=None):
        if t is None:
            (b, t) = b
        t = int(t)
        (self.branch, self.tick) = (b, t)

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
                'language': 'eng',
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
                    ('RLTiles: Body', 'rltiles/base.atlas'),
                    ('RLTiles: Basic clothes', 'rltiles/body.atlas'),
                    ('RLTiles: Armwear', 'rltiles/arm.atlas'),
                    ('RLTiles: Legwear', 'rltiles/leg.atlas'),
                    ('RLTiles: Right hand', 'rltiles/hand1.atlas'),
                    ('RLTiles: Left hand', 'rltiles/hand2.atlas'),
                    ('RLTiles: Boots', 'rltiles/boot.atlas'),
                    ('RLTiles: Hair', 'rltiles/hair.atlas'),
                    ('RLTiles: Beard', 'rltiles/beard.atlas'),
                    ('RLTiles: Headwear', 'rltiles/head.atlas')
                ]),
                'place_graphics': json.dumps([
                    ("Marsh Davies' Island", 'marsh_davies_island_bg.atlas'),
                    ("Marsh Davies' Crypt", 'marsh_davies_crypt.atlas'),
                    ('RLTiles: Dungeon', 'rltiles/dungeon.atlas'),
                    ('Pixel City: Storeys', 'pixelcity/32x_outlineless/floors.atlas'),
                    ('Pixel City: Ground', 'pixelcity/32x_outlineless/ground.atlas'),
                    ('Pixel City: Road', 'pixelcity/32x_outlineless/road.atlas')
                ]),
                'stacking_heights': json.dumps({
                    'pixelcity/32x_outlineless/floors.atlas': 8
                })
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
        enkw = {'logger': Logger}
        if config['LiSE'].get('logfile'):
            enkw['logfile'] = config['LiSE']['logfile']
        if config['LiSE'].get('loglevel'):
            enkw['loglevel'] = config['LiSE']['loglevel']
        self.engine = self.procman.start(
            config['LiSE']['world'],
            config['LiSE']['code'],
            **enkw
        )
        self.pull_time()

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

        thing_graphics = json.loads(config['ELiDE']['thing_graphics'])
        place_graphics = json.loads(config['ELiDE']['place_graphics'])
        stacking_heights = json.loads(config['ELiDE']['stacking_heights'])

        self.pawncfg = ELiDE.spritebuilder.PawnConfigScreen(
            toggle=toggler('pawncfg'),
            data=[
                (name, path, stacking_heights.get(path, 0))
                for name, path in thing_graphics
            ]
        )

        self.spotcfg = ELiDE.spritebuilder.SpotConfigScreen(
            toggle=toggler('spotcfg'),
            data=[
                (name, path, stacking_heights.get(path, 0))
                for name, path in place_graphics
            ]
        )

        self.rules = ELiDE.rulesview.RulesScreen(
            engine=self.engine,
            toggle=toggler('rules')
        )

        self.chars = ELiDE.charsview.CharactersScreen(
            engine=self.engine,
            toggle=toggler('chars')
        )

        self.strings = ELiDE.stores.StringsEdScreen(
            language=self.engine.string.language,
            language_setter=self._set_language,
            toggle=toggler('strings')
        )

        self.funcs = ELiDE.stores.FuncsEdScreen(
            name='funcs',
            toggle=toggler('funcs')
        )

        self.select_character(
            self.engine.character[
                config['ELiDE']['boardchar']
            ]
        )

        self.statcfg = ELiDE.statcfg.StatScreen(
            toggle=toggler('statcfg'),
            branch=self.branch,
            tick=self.tick,
            engine=self.engine
        )
        self.bind(
            selected_remote=self.statcfg.setter('remote'),
            branch=self.statcfg.setter('branch'),
            tick=self.statcfg.setter('tick')
        )

        self.mainscreen = ELiDE.screen.MainScreen(
            use_kv=config['ELiDE']['user_kv'] == 'yes',
            play_speed=int(config['ELiDE']['play_speed'])
        )
        if self.mainscreen.statlist:
            self.statcfg.statlist = self.mainscreen.statlist
        self.mainscreen.bind(statlist=self.statcfg.setter('statlist'))
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

        @self.engine.time.connect
        def pull_time(inst, **kwargs):
            self.branch = inst.branch
            self.tick = inst.tick

        @self.engine.string.language.connect
        def pull_lang(inst, **kwargs):
            self.strings.language = kwargs['language']

        self.bind(
            branch=self._push_time,
            tick=self._push_time
        )
        return self.manager

    def _set_language(self, lang):
        self.engine.string.language = lang

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

    def delete_selection(self):
        """Delete both the selected widget and whatever it represents."""
        selection = self.selection
        if selection is None:
            return
        if isinstance(selection, ArrowWidget):
            self.selection = None
            self.screen.board.rm_arrow(
                selection.origin.name,
                selection.destination.name
            )
            selection.portal.delete()
        elif isinstance(selection, Spot):
            self.selection = None
            self.screen.board.rm_spot(selection.name)
            selection.remote.delete()
        else:
            assert isinstance(selection, Pawn)
            self.selection = None
            self.screen.board.rm_pawn(selection.name)
            selection.remote.delete()


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
