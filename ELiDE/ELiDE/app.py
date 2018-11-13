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
"""Object to configure, start, and stop ELiDE."""

import json

from kivy.logger import Logger
from kivy.app import App
from kivy.clock import Clock
from kivy.resources import resource_add_path

from kivy.uix.screenmanager import ScreenManager, NoTransition

from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty
)
import LiSE
from LiSE.proxy import EngineProcessManager
import ELiDE
import ELiDE.dialog
import ELiDE.screen
import ELiDE.stores
import ELiDE.statcfg
import ELiDE.spritebuilder
import ELiDE.rulesview
import ELiDE.charsview
from ELiDE.board.board import Board
from ELiDE.board.arrow import ArrowWidget
from ELiDE.board.spot import Spot
from ELiDE.board.pawn import Pawn
from .util import trigger

resource_add_path(ELiDE.__path__[0] + "/assets")
resource_add_path(ELiDE.__path__[0] + "/assets/rltiles")


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    """
    title = 'ELiDE'

    engine = ObjectProperty()
    branch = StringProperty('trunk')
    turn = NumericProperty(0)
    tick = NumericProperty(0)
    character = ObjectProperty()
    selection = ObjectProperty(None, allownone=True)
    selected_proxy = ObjectProperty()

    def on_selection(self, *args):
        Logger.debug("App: {} selected".format(self.selection))

    def _get_character_name(self, *args):
        if self.character is None:
            return
        return self.character.name

    def _set_character_name(self, name):
        if self.character.name != name:
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
        branch, turn, tick = self.engine.btt()
        self.branch = branch
        self.turn = turn
        self.tick = tick
    pull_time = trigger(_pull_time)

    @trigger
    def _push_time(self, *args):
        branch, turn, tick = self.engine.btt()
        if (self.branch, self.turn, self.tick) != (branch, turn, tick):
            self.engine.time_travel(
                self.branch, self.turn, self.tick if self.tick != tick else None,
                chars=[self.character.name],
                cb=self.mainscreen._update_from_time_travel
            )

    def set_tick(self, t):
        """Set my tick to the given value, cast to an integer."""
        self.tick = int(t)

    def set_turn(self, t):
        self.turn = int(t)

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
        self.icon = 'icon_24px.png'
        config = self.config
        Logger.debug(
            "ELiDEApp: starting with world {}, path {}".format(
                config['LiSE']['world'],
                LiSE.__path__[-1]
            )
        )

        if config['ELiDE']['debugger'] == 'yes':
            import pdb
            pdb.set_trace()


        self.manager = ScreenManager(transition=NoTransition())
        if config['ELiDE']['inspector'] == 'yes':
            from kivy.core.window import Window
            from kivy.modules import inspector
            inspector.create_inspector(Window, self.manager)
        
        self._start_subprocess()
        self._add_screens()
        return self.manager

    def _pull_lang(self, *args, **kwargs):
        self.strings.language = kwargs['language']

    def _pull_chars(self, *args, **kwargs):
        self.chars.names = list(self.engine.character)

    def _pull_time_from_signal(self, *args, branch, turn, tick):
        self.branch, self.turn, self.tick = branch, turn, tick

    def _start_subprocess(self, *args):
        if hasattr(self, '_started'):
            raise ChildProcessError("Subprocess already running")
        config = self.config
        self.procman = EngineProcessManager()
        enkw = {'logger': Logger}
        if config['LiSE'].get('logfile'):
            enkw['logfile'] = config['LiSE']['logfile']
        if config['LiSE'].get('loglevel'):
            enkw['loglevel'] = config['LiSE']['loglevel']
        self.engine = self.procman.start(
            config['LiSE']['world'],
            **enkw
        )
        self.pull_time()

        self.engine.time.connect(self._pull_time_from_signal, weak=False)
        self.engine.string.language.connect(self._pull_lang, weak=False)
        self.engine.character.connect(self._pull_chars, weak=False)

        self.bind(
            branch=self._push_time,
            turn=self._push_time,
            tick=self._push_time
        )

        char = config['ELiDE']['boardchar']
        if char not in self.engine.character:
            self.engine.add_character(char)
        self._started = True

    def _add_screens(self, *args):
        if not getattr(self, '_started'):
            Clock.schedule_once(self._add_screens, 0)
            return
        def toggler(screenname):
            def tog(*args):
                if self.manager.current == screenname:
                    self.manager.current = 'main'
                else:
                    self.manager.current = screenname
            return tog
        config = self.config

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

        self.charrules = ELiDE.rulesview.CharacterRulesScreen(
            engine=self.engine,
            character=self.character,
            toggle=toggler('charrules')
        )
        self.bind(character=self.charrules.setter('character'))

        self.chars = ELiDE.charsview.CharactersScreen(
            engine=self.engine,
            toggle=toggler('chars'),
            names=list(self.engine.character),
            new_board=self.new_board
        )
        self.bind(character_name=self.chars.setter('character_name'))

        def chars_push_character_name(*args):
            self.unbind(character_name=self.chars.setter('character_name'))
            self.character_name = self.chars.character_name
            self.bind(character_name=self.chars.setter('character_name'))

        self.chars.push_character_name = chars_push_character_name

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
            engine=self.engine
        )
        self.bind(
            selected_proxy=self.statcfg.setter('proxy')
        )

        self.mainscreen = ELiDE.screen.MainScreen(
            use_kv=config['ELiDE']['user_kv'] == 'yes',
            play_speed=int(config['ELiDE']['play_speed']),
            boards={
                name: Board(
                    character=char
                ) for name, char in self.engine.character.items()
            }
        )
        if self.mainscreen.statlist:
            self.statcfg.statlist = self.mainscreen.statlist
        self.mainscreen.bind(statlist=self.statcfg.setter('statlist'))
        self.bind(
            selection=self.refresh_selected_proxy,
            character=self.refresh_selected_proxy
        )
        self.selected_proxy = self._get_selected_proxy()
        for wid in (
                self.mainscreen,
                self.pawncfg,
                self.spotcfg,
                self.statcfg,
                self.rules,
                self.charrules,
                self.chars,
                self.strings,
                self.funcs
        ):
            self.manager.add_widget(wid)

    def _set_language(self, lang):
        self.engine.string.language = lang

    def _get_selected_proxy(self):
        if self.selection is None:
            return self.character.stat
        elif hasattr(self.selection, 'proxy'):
            return self.selection.proxy
        elif (
                hasattr(self.selection, 'portal') and
                self.selection.portal is not None
        ):
            return self.selection.portal
        else:
            raise ValueError("Invalid selection: {}".format(self.selection))

    def refresh_selected_proxy(self, *args):
        self.selected_proxy = self._get_selected_proxy()

    def on_character_name(self, *args):
        if self.config['ELiDE']['boardchar'] != self.character_name:
            self.config['ELiDE']['boardchar'] = self.character_name

    def on_character(self, *args):
        if not hasattr(self, 'mainscreen'):
            Clock.schedule_once(self.on_character, 0)
            return
        if hasattr(self, '_oldchar'):
            self.mainscreen.boards[self._oldchar.name].unbind(selection=self.setter('selection'))
        self.selection = None
        self.mainscreen.boards[self.character.name].bind(selection=self.setter('selection'))

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.strings.save()
        self.funcs.save()
        self.config.write()

    def on_stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.strings.save()
        self.funcs.save()
        self.engine.commit()
        self.procman.shutdown()
        self.config.write()

    def delete_selection(self):
        """Delete both the selected widget and whatever it represents."""
        selection = self.selection
        if selection is None:
            return
        if isinstance(selection, ArrowWidget):
            self.mainscreen.boardview.board.rm_arrow(
                selection.origin.name,
                selection.destination.name
            )
            selection.portal.delete()
        elif isinstance(selection, Spot):
            self.mainscreen.boardview.board.rm_spot(selection.name)
            selection.proxy.delete()
        else:
            assert isinstance(selection, Pawn)
            self.mainscreen.boardview.board.rm_pawn(selection.name)
            selection.proxy.delete()
        self.selection = None

    def new_board(self, name):
        """Make a board for a character name, and switch to it."""
        char = self.engine.character[name]
        board = Board(character=char)
        self.mainscreen.boards[name] = board
        self.character = char
