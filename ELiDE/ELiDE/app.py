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
from LiSE.proxy import EngineProcessManager, CharStatProxy
import ELiDE
import ELiDE.dialog
import ELiDE.screen
import ELiDE.stores
import ELiDE.statcfg
import ELiDE.spritebuilder
import ELiDE.rulesview
import ELiDE.charsview
from ELiDE.graph.board import GraphBoard
from ELiDE.graph.arrow import GraphArrowWidget
from ELiDE.graph.spot import GraphSpot
from ELiDE.graph.pawn import Pawn
from ELiDE.grid.board import GridBoard
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
    selected_proxy_name = StringProperty('')
    statcfg = ObjectProperty()

    def on_selection(self, *args):
        Logger.debug("App: {} selected".format(self.selection))

    def on_selected_proxy(self, *args):
        if hasattr(self.selected_proxy, 'name'):
            self.selected_proxy_name = str(self.selected_proxy.name)
            return
        selected_proxy = self.selected_proxy
        assert hasattr(selected_proxy, 'origin'), '{} has no origin'.format(type(selected_proxy))
        assert hasattr(selected_proxy, 'destination'), '{} has no destination'.format(type(selected_proxy))
        origin = selected_proxy.origin
        destination = selected_proxy.destination
        reciprocal = selected_proxy.reciprocal
        if selected_proxy.get('is_mirror', False) or (reciprocal and reciprocal.get('is_mirror', False)):
            link = '<>'
        else:
            link = '->'
        self.selected_proxy_name = str(origin.name) + link + str(destination.name)

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
        branch, turn, tick = self.engine._btt()
        self.branch = branch
        self.turn = turn
        self.tick = tick
    pull_time = trigger(_pull_time)

    @trigger
    def _push_time(self, *args):
        branch, turn, tick = self.engine._btt()
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
        """Set the turn to the given value, cast to an integer"""
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
                'world': 'sqlite:///world.db',
                'language': 'eng',
                'logfile': '',
                'loglevel': 'info'
            }
        )
        config.setdefaults(
            'ELiDE',
            {
                'debugger': 'no',
                'inspector': 'no',
                'user_kv': 'yes',
                'play_speed': '1',
                'thing_graphics': json.dumps([
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
                    ('RLTiles: Dungeon', 'dungeon.atlas'),
                    ('RLTiles: Floor', 'floor.atlas')
                ])
            }
        )
        config.write()

    def build(self):
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
        
        self._add_screens()
        return self.manager

    def _pull_lang(self, *args, **kwargs):
        self.strings.language = kwargs['language']

    def _pull_chars(self, *args, **kwargs):
        self.chars.names = list(self.engine.character)

    def _pull_time_from_signal(self, *args, branch, turn, tick):
        self.branch, self.turn, self.tick = branch, turn, tick

    def start_subprocess(self, *args):
        """Start the LiSE core and get a proxy to it

        Must be called before ``init_board``

        """
        if hasattr(self, '_started'):
            raise ChildProcessError("Subprocess already running")
        config = self.config
        enkw = {'logger': Logger}
        if config['LiSE'].get('logfile'):
            enkw['logfile'] = config['LiSE']['logfile']
        if config['LiSE'].get('loglevel'):
            enkw['loglevel'] = config['LiSE']['loglevel']
        self.procman = EngineProcessManager()
        self.engine = engine = self.procman.start(**enkw)
        self.pull_time()

        self.engine.time.connect(self._pull_time_from_signal, weak=False)
        self.engine.string.language.connect(self._pull_lang, weak=False)
        self.engine.character.connect(self._pull_chars, weak=False)

        self.bind(
            branch=self._push_time,
            turn=self._push_time,
            tick=self._push_time
        )

        self.strings.store = self.engine.string
        self._started = True
        return engine
    trigger_start_subprocess = trigger(start_subprocess)

    def init_board(self, *args):
        """Get the board widgets initialized to display the game state

        Must be called after start_subprocess

        """
        if 'boardchar' not in self.engine.eternal:
            if 'physical' in self.engine.character:
                self.engine.eternal['boardchar'] = self.engine.character['physical']
            else:
                chara = self.engine.eternal['boardchar'] = self.engine.new_character('physical')
        self.chars.names = list(self.engine.character)
        self.mainscreen.graphboards = {
                name: GraphBoard(
                    character=char
                ) for name, char in self.engine.character.items()
            }
        self.mainscreen.gridboards = {
                name: GridBoard(character=char)
                for name, char in self.engine.character.items()
            }
        self.select_character(self.engine.eternal['boardchar'])
        self.selected_proxy = self._get_selected_proxy()

    def _add_screens(self, *args):
        def toggler(screenname):
            def tog(*args):
                if self.manager.current == screenname:
                    self.manager.current = 'main'
                else:
                    self.manager.current = screenname
            return tog
        config = self.config

        self.mainmenu = ELiDE.menu.DirPicker(
            toggle=toggler('mainmenu'),
            start=self.start_subprocess
        )

        self.pawncfg = ELiDE.spritebuilder.PawnConfigScreen(
            toggle=toggler('pawncfg'),
            data=json.loads(config['ELiDE']['thing_graphics'])
        )

        self.spotcfg = ELiDE.spritebuilder.SpotConfigScreen(
            toggle=toggler('spotcfg'),
            data=json.loads(config['ELiDE']['place_graphics'])
        )

        self.statcfg = ELiDE.statcfg.StatScreen(
            toggle=toggler('statcfg'),
            engine=self.engine
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
            new_board=self.new_board
        )
        self.bind(character_name=self.chars.setter('character_name'))

        def chars_push_character_name(*args):
            self.unbind(character_name=self.chars.setter('character_name'))
            self.character_name = self.chars.character_name
            self.bind(character_name=self.chars.setter('character_name'))

        self.chars.push_character_name = chars_push_character_name

        self.strings = ELiDE.stores.StringsEdScreen(
            toggle=toggler('strings')
        )

        self.funcs = ELiDE.stores.FuncsEdScreen(
            name='funcs',
            toggle=toggler('funcs')
        )

        self.bind(
            selected_proxy=self.statcfg.setter('proxy')
        )

        self.mainscreen = ELiDE.screen.MainScreen(
            use_kv=config['ELiDE']['user_kv'] == 'yes',
            play_speed=int(config['ELiDE']['play_speed'])
        )
        if self.mainscreen.statlist:
            self.statcfg.statlist = self.mainscreen.statlist
        self.mainscreen.bind(statlist=self.statcfg.setter('statlist'))
        self.bind(
            selection=self.refresh_selected_proxy,
            character=self.refresh_selected_proxy
        )
        for wid in (
                self.mainmenu,
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
        self.manager.current = 'mainmenu'

    def update_calendar(self, calendar, past_turns=1, future_turns=5):
        """Fill in a calendar widget with actual simulation data"""
        startturn = self.turn - past_turns
        endturn = self.turn + future_turns
        stats = ['_config'] + [
            stat for stat in self.selected_proxy if not stat.startswith('_')
            and stat not in ('character', 'name')
        ]
        if isinstance(self.selected_proxy, CharStatProxy):
            sched_entity = self.engine.character[self.selected_proxy.name]
        else:
            sched_entity = self.selected_proxy
        calendar.entity = sched_entity
        calendar.from_schedule(
            self.engine.handle(
                'get_schedule', entity=sched_entity,
                stats=stats, beginning=startturn, end=endturn
            ),
            start_turn=startturn
        )

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
        if not self.engine:
            Clock.schedule_once(self.on_character_name, 0)
            return
        self.engine.eternal['boardchar'] = self.engine.character[self.character_name]

    def on_character(self, *args):
        if not hasattr(self, 'mainscreen'):
            Clock.schedule_once(self.on_character, 0)
            return
        if hasattr(self, '_oldchar'):
            self.mainscreen.graphboards[self._oldchar.name].unbind(selection=self.setter('selection'))
            self.mainscreen.gridboards[self._oldchar.name].unbind(selection=self.setter('selection'))
        self.selection = None
        self.mainscreen.graphboards[self.character.name].bind(selection=self.setter('selection'))
        self.mainscreen.gridboards[self.character.name].bind(selection=self.setter('selection'))

    def on_pause(self):
        """Sync the database with the current state of the game."""
        if hasattr(self, 'engine'):
            self.engine.commit()
        self.strings.save()
        self.funcs.save()
        self.config.write()

    def on_stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.strings.save()
        self.funcs.save()
        if self.engine:
            self.engine.commit()
        if hasattr(self, 'procman'):
            self.procman.shutdown()
        self.config.write()

    def delete_selection(self):
        """Delete both the selected widget and whatever it represents."""
        selection = self.selection
        if selection is None:
            return
        if isinstance(selection, GraphArrowWidget):
            if selection.reciprocal and selection.reciprocal.portal.get('is_mirror', False):
                selection.reciprocal.portal.delete()
                self.mainscreen.boardview.board.rm_arrow(
                    selection.destination.name,
                    selection.origin.name
                )
            self.mainscreen.boardview.board.rm_arrow(
                selection.origin.name,
                selection.destination.name
            )
            selection.portal.delete()
        elif isinstance(selection, GraphSpot):
            charn = selection.board.character.name
            self.mainscreen.graphboards[charn].rm_spot(selection.name)
            gridb = self.mainscreen.gridboards[charn]
            if selection.name in gridb.spot:
                gridb.rm_spot(selection.name)
            selection.proxy.delete()
        else:
            assert isinstance(selection, Pawn)
            charn = selection.board.character.name
            self.mainscreen.graphboards[charn].rm_pawn(selection.name)
            self.mainscreen.gridboards[charn].rm_pawn(selection.name)
            selection.proxy.delete()
        self.selection = None

    def new_board(self, name):
        """Make a graph for a character name, and switch to it."""
        char = self.engine.character[name]
        self.mainscreen.graphboards[name] = GraphBoard(character=char)
        self.mainscreen.gridboards[name] = GridBoard(character=char)
        self.character = char
