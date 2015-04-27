# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.logger import Logger
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.resources import resource_add_path

import LiSE
from LiSE.proxy import EngineProcessManager

import ELiDE
import ELiDE.layout

"""Object to configure, start, and stop ELiDE."""

resource_add_path(ELiDE.__path__[0] + "/assets")
resource_add_path(ELiDE.__path__[0] + "/assets/rltiles")


def proxylog(typ, data):
    if typ == 'command':
        (cmd, args) = data[1:]
        Logger.debug(
            "LiSE.proxy: calling {}{}".format(
                cmd,
                tuple(args)
            )
        )
    else:
        Logger.debug(
            "LiSE.proxy: returning {}".format(data)
        )


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    """
    engine = ObjectProperty()

    def build_config(self, config):
        """Set config defaults"""
        for sec in 'LiSE', 'ELiDE':
            config.adddefaultsection(sec)
        config.setdefaults(
            'LiSE',
            {
                'world': 'sqlite:///LiSEworld.db',
                'code': 'LiSEcode.db',
                'language': 'en'
            }
        )
        config.setdefaults(
            'ELiDE',
            {
                'boardchar': 'physical',
                'debugger': 'no',
                'inspector': 'no',
                'user_kv': 'yes',
                'user_message': 'yes',
                'play_speed': '1'
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
        self.manager = EngineProcessManager()
        self.engine = self.manager.start(
            config['LiSE']['world'],
            config['LiSE']['code'],
            logger=proxylog
        )

        Clock.schedule_interval(self._check_stats, 0.01)
        char = config['ELiDE']['boardchar']
        if char not in self.engine.character:
            print("adding character: {}".format(char))
            self.engine.add_character(char)
        l = ELiDE.layout.ELiDELayout(
            engine=self.engine,
            character_name=config['ELiDE']['boardchar'],
            use_kv=config['ELiDE']['user_kv'] == 'yes',
            use_message=config['ELiDE']['user_message'] == 'yes',
            play_speed=int(config['ELiDE']['play_speed'])
        )
        if config['ELiDE']['inspector'] == 'yes':
            from kivy.core.window import Window
            from kivy.modules import inspector
            inspector.create_inspector(Window, l)

        def upd_boardchar(*args):
            if config['ELiDE']['boardchar'] != l.character_name:
                config['ELiDE']['boardchar'] = l.character_name

        def upd_use_kv(*args):
            v = 'yes' if l.use_kv else 'no'
            if v != config['ELiDE']['user_kv']:
                config['ELiDE']['user_kv'] = v

        def upd_use_message(*args):
            v = 'yes' if l.use_message else 'no'
            if v != config['ELiDE']['use_message']:
                config['ELiDE']['user_message'] = v

        def upd_play_speed(*args):
            v = str(l.play_speed)
            if v != config['ELiDE']['play_speed']:
                config['ELiDE']['play_speed'] = v

        l.bind(
            character_name=upd_boardchar,
            use_kv=upd_use_kv,
            use_message=upd_use_message,
            play_speed=upd_play_speed
        )
        return l

    def _check_stats(self, *args):
        """Ask the engine to poll changes."""
        self.engine.poll_changes()

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()
        self.config.write()

    def on_stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        Clock.unschedule(self._check_stats)
        self.manager.shutdown()
        self.config.write()


kv = """
<SymbolLabel@Label>:
    font_name: "Symbola.ttf"
    font_size: 50
<SymbolButton@Button>:
    font_name: "Symbola.ttf"
    font_size: 50
"""
Builder.load_string(kv)
