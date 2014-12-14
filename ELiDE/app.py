# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.logger import Logger
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.resources import resource_add_path

import LiSE

import ELiDE
import ELiDE.layout

resource_add_path(ELiDE.__path__[0] + "/assets")


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    As it's a Kivy app, this implements the things required of the App
    class. I also keep \"globals\" here.

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
                'wallpaper': "wallpape.jpg",
                'boardchar': 'physical',
                'sheetchar': 'player',
                'debug': 'no'
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
        if config['ELiDE']['debug'] == 'yes':
            import pdb
            pdb.set_trace()

        self.engine = LiSE.Engine(
            config['LiSE']['world'],
            config['LiSE']['code']
        )
        for char in config['ELiDE']['boardchar'], config['ELiDE']['sheetchar']:
            if char not in self.engine.character:
                print("adding character: {}".format(char))
                self.engine.add_character(char)
        l = ELiDE.layout.ELiDELayout(app=self)
        from kivy.core.window import Window
        from kivy.modules import inspector
        inspector.create_inspector(Window, l)
        return l

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()

    def stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.engine.commit()
        self.engine.close()
        super().stop(*largs)


kv = """
<SymbolLabel@Label>:
    font_name: "Symbola.ttf"
    font_size: 50
<SymbolButton@Button>:
    font_name: "Symbola.ttf"
    font_size: 50
"""
Builder.load_string(kv)
