# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""The database backend, with dictionaries of loaded objects.

This is a caching database connector. There are dictionaries for all
objects that can be loaded from the database.

This module does not contain the code used to generate
SQL. That's in util.py, the class SaveableMetaclass.

"""
from __future__ import print_function
import sqlite3
import re
import os
import igraph

from logging import getLogger
from dimension import Dimension
from spot import Spot
from pawn import Pawn
from board import Board
from card import Card
from style import LiSEStyle, LiSEColor
from util import (
    dictify_row,
    schemata,
    saveables,
    saveable_classes,
    Fabulator,
    Skeleton,
    Timestream,
    TimestreamException)
from portal import Portal
from thing import Thing
from character import Character
from charsheet import CharSheet
from menu import Menu
from event import Implicator


logger = getLogger(__name__)


def noop(*args, **kwargs):
    """Do nothing."""
    pass


class ListItemIterator:
    """Iterate over a list in a way that resembles dict.iteritems()"""
    def __init__(self, l):
        self.l = l
        self.l_iter = iter(l)
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.l)

    def __next__(self):
        it = next(self.l_iter)
        i = self.i
        self.i += 1
        return (i, it)


ONE_ARG_RE = re.compile("(.+)")
TWO_ARG_RE = re.compile("(.+), ?(.+)")
ITEM_ARG_RE = re.compile("(.+)\.(.+)")
MAKE_SPOT_ARG_RE = re.compile(
    "(.+)\."
    "(.+),([0-9]+),([0-9]+),?(.*)")
MAKE_PORTAL_ARG_RE = re.compile(
    "(.+)\.(.+)->"
    "(.+)\.(.+)")
MAKE_THING_ARG_RE = re.compile(
    "(.+)\.(.+)@(.+)")
PORTAL_NAME_RE = re.compile(
    "Portal\((.+)->(.+)\)")


class Closet(object):
    """This is where you should get all your LiSE objects from, generally.

A RumorMill is a database connector that can load and generate LiSE
objects. When loaded or generated, the object will be kept in the
RumorMill, even if nowhere else.

There are some special facilities here for the convenience of
particular LiSE objects: Things look up their location here; Items
(including Things) look up their contents here; and Effects look up
their functions here. That means you need to register functions here
when you want Effects to use them. Supply callback
functions for Effects in a list in the keyword argument "effect_cbs".

Supply boolean callback functions for Causes and the like in the
keyword argument "test_cbs".

You need to create a SQLite database file with the appropriate schema
before RumorMill will work. For that, run mkdb.sh.

    """
    working_dicts = [
        "boardhanddict",
        "calendardict",
        "colordict",
        "dimensiondict",
        "boarddict",
        "dimensiondict",
        "boarddict",
        "effectdict",
        "effectdeckdict",
        "imgdict",
        "texturedict",
        "menudict",
        "menuitemdict",
        "styledict",
        "tickdict",
        "eventdict",
        "characterdict"]

    @property
    def dimensions(self):
        return self.dimensiondict.itervalues()

    @property
    def characters(self):
        return self.characterdict.itervalues()

    @property
    def dimension(self):
        return self.skeleton["game"]["dimension"]

    @property
    def branch(self):
        return self.skeleton["game"]["branch"]

    @property
    def tick(self):
        return self.skeleton["game"]["tick"]

    @property
    def language(self):
        return self.skeleton["game"]["language"]

    def __setattr__(self, attrn, val):
        if attrn in ("dimension", "branch", "tick", "language"):
            self.skeleton["game"][attrn] = val
            if hasattr(self, 'USE_KIVY'):
                setattr(self.kivy_connector, attrn, val)
        else:
            super(Closet, self).__setattr__(attrn, val)

    def __init__(self, connector, USE_KIVY=False, **kwargs):
        """Return a database wrapper around the SQLite database file by the
given name.

        """
        self.connector = connector

        self.c = self.connector.cursor()

        # This dict is special. It contains all the game
        # data--represented only as those types which sqlite3 is
        # capable of storing. All my objects are ultimately just
        # views on this thing.
        self.skeleton = Skeleton()
        for saveable in saveables:
            for tabn in saveable[3]:
                self.skeleton[tabn] = {}
        self.c.execute(
            "SELECT language, seed, dimension, branch, tick FROM game")
        self.skeleton.update(
            {"game": dictify_row(
                self.c.fetchone(),
                ("language", "seed", "dimension", "branch", "tick"))})
        if "language" in kwargs:
            self.skeleton["game"]["language"] = kwargs["language"]
        # This is a copy of the skeleton as it existed at the time of
        # the last save. I'll be finding the differences between it
        # and the current skeleton in order to decide what to write to
        # disk.
        self.old_skeleton = self.skeleton.copy()

        if USE_KIVY:
            from kivybits import load_textures
            self.load_textures = lambda names: load_textures(
                self.c, self.skeleton, self.texturedict, names)
            from kivybits import KivyConnector
            self.kivy_connector = KivyConnector(
                language=self.language,
                dimension=self.dimension,
                branch=self.branch,
                tick=self.tick)
            self.USE_KIVY = True

        self.timestream = Timestream(self)

        for wd in self.working_dicts:
            setattr(self, wd, dict())

        self.game_speed = 1
        self.updating = False

        self.timestream = Timestream(self)
        self.time_travel_history = []

        placeholder = (noop, ITEM_ARG_RE)
        if "effect_cbs" in kwargs:
            effect_cb_fabdict = dict(
                [(
                    cls.__name__, self.constructorate(cls))
                 for cls in kwargs["effect_cbs"]])
        else:
            effect_cb_fabdict = {}
        self.get_effect_cb = Fabulator(effect_cb_fabdict)
        if "test_cbs" in kwargs:
            test_cb_fabdict = dict(
                [(
                    cls.__name__, self.constructorate(cls))
                 for cls in kwargs["test_cbs"]])
        else:
            test_cb_fabdict = {}
        self.get_test_cb = Fabulator(test_cb_fabdict)
        if "effect_cb_makers" in kwargs:
            effect_cb_maker_fabdict = dict(
                [(
                    maker.__name__, self.constructorate(maker))
                 for maker in kwargs["effect_makers"]])
        else:
            effect_cb_maker_fabdict = {}
        for (name, cb) in effect_cb_fabdict.iteritems():
            effect_cb_maker_fabdict[name] = lambda: cb
        self.make_effect_cb = Fabulator(effect_cb_maker_fabdict)
        if "test_cb_makers" in kwargs:
            test_cb_maker_fabdict = dict(
                [(
                    maker.__name__, self.constructorate(maker))
                 for maker in kwargs["test_cb_makers"]])
        else:
            test_cb_maker_fabdict = {}
        for (name, cb) in test_cb_fabdict.iteritems():
            test_cb_maker_fabdict[name] = lambda: cb
        self.make_test_cb = Fabulator(test_cb_maker_fabdict)
        self.menu_cbs = {
            'play_speed':
            (self.play_speed, ONE_ARG_RE),
            'back_to_start':
            (self.back_to_start, ''),
            'one': placeholder,
            'two': placeholder,
            'noop': placeholder,
            'toggle_menu':
            (self.toggle_menu, ONE_ARG_RE),
            'hide_menu':
            (self.hide_menu, ONE_ARG_RE),
            'show_menu':
            (self.show_menu, ONE_ARG_RE),
            'make_generic_place':
            (self.make_generic_place, ''),
            'increment_branch':
            (self.increment_branch, ONE_ARG_RE),
            'time_travel_inc_tick':
            (lambda mi, ticks:
             self.time_travel_inc_tick(int(ticks)), ONE_ARG_RE),
            'time_travel':
            (self.time_travel_menu_item, TWO_ARG_RE),
            'time_travel_inc_branch':
            (lambda mi, branches: self.time_travel_inc_branch(int(branches)),
             ONE_ARG_RE),
            'go':
            (self.go, ""),
            'stop':
            (self.stop, ""),
            'start_new_map': placeholder,
            'open_map': placeholder,
            'save_map': placeholder,
            'quit_map_editor': placeholder,
            'editor_select': placeholder,
            'editor_copy': placeholder,
            'editor_paste': placeholder,
            'editor_delete': placeholder,
            'mi_create_place':
            (self.mi_create_place, ONE_ARG_RE),
            'mi_create_thing':
            (self.mi_create_thing, ONE_ARG_RE),
            'mi_create_portal':
            (self.mi_create_portal, ONE_ARG_RE)}

    def __del__(self):
        """Try to write changes to disk before dying.

        """
        self.c.close()
        self.connector.commit()
        self.connector.close()

    def constructorate(self, cls):

        def construct(*args):
            return cls(self, *args)
        return construct

    def insert_rowdicts_table(self, rowdict, clas, tablename):
        """Insert the given rowdicts into the table of the given name, as
defined by the given class.

For more information, consult SaveableMetaclass in util.py.

        """
        if rowdict != []:
            clas.dbop['insert'](self, rowdict, tablename)

    def delete_keydicts_table(self, keydict, clas, tablename):
        """Delete the records identified by the keydicts from the given table,
as defined by the given class.

For more information, consult SaveableMetaclass in util.py.

        """
        if keydict != []:
            clas.dbop['delete'](self, keydict, tablename)

    def detect_keydicts_table(self, keydict, clas, tablename):
        """Return the rows in the given table, as defined by the given class,
matching the given keydicts.

For more information, consult SaveableMetaclass in util.py.

        """
        if keydict != []:
            return clas.dbop['detect'](self, keydict, tablename)
        else:
            return []

    def missing_keydicts_table(self, keydict, clas, tablename):
        """Return rows in the given table, as defined by the given class,
*not* matching any of the given keydicts.

For more information, consult SaveableMetaclass in util.py.

        """
        if keydict != []:
            return clas.dbop['missing'](self, keydict, tablename)
        else:
            return []

    def toggle_menu(self, menuitem, menuname,
                    effect=None, deck=None, event=None):
        window = menuitem.menu.window
        menu = window.menus_by_name[menuname]
        menu.visible = not menu.visible
        menu.tweaks += 1

    def hide_menu(self, menuitem, menuname,
                  effect=None, deck=None, event=None):
        window = menuitem.menu.window
        menu = window.menus_by_name[menuname]
        menu.visible = False
        menu.tweaks += 1

    def show_menu(self, menuitem, menuname,
                  effect=None, deck=None, event=None):
        window = menuitem.menu.window
        menu = window.menus_by_name[menuname]
        menu.visible = True
        menu.tweaks += 1

    def get_text(self, strname):
        """Get the string of the given name in the language set at startup."""
        if strname is None:
            return ""
        elif strname[0] == "@":
            if strname[1:] == "branch":
                return str(self.branch)
            elif strname[1:] == "tick":
                return str(self.tick)
            else:
                assert(strname[1:] in self.skeleton["strings"])
                return self.skeleton["strings"][
                    strname[1:]][self.language]["string"]
        else:
            return strname

    def mi_create_place(self, menuitem):
        return menuitem.window.create_place()

    def mi_create_thing(self, menuitem):
        return menuitem.window.create_thing()

    def mi_create_portal(self, menuitem):
        return menuitem.window.create_portal()

    def make_igraph_graph(self, name):
        self.graphdict[name] = igraph.Graph(directed=True)

    def get_igraph_graph(self, name):
        if name not in self.graphdict:
            self.make_igraph_graph(name)
        return self.graphdict[name]

    def save_game(self):
        to_save = self.skeleton - self.old_skeleton
        to_delete = self.old_skeleton - self.skeleton
        logger.debug(
            "Saving the skeleton:\n%s", repr(to_save))
        logger.debug(
            "Deleting the skeleton:\n%s", repr(to_delete))
        for clas in saveable_classes:
            for tabname in clas.tablenames:
                if tabname in to_delete:
                    clas._delete_keydicts_table(
                        self.c, to_delete[tabname], tabname)
                if tabname in to_save:
                    clas._delete_keydicts_table(
                        self.c, to_save[tabname], tabname)
                    clas._insert_rowdicts_table(
                        self.c, to_save[tabname], tabname)
        self.c.execute("DELETE FROM game")
        keys = self.skeleton["game"].keys()
        self.c.execute(
            "INSERT INTO game ({0}) VALUES ({1})".format(
                ", ".join(keys),
                ", ".join(["?"] * len(self.skeleton["game"]))),
            [self.skeleton["game"][k] for k in keys])
        self.old_skeleton = self.skeleton.copy()

    def load_strings(self):
        self.c.execute("SELECT stringname, language, string FROM strings")
        if "strings" not in self.skeleton:
            self.skeleton["strings"] = {}
        for row in self.c:
            rowd = dictify_row(row, ("stringname", "language", "string"))
            if rowd["stringname"] not in self.skeleton["strings"]:
                self.skeleton["strings"][rowd["stringname"]] = {}
            self.skeleton["strings"][
                rowd["stringname"]][rowd["language"]] = rowd

    def make_generic_place(self, dimension):
        placen = "generic_place_{0}".format(len(dimension.graph.vs))
        dimension.make_place(placen)
        return dimension.get_place(placen)

    def make_generic_thing(self, dimension, location):
        thingn = "generic_thing_{0}".format(len(dimension.thingdict))
        dimension.make_thing(thingn)
        th = dimension.get_thing(thingn)
        th.set_location(location)
        return th

    def make_spot(self, board, place, x, y):
        spot = Spot(board, place)
        if not hasattr(place, 'spots'):
            place.spots = []
        while len(place.spots) <= int(board):
            place.spots.append(None)
        place.spots[int(board)] = spot
        spot.set_img(self.imgdict['default_spot'])
        spot.set_coords(x, y)

    def make_portal(self, orig, dest):
        return orig.dimension.make_portal(orig, dest)

    def load_charsheet(self, character):
        character = str(character)
        kd = {
            "charsheet": {
                "character": character},
            "charsheet_item": {
                "character": character}}
        self.skeleton.update(
            CharSheet._select_skeleton(self.c, kd))
        return CharSheet(character=self.get_character(character))

    def load_characters(self, names):
        qtd = {
            "character_things": {},
            "character_places": {},
            "character_portals": {},
            "character_stats": {},
            "character_skills": {}}
        for name in names:
            for tabn in qtd.keys():
                qtd[tabn][name] = {"character": name}
        self.skeleton.update(
            Character._select_skeleton(self.c, qtd))
        r = {}
        for name in names:
            char = Character(self, name)
            r[name] = char
            self.characterdict[name] = char
        return r

    def get_characters(self, names):
        r = {}
        unhad = set()
        for name in names:
            if isinstance(name, Character):
                r[str(name)] = name
            elif name in self.characterdict:
                r[name] = self.characterdict[name]
            else:
                unhad.add(name)
        if len(unhad) > 0:
            r.update(self.load_characters(names))
        return r

    def get_character(self, name):
        return self.get_characters([str(name)])[str(name)]

    def get_thing(self, dimn, thingn):
        return self.get_dimension(dimn).get_thing(thingn)

    def get_effects(self, names):
        r = {}
        for name in names:
            r[name] = Implicator.make_effect(name)
        return r

    def get_effect(self, name):
        return self.get_effects([name])[name]

    def get_causes(self, names):
        r = {}
        for name in names:
            r[name] = Implicator.make_cause(name)
        return r

    def get_cause(self, cause):
        return self.get_causes([cause])[cause]

    def load_dimensions(self, names):
        # I think it might eventually *make sense* to load the same
        # dimension more than once without unloading it first. Perhaps
        # you want to selectively load the parts of it that the player
        # is interested in at the moment, the game world being too
        # large to practically load all at once.
        kd = {"portal": {},
              "thing_location": {}}
        for name in names:
            kd["portal"][name] = {"dimension": name}
            kd["thing_location"][name] = {"dimension": name}
        dimtd = Portal._select_skeleton(self.c, kd)
        dimtd.update(Thing._select_skeleton(self.c, kd))
        self.skeleton.update(dimtd)
        r = {}
        for name in names:
            r[name] = Dimension(self, name)
        return r

    def load_dimension(self, name):
        return self.load_dimensions([name])[name]

    def get_dimensions(self, names=None):
        if names is None:
            self.c.execute("SELECT name FROM dimension")
            names = [row[0] for row in self.c.fetchall()]
        r = {}
        unhad = set()
        for name in names:
            if name in self.dimensiondict:
                r[name] = self.dimensiondict[name]
            else:
                unhad.add(name)
        if len(unhad) > 0:
            r.update(self.load_dimensions(unhad))
        return r

    def get_dimension(self, name):
        return self.get_dimensions([name])[name]

    def load_board(self, name):
        self.skeleton.update(Board._select_skeleton(self.c, {
            "board": {"dimension": name}}))
        self.skeleton.update(Spot._select_skeleton(self.c, {
            "spot_img": {"dimension": name},
            "spot_interactive": {"dimension": name},
            "spot_coords": {"dimension": name}}))
        self.skeleton.update(Pawn._select_skeleton(self.c, {
            "pawn_img": {"dimension": name},
            "pawn_interactive": {"dimension": name}}))
        return self.get_board(name)

    def get_board(self, name):
        dim = self.get_dimension(name)
        return Board(closet=self, dimension=dim)

    def get_place(self, dim, placen):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        return dim.get_place(placen)

    def get_portal(self, dim, origin, destination):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        return dim.get_portal(str(origin), str(destination))

    def get_textures(self, imgnames):
        r = {}
        unloaded = set()
        for imgn in imgnames:
            if imgn in self.texturedict:
                r[imgn] = self.texturedict[imgn]
            else:
                unloaded.add(imgn)
        if len(unloaded) > 0:
            r.update(self.load_textures(unloaded))
        return r

    def get_texture(self, imgn):
        return self.get_textures([imgn])[imgn]

    def load_colors(self, names):
        kd = {"color": {}}
        for name in names:
            kd["color"][name] = {"name": name}
        self.skeleton.update(
            LiSEColor._select_skeleton(self.c, kd))
        r = {}
        for name in names:
            r[name] = LiSEColor(self, name)
            self.colordict[name] = r[name]
        return r

    def get_colors(self, colornames):
        r = {}
        unloaded = set()
        for color in colornames:
            if color in self.colordict:
                r[color] = self.colordict[color]
            else:
                unloaded.add(color)
        if len(unloaded) > 0:
            r.update(self.load_colors(unloaded))
        return r

    def get_color(self, name):
        return self.get_colors([name])[name]

    def load_styles(self, stylenames):
        kd = {"style": {}}
        for name in stylenames:
            kd["style"][name] = {"name": name}
        self.skeleton.update(
            LiSEStyle._select_skeleton(self.c, kd))
        colornames = set()
        colorcols = set([
            'bg_inactive', 'bg_active', 'fg_inactive', 'fg_active',
            'textcolor'])
        for name in stylenames:
            rd = self.skeleton["style"][name]
            for colorcol in colorcols:
                colornames.add(rd[colorcol])
        self.get_colors(colornames)
        r = {}
        for name in stylenames:
            r[name] = LiSEStyle(self, name)
        return r

    def get_styles(self, stylenames):
        r = {}
        unloaded = set()
        for style in stylenames:
            if style in self.styledict:
                r[style] = self.styledict[style]
            else:
                unloaded.add(style)
        if len(unloaded) > 0:
            r.update(self.load_styles(unloaded))
        return r

    def get_style(self, name):
        if isinstance(name, LiSEStyle):
            return name
        return self.get_styles([name])[name]

    def load_cards(self, names):
        effectdict = self.get_effects(names)
        kd = {"card": {}}
        for name in names:
            kd["card"][name] = {"effect": name}
        td = Card._select_skeleton(self.c, kd)
        r = {}
        for rd in td.iterrows():
            r[rd["effect"]] = Card(self, effectdict[rd["effect"]], td)
        return r

    def get_cards(self, names):
        r = {}
        unhad = set()
        for name in names:
            if name in self.carddict:
                r[name] = self.carddict[name]
            else:
                unhad.add(name)
        if len(unhad) > 0:
            r.update(self.load_cards(unhad))
        return r

    def get_card(self, name):
        return self.get_cards([name])[name]

    def load_menus(self, names):
        kd = {"menu": {}}
        for name in names:
            kd["menu"][name] = {"name": name}
        skel = Menu._select_skeleton(self.c, kd)
        self.skeleton.update(skel)
        r = {}
        for rd in skel.iterrows():
            self.load_menu_items(rd["name"])
            r[rd["name"]] = Menu(closet=self, name=rd["name"])
        return r

    def load_menu(self, name):
        return self.load_menus([name])[name]

    def load_menu_items(self, menu):
        kd = {"menu_item": {"menu": menu}}
        skel = Menu._select_skeleton(self.c, kd)
        self.skeleton.update(skel)

    def load_timestream(self):
        self.skeleton.update(
            Timestream._select_table_all(self.c, 'timestream'))
        self.timestream = Timestream(self)

    def time_travel_menu_item(self, mi, branch, tick):
        return self.time_travel(branch, tick)

    def time_travel(self, branch, tick):
        if branch > self.timestream.hi_branch + 1:
            raise TimestreamException("Tried to travel too high a branch")
        if branch == self.timestream.hi_branch + 1:
            self.new_branch(self.branch, branch, tick)
        # will need to take other games-stuff into account than the
        # thing_location
        mintick = self.timestream.min_tick(branch, "thing_location")
        if tick < mintick:
            tick = mintick
        self.time_travel_history.append((self.branch, self.tick))
        if tick > self.timestream.hi_tick:
            self.timestream.hi_tick = tick
            if hasattr(self, 'USE_KIVY'):
                self.kivy_connector.hi_tick = tick
        self.branch = branch
        self.tick = tick

    def increment_branch(self, branches=1):
        b = self.branch + int(branches)
        mb = self.timestream.max_branch()
        if b > mb:
            # I dunno where you THOUGHT you were going
            self.new_branch(self.branch, self.branch+1, self.tick)
            return self.branch + 1
        else:
            return b

    def new_branch(self, parent, branch, tick):
        for dimension in self.dimensiondict.itervalues():
            dimension.new_branch(parent, branch, tick)
        for board in self.boarddict.itervalues():
            board.new_branch(parent, branch, tick)
        for character in self.characterdict.itervalues():
            character.new_branch(parent, branch, tick)
        self.skeleton["timestream"][branch] = {
            "branch": branch,
            "parent": parent}

    def time_travel_inc_tick(self, ticks=1):
        self.time_travel(self.branch, self.tick+ticks)

    def time_travel_inc_branch(self, branches=1):
        self.time_travel(self.branch+branches, self.tick)

    def go(self, nope=None):
        self.updating = True

    def stop(self, nope=None):
        self.updating = False

    def set_speed(self, newspeed):
        self.game_speed = newspeed

    def play_speed(self, mi, gamespeed):
        self.set_speed(int(gamespeed))
        self.go()

    def back_to_start(self, nope):
        self.stop()
        self.time_travel(self.branch, 0)

    def update(self, *args):
        if self.updating:
            self.time_travel_inc_tick(ticks=self.game_speed)

    def end_game(self):
        self.c.close()
        self.connector.commit()
        self.connector.close()

    def checkpoint(self):
        self.old_skeleton = self.skeleton.copy()

    def uptick_rd(self, rd):
        if "branch" in rd and rd["branch"] > self.timestream.hi_branch:
            self.timestream.hi_branch = rd["branch"]
        if "tick_from" in rd and rd["tick_from"] > self.timestream.hi_tick:
            self.timestream.hi_tick = rd["tick_from"]
            if hasattr(self, 'USE_KIVY'):
                self.kivy_connector.hi_tick = rd["tick_from"]
        if "tick_to" in rd and rd["tick_to"] > self.timestream.hi_tick:
            self.timestream.hi_tick = rd["tick_to"]
            if hasattr(self, 'USE_KIVY'):
                self.kivy_connector.hi_tick = rd["tick_to"]

    def uptick_skel(self):
        for rd in self.skeleton.iterrows():
            self.uptick_rd(rd)


def mkdb(DB_NAME='default.sqlite'):
    def isdir(p):
        try:
            os.chdir(p)
            return True
        except:
            return False

    def allsubdirs_core(doing, done):
        if len(doing) == 0:
            return done
        here = doing.pop()
        if isdir(here):
            done.add(here + '/')
            inside = (
                [here + '/' + there for there in
                 os.listdir(here) if there[0] != '.'])
            doing.update(set(inside))

    def allsubdirs(path):
        inpath = os.path.realpath(path)
        indoing = set()
        indoing.add(inpath)
        indone = set()
        result = None
        while result is None:
            result = allsubdirs_core(indoing, indone)
        return iter(result)

    def ins_rltiles(curs, dirname):
        here = os.getcwd()
        directories = os.path.abspath(dirname).split("/")
        home = "/".join(directories[:-1]) + "/"
        dirs = allsubdirs(dirname)
        for dir in dirs:
            for bmp in os.listdir(dir):
                if bmp[-4:] != ".bmp":
                    continue
                qrystr = """insert or replace into img
    (name, path, rltile) values (?, ?, ?)"""
                bmpr = bmp.replace('.bmp', '')
                dirr = dir.replace(home, '') + bmp
                curs.execute(qrystr, (bmpr, dirr, True))
        os.chdir(here)

    try:
        os.remove(DB_NAME)
    except OSError:
        pass
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    def read_sql(filen):
        sqlfile = open(filen, "r")
        sql = sqlfile.read()
        sqlfile.close()
        c.executescript(sql)

    c.execute(
        "CREATE TABLE game"
        " (language TEXT DEFAULT 'eng',"
        "dimension TEXT DEFAULT 'Physical', "
        "branch INTEGER DEFAULT 0, "
        "tick INTEGER DEFAULT 0,"
        " seed INTEGER DEFAULT 0);")
    c.execute(
        "CREATE TABLE strings (stringname TEXT NOT NULL, language TEXT NOT"
        " NULL DEFAULT 'eng', string TEXT NOT NULL, "
        "PRIMARY KEY(stringname,  language));")

    done = set()
    while saveables != []:
        (demands, provides, prelude,
         tablenames, postlude) = saveables.pop(0)
        print(tablenames)
        breakout = False
        for demand in iter(demands):
            if demand not in done:
                saveables.append(
                    (demands, provides, prelude,
                     tablenames, postlude))
                breakout = True
                break
        if breakout:
            continue
        while prelude != []:
            pre = prelude.pop()
            if isinstance(pre, tuple):
                c.execute(*pre)
            else:
                c.execute(pre)
        if tablenames == []:
            while postlude != []:
                post = postlude.pop()
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
            continue
        try:
            while prelude != []:
                pre = prelude.pop()
                if isinstance(pre, tuple):
                    c.execute(*pre)
                else:
                    c.execute(pre)
        except sqlite3.OperationalError as e:
            saveables.append(
                (demands, provides, prelude, tablenames, postlude))
            continue
        breakout = False
        while tablenames != []:
            tn = tablenames.pop(0)
            try:
                c.execute(schemata[tn])
                done.add(tn)
            except sqlite3.OperationalError as e:
                print("OperationalError while creating table {0}:".format(tn))
                print(e)
                breakout = True
                break
        if breakout:
            saveables.append(
                (demands, provides, prelude, tablenames, postlude))
            continue
        try:
            while postlude != []:
                post = postlude.pop()
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
        except sqlite3.OperationalError as e:
            print("OperationalError during postlude from {0}:".format(tn))
            print(e)
            saveables.append(
                (demands, provides, prelude, tablenames, postlude))
            continue
        done.update(provides)

    oldhome = os.getcwd()
    os.chdir('sql')
    initfiles = sorted(os.listdir('.'))
    for initfile in initfiles:
        if initfile[-3:] == "sql":  # weed out automatic backups and so forth
            print("reading SQL from file " + initfile)
            read_sql(initfile)

    os.chdir(oldhome)

    print("indexing the RLTiles")
    ins_rltiles(c, 'rltiles')

    c.close()
    conn.commit()


def load_closet(dbfn, lang="eng", kivy=False):
    conn = sqlite3.connect(dbfn)
    r = Closet(connector=conn, lang=lang, USE_KIVY=kivy)
    r.load_strings()
    r.load_timestream()
    return r
