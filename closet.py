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
from effect import Effect, EffectDeck
from style import LiSEStyle, LiSEColor
from util import (
    dictify_row,
    Skeleton,
    empty_skel,
    schemata,
    saveables,
    saveable_classes,
    Timestream,
    TimestreamException)
from portal import Portal
from thing import Thing
from character import Character
from charsheet import CharSheet
from img import LiSEImage
from kivy.uix.image import Image
from kivy.core.image import ImageData
from kivy.properties import NumericProperty
from kivy.event import EventDispatcher
from menu import Menu


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


class Closet(EventDispatcher):
    """This is where you should get all your LiSE objects from, generally.

A RumorMill is a database connector that can load and generate LiSE
objects. When loaded or generated, the object will be kept in the
RumorMill, even if nowhere else.

There are some special facilities here for the convenience of
particular LiSE objects: Things look up their location here; Items
(including Things) look up their contents here; and Effects look up
their functions here. That means you need to register functions here
when you want Effects to be able to trigger them. To do that, put your
functions in a dictionary keyed with the strings LiSE should use to
refer to your functions, and pass that dictionary to the RumorMill
method xfunc(). You may also pass such a dictionary to the
constructor, just after the name of the database file.

You need to create a SQLite database file with the appropriate schema
before RumorMill will work. For that, run mkdb.sh.

    """

    atrdic = {
        "game": lambda self: self.skeleton["game"],
        "seed": lambda self: self.game["seed"],
        "dimensions": lambda self: iter(self.dimensiondict.values()),
        "characters": lambda self: iter(self.characterdict.values())}
    branch = NumericProperty()
    tick = NumericProperty()

    def __init__(self, connector, xfuncs={}, lang="eng",
                 front_dimension="Physical", front_board=0,
                 front_branch=0, seed=0, tick=0, **kwargs):
        """Return a database wrapper around the SQLite database file by the
given name.

        """
        EventDispatcher.__init__(self, **kwargs)
        self.bind(branch=self.upd_branch)
        self.bind(tick=self.upd_tick)
        self.branch = front_branch
        self.tick = tick
        self.language = lang
        self.conn = connector
        self.cursor = self.conn.cursor()
        self.c = self.cursor

        # This dict is special. It contains all the game
        # data--represented only as those types which sqlite3 is
        # capable of storing. All my objects are ultimately just
        # views on this thing.
        self.skeleton = empty_skel()
        # This is a copy of the skeleton as it existed at the time of
        # the last save. I'll be finding the differences between it
        # and the current skeleton in order to decide what to write to
        # disk.
        self.old_skeleton = empty_skel()

        self.windowdict = {}
        self.boardhanddict = {}
        self.calendardict = {}
        self.carddict = {}
        self.colordict = {}
        self.dimensiondict = {}
        self.effectdict = {}
        self.effectdeckdict = {}
        self.imgdict = {}
        self.texturedict = {}
        self.menudict = {}
        self.menuitemdict = {}
        self.styledict = {}
        self.tickdict = {}
        self.eventdict = {}
        self.characterdict = {}
        self.windowdict = {}
        self.lang = lang

        self.game_speed = 1
        self.updating = False

        self.time_travel_history = []

        placeholder = (noop, ITEM_ARG_RE)
        self.effect_cbs = {}
        self.menu_cbs = {
            'play_speed':
            (self.play_speed, ONE_ARG_RE),
            'back_to_start':
            (self.back_to_start, ""),
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
            (self.make_generic_place, ONE_ARG_RE),
            'increment_branch':
            (self.increment_branch, ONE_ARG_RE),
            'time_travel_inc_tick':
            (lambda mi, ticks: self.time_travel_inc_tick(ticks), ONE_ARG_RE),
            'time_travel':
            (self.time_travel_menu_item, TWO_ARG_RE),
            'time_travel_inc_branch':
            (lambda mi, branches: self.time_travel_inc_branch(branches),
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
        self.menu_cbs.update(xfuncs)
        self.skeleton["game"] = {
            "front_dimension": front_dimension,
            "front_branch": front_branch,
            "seed": seed,
            "tick": tick}

    def __getattr__(self, attrn):
        try:
            return Closet.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Closet doesn't have the attribute " + attrn)

    def __setattr__(self, attrn, val):
        if attrn in ("seed", "age"):
            getattr(self, "game")[attrn] = val
        else:
            super(Closet, self).__setattr__(attrn, val)

    def __del__(self):
        """Try to write changes to disk before dying.

        """
        self.c.close()
        self.conn.commit()
        self.conn.close()

    def upd_branch(self, *args):
        self.skeleton["game"]["front_branch"] = self.branch

    def upd_tick(self, *args):
        self.skeleton["game"]["tick"] = self.tick

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

    def get_age(self):
        """Get the number of ticks since the start of the game. Persists
between sessions.

This is game-world time. It doesn't always go forwards.

        """
        return self.game["age"]

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

    def get_card_base(self, name):
        """Return the CardBase named thus, loading it first if necessary."""
        if name not in self.carddict:
            self.load_card(name)
        return self.carddict[name]

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
        keys = list(self.game.keys())
        self.c.execute(
            "INSERT INTO game ({0}) VALUES ({1})".format(
                ", ".join(keys),
                ", ".join(["?"] * len(self.game))),
            tuple([self.game[k] for k in keys]))
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
        skel = Skeleton({
            "charsheet": {
                "character": character},
            "charsheet_item": {
                "character": character}})
        self.skeleton.update(
            CharSheet._select_skeleton(self.c, skel))
        return CharSheet(self.get_character(character))

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

    def load_effects(self, names):
        r = {}
        kd = {"effect": {}}
        for name in names:
            kd["effect"][name] = {"name": name}
        self.skeleton.update(
            Effect._select_skeleton(self.c, kd))
        need_chars = set()
        for name in names:
            rd = self.skeleton["effect"][name]
            need_chars.add(rd["character"])
        self.get_characters(need_chars)
        for name in names:
            r[name] = Effect(self, name)

    def get_effects(self, names):
        if len(names) == 0:
            return {}
        r = {}
        unloaded = set()
        for name in names:
            if name in self.effectdict:
                r[name] = self.effectdict[name]
            else:
                unloaded.add(name)
        if len(unloaded) > 0:
            r.update(self.load_effects(unloaded))
        return r

    def load_effect_decks(self, names):
        r = {}
        kd = {
            "effect_deck": {},
            "effect_deck_link": {}}
        for name in names:
            kd["effect_deck"][name] = {"name": name}
            kd["effect_deck_link"][name] = {"deck": name}
        self.skeleton.update(
            EffectDeck._select_skeleton(self.c, kd))
        for name in names:
            r[name] = EffectDeck(self, name)
        return r

    def get_effect_decks(self, names):
        r = {}
        unloaded = set()
        for name in names:
            if name in self.effectdeckdict:
                r[name] = self.effectdeckdict[name]
            else:
                unloaded.add(name)
        if len(unloaded) > 0:
            r.update(self.load_effect_decks(unloaded))
        return r

    def load_dimensions(self, names):
        # I think it might eventually *make sense* to load the same
        # dimension more than once without unloading it first. Perhaps
        # you want to selectively load the parts of it that the player
        # is interested in at the moment, the game world being too
        # large to practically load all at once.
        kd = Skeleton({"portal": {},
                       "thing_location": {}})
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

    def load_textures(self, names):
        kd = {"img": {}}
        for name in names:
            kd["img"][name] = {"name": name}
        self.skeleton.update(
            LiSEImage._select_skeleton(
                self.c, kd))
        r = {}
        for name in names:
            if self.skeleton["img"][name]["rltile"] != 0:
                rltex = Image(
                    source=self.skeleton["img"][name]["path"]).texture
                imgd = ImageData(rltex.width, rltex.height,
                                 rltex.colorfmt, rltex.pixels,
                                 source=self.skeleton["img"][name]["path"])
                fixed = ImageData(
                    rltex.width, rltex.height,
                    rltex.colorfmt, imgd.data.replace(
                    '\xffGll', '\x00Gll').replace(
                        '\xff.', '\x00.'),
                    source=self.skeleton["img"][name]["path"])
                rltex.blit_data(fixed)
                r[name] = rltex
            else:
                r[name] = Image(
                    source=self.skeleton["img"][name]["path"]).texture
        self.texturedict.update(r)
        return r

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

    def get_images(self, names):
        r = {}
        for (name, tex) in self.get_textures(names).iteritems():
            r[name] = Image(texture=tex, size=tex.size)
        return r

    def get_image(self, name):
        return self.get_images([name])[name]

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
        r = Skeleton({})
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
        kd = Skeleton({"menu": {}})
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
        kd = Skeleton({"menu_item": {"menu": menu}})
        skel = Menu._select_skeleton(self.c, kd)
        self.skeleton.update(skel)

    def get_effects_in_decks(self, decks):
        effds = self.get_effect_decks(decks)
        effects = set()
        for effd in effds.values():
            for rd in effd._card_links:
                effects.add(rd["effect"])
        return self.get_effects(effects)

    def get_cards_in_hands(self, hands):
        effects = self.get_effects_in_decks(hands)
        return self.get_cards([
            str(effect) for effect in effects.values()])

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
        for dimension in self.dimensions:
            dimension.new_branch(parent, branch, tick)
        for character in self.characters:
            character.new_branch(parent, branch, tick)
        self.skeleton["timestream"][branch] = {
            "branch": branch,
            "parent": parent}
        assert(branch == self.timestream.hi_branch + 1)
        self.timestream.hi_branch = branch

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

    def update(self, ts=None):
        self.time_travel_inc_tick(ticks=self.game_speed)

    def end_game(self):
        self.c.close()
        self.conn.commit()
        self.conn.close()

    def checkpoint(self):
        self.old_skeleton = self.skeleton.copy()


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
        " (front_dimension TEXT DEFAULT 'Physical', "
        "front_branch INTEGER DEFAULT 0, "
        "tick INTEGER DEFAULT 0,"
        " seed INTEGER DEFAULT 0);")
    c.execute(
        "CREATE TABLE strings (stringname TEXT NOT NULL, language TEXT NOT"
        " NULL DEFAULT 'English', string TEXT NOT NULL, "
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


def load_closet(dbfn, lang="eng", xfuncs={}):
    conn = sqlite3.connect(dbfn)
    c = conn.cursor()
    c.execute(
        "SELECT front_dimension, front_branch, seed, tick "
        "FROM game")
    row = c.fetchone()
    c.close()
    initargs = (conn, xfuncs, lang) + row
    r = Closet(*initargs)
    r.load_strings()
    r.load_timestream()
    return r
