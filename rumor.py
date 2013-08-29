# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""The database backend, with dictionaries of loaded objects.

This is a caching database connector. There are dictionaries for all
objects that can be loaded from the database.

This module does not contain the code used to generate
SQL. That's in util.py, the class SaveableMetaclass.

"""

import sqlite3
import re
import os
import igraph
from collections import OrderedDict, defaultdict
from copy import deepcopy
from logging import getLogger
from dimension import Dimension
from spot import Spot
from pawn import Pawn
from board import Board, BoardViewport
from card import Card
from effect import Effect, EffectDeck
from img import Img
from style import Style, Color
from timestream import Timestream, TimestreamException
from gui import GameWindow
from util import (
    dictify_row,
    TabdictIterator,
    schemata,
    saveables)
from portal import Portal
from thing import Thing
from character import Character


logger = getLogger(__name__)


def noop(*args, **kwargs):
    """Do nothing."""
    pass


def dd():
    return defaultdict(dd)


def updd(d1, d2):
    """Deep update"""
    for (k, v) in d2.iteritems():
        if k not in d1:
            d1[k] = v
        elif isinstance(v, dict):
            assert isinstance(d1[k], dict)
            updd(d1[k], v)
        else:
            d1[k] = v

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


class RumorMill(object):
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

    def __init__(self, dbfilen, xfuncs={},
                 front_board="default_board", front_branch=0, seed=0, tick=0,
                 hi_branch=0, hi_place=0, hi_portal=0, hi_thing=0, lang="eng"):
        """Return a database wrapper around the SQLite database file by the
given name.

        """
        self.conn = sqlite3.connect(dbfilen)
        self.cursor = self.conn.cursor()
        self.c = self.cursor

        # This dict is special. It contains all the game
        # data--represented only as those types which sqlite3 is
        # capable of storing. All my objects are ultimately just
        # views on this thing.
        self.tabdict = {}
        # This is a copy of the tabdict as it existed at the time of
        # the last save. I'll be finding the differences between it
        # and the current tabdict in order to decide what to write to
        # disk.
        self.old_tabdict = {}

        self.windowdict = {}
        self.boardhanddict = {}
        self.calendardict = {}
        self.carddict = {}
        self.colordict = {}
        self.dimensiondict = {}
        self.effectdict = {}
        self.effectdeckdict = {}
        self.imgdict = OrderedDict()
        self.menudict = {}
        self.menuitemdict = {}
        self.stringdict = {}
        self.styledict = {}
        self.tickdict = {}
        self.eventdict = {}
        self.characterdict = {}
        self.windowdict = {}
        self.lang = lang

        self.game_speed = 1
        self.updating = False

        self.c.execute(
            "SELECT branch, parent, tick_from, tick_to FROM timestream")
        self.timestream = Timestream(self.c.fetchall())
        self.time_travel_history = []

        placeholder = (noop, ITEM_ARG_RE)
        self.effect_cbs = {}
        self.func = {
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
            'increment_tick':
            (self.increment_tick, ONE_ARG_RE),
            'time_travel':
            (self.time_travel, TWO_ARG_RE),
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
        self.func.update(xfuncs)
        self.game = {
            "front_board": front_board,
            "front_branch": front_branch,
            "seed": seed,
            "tick": tick,
            "hi_place": hi_place,
            "hi_portal": hi_portal,
            "hi_thing": hi_thing}

    def __getattr__(self, attrn):
        if attrn in ("board", "front_board"):
            return self.game["front_board"]
        elif attrn in ("branch", "front_branch"):
            return self.game["front_branch"]
        elif attrn == "seed":
            return self.game["seed"]
        elif attrn == "tick":
            return self.game["tick"]
        elif attrn == "hi_place":
            return self.game["hi_place"]
        elif attrn == "hi_portal":
            return self.game["hi_portal"]
        elif attrn == "hi_thing":
            return self.game["hi_thing"]
        elif attrn == "dimensions":
            return self.dimensiondict.itervalues()
        else:
            raise AttributeError(
                "RumorMill doesn't have the attribute " + attrn)

    def __setattr__(self, attrn, val):
        if attrn in ("front_board", "seed", "age",
                     "hi_place", "hi_portal", "hi_thing"):
            getattr(self, "game")[attrn] = val
        else:
            super(RumorMill, self).__setattr__(attrn, val)

    def __del__(self):
        """Try to write changes to disk before dying.

        """
        self.c.close()
        self.conn.commit()
        self.conn.close()

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
        if strname == "tick":
            return str(self.tick)
        elif strname == "branch":
            return str(self.branch)
        else:
            return self.stringdict[strname][self.lang]

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
        self.c.execute("DELETE FROM game")
        fieldnames = self.game.keys()
        qrystr = "INSERT INTO game ({0}) VALUES ({1})".format(
            ", ".join(fieldnames), ", ".join(["?"] * len(fieldnames)))
        qrylst = [self.game[field] for field in fieldnames]
        qrytup = tuple(qrylst)
        self.c.execute(qrystr, qrytup)
        for dimension in self.dimensiondict.itervalues():
            dimension.save()
        self.old_tabdict = deepcopy(self.tabdict)

    # TODO: For all these schedule functions, handle the case where I
    # try to schedule something for a time outside of the given
    # branch. These functions may not be the most appropriate *place*
    # to handle that.

    def schedule_something(
            self, scheddict, dictkeytup,
            val=None, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.branch
        if tick_from is None:
            tick_from = self.tick
        ptr = scheddict
        for key in dictkeytup + (branch,):
            if key not in ptr:
                ptr[key] = {}
            ptr = ptr[key]
        if val is None:
            ptr[tick_from] = tick_to
        else:
            ptr[tick_from] = (val, tick_to)

    def schedule_event(self, ev, branch=None, tick_from=None, tick_to=None):
        if not hasattr(ev, 'schedule'):
            ev.schedule = {}
        if branch not in ev.schedule:
            ev.schedule[branch] = {}
        ev.schedule[branch][tick_from] = tick_to

    def event_is_commencing(self, ev, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick

    def event_is_concluding(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return False
        for prevstart in self.event_scheduled[name][branch]:
            if (
                    prevstart < tick and
                    self.event_scheduled[
                        name][branch][prevstart] == tick):
                return True
        return False

    def event_is_proceeding(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return False
        for prevstart in self.event_scheduled[name][branch]:
            if (
                    prevstart < tick and
                    self.event_scheduled[
                        name][branch][prevstart] > tick):
                return True
        return False

    def event_is_starting_or_proceeding(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return False
        for prevstart in self.event_scheduled[name][branch]:
            if prevstart == tick:
                return True
            elif (
                    prevstart < tick and
                    self.event_scheduled[
                        name][branch][prevstart] > tick):
                return True
        return False

    def event_is_proceeding_or_concluding(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return False
        for prevstart in self.event_scheduled[name][branch]:
            if prevstart < tick:
                if self.event_scheduled[name][branch][prevstart] >= tick:
                    return True
        return False

    def event_is_happening(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return False
        for prevstart in self.event_scheduled[name][branch]:
            if prevstart == tick:
                return True
            elif prevstart < tick:
                if self.event_scheduled[name][branch][prevstart] >= tick:
                    return True
        return False

    def get_event_start(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return None
        for (tick_from, tick_to) in (
                self.event_scheduled[name][branch].iteritems()):
            if tick_from <= tick and tick <= tick_to:
                return tick_from
        return None

    def get_event_end(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        if (
                name not in self.event_scheduled or
                branch not in self.event_scheduled[name]):
            return None
        for (tick_from, tick_to) in (
                self.event_scheduled[name][branch].iteritems()):
            if tick_from <= tick and tick <= tick_to:
                return tick_to
        return None

    def schedule_effect_deck(self, name, cards,
                             branch=None, tick_from=None, tick_to=None):
        self.schedule_something(
            self.effect_deck_scheduled,
            (name,), cards, branch, tick_from, tick_to)

    def get_effect_deck_card_names(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        for (commencement, (cards, conclusion)) in self.effect_deck_scheduled[
                name][branch].iteritems():
            if commencement <= tick and conclusion >= tick:
                return cards
        return None

    def schedule_char_att(self, char_s, att_s, val,
                          branch=None, tick_from=None, tick_to=None):
        self.schedule_something(
            self.char_att_scheduled,
            (char_s, att_s), val, branch, tick_from, tick_to)

    def get_char_att_val(self, char_s, att_s, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        for (commencement, (value, conclusion)) in self.char_att_scheduled[
                char_s][att_s][branch].iteritems():
            if commencement <= tick and conclusion >= tick:
                return value
        return None

    def schedule_char_skill(self, char_s, skill_s, effect_deck_s,
                            branch=None, tick_from=None, tick_to=None):
        self.schedule_something(
            self.char_skill_scheduled, (char_s, skill_s), effect_deck_s,
            branch, tick_from, tick_to)

    def get_char_skill_deck_name(
            self, char_s, skill_s, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        for (commencement, (deck, conclusion)) in self.char_skill_scheduled[
                char_s][skill_s][branch].iteritems():
            if commencement <= tick and conclusion >= tick:
                return deck
        return ''

    def schedule_char_item(self, char_s, dimension_s, item_s,
                           branch=None, tick_from=None, tick_to=None):
        self.schedule_something(
            self.char_item_scheduled, (char_s, dimension_s, item_s),
            None, branch, tick_from, tick_to)

    def character_is_item(self, char_s, dimension_s, item_s,
                          branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        for (commencement, conclusion) in (
                self.char_item_scheduled[
                    char_s][dimension_s][item_s][branch]):
            if commencement <= tick and conclusion >= tick:
                return True
        return False

    def schedule_hand_card(self, hand_s, card_s, card_n,
                           branch=None, tick_from=None, tick_to=None):
        self.schedule_something(
            self.hand_card_scheduled, (hand_s, card_s), card_n,
            branch, tick_from, tick_to)

    def card_copies_in_hand(self, hand_s, card_s, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        for (commencement, (n, conclusion)) in self.hand_card_scheduled[
                hand_s][card_s][branch].iteritems():
            if commencement <= tick and conclusion >= tick:
                return n
        return 0

    def load_game(self):
        self.c.execute(
            "SELECT front_board, front_branch, tick, seed, "
            "hi_branch, hi_place, hi_portal, hi_thing FROM game")
        for row in self.c:
            self.game = {
                "front_board": row[0],
                "front_branch": row[1],
                "tick": row[2],
                "seed": row[3],
                "hi_branch": row[4],
                "hi_place": row[5],
                "hi_portal": row[6],
                "hi_thing": row[7]}

    def load_strings(self):
        self.c.execute("SELECT stringname, language, string FROM strings")
        for row in self.c:
            rowd = dictify_row(row, ("stringname", "language", "string"))
            if rowd["stringname"] not in self.stringdict:
                self.stringdict[rowd["stringname"]] = {}
            self.stringdict[rowd[
                "stringname"]][rowd["language"]] = rowd["string"]

    def make_generic_place(self, dimension):
        placen = "generic_place_{0}".format(self.hi_place)
        self.hi_place += 1
        dimension.make_place(placen)
        return dimension.get_place(placen)

    def make_generic_thing(self, dimension, location):
        thingn = "generic_thing_{0}".format(self.hi_thing)
        self.hi_thing += 1
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
        dimension = orig.dimension
        dimension.make_portal(orig, dest)
        port = dimension.get_portal(orig, dest)
        port.persist()
        return port

    def load_characters(self, names):
        qtd = {
            "character_things": {},
            "character_stats": {},
            "character_skills": {}}
        for name in names:
            for tabn in qtd.iterkeys():
                qtd[tabn][name] = {"character": name}
        updd(self.tabdict,
             Character._select_tabdict(self.c, qtd))
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
            if name in self.characterdict:
                r[name] = self.characterdict[name]
            else:
                unhad.add(name)
        r.update(self.load_characters(names))
        return r

    def get_character(self, name):
        return self.get_characters([name])[name]

    def get_thing(self, dimn, thingn):
        return self.get_dimension(dimn).get_thing(thingn)

    def load_effects(self, names):
        r = {}
        kd = {"effect": {}}
        for name in names:
            kd["effect"][name] = {"name": name}
        updd(self.tabdict,
             Effect._select_tabdict(self.c, kd))
        need_chars = set()
        for name in names:
            rd = self.tabdict["effect"][name]
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
        updd(self.tabdict,
             EffectDeck._select_tabdict(self.c, kd))
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
        r.update(self.load_effect_decks(unloaded))
        return r

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
        updd(self.tabdict,
             Portal._select_tabdict(self.c, kd))
        updd(self.tabdict,
             Thing._select_tabdict(self.c, kd))
        r = {}
        for name in names:
            r[name] = Dimension(self, name)
        return r

    def get_dimensions(self, names):
        r = {}
        unhad = set()
        for name in names:
            if name in self.dimensiondict:
                r[name] = self.dimensiondict[name]
            else:
                unhad.add(name)
        r.update(self.load_dimensions(unhad))
        return r

    def get_dimension(self, name):
        return self.get_dimensions([name])[name]

    def get_place(self, dim, placen):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        return dim.get_place(placen)

    def get_portal(self, dim, origin, destination):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        return dim.get_portal(str(origin), str(destination))

    def get_board(self, dim, i):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        if len(dim.boards) <= i or dim.boards[i] is None:
            return self.load_board(dim, i)
        else:
            return dim.boards[i]

    def load_board(self, dim, i):
        dimn = str(dim)
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dimn)
        updd(self.tabdict,
             Board._select_tabdict(
                 self.c,
                 {"board":
                  {"dimension": dimn,
                   "idx": i}}))
        updd(self.tabdict,
             Spot._select_tabdict(
                 self.c,
                 {"spot_img":
                  {"dimension": dimn,
                   "board": i},
                  "spot_interactive":
                  {"dimension": dimn,
                   "board": i},
                  "spot_coords":
                  {"dimension": dimn,
                   "board": i}}))
        updd(self.tabdict,
             Pawn._select_tabdict(
                 self.c,
                 {"pawn_img":
                  {"dimension": dimn,
                   "board": i},
                  "pawn_interactive":
                  {"dimension": dimn,
                   "board": i}}))
        return Board(self, dim, i)

    def load_viewport(self, win, dim, board, viewi):
        winn = str(win)
        dimn = str(dim)
        boardi = int(board)
        if not isinstance(win, GameWindow):
            win = self.get_window(winn)
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dimn)
        if not isinstance(board, Board):
            board = self.get_board(dim, boardi)
        kd = {"board_viewport":
              {"window": winn,
               "dimension": dimn,
               "board": boardi,
               "idx": viewi}}
        updd(self.tabdict,
             BoardViewport._select_tabdict(self.c, kd))
        return BoardViewport(
            self, win, dim, board, viewi)

    def get_viewport(self, win, dim, boardidx, viewi):
        if isinstance(boardidx, Board):
            board = boardidx
        else:
            board = self.get_board(dim, boardidx)
        if (
                len(board.views) > viewi and
                board.views[viewi] is not None):
            return board.views[viewi]
        else:
            return self.load_viewport(win, dim, board, viewi)

    def load_imgs(self, names):
        kd = {"img": {}}
        for name in names:
            kd["img"][name] = {"name": name}
        updd(self.tabdict,
             Img._select_tabdict(
                 self.c, kd))
        r = {}
        for name in names:
            r[name] = Img(self, name)
        return r

    def get_imgs(self, imgnames):
        r = {}
        unloaded = set()
        for imgn in imgnames:
            if imgn in self.imgdict:
                r[imgn] = self.imgdict[imgn]
            else:
                unloaded.add(imgn)
        r.update(self.load_imgs(unloaded))
        return r

    def get_img(self, imgn):
        return self.get_imgs([imgn])[imgn]

    def load_colors(self, names):
        kd = {"color": {}}
        for name in names:
            kd["color"][name] = {"name": name}
        updd(self.tabdict,
             Color._select_tabdict(self.c, kd))
        r = {}
        for name in names:
            r[name] = Color(self, name)
        return r

    def get_colors(self, colornames):
        r = {}
        unloaded = set()
        for color in colornames:
            if color in self.colordict:
                r[color] = self.colordict[color]
            else:
                unloaded.add(color)
        r.update(self.load_colors(unloaded))
        return r

    def get_color(self, name):
        return self.get_colors([name])[name]

    def load_styles(self, stylenames):
        kd = {"style": {}}
        for name in stylenames:
            kd["style"][name] = {"name": name}
        updd(self.tabdict,
             Style._select_tabdict(self.c, kd))
        colornames = set()
        for name in stylenames:
            rd = self.tabdict["style"][name]
            for colorcol in Style.color_cols:
                colornames.add(rd[colorcol])
        self.get_colors(colornames)
        r = {}
        for name in stylenames:
            r[name] = Style(self, name)
        return r

    def get_styles(self, stylenames):
        r = {}
        unloaded = set()
        for style in stylenames:
            if style in self.styledict:
                r[style] = self.styledict[style]
            else:
                unloaded.add(style)
        r.update(self.load_styles(unloaded))
        return r

    def get_style(self, name):
        return self.get_styles([name])[name]

    def load_windows(self, names):
        kd = {
            "window": {},
            "board_viewport": {},
            "menu": {},
            "hand": {},
            "menu_item": {},
            "calendar": {},
            "calendar_col_thing": {},
            "calendar_col_stat": {},
            "calendar_col_skill": {}}
        for name in names:
            kd["window"][name] = {"name": name}
            for col in kd.iterkeys():
                if col == "window":
                    continue
                kd[col][name] = {"window": name}
        updd(self.tabdict,
             GameWindow._select_tabdict(self.c, kd))
        r = {}
        for name in names:
            r[name] = GameWindow(self, name)
        return r

    def get_windows(self, names):
        r = {}
        unhad = set()
        for name in iter(names):
            if name in self.windowdict:
                r[name] = self.windowdict[name]
            else:
                unhad.add(name)
        r.update(self.load_windows(unhad))
        return r

    def get_window(self, name):
        return self.get_windows([name])[name]

    def load_cards(self, names):
        effectdict = self.get_effects(names)
        kd = {"card": {}}
        for name in names:
            kd["card"][name] = {"effect": name}
        td = Card._select_tabdict(self.c, kd)
        r = {}
        for rd in TabdictIterator(td):
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
        r.update(self.load_cards(unhad))
        return r

    def get_card(self, name):
        return self.get_cards([name])[name]

    def get_effects_in_decks(self, decks):
        effds = self.get_effect_decks(decks)
        effects = set()
        for effd in effds.itervalues():
            for rd in TabdictIterator(effd._card_links):
                effects.add(rd["effect"])
        return self.get_effects(effects)

    def get_cards_in_hands(self, hands):
        effects = self.get_effects_in_decks(hands)
        return self.get_cards([
            str(effect) for effect in effects.itervalues()])

    def time_travel(self, mi, branch, tick):
        if branch not in self.timestream.branchdict:
            raise Exception(
                "Tried to time-travel to a branch that didn't exist yet")
        (parent, tick_from, tick_to) = self.timestream.branchdict[branch]
        if tick < tick_from or tick > tick_to:
            raise Exception(
                "Tried to time-travel to a tick that hadn't passed yet")
        self.time_travel_history.append((self.branch, self.tick))
        self.branch = branch
        self.tick = tick
        if mi is not None:
            for calendar in mi.window.calendars:
                calendar.refresh()

    def more_time(self, branch_from, branch_to, tick_from, tick_to):
        if branch_to in self.timestream.branchdict:
            (parent, old_tick_from, old_tick_to) = (
                self.timestream.branchdict[branch_to])
            if tick_to < old_tick_from:
                raise TimestreamException(
                    "Can't make a new branch that starts "
                    "earlier than its parent.")
            if tick_to > old_tick_to:
                # TODO: This really demands special handling--
                # STUFF may happen between old_tick_to and tick_to
                self.timestream.branchdict[branch_to] = (
                    parent, old_tick_from, tick_to)
                e = self.timestream.latest_edge(branch_to)
                self.timestream.graph.vs[e.target]["tick"] = tick_to
        else:
            e = self.timestream.split_branch(
                branch_from,
                branch_to,
                tick_from,
                tick_to)
            v = self.timestream.graph.vs[e.source]
            self.timestream.branch_head[branch_to] = v
            self.timestream.branchdict[branch_to] = (
                branch_from, tick_from, tick_to)
            for dimension in self.dimensions:
                dimension.new_branch(branch_from, branch_to, tick_from)
                for board in dimension.boards:
                    board.new_branch(branch_from, branch_to, tick_from)
            if self.game["hi_branch"] < branch_to:
                self.game["hi_branch"] = branch_to

    def increment_branch(self, mi=None, branches=1):
        try:
            self.more_time(
                self.branch, self.branch + int(branches),
                self.tick, self.tick)
        except TimestreamException:
            return self.increment_branch(mi, int(branches) + 1)
        self.time_travel(mi, self.branch + int(branches), self.tick)

    def increment_tick(self, mi=None, ticks=1):
        self.more_time(
            self.branch, self.branch,
            self.tick, self.tick + int(ticks))
        self.time_travel(mi, self.branch, self.tick + int(ticks))

    def go(self, nope=None):
        self.updating = True

    def stop(self, nope=None):
        self.updating = False

    def set_speed(self, newspeed):
        self.game_speed = newspeed

    def play_speed(self, mi, gamespeed):
        self.set_speed(float(gamespeed))
        self.go()

    def back_to_start(self, nope):
        self.stop()
        self.time_travel(None, self.branch, 0)

    def update(self, ts):
        if self.updating:
            self.increment_tick(ticks=self.game_speed)

    def end_game(self):
        self.c.close()
        self.conn.commit()
        self.conn.close()


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
        " (front_board TEXT DEFAULT 'Physical', "
        "front_branch INTEGER DEFAULT 0, "
        "tick INTEGER DEFAULT 0,"
        " seed INTEGER DEFAULT 0, hi_place INTEGER DEFAULT 0, "
        "hi_portal INTEGER DEFAULT 0, "
        "hi_thing INTEGER DEFAULT 0, hi_branch INTEGER DEFAULT 0);")
    c.execute(
        "CREATE TABLE strings (stringname TEXT NOT NULL, language TEXT NOT"
        " NULL DEFAULT 'English', string TEXT NOT NULL, "
        "PRIMARY KEY(stringname,  language));")

    done = set()
    while saveables != []:
        (demands, provides, prelude,
         tablenames, postlude) = saveables.pop(0)
        print tablenames
        if 'character_things' in tablenames:
            pass
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
        if tablenames == []:
            while prelude != []:
                pre = prelude.pop()
                if isinstance(pre, tuple):
                    c.execute(*pre)
                else:
                    c.execute(pre)
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
            if tn == "calendar":
                pass
            try:
                c.execute(schemata[tn])
                done.add(tn)
            except sqlite3.OperationalError as e:
                print "OperationalError while creating table {0}:".format(tn)
                print e
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
            print "OperationalError during postlude from {0}:".format(tn)
            print e
            saveables.append(
                (demands, provides, prelude, tablenames, postlude))
            continue
        done.update(provides)

    oldhome = os.getcwd()
    os.chdir('sql')
    initfiles = sorted(os.listdir('.'))
    for initfile in initfiles:
        if initfile[-3:] == "sql":  # weed out automatic backups and so forth
            print "reading SQL from file " + initfile
            read_sql(initfile)

    os.chdir(oldhome)

    print "indexing the RLTiles"
    ins_rltiles(c, 'rltiles')

    # print "indexing the dumb effects"
    # efns = db.c.execute("SELECT on_click FROM menu_item").fetchall()
    # for row in efns:
    #     print row[0]
    #     dumb_effect(db, row[0])

    c.close()
    conn.commit()


def load_game(dbfn, lang="eng"):
    db = RumorMill(dbfn, lang=lang)
    db.load_game()
    db.load_strings()
    return db
