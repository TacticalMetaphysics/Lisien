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
import igraph
import effect
from dimension import Dimension
from spot import Spot
from pawn import Pawn
from arrow import Arrow
from board import Board, BoardViewport
from card import Card, Hand
from calendar import Calendar
from effect import Effect, EffectDeck
from img import Img
from menu import Menu, MenuItem
from style import Style, Color
from timestream import Timestream
from gui import GameWindow
from collections import OrderedDict, defaultdict
from logging import getLogger
from util import dictify_row, colnames, colnamestr
from portal import Portal
from thing import Thing
from character import Character


logger = getLogger(__name__)


def noop(*args, **kwargs):
    """Do nothing."""
    pass

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
READ_IMGS_QRYFMT = (
    "SELECT {0} FROM img WHERE name IN ({1})".format(
        colnamestr["img"], "{0}"))
LOAD_CARDS_QRYFMT = (
    "SELECT {0} FROM card WHERE effect IN ({1})".format(
        colnamestr["card"], "{0}"))
BOARD_QRYFMT = (
    "SELECT {0} FROM board WHERE dimension=? AND idx=?".format(
        colnamestr["board"]))
BOARD_VIEWPORTS_QRYFMT = (
    "SELECT {0} FROM board_viewport WHERE "
    "window=? AND dimension=? AND board=?".format(
        colnamestr["board_viewport"]))
BOARD_VIEWPORT_WINDOW_QRYFMT = (
    "SELECT {0} FROM board_viewport WHERE window=?".format(
        colnamestr["board_viewport"]))
COLOR_QRYFMT = (
    "SELECT {0} FROM color WHERE name IN ({1})".format(colnamestr["color"], "{0}"))
STYLE_QRYFMT = (
    "SELECT {0} FROM style WHERE name IN ({1})".format(colnamestr["style"], "{0}"))
MENU_NAME_QRYFMT = (
    "SELECT {0} FROM menu WHERE name IN ({1})".format(colnamestr["menu"], "{0}"))
MENU_WINDOW_QRYFMT = (
    "SELECT {0} FROM menu WHERE window=?".format(colnamestr["menu"]))
MENU_ITEM_MENU_QRYFMT = (
    "SELECT {0} FROM menu_item WHERE menu IN ({1})".format(
        colnamestr["menu_item"], "{0}"))
MENU_ITEM_WINDOW_QRYFMT = (
    "SELECT {0} FROM menu_item WHERE window=?".format(colnamestr["menu_item"]))
CALENDAR_WINDOW_QRYFMT = (
    "SELECT {0} FROM calendar WHERE window=? ORDER BY idx".format(
        colnamestr["calendar"]))
WINDOW_QRYFMT = (
    "SELECT {0} FROM window WHERE name=?".format(GameWindow.colnstr))
EFFECT_QRYFMT = (
    "SELECT {0} FROM effect WHERE name IN ({1})".format(
        colnamestr["effect"], "{0}"))
EFFECT_DECK_QRYFMT = (
    "SELECT {0} FROM effect_deck WHERE name IN ({1})".format(
        colnamestr["effect_deck"], "{0}"))
EFFECT_IN_DECK_QRYFMT = (
    "SELECT {0} FROM effect WHERE name IN "
    "(SELECT effect FROM effect_deck_link WHERE deck=?)".format(
        colnamestr["effect"]))
CARD_IN_DECK_QRYFMT = (
    "SELECT {0} FROM card WHERE effect IN "
    "(SELECT effect FROM effect_deck_link WHERE deck=?)".format(
        colnamestr["card"]))
HAND_WINDOW_QRYFMT = (
    "SELECT {0} FROM hand WHERE window=?".format(colnamestr["hand"]))
PORT_DIM_QRYFMT = (
    "SELECT {0} FROM portal WHERE dimension=?".format(colnamestr["portal"]))
THING_LOC_QRYFMT = (
    "SELECT {0} FROM thing_location WHERE dimension=?".format(colnamestr["thing_location"]))
SPOT_BOARD_COORD_QRYFMT = (
    "SELECT {0} FROM spot_coords WHERE dimension=? AND board=?".format(colnamestr["spot_coords"]))
SPOT_BOARD_IMG_QRYFMT = (
    "SELECT {0} FROM spot_img WHERE dimension=? AND board=?".format(colnamestr["spot_img"]))
SPOT_BOARD_INTER_QRYFMT = (
    "SELECT {0} FROM spot_interactive WHERE dimension=? AND board=?".format(colnamestr["spot_interactive"]))
PAWN_BOARD_INTER_QRYFMT = (
    "SELECT {0} FROM pawn_interactive WHERE dimension=? AND board=?".format(
        colnamestr["pawn_interactive"]))
PAWN_BOARD_IMG_QRYFMT = (
    "SELECT {0} FROM pawn_img WHERE dimension=? AND board=?".format(
        colnamestr["pawn_img"]))
IMG_QRYFMT = (
    "SELECT {0} FROM img WHERE name IN ({1})".format(
        colnamestr["img"], "{0}"))
CHAR_THING_QRYFMT = (
    "SELECT {0} FROM character_things WHERE character IN ({1})".format(
        colnamestr["character_things"], "{0}"))
CHAR_SKILL_QRYFMT = (
    "SELECT {0} FROM character_skills WHERE character IN ({1})".format(
        colnamestr["character_skills"], "{0}"))
CHAR_STAT_QRYFMT = (
    "SELECT {0} FROM character_stats WHERE character IN ({1})".format(
        colnamestr["character_stats"], "{0}"))


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
        self.lang = lang

        self.game_speed = 1
        self.updating = False

        self.timestream = Timestream({0: (0, 0)}, {})
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

    def get_style(self, name):
        """Return the Style by the given name, loading it first if
necessary."""
        if name not in self.styledict:
            self.load_styles(name)
        return self.styledict[name]

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
        r = {}
        qmstr = ", ".join(["?"] * len(names))
        namet = tuple(names)
        things2load = set()
        decks2load = set()
        qrystr = CHAR_THING_QRYFMT.format(qmstr)
        self.c.execute(qrystr, namet)
        getdefd = lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        charthing_d = getdefd()
        for row in self.c:
            rd = dictify_row(row, colnames["character_things"])
            charthing_d[rd["character"]][rd["dimension"]][rd["branch"]][rd["tick_from"]] = (
                rd["thing"], rd["tick_to"])
            things2load.add(rd["thing"])
        charskil_d = getdefd()
        qrystr = CHAR_SKILL_QRYFMT.format(qmstr)
        self.c.execute(qrystr, namet)
        for row in self.c:
            rd = dictify_row(row, colnames["character_skills"])
            charskil_d[rd["character"]][rd["skill"]][rd["branch"]][rd["tick_from"]] = (
                rd["effect_deck"], rd["tick_to"])
            decks2load.add(rd["effect_deck"])
        charstat_d = getdefd()
        qrystr = CHAR_STAT_QRYFMT.format(qmstr)
        self.c.execute(qrystr, namet)
        for row in self.c:
            rd = dictify_row(row, colnames["character_stats"])
            charstat_d[rd["character"]][rd["stat"]][rd["branch"]][
                rd["tick_from"]] = (rd["value"], rd["tick_to"])
        for name in iter(names):
            r[name] = Character(
                self, name,
                charthing_d[name],
                charskil_d[name],
                charstat_d[name])
        self.characterdict.update(r)
        return r

    def get_characters(self, names):
        r = {}
        unloaded = set()
        for name in iter(names):
            if name in self.characterdict:
                r[name] = self.characterdict[name]
            else:
                unloaded.add(name)
        r.update(self.load_characters(unloaded))
        return r

    def load_effects(self, names):
        r = {}
        qrystr = EFFECT_QRYFMT.format(", ".join(["?"] * len(names)))
        self.c.execute(qrystr, tuple(names))
        chars = set()
        effect_rds = {}
        for row in self.c:
            rd = dictify_row(row, Effect.colns)
            chars.add(rd["character"])
            effect_rds[rd["name"]] = rd
        chard = self.get_characters(chars)
        for rd in effect_rds.itervalues():
            rd["character"] = chard[rd["character"]]
            r[rd["name"]] = Effect(**rd)
        return r

    def get_effects(self, names):
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
        qrystr = EFFECT_DECK_QRYFMT.format(
            ", ".join(["?"] * len(names)))
        self.c.execute(qrystr, tuple(names))
        effect_deck_rdd = {}
        effect_names = set()
        for row in self.c:
            rd = dictify_row(row, EffectDeck.colns)
            effect_deck_rdd[rd["deck"]] = rd
            effect_names.add(rd["effect"])
        effectd = self.get_effects(effect_names)
        self.c.execute(
            "SELECT name, draw_order FROM effect_deck "
            "WHERE name IN ({0})".format(
                ", ".join(["?"] * len(names))))
        for row in self.c:
            effect_deck_rdd[row[0]]["draw_order"] = row[1]
        effect_deck_d = {}
        for name in iter(names):
            effect_deck_d[name] = EffectDeck(
                self, name, effect_deck_rdd[name]["draw_order"])
        for rd in effect_deck_rdd.itervalues():
            effect_deck = effect_deck_d[rd["deck"]]
            try:
                (effects, tick_to) = effect_deck.effects[
                    rd["branch"]][rd["tick_from"]]
                assert(rd["tick_to"] == tick_to)
                while len(effects) <= rd["idx"]:
                    effects.append(None)
                effects[rd["idx"]] = effectd[rd["effect"]]
                effect_deck.set_effects(
                    effects, rd["branch"], rd["tick_from"], rd["tick_to"])
            except:
                effect_deck.set_effects(
                    [effectd[rd["effect"]]], rd["branch"],
                    rd["tick_from"], rd["tick_to"])
        self.effectdeckdict.update(effect_deck_d)
        return effect_deck_d

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

    def load_dimension(
            self, dimn, branches=None, tick_from=None, tick_to=None):
        # I think it might eventually *make sense* to load the same
        # dimension more than once without unloading it first. Perhaps
        # you want to selectively load the parts of it that the player
        # is interested in at the moment, the game world being too
        # large to practically load all at once.
        if dimn not in self.dimensiondict:
            self.dimensiondict[dimn] = Dimension(self, dimn)
        dim = self.dimensiondict[dimn]
        branchexpr = ""
        tickfromexpr = ""
        ticktoexpr = ""
        if branches is not None:
            branchexpr = " AND branch IN ({0})".format(
                ", ".join(["?"] * len(branches)))
        if tick_from is not None:
            tickfromexpr = " AND tick_from>=?"
        if tick_to is not None:
            ticktoexpr = " AND tick_to<=?"
        extrastr = "".join((branchexpr, tickfromexpr, ticktoexpr))
        valtup = tuple([dimn] + [
            b for b in (branches, tick_from, tick_to)
            if b is not None])
        self.c.execute(PORT_DIM_QRYFMT + extrastr, valtup)
        for row in self.c:
            rd = dictify_row(row, Portal.colns)
            if not dim.have_place(rd["origin"]):
                dim.make_place(rd["origin"])
            if not dim.have_place(rd["destination"]):
                dim.make_place(rd["destination"])
            if not dim.have_portal(rd["origin"], rd["destination"]):
                dim.make_portal(rd["origin"], rd["destination"])
            po = dim.get_portal(rd["origin"], rd["destination"])
            if not po.extant_between(rd["branch"], rd["tick_from"], rd["tick_to"]):
                po.persist(rd["branch"], rd["tick_from"], rd["tick_to"])
        self.c.execute(THING_LOC_QRYFMT + extrastr, valtup)
        for row in self.c:
            rd = dictify_row(row, Thing.colns)
            if rd["thing"] not in dim.thingdict:
                dim.make_thing(rd["thing"])
            try:
                loc = dim.get_place(rd["location"])
            except ValueError:
                loc = dim.get_portal(*re.match(PORTAL_NAME_RE, rd["location"]).groups())
            logger.debug("putting thing %s in place %s", rd["thing"], str(loc))
            thing = dim.get_thing(rd["thing"])
            thing.set_location(loc, rd["branch"], rd["tick_from"], rd["tick_to"])
        self.dimensiondict[dimn] = dim
        return dim

    def get_dimension(self, dimn):
        if dimn not in self.dimensiondict:
            self.load_dimension(dimn)
        return self.dimensiondict[dimn]

    def get_dimensions(self, names):
        r = {}
        for name in names:
            r[name] = self.get_dimension(name)
        return r

    def get_board(self, dim, i):
        if not isinstance(dim, Dimension):
            dim = self.get_dimension(dim)
        if len(dim.boards) <= i:
            return self.load_board(dim, i)
        else:
            return dim.boards[i]

    def load_board(self, dim, i):
        while len(dim.boards) <= i:
            dim.boards.append(None)
        # basic information for this board
        self.c.execute(BOARD_QRYFMT, (str(dim), i))
        rd = dictify_row(self.c.fetchone(), Board.colns)
        walln = rd["wallpaper"]
        width = rd["width"]
        height = rd["height"]
        imgs2load = set([walln, 'default_pawn', 'default_spot'])
        # images for the spots
        self.c.execute(SPOT_BOARD_IMG_QRYFMT, (str(dim), i))
        spot_rows = self.c.fetchall()
        for row in spot_rows:
            imgs2load.add(row[4])
        # images for the pawns
        self.c.execute(PAWN_BOARD_IMG_QRYFMT, (str(dim), i))
        pawn_rows = self.c.fetchall()
        for row in pawn_rows:
            imgs2load.add(row[4])
        imgs = self.load_imgs(imgs2load)
        dim.boards[i] = Board(dim, i, width, height, imgs[walln])
        # actually assign images instead of just collecting the names
        for row in pawn_rows:
            rd = dictify_row(row, colnames["pawn_img"])
            thing = dim.thingdict[rd["thing"]]
            pawn = dim.boards[i].get_pawn(thing)
            pawn.set_img(imgs[rd["img"]], rd["branch"],
                         rd["tick_from"], rd["tick_to"])
        # interactivity for the pawns
        self.c.execute(PAWN_BOARD_INTER_QRYFMT, (str(dim), i))
        for row in self.c:
            rd = dictify_row(row, colnames["pawn_interactive"])
            pawn = dim.boards[i].get_pawn(dim.thingdict[rd["thing"]])
            pawn.set_interactive(rd["branch"], rd["tick_from"], rd["tick_to"])
        # spots in this board
        self.c.execute(SPOT_BOARD_COORD_QRYFMT, (str(dim), i))
        for row in self.c:
            rd = dictify_row(row, colnames["spot_coords"])
            if rd["place"] not in dim.placenames:
                dim.make_place(rd["place"])
            place = dim.get_place(rd["place"])
            spot = dim.boards[i].get_spot(place)
            logger.debug(
                "Loaded the spot for %s. Setting its coords to "
                "(%d, %d) in branch %d from tick %d.",
                str(place), rd["x"], rd["y"], rd["branch"], rd["tick_from"])
            spot.set_coords(rd["x"], rd["y"], rd["branch"], rd["tick_from"], rd["tick_to"])
        #their images
        for row in spot_rows:
            rd = dictify_row(row, colnames["spot_img"])
            spot = dim.boards[i].get_spot(rd["place"])
            spot.set_img(imgs[rd["img"]], rd["branch"], rd["tick_from"], rd["tick_to"])
        # interactivity for the spots
        self.c.execute(SPOT_BOARD_INTER_QRYFMT, (str(dim), i))
        for row in self.c:
            rd = dictify_row(row, colnames["spot_interactive"])
            spot = dim.boards[i].get_spot(rd["place"])
            spot.set_interactive(rd["branch"], rd["tick_from"], rd["tick_to"])
        # arrows in this board
        for port in dim.portals:
            dim.boards[i].get_arrow(port)
        return dim.boards[i]

    def load_imgs(self, imgs):
        qryfmt = IMG_QRYFMT
        qrystr = qryfmt.format(", ".join(["?"] * len(imgs)))
        self.c.execute(qrystr, tuple(imgs))
        r = {}
        for row in self.c:
            img = Img(self, row[0], row[1], row[2])
            r[row[0]] = img
        self.imgdict.update(r)
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

    def read_colors(self, colornames):
        qrystr = COLOR_QRYFMT.format(", ".join(["?"] * len(colornames)))
        self.c.execute(qrystr, tuple(colornames))
        r = {}
        for row in self.c:
            rowdict = dictify_row(row, Color.colns)
            c = Color(**rowdict)
            r[rowdict["name"]] = c
            self.colordict[rowdict["name"]] = c
        return r

    def get_colors(self, colornames):
        r = {}
        unloaded = set()
        for color in colornames:
            if color in self.colordict:
                r[color] = self.colordict[color]
            else:
                unloaded.add(color)
        r.update(self.read_colors(unloaded))
        return r

    def read_styles(self, stylenames):
        qrystr = STYLE_QRYFMT.format(", ".join(["?"] * len(stylenames)))
        self.c.execute(qrystr, tuple(stylenames))
        style_rows = self.c.fetchall()
        colornames = set()
        colorcols = (
            "textcolor", "fg_inactive",
            "fg_active", "bg_inactive", "bg_active")
        for row in style_rows:
            rowdict = dictify_row(row, Style.colns)
            for colorcol in colorcols:
                colornames.add(rowdict[colorcol])
        colors = self.get_colors(tuple(colornames))
        r = {}
        for row in style_rows:
            rowdict = dictify_row(row, Style.colns)
            for colorcol in colorcols:
                rowdict[colorcol] = colors[rowdict[colorcol]]
            s = Style(**rowdict)
            r[rowdict["name"]] = s
            self.styledict[rowdict["name"]] = s
        return r

    def get_styles(self, stylenames):
        r = {}
        unloaded = set()
        for style in stylenames:
            if style in self.styledict:
                r[style] = self.styledict[style]
            else:
                unloaded.add(style)
        r.update(self.read_styles(unloaded))
        return r

    def load_window(self, name):
        self.c.execute(WINDOW_QRYFMT, (name,))
        window_row = dictify_row(self.c.fetchone(), GameWindow.colns)
        self.c.execute(MENU_WINDOW_QRYFMT, (name,))
        menu_rds = {}
        for row in self.c:
            rd = dictify_row(row, Menu.colns)
            menu_rds[rd["name"]] = rd
        self.c.execute(MENU_ITEM_WINDOW_QRYFMT, (name,))
        menu_item_rds = defaultdict(list)
        for row in self.c:
            rd = dictify_row(row, MenuItem.colns)
            while len(menu_item_rds[rd["menu"]]) <= rd["idx"]:
                menu_item_rds[rd["menu"]].append(None)
            menu_item_rds[rd["menu"]][rd["idx"]] = rd
        self.c.execute(CALENDAR_WINDOW_QRYFMT, (name,))
        calendar_rds = []
        for row in self.c:
            rd = dictify_row(row, Calendar.colns)
            while len(calendar_rds) <= rd["idx"]:
                calendar_rds.append(None)
            calendar_rds[rd["idx"]] = rd
        self.c.execute(HAND_WINDOW_QRYFMT, (name,))
        hand_rds = {}
        for row in self.c:
            rd = dictify_row(row, Hand.colns)
            hand_rds[rd["effect_deck"]] = rd
        if len(hand_rds) > 0:
            self.c.execute(CARD_IN_DECK_QRYFMT, tuple(hand_rds.keys()))
        card_rds = {}
        for row in self.c:
            rd = dictify_row(row, Card.colns)
            card_rds[rd["effect"]] = rd
        self.c.execute(BOARD_VIEWPORT_WINDOW_QRYFMT, (name,))
        viewport_rds = []
        for row in self.c:
            rd = dictify_row(row, BoardViewport.colns)
            viewport_rds.append(rd)
        window_row["menu_rds"] = menu_rds
        window_row["menu_item_rds"] = menu_item_rds
        window_row["calendar_rds"] = calendar_rds
        window_row["hand_rds"] = hand_rds
        window_row["card_rds"] = card_rds
        window_row["viewport_rds"] = viewport_rds
        window_row["rumor"] = self
        return GameWindow(**window_row)

    def time_travel(self, mi, branch, tick):
        if branch not in self.timestream.branchdict:
            raise Exception(
                "Tried to time-travel to a branch that didn't exist yet")
        (tick_from, tick_to) = self.timestream.branchdict[branch]
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
            (old_tick_from, old_tick_to) = self.timestream.branchdict[branch_to]
            if tick_to < old_tick_from:
                raise TimestreamException(
                    "Can't make a new branch that starts earlier than its parent.")
            if tick_to > old_tick_to:
                # TODO: This really demands special handling--
                # STUFF may happen between old_tick_to and tick_to
                self.timestream.branchdict[branch_to] = (old_tick_from, tick_to)
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
            self.timestream.branchdict[branch_to] = (tick_from, tick_to)
            for dimension in self.dimensions:
                dimension.new_branch(branch_from, branch_to, tick_from)
                for board in dimension.boards:
                    board.new_branch(branch_from, branch_to, tick_from)
            if self.game["hi_branch"] < branch_to:
                self.game["hi_branch"] = branch_to
        

    def increment_branch(self, mi=None, branches=1):
        try:
            self.more_time(self.branch, self.branch+int(branches), self.tick, self.tick)
        except TimestreamException:
            return self.increment_branch(mi, int(branches)+1)
        self.time_travel(mi, self.branch+int(branches), self.tick)

    def increment_tick(self, mi=None, ticks=1):
        self.more_time(self.branch, self.branch, self.tick, self.tick+int(ticks))
        self.time_travel(mi, self.branch, self.tick+int(ticks))

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

    def save_game(self):
        self.c.execute("DELETE FROM game")
        keynames = self.game.keys()
        values = tuple([self.game[key] for key in keynames])
        qrystr = (
            "INSERT INTO game ({0}) VALUES ({1})".format(
                ", ".join(keynames),
                ", ".join(["?"] * len(keynames))))
        self.c.execute(qrystr, values)

    def end_game(self):
        self.c.close()
        self.conn.commit()
        self.conn.close()


def load_game(dbfn, lang="eng"):
    db = RumorMill(dbfn, lang=lang)
    db.load_game()
    db.load_strings()
    return db
