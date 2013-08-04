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
from place import Place
from portal import Portal
from thing import Thing
from spot import Spot
from pawn import Pawn
from arrow import Arrow
from board import Board
from img import Img
from style import Style, Color
from menu import Menu, MenuItem
from card import Card, Hand
from calendar import Calendar
from gui import GameWindow
from collections import OrderedDict
from logging import getLogger
from util import dictify_row


logger = getLogger(__name__)


def noop(*args, **kwargs):
    """Do nothing."""
    pass

ONE_ARG_RE = re.compile("(.+)")
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
        ", ".join(Img.colnames["img"]), "{0}"))
LOAD_CARDS_QRYFMT = (
    "SELECT {0} FROM card WHERE effect IN ({1})".format(
        ", ".join(Card.colns), "{0}"))
THING_LOC_QRY_START = (
    "SELECT thing, branch, tick_from, tick_to, "
    "location FROM thing_location WHERE dimension=?")
THING_SPEED_QRY_START = (
    "SELECT thing, branch, tick_from, tick_to, "
    "ticks_per_span FROM thing_speed WHERE dimension=?")
PORT_QRY_START = (
    "SELECT origin, destination, branch, tick_from, "
    "tick_to FROM portal_existence WHERE dimension=?")
BOARD_QRY_START = (
    "SELECT wallpaper, width, height "
    "FROM board WHERE dimension=? AND i=?")
SPOT_BOARD_QRY_START = (
    "SELECT place, branch, tick_from, tick_to, "
    "x, y FROM spot_coords WHERE dimension=? AND board=?")
SPOT_IMG_QRY_START = (
    "SELECT place, branch, tick_from, tick_to, "
    "img FROM spot_img WHERE dimension=? AND board=?")
SPOT_INTER_QRY_START = (
    "SELECT place, branch, tick_from, tick_to "
    "FROM spot_interactive WHERE dimension=? AND board=?")
PAWN_IMG_QRY_START = (
    "SELECT thing, branch, tick_from, tick_to, "
    "img FROM pawn_img WHERE dimension=? AND board=?")
PAWN_INTER_QRY_START = (
    "SELECT thing, branch, tick_from, tick_to "
    "FROM pawn_interactive WHERE dimension=? AND board=?")
IMG_QRYFMT = (
    "SELECT name, path, rltile FROM img WHERE "
    "name IN ({0})")
COLOR_QRYFMT = (
    "SELECT {0} FROM color WHERE name IN ({1})".format(Color.colnstr, "{0}"))
STYLE_QRYFMT = (
    "SELECT {0} FROM style WHERE name IN ({1})".format(Style.colnstr, "{0}"))


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
                 hi_branch=0, hi_place=0, hi_portal=0, lang="eng"):
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
        self.branchgraph = igraph.Graph(directed=True)
        self.eventdict = {}
        self.lang = lang

        placeholder = (noop, ITEM_ARG_RE)
        self.effect_cbs = {}
        self.func = {
            'one': placeholder,
            'two': placeholder,
            'toggle_menu':
            (self.toggle_menu, ONE_ARG_RE),
            'hide_menu':
            (self.hide_menu, ONE_ARG_RE),
            'show_menu':
            (self.show_menu, ONE_ARG_RE),
            'make_generic_place':
            (self.make_generic_place, ONE_ARG_RE),
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
            "hi_branch": hi_branch,
            "hi_place": hi_place,
            "hi_portal": hi_portal}

    def __getattr__(self, attrn):
        if attrn in ("board", "front_board"):
            return self.game["front_board"]
        elif attrn in ("branch", "front_branch"):
            return self.game["front_branch"]
        elif attrn == "seed":
            return self.game["seed"]
        elif attrn == "tick":
            return self.game["tick"]
        elif attrn == "hi_branch":
            return self.game["hi_branch"]
        elif attrn == "hi_place":
            return self.game["hi_place"]
        elif attrn == "hi_portal":
            return self.game["hi_portal"]
        else:
            raise AttributeError(
                "RumorMill doesn't have the attribute " + attrn)

    def __setattr__(self, attrn, val):
        if attrn in ("front_board", "seed", "age", "hi_place", "hi_portal"):
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

    def xfunc(self, func, name=None):
        """Add the given function to those accessible to effects.

Optional argument name is the one the effect needs to use to access
the function.

        """
        if name is None:
            self.func[func.__name__] = func
        else:
            self.func[name] = func

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

    def handle_effect(self, effect, deck, event):
        (fun, ex) = self.func[effect._func]
        argmatch = re.match(ex, effect.arg)
        args = argmatch.groups() + (effect, deck, event)
        return fun(*args)

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

    def get_effect_decks(self, names):
        r = {}
        for name in names:
            r[name] = self.effectdeckdict[name]
        r.update(self.load_effect_decks(names))
        return r

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
            "hi_branch, hi_place, hi_portal FROM game")
        for row in self.c:
            self.game = {
                "front_board": row[0],
                "front_branch": row[1],
                "tick": row[2],
                "seed": row[3],
                "hi_branch": row[4],
                "hi_place": row[5],
                "hi_portal": row[6]}

    def load_strings(self):
        self.c.execute("SELECT stringname, language, string FROM strings")
        for row in self.c:
            rowd = dictify_row(row, ("stringname", "language", "string"))
            if rowd["stringname"] not in self.stringdict:
                self.stringdict[rowd["stringname"]] = {}
            self.stringdict[rowd["stringname"]][rowd["language"]] = rowd["string"]

    def get_dimension(self, dimn):
        if dimn not in self.dimensiondict:
            self.load_dimension(dimn)
        return self.dimensiondict[dimn]

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
        port = Portal(dimension, orig, dest)
        port.exist()
        if str(orig) not in dimension.portals_by_orign_destn:
            dimension.portals_by_orign_destn[str(orig)] = {}
        dimension.portals_by_orign_destn[str(orig)][str(dest)] = port
        return port

    def load_cards(self, names):
        r = {}
        qryfmt = load_cards_qryfmt
        qrystr = qryfmt.format(", ".join(["?"] * len(names)))
        self.c.execute(qrystr, tuple(names))
        for row in self.c:
            rowdict = dictify_row(row, Card.colns)
            effn = rowdict["effect"]
            rowdict["db"] = self
            r[effn] = Card(**rowdict)
        return r

    def get_cards(self, cardnames):
        r = {}
        unloaded = set()
        for card in cardnames:
            if card in self.carddict:
                r[card] = self.carddict[card]
            else:
                unloaded.add(card)
        r.update(self.load_cards(unloaded))
        return r

    def load_dimension(self, dimn, branches=None, tick_from=None, tick_to=None):
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
        self.c.execute(PORT_QRY_START + extrastr, valtup)
        for row in self.c:
            (orign, destn, branch, tick_from, tick_to) = row
            if not dim.have_place(orign):
                dim.make_place(orign)
            if not dim.have_place(destn):
                dim.make_place(destn)
            if not dim.have_portal(orign, destn):
                dim.make_portal(orign, destn)
            po = dim.get_portal(orign, destn)
            if not po.extant_between(branch, tick_from, tick_to):
                po.persist(branch, tick_from, tick_to)
        self.c.execute(THING_LOC_QRY_START + extrastr, valtup)
        for row in self.c:
            (thingn, branch, tick_from, tick_to, locn) = row
            if thingn not in dim.thingdict:
                dim.make_thing(thingn)
            loc = dim.get_place(locn)
            logger.debug("putting thing %s in place %s", thingn, str(loc))
            thing = dim.get_thing(thingn)
            thing.set_location(loc, branch, tick_from, tick_to)
        return dim

    def get_board(self, i, window):
        if len(window.dimension.boards) <= i or window.dimension.boards[i] is None:
            return self.load_board(i, window)
        else:
            return window.dimension.boards[i]

    def load_board(self, i, window):
        dim = window.dimension
        while len(dim.boards) <= i:
            dim.boards.append(None)
        # basic information for this board
        self.c.execute(BOARD_QRY_START, (str(dim), i))
        imgs2load = set()
        walln = None
        width = None
        height = None
        (walln, width, height) = self.c.fetchone()
        imgs2load.add(walln)
        # images for the spots
        self.c.execute(SPOT_IMG_QRY_START, (str(dim), i))
        spot_rows = self.c.fetchall()
        for row in spot_rows:
            imgs2load.add(row[4])
        # images for the pawns
        self.c.execute(PAWN_IMG_QRY_START, (str(dim), i))
        pawn_rows = self.c.fetchall()
        for row in pawn_rows:
            imgs2load.add(row[4])
        imgnames = tuple(imgs2load)
        imgs = self.load_imgs(imgs2load)
        dim.boards[i] = Board(window, i, width, height, imgs[walln])
        # actually assign images instead of just collecting the names
        for row in pawn_rows:
            (thingn, branch, tick_from, tick_to, imgn) = row
            thing = dim.thingdict[thingn]
            while len(thing.pawns) <= i:
                thing.pawns.append(None)
            thing.pawns[i] = Pawn(dim.boards[i], thing)
            thing.pawns[i].set_img(imgs[imgn], branch, tick_from, tick_to)
        # interactivity for the pawns
        self.c.execute(PAWN_INTER_QRY_START, (str(dim), i))
        for row in self.c:
            (thingn, branch, tick_from, tick_to) = row
            thing = dim.thingdict[thingn]
            thing.pawns[i].set_interactive(branch, tick_from, tick_to)
        # spots in this board
        self.c.execute(SPOT_BOARD_QRY_START, (str(dim), i))
        for row in self.c:
            (placen, branch, tick_from, tick_to, x, y) = row
            place = dim.get_place(placen)
            while len(place.spots) <= i:
                place.spots.append(None)
            if place.spots[i] is None:
                place.spots[i] = Spot(dim.boards[i], place)
            logger.debug(
                "Loaded the spot for %s. Setting its coords to "
                "(%d, %d) in branch %d from tick %d.",
                str(place), x, y, branch, tick_from)
            place.spots[i].set_coords(x, y, branch, tick_from, tick_to)
        #their images
        for row in spot_rows:
            (placen, branch, tick_from, tick_to, imgn) = row
            place = dim.get_place(placen)
            place.spots[i].set_img(imgs[imgn], branch, tick_from, tick_to)
        # their locations
        
        # interactivity for the spots
        self.c.execute(SPOT_INTER_QRY_START, (str(dim), i))
        for row in self.c:
            (placen, branch, tick_from, tick_to) = row
            place = dim.get_place(placen)
            place.spots[i].set_interactive(branch, tick_from, tick_to)
        # arrows in this board
        for port in dim.portals:
            if (
                    len(port.arrows) <= i or
                    port.arrows[i] is None):
                while len(port.arrows) <= i:
                    port.arrows.append(None)
                port.arrows[i] = Arrow(dim.boards[i], port)
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
        colorcols = ("textcolor", "fg_inactive", "fg_active", "bg_inactive", "bg_active")
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

    def get_effects(self, names):
        r = {}
        unloaded = set()
        for eff in names:
            if eff in self.effectdict:
                r[eff] = self.effectdict[eff]
            else:
                unloaded.add(eff)
        r.update(effect.load_effects(self, unloaded))
        return r

    def get_effect_decks(self, names):
        r = {}
        unloaded = set()
        for effd in names:
            if effd in self.effectdeckdict:
                r[effd] = self.effectdeckdict[effd]
            else:
                unloaded.add(effd)
        r.update(effect.load_effect_decks(self, unloaded))
        return r

    def load_window(self, name):
        self.c.execute(
            "SELECT min_width, min_height, dimension, board, arrowhead_size, "
            "arrow_width, view_left, view_bot, main_menu FROM window "
            "WHERE name=?", (name,))
        (min_width, min_height, dimn, boardi, arrowhead_size,
         arrow_width, view_left, view_bot, main_menu) = self.c.fetchone()
        stylenames = set()
        menunames = set()
        cards_needed = set()
        imgs_needed = set()
        dim = self.get_dimension(dimn)
        self.c.execute(
            "SELECT name, left, bottom, top, right, style FROM menu "
            "WHERE window=?", (name,))
        menu_rows = self.c.fetchall()
        for row in menu_rows:
            menunames.add(row[0])
            stylenames.add(row[5])
        self.c.execute(
            "SELECT i, left, right, top, bot, style, interactive, "
            "rows_shown, scrolled_to, scroll_factor FROM calendar "
            "WHERE window=? ORDER BY i", (name,))
        cal_rows = self.c.fetchall()
        for row in cal_rows:
            stylenames.add(row[5])
        self.c.execute(
            "SELECT effect_deck, left, right, top, bot, style, visible, interactive FROM hand "
            "WHERE window=?", (name,))
        hand_rows = self.c.fetchall()
        effect_decks_needed = [row[0] for row in hand_rows]
        effect_decks_loaded = self.get_effect_decks(effect_decks_needed)
        for effect_deck in effect_decks_loaded.itervalues():
            for effect in effect_deck.get_effects():
                cards_needed.add(str(effect))
        self.c.execute("SELECT effect, display_name, image, text, style FROM card WHERE effect IN ({0})".format(", ".join(["?"] * len(cards_needed))), tuple(cards_needed))
        card_rows = self.c.fetchall()
        for row in card_rows:
            stylenames.add(row[4])
            if row[2] is not None:
                imgs_needed.add(row[2])
        styles_loaded = self.get_styles(stylenames)
        imgs_loaded = self.get_imgs(imgs_needed)
        cards_loaded = {}
        for row in card_rows:
            (effn, dispn, imgn, text, style) = row
            if imgn is None:
                img = None
            else:
                img = imgs_loaded[imgn]
            cards_loaded[effn] = Card(self,
                self.effectdict[effn], dispn, 
                img, text, styles_loaded[style])
        hands_by_name = OrderedDict()
        self.c.execute(
            "SELECT menu, idx, text, on_click, closer FROM menu_item WHERE window=? AND menu IN ({0})".format(", ".join(["?"] * len(menunames))), (name,) + tuple(menunames))
        menu_item_rows = self.c.fetchall()
        return GameWindow(
            self, name, min_width, min_height, dim, boardi, arrowhead_size,
            arrow_width, view_left, view_bot, main_menu, hand_rows, cal_rows,
            menu_rows, menu_item_rows)

    # def load_branch(self, i):
    #     self.c.execute(
    #         "SELECT {0} FROM branch WHERE idx=?".format(
    #             ", ".join(Branch.colns)))
    #     rowdict = dictify_row(self.c.fetchone(), Branch.colns)
    #     rowdict["rumor"] = self
    #     b = Branch(**rowdict)
    #     p = None
    #     if i == 0:
    #         self.branches.append(b)
    #         return
    #     if rowdict["parent"] < len(self.branches):
    #         p = self.branches[rowdict["parent"]]
    #     if p is None:
    #         self.load_branch(rowdict["parent"])
    #         p = self.branches[rowdict["parent"]]
    #     p.children.append(b)
    #     while len(self.branches) <= i:
    #         self.branches.append(None)
    #     self.branches[i] = b
    #     return b
    # def load_branch(self, i):
    #     self.c.execute(
    #         "SELECT {0} FROM branch WHERE idx=?".format(
    #             ", ".join(Branch.colns)), (i,))
    #     rowdict = dictify_row(self.c.fetchone(), Branch.colns)
    #     vattdict = {}
    #     for key in ("parent", "start", "parm"):
    #         vattdict[key] = rowdict[key]
    #     for key in ("parent", "start", "parm"):
    #         if key not in (
    #                 self.branchgraph.vs[
    #                     rowdict["parent"]].attribute_names()):
    #             # I think this recursion will result in the addition
    #             # of my parent vertex
    #             self.load_branch(rowdict["parent"])
    #             break
    #     self.branchgraph.add_vertex(**vattdict)
    #     self.branchgraph.add_edge(rowdict["parent"], i)


    # def get_branch(self, i):
    #     if (
    #             i >= len(self.branches) or
    #             self.branches[i] is None):
    #         return self.load_branch[i]
    #     else:
    #         return self.branches[i]

    # def make_branch(self, i, parent, start, parm):
    #     b = Branch(self, i, parent, start, parm)
    #     if (
    #             parent >= len(self.branches) or
    #             self.branches[parent] is None):
    #         self.load_branch(parent)
    #     self.branches[parent].children.append(b)
    #     return b


def load_game(dbfn, lang="eng"):
    db = RumorMill(dbfn, lang=lang)
    db.load_game()
    db.load_strings()
    return db
