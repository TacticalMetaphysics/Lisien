# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""The database backend, with dictionaries of loaded objects.

This is a caching database connector. There are dictionaries for all
objects that can be loaded from the database.

This module does not contain the code used to generate
SQL. That's in :class:`~LiSE.util.SaveableMetaclass`.

"""
import os
from os.path import sep
import re
import sqlite3

from gui.board import (
    Board,
    Spot,
    Pawn)
from gui.charsheet import CharSheet, CharSheetView
from gui.menu import Menu
from gui.img import Img
from model import (
    Character,
    Place,
    Portal,
    Timestream,
    Thing)
from model.event import Implicator
from util import (
    passthru,
    TimestreamException,
    SaveableMetaclass,
    int2pytype,
    pytype2int,
    PlaceBone,
    schemata,
    saveables,
    saveable_classes,
    Skeleton)
from kivy.atlas import Atlas
from LiSE import __path__


def noop(*args, **kwargs):
    """Do nothing."""
    pass


###
# These regexes serve to parse certain database records that represent
# function calls.
#
# Mainly, that means menu items and Effects.
###
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
NEW_THING_RE = re.compile(
    "new_thing\((.+)+\)")
NEW_PLACE_RE = re.compile(
    "new_place\((.+)\)")
CHARACTER_RE = re.compile(
    "character\((.+)\)")


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
    __metaclass__ = SaveableMetaclass
    tables = [
        ("globals", {
            "columns": {
                "key": "text not null",
                "type": "integer not null default 3",
                "value": "text"},
            "primary_key": ("key",),
            "checks": ["type in ({})".format(", ".join([
                str(typ) for typ in int2pytype]))]
        }),
        ("strings", {
            "columns": {
                "stringname": "text not null",
                "language": "text not null default 'eng'",
                "string": "text not null"},
            "primary_key": ("stringname", "language")})]
    globs = ("branch", "tick", "observer", "observed", "host")
    working_dicts = [
        "boardhand_d",
        "calendar_d",
        "cause_d",
        "color_d",
        "board_d",
        "effect_d",
        "img_d",
        "texture_d",
        "textag_d",
        "menu_d",
        "menuitem_d",
        "style_d",
        "event_d",
        "character_d",
        "facade_d"]
    """The names of dictionaries where I keep objects after
    instantiation.

    """

    def __setattr__(self, attrn, val):
        if attrn == "branch" and hasattr(self, 'branch'):
            self.upd_branch(val)
        elif attrn == "tick" and hasattr(self, 'tick'):
            self.upd_tick(val)
        elif attrn == "language" and hasattr(self, 'language'):
            self.upd_lang(val)
        else:
            super(Closet, self).__setattr__(attrn, val)

    def __init__(self, connector, lgettext=passthru,
                 USE_KIVY=False, **kwargs):
        """Initialize a Closet for the given connector and path.

        With USE_KIVY, I will use the kivybits module to load images.

        """
        self.connector = connector
        self.skeleton = Skeleton({"place": {}})

        self.c = self.connector.cursor()
        self.branch_listeners = []
        self.tick_listeners = []
        self.time_listeners = []

        for glob in self.globs:
            setattr(self, glob, self.get_global(glob))

        self.lisepath = __path__
        self.lgettext = lgettext

        for wd in self.working_dicts:
            setattr(self, wd, dict())

        if USE_KIVY:
            from gui.kivybits import (
                load_textures,
                load_textures_tagged,
                load_all_textures)
            self.load_textures = lambda names: load_textures(
                self.c, self.skeleton, self.texture_d,
                self.textag_d, names)
            self.load_all_textures = lambda: load_all_textures(
                self.c, self.skeleton, self.texture_d, self.textag_d)
            self.load_textures_tagged = lambda tags: load_textures_tagged(
                self.c, self.skeleton, self.texture_d, self.textag_d,
                tags)
            self.USE_KIVY = True

        self.timestream = Timestream(self)
        self.time_travel_history = []
        self.game_speed = 1
        self.updating = False
        placeholder = (noop, ITEM_ARG_RE)
        self.menu_cbs = {
            'play_speed':
            (self.play_speed, ONE_ARG_RE),
            'back_to_start':
            (self.back_to_start, ''),
            'noop': placeholder,
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
            'mi_connect_portal':
            (self.mi_connect_portal, ""),
            'mi_show_popup':
            (self.mi_show_popup, ONE_ARG_RE)}

    def __del__(self):
        """Try to write changes to disk before dying.

        """
        self.c.close()
        self.connector.commit()
        self.connector.close()

    def select_class_all(self, cls):
        """Return a Skeleton with all records for all tables defined in the
        class."""
        td = {}
        for tabn in cls.tablenames:
            bonetype = cls.bonetypes[tabn]
            bone = bonetype._null()
            td[tabn] = [bone]
        r = cls._select_skeleton(self.c, td)
        for bone in r.iterbones():
            self.upbone(bone)
        return r

    def select_keybones(self, kbs):
        """Take an iterable of "keybones," being regular bones whose key
        fields are to be matched with other bones loaded from
        disc. Return a skeleton of bones that match, keyed the same
        way the database is.

        """
        clasname2clas = {}
        clas_qd = {}
        for kb in kbs:
            if kb.cls.__name__ not in clas_qd:
                clas_qd[kb.cls.__name__] = {}
                clasname2clas[kb.cls.__name__] = kb.cls
            if kb.__class__.__name__ not in clas_qd[kb.cls.__name__]:
                clas_qd[kb.cls.__name__][kb.__class__.__name__] = set()
            clas_qd[kb.cls.__name__][kb.__class__.__name__].add(kb)
        r = Skeleton()
        for clasn in clas_qd:
            clas = clasname2clas[clasn]
            r.update(clas._select_skeleton(self.c, clas_qd[clasn]))
        for bone in r.iterbones():
            self.upbone(bone)
        return r

    def update_keybones(self, kbs):
        """Update my skeleton with database records matching at least one of
        the keybones."""
        self.skeleton.update(self.select_keybones(kbs))

    def update_keybone(self, kb):
        """Query the database for records matching the given keybone. Make
        bones for them, and add those to the appropriate place in my
        skeleton, overwriting anything already there.

        """
        self.update_keybones([kb])

    def upd_branch(self, b):
        """Set the active branch, alerting any branch_listeners"""
        super(Closet, self).__setattr__('branch', b)
        self.upd_time(b, self.tick)
        for listener in self.branch_listeners:
            listener(b)

    def upd_tick(self, t):
        """Set the current tick, alerting any tick_listeners"""
        super(Closet, self).__setattr__('tick', t)
        self.upd_time(self.branch, t)
        for listener in self.tick_listeners:
            listener(t)

    def upd_time(self, b, t):
        """Set the current branch and tick, alerting any time_listeners"""
        for listener in self.time_listeners:
            listener(b, t)

    def upd_lang(self, l):
        """Set the current language, alerting any lang_listeners"""
        for listener in self.lang_listeners:
            listener(self, l)
        super(Closet, self).__setattr__('language', l)

    def get_global(self, key):
        self.c.execute("SELECT type, value FROM globals WHERE key=?;", (key,))
        (typ_i, val_s) = self.c.fetchone()
        return int2pytype[typ_i](val_s)

    def set_global(self, key, value):
        self.c.execute("DELETE FROM globals WHERE key=?;", (key,))
        self.c.execute(
            "INSERT INTO globals (key, type, value) VALUES (?, ?, ?);",
            (key, pytype2int[type(value)], unicode(value)))

    def get_text(self, strname):
        """Get the string of the given name in the language set at startup."""
        if strname is None:
            return ""
        elif strname == "@branch":
            return unicode(self.branch)
        elif strname == "@tick":
            return unicode(self.tick)
        else:
            return unicode(self.lgettext(strname))

    def save_game(self):
        """Save all pending changes to disc."""
        # save globals first
        for glob in self.globs:
            self.set_global(glob, getattr(self, glob))
        # find out what's changed since last checkpoint
        to_save = self.skeleton - self.old_skeleton
        to_delete = self.old_skeleton - self.skeleton
        # save each class in turn
        for clas in saveable_classes:
            assert(len(clas.tablenames) > 0)
            for tabname in clas.tablenames:
                if tabname in to_delete:
                    clas._delete_keybones_table(
                        self.c, to_delete[tabname].iterbones(), tabname)
                if tabname in to_save:
                    clas._delete_keybones_table(
                        self.c, to_save[tabname].iterbones(), tabname)
                    try:
                        clas._insert_bones_table(
                            self.c, to_save[tabname].iterbones(), tabname)
                    except ValueError:
                        pass
        # remember how things are now
        self.checkpoint()

    def load_strings(self):
        """Load all strings available."""
        self.update_keybone(self.bonetypes["strings"]._null())

    def make_generic_place(self, character):
        """Make a place hosted by the given character, and give it a boring
        name.

        """
        character = self.get_character(character)
        placen = "generic_place_{0}".format(len(character.graph.vs))
        return character.make_place(placen)

    def make_generic_thing(self, character, host, location,
                           branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        character = self.get_character(character)
        charn = unicode(character)
        hostn = unicode(host)
        locn = unicode(location)
        if charn not in self.skeleton[u"thing"]:
            self.skeleton[u"thing"][charn] = {}
        thingn = u"generic_thing_{}".format(
            len(self.skeleton[u"thing"][charn]))
        thing_core_bone = Thing.bonetypes["thing"](
            character=charn,
            name=thingn,
            host=hostn)
        self.set_bone(thing_core_bone)
        thing_loc_bone = Thing.bonetypes["thing_loc"](
            character=charn,
            name=thingn,
            branch=branch,
            tick=tick,
            location=locn)
        self.set_bone(thing_loc_bone)
        return character.make_thing(thingn)

    def load_charsheet(self, character):
        """Return a CharSheetView displaying the CharSheet for the character
        specified, perhaps loading it if necessary."""
        def gen_keybones():
            for bonetype in CharSheet.bonetypes.itervalues():
                yield bonetype._null()._replace(character=character)
        # if the character is not loaded yet, make it so
        character = unicode(self.get_character(character))
        self.update_keybones(gen_keybones())
        return CharSheetView(character=self.get_character(character))

    def load_characters(self, names):
        """Load all the named characters and return them in a dict."""
        char_stat_bones = [
            Character.bonetypes["character_stat"]._null()._replace(
                character=name) for name in names]
        portal_bones = [
            Portal.bonetypes["portal"]._null()._replace(character=name)
            for name in names]
        portal_loc_bones = [
            Portal.bonetypes["portal_loc"]._null()._replace(character=name)
            for name in names]
        portal_stat_bones = [
            Portal.bonetypes["portal_stat"]._null()._replace(character=name)
            for name in names]
        thing_bones = [
            Thing.bonetypes["thing"]._null()._replace(character=name)
            for name in names]
        thing_loc_bones = [
            Thing.bonetypes["thing_loc"]._null()._replace(character=name)
            for name in names]
        thing_stat_bones = [
            Thing.bonetypes["thing_stat"]._null()._replace(character=name)
            for name in names]
        place_stat_bones = [
            Place.bonetypes["place_stat"]._null()._replace(character=name)
            for name in names]
        self.update_keybones(
            char_stat_bones + portal_bones + portal_loc_bones +
            portal_stat_bones + thing_bones + thing_loc_bones +
            thing_stat_bones + place_stat_bones)
        r = {}
        for name in names:
            char = Character(self, name)
            char.update()
            r[name] = char
            self.character_d[name] = char
        return r

    def get_characters(self, names):
        """Return the named characters in a dict. Load them as needed."""
        r = {}
        unhad = set()
        for name in names:
            if isinstance(name, Character):
                r[str(name)] = name
            elif name in self.character_d:
                r[name] = self.character_d[name]
            else:
                unhad.add(name)
        if len(unhad) > 0:
            r.update(self.load_characters(names))
        return r

    def get_character(self, name):
        """Return the named character. Load it if needed.

        When supplied with a Character object, this will simply return
        it, so you may use it to *ensure* that an object is a
        Character.

        """
        if isinstance(name, Character):
            return name
        return self.get_characters([str(name)])[str(name)]

    def get_effects(self, names):
        """Return the named effects in a dict"""
        r = {}
        for name in names:
            r[name] = Implicator.make_effect(name)
        return r

    def get_effect(self, name):
        """Return the named effect in a dict"""
        return self.get_effects([name])[name]

    def get_causes(self, names):
        """Return the named causes in a dict"""
        r = {}
        for name in names:
            r[name] = Implicator.make_cause(name)
        return r

    def get_cause(self, cause):
        """Return the named cause in a dict"""
        return self.get_causes([cause])[cause]

    def load_board(self, observer, observed, host):
        """Load and return a graphical board widget displaying the contents of
        the host that are parts of the observed character, as seen by
        the observer character.

        """
        obsrvr = unicode(observer)
        obsrvd = unicode(observed)
        hst = unicode(host)
        keybones = [
            Board.bonetypes["board"]._null()._replace(
                observer=obsrvr, observed=obsrvd, host=hst),
            Spot.bonetypes["spot"]._null()._replace(
                host=hst),
            Spot.bonetypes["spot_coords"]._null()._replace(
                host=hst),
            Pawn.bonetypes["pawn"]._null()._replace(
                host=hst)]
        self.update_keybones(keybones)
        return self.get_board(observer, observed, host)

    def get_board(self, observer, observed, host):
        """Return a graphical board widget displaying the contents of the host
        that are parts of the observed character, as seen by the
        observer character. Load it if needed.

        """
        observer = self.get_character(observer)
        observed = self.get_character(observed)
        host = self.get_character(host)
        facade = observed.get_facade(observer)
        return Board(facade=facade, host=host)

    def get_place(self, char, placen):
        """Get a place from a character"""
        return self.get_character(char).get_place(placen)

    def get_portal(self, char, name):
        """Get a portal from a character"""
        return self.get_character(char).get_portal(name)

    def get_thing(self, char, name):
        """Get a thing from a character"""
        return self.get_character(char).get_thing(name)

    def get_textures(self, imgnames):
        """Return a dictionary full of textures by the given names, loading
        them as needed."""
        r = {}
        unloaded = set()
        for imgn in imgnames:
            if imgn in self.texture_d:
                r[imgn] = self.texture_d[imgn]
            else:
                unloaded.add(imgn)
        if len(unloaded) > 0:
            r.update(self.load_textures(unloaded))
        return r

    def get_texture(self, imgn):
        """Return the texture by the given name"""
        return self.get_textures([imgn])[imgn]

    def load_menus(self, names):
        """Return a dictionary full of menus by the given names, loading them
        as needed."""
        r = {}
        for name in names:
            r[name] = self.load_menu(name)
        return r

    def load_menu(self, name):
        """Load and return the named menu"""
        self.load_menu_items(name)
        return Menu(closet=self, name=name)

    def load_menu_items(self, menu):
        """Load a dictionary of menu item infos. Don't return anything."""
        self.update_keybone(Menu.bonetypes["menu_item"]._null()._replace(
            menu=menu))

    def load_timestream(self):
        """Load and return the timestream"""
        self.skeleton.update(self.select_class_all(Timestream))
        self.timestream = Timestream(self)

    def time_travel_menu_item(self, mi, branch, tick):
        """Tiny wrapper for ``time_travel``"""
        return self.time_travel(branch, tick)

    def time_travel(self, branch, tick):
        """"Set the diegetic time to the given branch and tick.

        If the branch is one higher than the known highest branch,
        create it.

        """
        assert branch <= self.timestream.hi_branch + 1, (
            "Tried to travel to too high a branch")
        if branch == self.timestream.hi_branch + 1:
            self.new_branch(self.branch, branch, tick)
        # will need to take other games-stuff into account than the
        # thing_location
        if tick < 0:
            raise TimestreamException("Tick before start of time")
        # make it more general
        mintick = self.timestream.min_tick(branch, "thing_loc")
        if tick < mintick:
            raise TimestreamException("Tick before start of branch")
        if branch < 0:
            raise TimestreamException("Branch can't be less than zero")
        self.time_travel_history.append((self.branch, self.tick))
        if tick > self.timestream.hi_tick:
            self.timestream.hi_tick = tick
        self.branch = branch
        self.tick = tick

    def increment_branch(self, branches=1):
        """Go to the next higher branch. Might result in the creation of said
        branch."""
        b = self.branch + int(branches)
        mb = self.timestream.max_branch()
        if b > mb:
            # I dunno where you THOUGHT you were going
            self.new_branch(self.branch, self.branch+1, self.tick)
            return self.branch + 1
        else:
            return b

    def new_branch(self, parent, child, tick):
        """Copy records from the parent branch to the child, starting at
        tick."""
        assert(parent != child)
        new_bones = set()
        for character in self.character_d.itervalues():
            for bone in character.new_branch(parent, child, tick):
                new_bones.add(bone)
        for observer in self.board_d:
            for observed in self.board_d[observer]:
                for host in self.board_d[observer][observed]:
                    for bone in self.board_d[observer][observed][
                            host].new_branch(parent, child, tick):
                        new_bones.add(bone)
        self.skeleton["timestream"][child] = Timestream.bonetype(
            branch=child, parent=parent, tick=tick)
        self.timestream.hi_branch += 1
        assert(self.timestream.hi_branch == child)
        for bone in new_bones:
            self.set_bone(bone)

    def time_travel_inc_tick(self, ticks=1):
        """Go to the next tick on the same branch"""
        self.time_travel(self.branch, self.tick+ticks)

    def time_travel_inc_branch(self, branches=1):
        """Go to the next branch on the same tick"""
        self.increment_branch(branches)
        self.time_travel(self.branch+branches, self.tick)

    def go(self, nope=None):
        """Pass time"""
        self.updating = True

    def stop(self, nope=None):
        """Stop time"""
        self.updating = False

    def set_speed(self, newspeed):
        """Change the rate of time passage"""
        self.game_speed = newspeed

    def play_speed(self, mi, n):
        """Set the rate of time passage, and start it passing"""
        self.game_speed = int(n)
        self.updating = True

    def back_to_start(self, nope):
        """Stop time and go back to the beginning"""
        self.stop()
        self.time_travel(self.branch, 0)

    def end_game(self):
        """Save everything and close the connection"""
        self.c.close()
        self.connector.commit()
        self.connector.close()

    def checkpoint(self):
        """Store an image of the skeleton in its present state, to compare
        later"""
        self.old_skeleton = self.skeleton.deepcopy()

    def upbone(self, bone):
        """Raise the timestream's hi_branch and hi_tick if the bone has new
        values for them"""
        if (
                hasattr(bone, "branch") and
                bone.branch > self.timestream.hi_branch):
            self.timestream.hi_branch = bone.branch
        if (
                hasattr(bone, "tick") and
                bone.tick > self.timestream.hi_tick):
            self.timestream.hi_tick = bone.tick
        if (
                hasattr(bone, "tick_from") and
                bone.tick_from > self.timestream.hi_tick):
            self.timestream.hi_tick = bone.tick_from
        if (
                hasattr(bone, "tick_to") and
                bone.tick_to > self.timestream.hi_tick):
            self.timestream.hi_tick = bone.tick_to

    def mi_show_popup(self, mi, name):
        """Get the root LiSELayout to show a popup of a kind appropriate to
        the name given."""
        root = mi.get_root_window().children[0]
        new_thing_match = re.match(NEW_THING_RE, name)
        if new_thing_match:
            return root.show_pawn_picker(
                new_thing_match.groups()[0].split(", "))
        new_place_match = re.match(NEW_PLACE_RE, name)
        if new_place_match:
            return root.show_spot_picker(
                new_place_match.groups()[0].split(", "))
        character_match = re.match(CHARACTER_RE, name)
        if character_match:
            argstr = character_match.groups()[0]
            if len(argstr) == 0:
                return root.show_charsheet_maker()

    def mi_connect_portal(self, mi):
        """Get the root LiSELayout to make an Arrow, representing a Portal."""
        mi.get_root_window().children[0].make_arrow()

    def register_text_listener(self, stringn, listener):
        """Notify the listener when the string called ``stringn`` changes its
        content."""
        if stringn == "@branch" and listener not in self.branch_listeners:
            self.branch_listeners.append(listener)
        elif stringn == "@tick" and listener not in self.tick_listeners:
            self.tick_listeners.append(listener)
        elif (
                stringn[0] == "@" and
                stringn[1:] in self.skeleton["strings"] and
                listener not in self.skeleton["strings"][
                    stringn[1:]].set_listeners):
            self.skeleton["strings"][stringn[1:]].register_set_listener(
                listener)
        else:
            return self.lgettext(stringn)

    def unregister_text_listener(self, stringn, listener):
        try:
            if stringn == "@branch":
                return self.unregister_branch_listener(listener)
            elif stringn == "@tick":
                return self.unregister_tick_listener(listener)
            else:
                self.skeleton["strings"][
                    stringn[1:]].set_listeners.remove(listener)
        except (KeyError, ValueError):
            raise ValueError("Listener isn't registered")

    def register_time_listener(self, listener):
        """Listener will be called when ``branch`` or ``tick`` changes"""
        if listener not in self.time_listeners:
            self.time_listeners.append(listener)

    def unregister_time_listener(self, listener):
        try:
            self.time_listeners.remove(listener)
        except ValueError:
            raise ValueError("Listener isn't registered")

    def register_branch_listener(self, listener):
        """Listener will be called when ``branch`` changes"""
        if listener not in self.branch_listeners:
            self.branch_listeners.append(listener)

    def unregister_branch_listener(self, listener):
        try:
            self.branch_listeners.remove(listener)
        except ValueError:
            raise ValueError("Listener isn't registered")

    def register_tick_listener(self, listener):
        """Listener will be called when ``tick`` changes"""
        if listener not in self.tick_listeners:
            self.tick_listeners.append(listener)

    def unregister_tick_listener(self, listener):
        try:
            self.tick_listeners.remove(listener)
        except ValueError:
            raise ValueError("Listener isn't registered")

    def register_img_listener(self, imgn, listener):
        try:
            skel = self.skeleton[u"img"][imgn]
        except KeyError:
            raise KeyError("Image unknown: {}".format(imgn))
        if listener not in skel.set_listeners:
            skel.set_listeners.append(listener)

    def unregister_img_listener(self, imgn, listener):
        try:
            skel = self.skeleton[u"img"][imgn]
        except KeyError:
            raise KeyError("Image unknown: {}".format(imgn))
        try:
            skel.set_listeners.remove(listener)
        except ValueError:
            raise ValueError("Listener isn't registered")

    def load_img_metadata(self):
        """Load all the records to do with img paths and tags and so forth."""
        self.skeleton.update(self.select_class_all(Img))

    def query_place(self, update=True):
        """Query the 'place' view, resulting in an up-to-date record of what
        places exist in the gameworld as it exists in the
        database.

        """
        self.c.execute("SELECT host, place, branch, tick FROM place;")
        if not update or u"place" not in self.skeleton:
            # empty it out if it exists, create it if it doesn't
            self.skeleton[u"place"] = {}
        for (host, place, branch, tick) in self.c:
            self.set_bone(PlaceBone(
                host=host,
                place=place,
                branch=branch,
                tick=tick))

    def have_place_bone(self, host, place, branch=None, tick=None):
        if branch is None:
            branch = self.branch
        if tick is None:
            tick = self.tick
        try:
            return self.skeleton[u"place"][host][place][
                branch].value_during(tick) is not None
        except (KeyError, IndexError):
            return False

    def set_bone(self, bone):
        """Take a bone of arbitrary type and put it in the right place in the
        skeleton.

        Additionally, if the bone is of a kind that may implicitly
        define a place, see if the place is a new one. If so, insert a
        PlaceBone to describe it.

        """
        def init_keys(skel, keylst):
            for key in keylst:
                if key not in skel:
                    skel[key] = {}
                skel = skel[key]
            return skel

        def set_place_maybe(host, place, branch, tick):
            if not self.have_place_bone(host, place, branch, tick):
                self.set_bone(PlaceBone(
                    host=host, place=place, branch=branch, tick=tick))
            else:
                print("Already knew about {}".format(place))

        def have_charsheet_type_bone(character, idx, type):
            try:
                return self.skeleton[u"character_sheet_item_type"][
                    character][idx] is not None
            except (KeyError, IndexError):
                return False

        def set_cstype_maybe(character, idx, type):
            if not have_charsheet_type_bone:
                self.set_bone(CharSheet.bonetype(
                    character=character,
                    idx=idx,
                    type=type))

        if isinstance(bone, PlaceBone):
            init_keys(
                self.skeleton[u"place"],
                [bone.host, bone.place, bone.branch])
            self.skeleton[u"place"][bone.host][bone.place][
                bone.branch][bone.tick] = bone
            return

        # Some bones implicitly declare a new place
        if isinstance(bone, Thing.bonetypes[u"thing_loc"]):
            core = self.skeleton[u"thing"][bone.character][bone.name]
            set_place_maybe(core.host, bone.location, bone.branch, bone.tick)
        elif isinstance(bone, Thing.bonetypes[u"thing_loc_facade"]):
            core = self.skeleton[u"thing"][bone.observed][bone.name]
            set_place_maybe(core.host, bone.location, bone.branch, bone.tick)
        elif isinstance(bone, Portal.bonetypes[u"portal_loc"]):
            core = self.skeleton[u"portal"][bone.character][bone.name]
            for loc in (bone.origin, bone.destination):
                set_place_maybe(core.host, loc, bone.branch, bone.tick)
        elif isinstance(bone, Portal.bonetypes[u"portal_stat_facade"]):
            core = self.skeleton[u"portal"][bone.observed][bone.name]
            for loc in (bone.origin, bone.destination):
                set_place_maybe(core.host, loc, bone.branch, bone.tick)
        elif isinstance(bone, Place.bonetypes[u"place_stat"]):
            set_place_maybe(bone.host, bone.name, bone.branch, bone.tick)
        elif isinstance(bone, Spot.bonetypes[u"spot"]):
            set_place_maybe(bone.host, bone.place, bone.branch, bone.tick)
        elif isinstance(bone, Spot.bonetypes[u"spot_coords"]):
            set_place_maybe(bone.host, bone.place, bone.branch, bone.tick)
        elif type(bone) is CharSheet.bonetype:
            pass
        elif type(bone) in CharSheet.bonetypes.values():
            set_cstype_maybe(bone.character, bone.idx, bone.type)

        keynames = bone.cls.keynames[bone._name]
        keys = [getattr(bone, keyn) for keyn in keynames[:-1]]
        skel = init_keys(self.skeleton[bone._name], keys)
        final_key = getattr(bone, keynames[-1])
        skel[final_key] = bone


def defaults(c):
    from LiSE.data import whole_imgrows
    c.executemany(
        "INSERT INTO img (name, path, off_x, off_y) VALUES (?, ?, ?, ?);",
        whole_imgrows)
    from LiSE.data import globs
    c.executemany(
        "INSERT INTO globals (key, type, value) VALUES (?, ?, ?);",
        globs)
    c.execute(
        "INSERT INTO timestream (branch, parent) VALUES (?, ?);",
        (0, 0))
    from LiSE.data import stackhs
    for (height, names) in stackhs:
        qrystr = (
            "UPDATE img SET stacking_height=? WHERE name IN ({});".format(
                ", ".join(["?"] * len(names))))
        qrytup = (height,) + names
        c.execute(qrystr, qrytup)
    from LiSE.data import offxs
    for (offx, names) in offxs:
        qrystr = (
            "UPDATE img SET off_x=? WHERE name IN ({});".format(
                ", ".join(["?"] * len(names))))
        qrytup = (offx,) + names
        c.execute(qrystr, qrytup)
    from LiSE.data import offys
    for (offy, names) in offys:
        qrystr = (
            "UPDATE img SET off_y=? WHERE name IN ({});".format(
                ", ".join(["?"] * len(names))))
        qrytup = (offy,) + names
        c.execute(qrystr, qrytup)
    c.execute("UPDATE img SET off_x=4, off_y=8 "
              "WHERE name IN (SELECT img FROM img_tag WHERE tag=?)",
              ('pawn',))
    from LiSE.data import boards
    for (obsrvr, obsrvd, hst) in boards:
        c.execute(
            "INSERT INTO board (observer, observed, host) VALUES (?, ?, ?);",
            (obsrvr, obsrvd, hst))
    from LiSE.data import things
    for character in things:
        for thing in things[character]:
            c.execute(
                "INSERT INTO thing (character, name, host) VALUES (?, ?, ?);",
                (character, thing, things[character][thing]["host"]))
            c.execute(
                "INSERT INTO thing_loc (character, name, location) "
                "VALUES (?, ?, ?);",
                (character, thing, things[character][thing]["location"]))
    from LiSE.data import reciprocal_portals
    for (orig, dest) in reciprocal_portals:
        name1 = "{}->{}".format(orig, dest)
        name2 = "{}->{}".format(dest, orig)
        c.executemany(
            "INSERT INTO portal (name) VALUES (?);",
            [(name1,), (name2,)])
        c.executemany(
            "INSERT INTO portal_loc (name, origin, destination) VALUES "
            "(?, ?, ?);", [(name1, orig, dest), (name2, dest, orig)])
    from LiSE.data import one_way_portals
    for (orig, dest) in one_way_portals:
        name = "{}->{}".format(orig, dest)
        c.execute(
            "INSERT INTO portal (name) VALUES (?);",
            (name,))
        c.execute(
            "INSERT INTO portal_loc (name, origin, destination) "
            "VALUES (?, ?, ?);", (name, orig, dest))
    from LiSE.data import charsheet_items
    for character in charsheet_items:
        i = 0
        for (typ, key0) in charsheet_items[character]:
            c.execute(
                "INSERT INTO charsheet_item (character, type, idx, key0) "
                "VALUES (?, ?, ?, ?);", (character, typ, i, key0))
            i += 1
    from LiSE.data import spot_coords
    for (place, x, y) in spot_coords:
        c.execute(
            "INSERT INTO spot (place) VALUES (?);",
            (place,))
    c.executemany(
        "INSERT INTO spot_coords (place, x, y) VALUES (?, ?, ?);",
        spot_coords)
    from LiSE.data import pawns
    for observed in pawns:
        for (thing, layers) in pawns[observed].iteritems():
            i = 0
            for layer in layers:
                c.execute(
                    "INSERT INTO pawn (observed, thing, layer, img) "
                    "VALUES (?, ?, ?, ?);",
                    (observed, thing, i, layer))
                i += 1


def mkdb(DB_NAME, lisepath):
    img_qrystr = (
        "INSERT INTO img (name, path) "
        "VALUES (?, ?);")
    tag_qrystr = (
        "INSERT INTO img_tag (img, tag) VALUES (?, ?);")

    def ins_atlas(curs, path, qualify=False, tags=[]):
        lass = Atlas(path)
        atlaspath = "atlas://{}".format(path[:-6])
        atlasn = path.split(sep)[-1][:-6]
        for tilen in lass.textures.iterkeys():
            imgn = atlasn + '.' + tilen if qualify else tilen
            curs.execute(img_qrystr, (
                imgn, "{}/{}".format(atlaspath, tilen)))
            for tag in tags:
                print (imgn, tag)
                curs.execute(tag_qrystr, (imgn, tag))

    def ins_atlas_dir(curs, dirname, qualify=False, tags=[]):
        for fn in os.listdir(dirname):
            if fn[-5:] == 'atlas':
                path = dirname + sep + fn
                ins_atlas(curs, path, qualify, [fn[:-6]] + tags)

    try:
        os.remove(DB_NAME)
    except OSError:
        pass
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    done = set()
    while saveables != []:
        (demands, provides, prelude,
         tablenames, postlude) = saveables.pop(0)
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
        prelude_todo = list(prelude)
        while prelude_todo != []:
            pre = prelude_todo.pop()
            if isinstance(pre, tuple):
                c.execute(*pre)
            else:
                c.execute(pre)
        if len(tablenames) == 0:
            for post in postlude:
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
            continue
        prelude_todo = list(prelude)
        try:
            while prelude_todo != []:
                pre = prelude_todo.pop()
                if isinstance(pre, tuple):
                    c.execute(*pre)
                else:
                    c.execute(pre)
        except sqlite3.OperationalError as e:
            saveables.append(
                (demands, provides, prelude_todo, tablenames, postlude))
            continue
        breakout = False
        tables_todo = list(tablenames)
        while tables_todo != []:
            tn = tables_todo.pop(0)
            print(tn)
            c.execute(schemata[tn])
            done.add(tn)
        if breakout:
            saveables.append(
                (demands, provides, prelude_todo, tables_todo, postlude))
            continue
        postlude_todo = list(postlude)
        try:
            while postlude_todo != []:
                post = postlude_todo.pop()
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
        except sqlite3.OperationalError as e:
            print("OperationalError during postlude from {0}.".format(tn))
            print(e)
            saveables.append(
                (demands, provides, prelude_todo, tables_todo, postlude_todo))
            continue
        done.update(provides)

    print("indexing the RLTiles")
    ins_atlas_dir(
        c, "LiSE/gui/assets/rltiles/hominid", True,
        ['hominid', 'rltile', 'pawn'])

    print("indexing Pixel City")
    ins_atlas(c, "LiSE/gui/assets/pixel_city.atlas", False,
              ['spot', 'pixel_city'])

    print("inserting default values")
    defaults(c)

    conn.commit()
    return conn


def load_closet(dbfn, lgettext=None, kivy=False):
    """Construct a ``Closet`` connected to the given database file. Use
    the LiSE library in the path given.

    If ``kivy`` == True, the closet will be able to load textures using
    Kivy's Image widget.

    Strings will be loaded for the language ``lang``. Use language codes
    from ISO 639-2, default "eng".

    """
    conn = sqlite3.connect(dbfn)
    kwargs = {
        "connector": conn,
        "USE_KIVY": kivy}
    if lgettext is not None:
        kwargs["lgettext"] = lgettext
    r = Closet(**kwargs)
    r.load_timestream()
    return r
