# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, dictify_row, stringlike
from random import randrange


class Effect:
    __metaclass__ = SaveableMetaclass
    """A function name and a string argument.

An effect is a function with a preselected string argument. These are
stored together under a name describing the both of them. The effect
may be fired by calling the do() method.

    """
    tables = [
        ("effect",
         {"name": "text not null",
          "func": "text not null",
          "arg": "text not null default ''"},
         ("name",),
         {},
         [])]

    def __init__(self, db, name, func, arg):
        """Return an Effect of the given name, where the given function is
called with the given argument. Register in db.effectdict."""
        self.db = db
        self.name = name
        self._func = str(func)
        self.arg = str(arg)
        self.db.effectdict[self.name] = self

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        if attrn == 'card':
            if str(self) in self.db.carddict:
                return self.db.carddict[str(self)]
            else:
                return None
        else:
            raise AttributeError("Effect instance has no attribute named " + attrn)

    def get_tabdict(self):
        return {
            "effect": {
                "name": self.name,
                "func": self._func,
                "arg": self.arg}}

    def do(self, deck=None, event=None):
        """Call the function with the argument."""
        return self.db.handle_effect(self, deck, event)


class CreatePlaceEffect(Effect):
    """Effect to make a place with the given dimension and name."""
    def __init__(self, db, dimension, name):
        arg = '{0}.{1}'.format(str(dimension), name)
        func = 'create_place'
        name = '{0}({1})'.format(func, arg)
        Effect.__init__(self, db, name, func, arg)


class CreateGenericPlaceEffect(Effect):
    """Effect to make a place in a dimension with no particular name."""
    def __init__(self, db, dimension):
        arg = str(dimension)
        func = 'create_generic_place'
        name = '{0}({1})'.format(func, arg)
        Effect.__init__(self, db, name, func, arg)


class CreateSpotEffect(Effect):
    """Effect to make a spot representing a place that already exists."""
    def __init__(self, db, dimension, placename):
        arg = "{0}.{1}".format(dimension, placename)
        func = "create_spot"
        name = "{0}({1})".format(func, arg)
        Effect.__init__(self, db, name, func, arg)


class PortalEntryEffect(Effect):
    """Effect to take a thing from a place and put it into a portal."""
    def __init__(self, db, thing, portal):
        self.thing = thing
        arg = "{0}.{1}->{2}".format(thing._dimension, thing.name, str(portal))
        name = "thing_into_portal({0})".format(arg)
        Effect.__init__(self, db, name, "thing_into_portal", arg)


class PortalProgressEffect(Effect):
    """Effect to move a thing some distance along a portal, but not out of
it."""
    def __init__(self, db, item):
        self.item = item
        arg = "{0}.{1}".format(item.dimension.name, item.name)
        name = "thing_along_portal({0})".format(arg)
        Effect.__init__(self, db, name, "thing_along_portal", arg)


class PortalExitEffect(Effect):
    """Effect to put an item into the destination of a portal it's already
in, incidentally taking it out of the portal."""
    def __init__(self, db, item):
        self.item = item
        self.portal = item.location
        arg = "{0}.{1}".format(item.dimension.name, item.name)
        name = "thing_out_of_portal({0})".format(arg)
        Effect.__init__(self, db, name, "thing_out_of_portal", arg)


class EffectWhile(Effect):
    """An effect that fires another effect in a loop, as long as the
condition passed to the constructor returns True.

The condition should be a string that is the key to a function stored
in the RumorMill.

    """
    def __init__(self, db, effect, condition):
        self.name = "while({0}):{1}".format(str(condition), str(effect))
        self.db = db
        if isinstance(effect, Effect):
            self.effect = effect
        else:
            self.effect = self.db.effectdict[str(effect)]
        self.condition = db.func[condition]
        self.db.effectdict[self.name] = self

    def do(self, deck=None, event=None):
        r = []
        while self.condition():
            r.extend(self.effect.do(deck, event))
        return r


class EffectDeck:
    """An ordered collection of Effects that may be executed in a
batch."""
    __metaclass__ = SaveableMetaclass
    draw_order = "FILO"
    reset = False
    depth = None
    tables = [
        ("effect_deck",
         {"deck": "text not null",
          "idx": "integer not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "effect": "text not null"},
         ("deck", "idx", "branch", "tick_from"),
         {"effect": ("effect", "name")},
         [])]

    def __init__(self, db, name):
        """Return an EffectDeck with the given name, containing the effects in
the given list.
        """
        self.name = name
        self.db = db
        self.cardhist = {}
        self.indefinite_cards = {}

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        if attrn in ("cards", "effects"):
            return self.get_effects()
        else:
            raise AttributeError(
                "Effect instance has no attribute named " + attrn)

    def __len__(self):
        return len(self.effects)

    def __iter__(self):
        return iter(self.effects)

    def do(self, event=None):
        """Fire all the Effects herein, in whatever order my iterator says to.

Return a list of the effects, paired with their return values."""
        fritter = iter(self)
        r = []
        for eff in fritter:
            r.append((eff, eff.do(self, event)))
        return r

    def append(self, eff):
        cont = self.get_effects()
        cont.append(eff)
        self.set_effects(cont)

    def insert(self, i, eff):
        cont = self.get_effects()
        cont.insert(i, eff)
        self.set_effects(cont)

    def remove(self, eff):
        cont = self.get_effects()
        cont.remove(eff)
        self.set_effects(cont)

    def index(self, eff):
        cont = self.get_effects()
        return cont.index(eff)

    def pop(self, i=None):
        cont = self.get_effects()
        try:
            r = cont.pop(i)
        except:
            r = cont.pop()
        self.set_effects(cont)
        return r

    def set_effects(self, effects, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.db.branch
        if tick_from is None:
            tick_from = self.db.tick
        if branch not in self.cardhist:
            self.cardhist[branch] = {}
        if branch in self.indefinite_cards:
            ifrom = self.indefinite_cards[branch]
            (icards, ito) = self.cardhist[branch][ifrom]
            if tick_to is None or tick_to > ito:
                del self.indefinite_cards[branch]
                if tick_from > ito:
                    self.cardhist[branch][ifrom] = (icards, tick_from - 1)
                else:
                    del self.cardhist[branch][ifrom]
            elif tick_to == ifrom:
                if effects == icards:
                    self.cardhist[branch][tick_from] = (effects, None)
                    self.indefinite_cards[branch] = tick_from
                    return
        if tick_to is None:
            self.indefinite_cards[branch] = tick_from
        self.cardhist[branch][tick_from] = (effects, tick_to)

    def get_effects(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.cardhist:
            return []
        for (tick_from, (cards, tick_to)) in self.cardhist[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return cards
        return []

    def index(self, that):
        return self.get_effects().index(that)


class PortalEntryEffectDeck(EffectDeck):
    reset = True

    def __init__(self, db, item, portal):
        effect = PortalEntryEffect(db, item, portal)
        EffectDeck.__init__(self, db, effect.name, [effect])


class PortalProgressEffectDeck(EffectDeck):
    reset = True

    def __init__(self, db, item):
        effect = PortalProgressEffect(db, item)
        EffectDeck.__init__(self, db, effect.name, [effect])


class PortalExitEffectDeck(EffectDeck):
    reset = True

    def __init__(self, db, item):
        effect = PortalExitEffect(db, item)
        EffectDeck.__init__(self, db, effect.name, [effect])

def make_toggle_menu_effect(db, board, menu):
    """Return an Effect that toggles the menu in the given board of the
given name."""
    if stringlike(board):
        boardname = board
    elif stringlike(board.dimension):
        boardname = board.dimension
    else:
        boardname = board.dimension.name
    if stringlike(menu):
        menuname = menu
    else:
        menuname = menu.name
    menuspec = boardname + "." + menuname
    return make_toggle_menu_effect_from_menuspec(db, menuspec)


def make_toggle_menu_effect_from_menuspec(db, menuspec):
    """Given a string consisting of a board dimension name, a dot, and a
menu name, return an Effect that toggles the menu of that name in that
board."""
    togglername = "toggle_menu_visibility({0})".format(menuspec)
    toggler = Effect(db, togglername, "toggle_menu_visibility", menuspec)
    toggler.unravel()
    return toggler


def make_toggle_calendar_effect_from_calspec(db, calspec):
    """Given a string consisting of a dimension name, a dot, and an item
name, return an Effect that toggles the calendar representing the
schedule of that item in that dimension."""
    togglername = "toggle_calendar_visibility({0})".format(calspec)
    toggler = Effect(togglername, "toggle_calendar_visibility", calspec, db)
    toggler.unravel()
    return toggler


def make_toggle_calendar_effect(db, dimname, itname):
    """Return an effect that toggles the calendar representing the
schedule of the given item in the given dimension."""
    calspec = dimname + "." + itname
    return make_toggle_calendar_effect_from_calspec(db, calspec)


def make_hide_menu_effect(db, boardname, menuname):
    """Return an effect that hides this menu in this board."""
    menuspec = boardname + "." + menuname
    return make_hide_menu_effect_from_menuspec(db, menuspec)


def make_hide_menu_effect_from_menuspec(db, menuspec):
    """Given a string consisting of a board dimension name, a dot, and a
menu name, return an effect that hides that menu in that board."""
    hidername = "hide_menu({0})".format(menuspec)
    hider = Effect(db, hidername, "hide_menu", menuspec)
    hider.unravel()
    return hider


def make_show_menu_effect(db, boardname, menuname):
    """Return an effect that shows this menu in this board."""
    menuspec = boardname + "." + menuname
    return make_show_menu_effect_from_menuspec(db, menuspec)


def make_show_menu_effect_from_menuspec(db, menuspec):
    """Given a string consisting of a board dimension name, a dot, and a
menu name, return an effect that shows that menu in that board."""
    showername = "show_menu({0})".format(menuspec)
    shower = Effect(db, showername, "show_menu", menuspec)
    shower.unravel()
    return shower


def make_hide_calendar_effect(db, dimname, itname):
    """Return an effect that hides the calendar column representing this
item in this dimension."""
    calspec = dimname + "." + itname
    return make_hide_calendar_effect_from_calspec(db, calspec)


def make_hide_calendar_effect_from_calspec(db, calspec):
    """Given a string consisting of a dimension name, a dot, and an item
name, return an effect that hides the calendar representing the
schedule for that item in that dimension."""
    hidername = "hide_calendar({0})".format(calspec)
    hider = Effect(db, hidername, "hide_calendar", calspec)
    hider.unravel()
    return hider


def make_show_calendar_effect(db, dimname, itname):
    """Return an effect that shows the calendar representing the schedule
of this item in this dimension."""
    calspec = dimname + "." + itname
    return make_show_calendar_effect_from_calspec(db, calspec)


def make_show_calendar_effect_from_calspec(db, calspec):
    """Given a string consisting of a dimension name, a dot, and an item
name, return an effect that shows the calendar representing the
schedule of that item in that dimension."""
    showername = "show_calendar({0})".format(calspec)
    shower = Effect(db, showername, "show_calendar", calspec)
    shower.unravel()
    return shower


def make_hide_all_menus_effect(db, boardname):
    """Return an effect that will hide all menus in the given board,
*unless* they are marked main_for_window."""
    hidername = "hide_menus_in_board({0})".format(boardname)
    hider = Effect(db, hidername, "hide_menus_in_board", boardname)
    hider.unravel()
    return hider


def make_hide_other_menus_effect(db, boardn, menun):
    """Return an effect that will hide all menus in the given board, save
the one given, as well as any marked main_for_window."""
    menuspec = boardn + "." + menun
    hidername = "hide_other_menus_in_board({0})".format(menuspec)
    hider = Effect(db, hidername, "hide_other_menus_in_board", menuspec)
    hider.unravel()
    return hider


def make_hide_all_calendars_effect(db, dimname):
    """Return an effect that will hide all the calendars in the given board."""
    hidername = "hide_calendars_in_board({0})".format(dimname)
    hider = Effect(db, hidername, "hide_calendars_in_board", dimname)
    hider.unravel()
    return hider


def make_hide_other_calendars_effect(db, dimname, itname):
    """Return an effect that will hide all calendars in this board, apart
from this one."""
    calspec = dimname + "." + itname
    hidername = "hide_other_calendars_in_board({0})".format(calspec)
    hider = Effect(db, hidername, "hide_other_calendars_in_board", calspec)
    hider.unravel()
    return hider


def make_show_only_menu_effect_deck(db, boardname, menuname):
    """Return an effect that will show the given menu in the given board,
but hide all the others (apart from main_for_window)."""
    hider = make_hide_all_menus_effect(db, boardname)
    shower = make_show_menu_effect(db, boardname, menuname)
    deckname = "show_only_menu({0}.{1})".format(boardname, menuname)
    deck = EffectDeck(db, deckname, [hider, shower])
    deck.unravel()
    return deck


def make_show_only_calendar_effect_deck(db, dimname, itname):
    """Return an effect that will hide all calendars in the given board,
except for the one for this item."""
    hider = make_hide_all_calendars_effect(db, dimname)
    shower = make_show_calendar_effect(db, dimname, itname)
    deckname = "show_only_calendar({0}.{1})".format(dimname, itname)
    deck = EffectDeck(db, deckname, [hider, shower])
    deck.unravel()
    return deck


def make_menu_toggler(db, board, menu):
    """Return an effect that will hide all non-main menus in the board,
except for this one, which will be hidden if visible, or shown if
invisible."""
    if stringlike(board):
        boardname = board
    else:
        if stringlike(board.dimension):
            boardname = board.dimension
        else:
            boardname = board.dimension.name
    if stringlike(menu):
        menuname = menu
    else:
        menuname = menu.name
    hide_effect = make_hide_other_menus_effect(db, boardname, menuname)
    toggle_effect = make_toggle_menu_effect(db, boardname, menuname)
    deckname = "toggle_menu_visibility({0}.{1})".format(boardname, menuname)
    deck = EffectDeck(db, deckname, [hide_effect, toggle_effect])
    deck.unravel()
    return deck


def make_calendar_toggler(dimname, itname, db):
    """Return an effect that will hide all calendar columns in the board,
except for this one, which will be hidden if visible, or shown if
invisible."""
    hide_effect = make_hide_all_calendars_effect(db, dimname)
    toggle_effect = make_toggle_calendar_effect(db, dimname, itname)
    deckname = "toggle_calendar_visibility({0}.{1})".format(dimname, itname)
    deck = EffectDeck(db, deckname, [hide_effect, toggle_effect])
    deck.unravel()
    return deck
