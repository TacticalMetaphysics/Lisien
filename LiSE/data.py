# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Module for extract, transform, load, as well as some constants"""


def _(arg):
    return arg

THING_LOC_TAB = 0
THING_STAT_TAB = 1
PLACE_STAT_TAB = 2
PORTAL_LOC_TAB = 3
PORTAL_STAT_TAB = 4
CHAR_STAT_TAB = 5
THING_LOC_CAL = 6
THING_STAT_CAL = 7
PLACE_STAT_CAL = 8
PORTAL_ORIG_CAL = 9
PORTAL_DEST_CAL = 10
PORTAL_STAT_CAL = 11
CHAR_STAT_CAL = 12


SHEET_ITEM_TYPES = [
    THING_LOC_TAB,
    THING_STAT_TAB,
    PLACE_STAT_TAB,
    PORTAL_LOC_TAB,
    PORTAL_STAT_TAB,
    CHAR_STAT_TAB,
    THING_LOC_CAL,
    THING_STAT_CAL,
    PLACE_STAT_CAL,
    PORTAL_ORIG_CAL,
    PORTAL_DEST_CAL,
    PORTAL_STAT_CAL,
    CHAR_STAT_CAL]


TABLE_TYPES = [
    THING_LOC_TAB,
    THING_STAT_TAB,
    PLACE_STAT_TAB,
    PORTAL_LOC_TAB,
    PORTAL_STAT_TAB,
    CHAR_STAT_TAB]


CALENDAR_TYPES = [
    THING_LOC_CAL,
    THING_STAT_CAL,
    PLACE_STAT_CAL,
    PORTAL_ORIG_CAL,
    PORTAL_DEST_CAL,
    PORTAL_STAT_CAL,
    CHAR_STAT_CAL]


ITEM_TYPE_TO_HEADERS = {
    THING_LOC_TAB: [_("Thing"), _("Location")],
    THING_STAT_TAB: [_("Thing")],
    PLACE_STAT_TAB: [_("Place")],
    PORTAL_LOC_TAB: [_("Portal"), _("Origin"), _("Destination")],
    PORTAL_STAT_TAB: [_("Portal")],
    CHAR_STAT_TAB: []}

ITEM_TYPE_TO_FIELD_NAMES = {
    THING_LOC_TAB: ["name", "location"],
    THING_STAT_TAB: ["name"],
    PLACE_STAT_TAB: ["name"],
    PORTAL_LOC_TAB: ["name", "origin", "destination"],
    PORTAL_STAT_TAB: ["name"],
    CHAR_STAT_TAB: []}

whole_imgrows = [
    ('default_wallpaper', 'LiSE/gui/assets/wallpape.jpg'),
    ('default_spot', 'LiSE/gui/assets/orb.png'),
    ('default_pawn', 'atlas://LiSE/gui/assets/rltiles/hominid/base/unseen')]

globs = [('branch',   1, '0'),
         ('tick',     1, '0'),
         ('language', 3, 'eng'),
         ('observer', 3, 'Omniscient'),
         ('observed', 3, 'Physical'),
         ('host',     3, 'Physical')]

stackhs = [(10, ('block', 'brutalist')),
           (6,  ('crossroad', 'corporate', 'modernist',
                 'brownstone', 'gray', 'lobby', 'bunker',
                 'red', 'orange')),
           (5,  ('4sidewalk', 'street-ne-sw', 'street-nw-se')),
           (4, ('spacer',))]

offys = [(-2, ('spacer',)),
         (1,  ('enterprise', 'block')),
         (-1, ('lobby', 'street-ne-sw', 'street-nw-se'))]

offxs = [(1, ('lobby', 'modernist', 'orange'))]

reciprocal_portals = []

one_way_portals = []

charsheet_items = {
    'Player': []}

spot_coords = []

boards = [('Omniscient', 'Player', 'Physical')]

things = {}

pawns = {}
