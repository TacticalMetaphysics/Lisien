# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Module for extract, transform, load, as well as some constants"""


def _(arg):
    return arg

THING_TAB = 0
PLACE_TAB = 2
PORTAL_TAB = 3
CHAR_TAB = 4
THING_LOC_CAL = 5
THING_STAT_CAL = 6
PLACE_STAT_CAL = 7
PORTAL_ORIG_CAL = 8
PORTAL_DEST_CAL = 9
PORTAL_STAT_CAL = 10
CHAR_STAT_CAL = 11


SHEET_ITEM_TYPES = [
    THING_LOC_CAL,
    THING_STAT_CAL,
    PLACE_STAT_CAL,
    PORTAL_ORIG_CAL,
    PORTAL_DEST_CAL,
    PORTAL_STAT_CAL,
    CHAR_STAT_CAL]


TABLE_TYPES = [
    THING_TAB,
    PLACE_TAB,
    PORTAL_TAB,
    CHAR_TAB]


CALENDAR_TYPES = [
    THING_LOC_CAL,
    THING_STAT_CAL,
    PLACE_STAT_CAL,
    PORTAL_ORIG_CAL,
    PORTAL_DEST_CAL,
    PORTAL_STAT_CAL,
    CHAR_STAT_CAL]


whole_imgrows = [
    ('default_wallpaper', 'LiSE/gui/assets/wallpape.jpg', 0, 0),
    ('default_spot', 'LiSE/gui/assets/orb.png', 0, 0),
    ('default_pawn', 'atlas://LiSE/gui/assets/rltiles/hominid/base/unseen',
     4, 8)]

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
