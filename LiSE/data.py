# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Data to initialize the database with"""


def _(arg):
    """Nothing!"""
    return arg


whole_imgrows = [
    ('default_wallpaper', 'LiSE/gui/assets/wallpape.jpg', 0),
    ('default_spot', 'LiSE/gui/assets/orb.png', 0),
    ('default_pawn', 'atlas://LiSE/gui/assets/rltiles/hominid/base/unseen', 0),
    ('locked', 'LiSE/gui/assets/locked.png', 0),
    ('unlocked', 'LiSE/gui/assets/unlocked.png', 0)]
"""Special imgs that need their own names, irrespective of the
user's wishes."""

graphics = {
    'locked': {'imgs': ['locked']},
    'unlocked': {'imgs': ['unlocked']},
    'default_wallpaper': {
        'imgs': ['default_wallpaper']},
    'default_spot': {
        'imgs': ['default_spot']},
    'default_pawn': {
        'offset_x': 4,
        'offset_y': 8,
        'imgs': ['default_pawn']}}
"""Special graphics"""

globs = [('branch',   'int', '0'),
         ('tick',     'int', '0'),
         ('language', 'unicode', 'eng'),
         ('observer', 'unicode', 'Omniscient'),
         ('observed', 'unicode', 'Player'),
         ('host',     'unicode', 'Physical'),
         ('top_generic_graphic', 'int', '0'),
         ('default_facade_cls', 'unicode', 'NullFacade')]
"""Default values for globals"""

stackhs = [(10, ('block', 'brutalist')),
           (6,  ('crossroad', 'corporate', 'modernist',
                 'brownstone', 'gray', 'lobby', 'bunker',
                 'red', 'orange')),
           (5,  ('4sidewalk', 'street-ne-sw', 'street-nw-se')),
           (4, ('spacer',))]
"""Stacking heights, presently only used for putting Pixel City tiles
one on the other"""


offys = [(-2, ('spacer',)),
         (1,  ('enterprise', 'block')),
         (-1, ('lobby', 'street-ne-sw', 'street-nw-se'))]
"""Y-offsets for certain imgs"""

offxs = [(1, ('lobby', 'modernist', 'orange'))]
"""X-offsets for certain imgs"""

reciprocal_portals = []
"""Portals that connect two places both ways"""

one_way_portals = []
"""Portals that connect one place to another place and only in that
direction"""

charsheet_items = {
    'Player': []}
"""Items to put on character sheets. Every character sheet needs a
list of items, though it may be empty."""

spot_coords = []
"""Pairs of spots with their coordinates"""

things = {}
"""Thing names mapped to their locations"""

pawns = {}
"""Thing names mapped to their graphics"""
