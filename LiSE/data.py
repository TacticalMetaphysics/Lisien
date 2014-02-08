# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Module for extract, transform, load, as well as some constants"""


def _(arg):
    return arg


whole_imgrows = [
    ('default_wallpaper', 'LiSE/gui/assets/wallpape.jpg', 0),
    ('default_spot', 'LiSE/gui/assets/orb.png', 0),
    ('default_pawn', 'atlas://LiSE/gui/assets/rltiles/hominid/base/unseen', 0),
    ('locked', 'LiSE/gui/assets/locked.png', 0),
    ('unlocked', 'LiSE/gui/assets/unlocked.png', 0)]

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

globs = [('branch',   'int', '0'),
         ('tick',     'int', '0'),
         ('language', 'unicode', 'eng'),
         ('observer', 'unicode', 'Omniscient'),
         ('observed', 'unicode', 'Player'),
         ('host',     'unicode', 'Physical')]

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
