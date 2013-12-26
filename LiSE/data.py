# coding=utf8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Module for extract, transform, load, as well as some constants"""

whole_imgrows = [
    ('default_wallpaper', ['wallpape.jpg']),
    ('default_spot', ['orb.png']),
    ('default_pawn', ['rltiles', 'hominid', 'unseen'])]

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

strings = [
    (u'game_menu', u'Game'),
    (u'editor_menu', u'Editor'),
    (u'place_menu', u'Place'),
    (u'thing_menu', u'Thing'),
    (u'portal_menu', u'Portal'),
    (u'new_map', u'New world'),
    (u'open_map', u'Open world...'),
    (u'save_map', u'Save'),
    (u'quit_maped', u'Quit'),
    (u'ed_select', u'Select...'),
    (u'ed_copy', u'Copy'),
    (u'ed_paste', u'Paste'),
    (u'ed_delete', u'Delete...'),
    (u'custplace', u'New place...'),
    (u'workplace', u'New workplace...'),
    (u'commonplace', u'New commons...'),
    (u'lairplace', u'New lair...'),
    (u'custthing', u'New thing...'),
    (u'decorthing', u'New decoration...'),
    (u'clothing', u'New clothing...'),
    (u'toolthing', u'New tool...'),
    (u'branch:', u'Branch:'),
    (u'tick:', u'Tick:'),
    (u'putthing', u'Drag this thing to the spot where you want it.'),
    (u'putplace', u'Drag this place where you want it.'),
    (u'putportalfrom',
     u'Draw a line between the spots where you want a portal.'),
    (u'putportalto', u'Drag the arrowhead where the portal leads.'),
    (u'play', u'▶'),
    (u'reverse', u''),
    (u'forward', u''),
    (u'switch', u'⇆'),
    (u'pause', u'‖'),
    (u'end', u'⏭'),
    (u'beginning', u'⏮'),
    (u'stepforward', u'↳'),
    (u'stepbackward', u'↰'),
    (u'speedup', u'⏩'),
    (u'slowdown', u'⏪'),
    (u'next_branch', u'➦'),
    (u'prev_branch', u''),
    (u'database', u'📸'),
    (u'calendar', u'📅'),
    (u'feed', u''),
    (u'edit', u'✎'),
    (u'tools', u'⚒'),
    (u'cog', u'⚙'),
    (u'map', u''),
    (u'save', u'💾'),
    (u'locked', u'🔒'),
    (u'unlocked', u'🔓'),
    (u'launch', u'🚀'),
    (u'split', u'🕪'),
    (u'bookmark', u'🔖'),
    (u'bookmarks', u'📑'),
    (u'character', u'👤'),
    (u'characters', u'👥'),
    (u'newchar', u''),
    (u'charsheet', u''),
    (u'delete', u''),
    (u'db', u'📸'),
    (u'network', u''),
    (u'night', u'☽'),
    (u'day', u'🔆')]

things = {}

pawns = {}
