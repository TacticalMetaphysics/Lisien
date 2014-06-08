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
    ('unlocked', 'LiSE/gui/assets/unlocked.png', 0)
]
"""Special imgs that need their own names, irrespective of the
user's wishes."""

graphics = {
    'locked': {'imgs': ['locked']},
    'unlocked': {'imgs': ['unlocked']},
    'default_wallpaper': {
        'imgs': ['default_wallpaper']
    },
    'default_spot': {
        'imgs': ['default_spot']
    },
    'default_pawn': {
        'offset_x': 4,
        'offset_y': 8,
        'imgs': ['default_pawn']
    }
}
"""Special graphics"""

globs = [
    ('branch',   'int', '0'),
    ('tick',     'int', '0'),
    ('language', 'unicode', 'eng'),
    ('observer', 'unicode', 'Omniscient'),
    ('observed', 'unicode', 'Player'),
    ('host',     'unicode', 'Physical'),
    ('top_generic_graphic', 'int', '0'),
    ('default_facade_cls', 'unicode', 'NullFacade')
]
"""Default values for globals"""

stackhs = [
    (10, ('block', 'brutalist')),
    (
        6,  (
            'crossroad',
            'corporate',
            'modernist',
            'brownstone',
            'gray',
            'lobby',
            'bunker',
            'red',
            'orange'
        )
    ),
           (5,  ('4sidewalk', 'street-ne-sw', 'street-nw-se')),
           (4, ('spacer',))]
"""Stacking heights, presently only used for putting Pixel City tiles
one on the other"""


offys = [
    (-2, ('spacer',)),
    (1,  ('enterprise', 'block')),
    (-1, ('lobby', 'street-ne-sw', 'street-nw-se'))
]
"""Y-offsets for certain imgs"""

offxs = [(1, ('lobby', 'modernist', 'orange'))]
"""X-offsets for certain imgs"""

reciprocal_portals = []
"""Portals that connect two places both ways"""

one_way_portals = []
"""Portals that connect one place to another place and only in that
direction"""

spot_coords = []
"""Pairs of spots with their coordinates"""

things = {}
"""Thing names mapped to their locations"""

pawns = {}
"""Thing names mapped to their graphics"""


def defaults(c, kivy=False):
    """Retrieve default values from ``LiSE.data`` and insert them with the
    cursor ``c``.

    With ``kivy``==``True``, this will include data about graphics."""
    if kivy:
        c.executemany(
            "INSERT INTO img (name, path, stacking_height) "
            "VALUES (?, ?, ?);",
            whole_imgrows
        )
        for (name, d) in graphics.iteritems():
            c.execute(
                "INSERT INTO graphic (name, offset_x, offset_y) "
                "VALUES (?, ?, ?);",
                (name, d.get('offset_x', 0), d.get('offset_y', 0))
            )
            for i in xrange(0, len(d['imgs'])):
                c.execute(
                    "INSERT INTO graphic_img (graphic, layer, img) "
                    "VALUES (?, ?, ?);",
                    (name, i, d['imgs'][i])
                )
        for (height, names) in stackhs:
            qrystr = (
                "UPDATE img SET stacking_height=? WHERE name IN ({});".format(
                    ", ".join(["?"] * len(names))
                )
            )
            qrytup = (height,) + names
            c.execute(qrystr, qrytup)
    c.executemany(
        "INSERT INTO globals (key, type, value) VALUES (?, ?, ?);",
        globs
    )
    c.execute(
        "INSERT INTO timestream (branch, parent) VALUES (?, ?);",
        (0, 0)
    )
    for character in things:
        for thing in things[character]:
            c.execute(
                "INSERT INTO thing (character, name, host) VALUES (?, ?, ?);",
                (character, thing, things[character][thing]["host"])
            )
            c.execute(
                "INSERT INTO thing_loc (character, name, location) "
                "VALUES (?, ?, ?);",
                (character, thing, things[character][thing]["location"])
            )
    from LiSE.data import reciprocal_portals
    for (orig, dest) in reciprocal_portals:
        name1 = "{}->{}".format(orig, dest)
        name2 = "{}->{}".format(dest, orig)
        c.executemany(
            "INSERT INTO portal (name) VALUES (?);",
            [(name1,), (name2,)]
        )
        c.executemany(
            "INSERT INTO portal_loc (name, origin, destination) VALUES "
            "(?, ?, ?);", [(name1, orig, dest), (name2, dest, orig)]
        )
    from LiSE.data import one_way_portals
    for (orig, dest) in one_way_portals:
        name = "{}->{}".format(orig, dest)
        c.execute(
            "INSERT INTO portal (name) VALUES (?);",
            (name,)
        )
        c.execute(
            "INSERT INTO portal_loc (name, origin, destination) "
            "VALUES (?, ?, ?);", (name, orig, dest)
        )


def mkdb(DB_NAME, lisepath, kivy=False):
    """Initialize a database file and insert default values.

    """
    import sqlite3
    img_qrystr = (
        "INSERT INTO img (name, path) "
        "VALUES (?, ?);"
    )
    tag_qrystr = (
        "INSERT INTO img_tag (img, tag) VALUES (?, ?);"
    )

    def ins_atlas(curs, path, qualify=False, tags=[]):
        """Grab all the images in an atlas and store them, optionally sticking
        the name of the atlas on the start.

        Apply the given tags if any.

        """
        from kivy.atlas import Atlas
        from os import sep
        lass = Atlas(path)
        atlaspath = "atlas://{}".format(path[:-6])
        atlasn = path.split(sep)[-1][:-6]
        for tilen in lass.textures.iterkeys():
            imgn = atlasn + '.' + tilen if qualify else tilen
            curs.execute(
                img_qrystr, (
                    imgn, "{}/{}".format(atlaspath, tilen)
                )
            )
            for tag in tags:
                curs.execute(tag_qrystr, (imgn, tag))

    def ins_atlas_dir(curs, dirname, qualify=False, tags=[]):
        from os import sep
        from os import listdir
        """Recurse into the directory and ins_atlas for all atlas therein."""
        for fn in listdir(dirname):
            if fn[-5:] == 'atlas':
                path = dirname + sep + fn
                ins_atlas(curs, path, qualify, [fn[:-6]] + tags)

    if kivy:
        import kivy.logger
        Logger = kivy.logger.Logger
        # I just need these modules to fill in the relevant bits of
        # SaveableMetaclass, which they do when imported. They don't
        # have to do anything else, so delete them.
        import LiSE.gui.img
        del LiSE.gui.img
        import LiSE.gui.board
        del LiSE.gui.board
    else:
        import logging
        logging.basicConfig()
        Logger = logging.getLogger()
        Logger.setLevel(0)

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    Logger.debug("initializing database")
    from LiSE.orm import SaveableMetaclass
    # contains classes with table declarations in them. 

    Logger.debug("inserting default values")
    defaults(c, kivy)

    if kivy:
        Logger.debug("indexing the RLTiles")
        ins_atlas_dir(
            c,
            "LiSE/gui/assets/rltiles/hominid",
            True,
            ['hominid', 'rltile', 'pawn']
        )

        Logger.debug("indexing Pixel City")
        ins_atlas(
            c,
            "LiSE/gui/assets/pixel_city.atlas",
            False,
            ['spot', 'pixel_city']
        )

    conn.commit()
    return conn
