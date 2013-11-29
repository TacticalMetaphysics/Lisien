# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Constants for use in kv files. Mostly colors."""


solarized_hex = {
    'base03': (0x00, 0x2b, 0x36),
    'base02': (0x07, 0x36, 0x42),
    'base01': (0x58, 0x6e, 0x75),
    'base00': (0x65, 0x7b, 0x83),
    'base0': (0x83, 0x94, 0x96),
    'base1': (0x93, 0xa1, 0xa1),
    'base2': (0xee, 0xe8, 0xd5),
    'base3': (0xfd, 0xf6, 0xe3),
    'yellow': (0xb5, 0x89, 0x00),
    'orange': (0xcb, 0x4b, 0x16),
    'red': (0xdc, 0x32, 0x2f),
    'magenta': (0xd3, 0x36, 0x82),
    'violet': (0x6c, 0x71, 0xc4),
    'blue': (0x26, 0x8b, 0xd2),
    'cyan': (0x2a, 0xa1, 0x98),
    'green': (0x85, 0x99, 0x00)}
"""Color values for Solarized, a color scheme by Ethan Schoonover:
http://ethanschoonover.com/solarized"""


macks = float(0xff)
"""The largest two-digit hexadecimal number, cast into float so that I
can use it to relativize colors given in 0-255 rather than 0.0-1.0."""


solarized = dict([
    (name, (r/macks, g/macks, b/macks, 1.0))
    for (name, (r, g, b)) in solarized_hex.iteritems()])
"""Color values for Solarized, a color scheme by Ethan Schoonover:
http://ethanschoonover.com/solarized"""
