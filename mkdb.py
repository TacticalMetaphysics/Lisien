# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import closet
from sys import argv

"""Make an empty database of LiSE's schema. By default it will be
called default.sqlite and include the RLTiles (in folder
./rltiles). Put sql files in the folder ./init and they'll be executed
in their sort order, after the schema is defined.

"""
closet.USE_KIVY = False
if argv[-1][-2:] != "py":
    closet.mkdb(argv[-1])
else:
    closet.mkdb()
