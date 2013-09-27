# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
import logging
# do I want to handle the timer here? that might be good
logger = logging.getLogger(__name__)


class GameState:
    """A single object through which you can access all the other
game-related objects, however indirectly."""
    logfmt = """%(ts)s Updating game state from tick %(old_age)s to tick
%(new_age)s."""

    def __init__(self, db):
        """Return a GameState controlling everything in the given database."""
        self.db = db
        self.window_by_name = {}
