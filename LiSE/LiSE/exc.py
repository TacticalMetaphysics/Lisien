# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Exception classes for use in LiSE."""
from .allegedb.query import IntegrityError, TimeError

try:
    from sqlalchemy.exc import OperationalError as alchemyOpError
    from sqlite3 import OperationalError as liteOpError
    OperationalError = (alchemyOpError, liteOpError)
except ImportError:
    from sqlite3 import OperationalError


class NonUniqueError(Exception):
    """For when you tried to look up the only one of something but there wasn't just one"""


class AmbiguousAvatarError(NonUniqueError, KeyError):
    """An AvatarMapping can't decide what you want."""


class AmbiguousUserError(NonUniqueError, AttributeError):
    """A user descriptor can't decide what you want."""


class UserFunctionError(SyntaxError):
    """Error condition for when I try to load a user-defined function and
    something goes wrong.

    """
    pass


class WorldIntegrityError(ValueError):
    """Error condition for when something breaks the world model, even if
    it might be allowed by the database schema.

    """


class CacheError(ValueError):
    """Error condition for something going wrong with a cache"""
    pass


class TravelException(Exception):
    """Exception for problems with pathfinding. Not necessarily an error
    because sometimes somebody SHOULD get confused finding a path.

    """
    def __init__(
            self,
            message,
            path=None,
            followed=None,
            traveller=None,
            branch=None,
            turn=None,
            lastplace=None
    ):
        """Store the message as usual, and also the optional arguments:

        ``path``: a list of Place names to show such a path as you found

        ``followed``: the portion of the path actually followed

        ``traveller``: the Thing doing the travelling

        ``branch``: branch during travel

        ``tick``: tick at time of error (might not be the tick at the
        time this exception is raised)

        ``lastplace``: where the traveller was, when the error happened

        """
        self.path = path
        self.followed = followed
        self.traveller = traveller
        self.branch = branch
        self.turn = turn
        self.lastplace = lastplace
        super().__init__(message)


class PlanError(AttributeError):
    """Tried to use an attribute that shouldn't be used while planning"""


class RulesEngineError(Exception):
    """For problems to do with the rules engine

    Rules themselves should never raise this. Only the engine should.

    """


class RuleError(RulesEngineError):
    """For problems to do with rules

    Rather than the operation of the rules engine as a whole.

    Don't use this in your trigger, prereq, or action functions.
    It's only for Rule objects as such.

    """


class RedundantRuleError(RuleError):
    """Error condition for when you try to run a rule on a (branch,
    turn) it's already been executed.

    """