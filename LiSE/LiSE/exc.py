# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Exception classes for use in LiSE."""
try:
    from sqlalchemy.exc import OperationalError as alchemyOpError
    from sqlite3 import OperationalError as liteOpError
    OperationalError = (alchemyOpError, liteOpError)
except ImportError:
    from sqlite3 import OperationalError


try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
    from sqlite3 import IntegrityError as liteIntegError
    IntegrityError = (alchemyIntegError, liteIntegError)
except ImportError:
    from sqlite3 import IntegrityError


class AvatarError(ValueError):
    pass


class AmbiguousAvatarError(AvatarError):
     """Error condition for when an AvatarMapping can't decide what you want."""
     pass


class RuleError(ValueError):
    pass


class RedundantRuleError(RuleError):
    """Error condition for when you try to run a rule on a (branch,
    tick) it's already been executed.

    """
    pass


class UserFunctionError(SyntaxError):
    """Error condition for when I try to load a user-defined function and
    something goes wrong.

    """
    pass


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
            tick=None,
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
        self.tick = tick
        self.lastplace = lastplace
        super().__init__(message)
