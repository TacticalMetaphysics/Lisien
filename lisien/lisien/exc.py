# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Exception classes for use in lisien."""


class GraphNameError(KeyError):
	"""For errors involving graphs' names"""


class TimeError(ValueError):
	"""Exception class for problems with the time model"""


class OutOfTimelineError(ValueError):
	"""You tried to access a point in time that didn't happen"""

	@property
	def branch_from(self):
		return self.args[1]

	@property
	def turn_from(self):
		return self.args[2]

	@property
	def tick_from(self):
		return self.args[3]

	@property
	def branch_to(self):
		return self.args[4]

	@property
	def turn_to(self):
		return self.args[5]

	@property
	def tick_to(self):
		return self.args[6]


class NonUniqueError(RuntimeError):
	"""You tried to look up the only one of something but there wasn't just one"""


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
	"""Exception for problems with pathfinding.

	Not necessarily an error because sometimes somebody SHOULD get
	confused finding a path.

	"""

	def __init__(
		self,
		message,
		path=None,
		followed=None,
		traveller=None,
		branch=None,
		turn=None,
		lastplace=None,
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


class RulesEngineError(RuntimeError):
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


class WorkerProcessError(RuntimeError):
	"""Something wrong to do with worker processes"""


class WorkerProcessReadOnlyError(WorkerProcessError):
	"""You tried to change the state of the world in a worker process"""


class HistoricKeyError(KeyError):
	"""Distinguishes deleted keys from those that were never set"""

	def __init__(self, *args, deleted=False):
		super().__init__(*args)
		self.deleted = deleted


class NotInKeyframeError(KeyError):
	pass


class KeyframeError(KeyError):
	pass


class TotalKeyError(KeyError):
	"""Error class for when a key is totally absent from a cache

	And was not, for instance, set at one point, then deleted later.

	"""
