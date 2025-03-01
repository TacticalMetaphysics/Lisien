# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
