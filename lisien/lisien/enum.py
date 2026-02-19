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
from __future__ import annotations

from enum import Enum


class MsgpackExtensionType(Enum):
	"""Type codes for packing special lisien types into msgpack"""

	tuple = 0x00
	frozenset = 0x01
	set = 0x02
	exception = 0x03
	graph = 0x04
	character = 0x7F
	place = 0x7E
	thing = 0x7D
	portal = 0x7C
	ellipsis = 0x7B
	function = 0x7A
	method = 0x79
	trigger = 0x78
	prereq = 0x77
	action = 0x76
	database = 0x75
	path = 0x74
	enum = 0x73


class Sub(Enum):
	"""Enum for the different forms of parallelism"""

	process = "process"
	interpreter = "interpreter"
	thread = "thread"
	serial = None


try:
	Sub.serial._add_value_alias_("serial")
except AttributeError:
	Sub._value2member_map_["serial"] = Sub.serial


class Direction(Enum):
	"""Directions a fast delta can go in -- just forward and backward

	Slow deltas can go 'sideways' to other branches, but direction isn't
	relevant to that algorithm.

	"""

	FORWARD = "forward"
	BACKWARD = "backward"
