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


class Direction(Enum):
	FORWARD = "forward"
	BACKWARD = "backward"
