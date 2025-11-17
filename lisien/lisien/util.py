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
"""Common utility functions and data structures."""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from functools import wraps
from pprint import pformat
from textwrap import dedent
from time import monotonic
from typing import Callable, Iterable, Protocol, TypeVar

try:
	import msgpack

	if msgpack.Packer.__module__.endswith("cmsgpack"):
		C_MSGPACK = True
	else:
		import umsgpack as msgpack

		C_MSGPACK = False
except ImportError:
	import umsgpack as msgpack

	C_MSGPACK = False

TRUE: bytes = msgpack.packb(True)
FALSE: bytes = msgpack.packb(False)
NONE: bytes = msgpack.packb(None)
EMPTY: bytes = msgpack.packb({})
EMPTY_LIST: bytes = msgpack.packb([])
ELLIPSIS: bytes = b"\xc7\x00{"
NAME: bytes = msgpack.packb("name")
NODES: bytes = msgpack.packb("nodes")
EDGES: bytes = msgpack.packb("edges")
UNITS: bytes = msgpack.packb("units")
RULEBOOK: bytes = msgpack.packb("rulebook")
RULEBOOKS: bytes = msgpack.packb("rulebooks")
NODE_VAL: bytes = msgpack.packb("node_val")
EDGE_VAL: bytes = msgpack.packb("edge_val")
ETERNAL: bytes = msgpack.packb("eternal")
UNIVERSAL: bytes = msgpack.packb("universal")
STRINGS: bytes = msgpack.packb("strings")
RULES: bytes = msgpack.packb("rules")
TRIGGERS: bytes = msgpack.packb("triggers")
PREREQS: bytes = msgpack.packb("prereqs")
ACTIONS: bytes = msgpack.packb("actions")
NEIGHBORHOOD: bytes = msgpack.packb("neighborhood")
BIG: bytes = msgpack.packb("big")
LOCATION: bytes = msgpack.packb("location")
BRANCH: bytes = msgpack.packb("branch")
EMPTY_MAPPING: bytes = msgpack.packb({})


@contextmanager
def timer(msg="", logfun: Callable | None = None):
	if logfun is None:
		logfun = print
	start = monotonic()
	yield
	logfun("{:,.3f} {}".format(monotonic() - start, msg))


_T = TypeVar("_T")


def singleton_get(s: Iterable[_T]) -> _T | None:
	"""Take an iterable and return its only item if possible, else None."""
	it = None
	for that in s:
		if it is not None:
			return None
		it = that
	return it


def dedent_source(source: str) -> str:
	nlidx = source.index("\n")
	if nlidx is None:
		raise ValueError("Invalid source")
	while source[:nlidx].strip().startswith("@"):
		source = source[nlidx + 1 :]
		nlidx = source.index("\n")
	return dedent(source)


def normalize_layout(l: dict) -> dict:
	"""Make sure all the spots in a layout are where you can click.

	Returns a copy of the layout with all spot coordinates are
	normalized to within (0.0, 0.98).

	"""
	import numpy as np

	xs = []
	ys = []
	ks = []
	for k, (x, y) in l.items():
		xs.append(x)
		ys.append(y)
		ks.append(k)
	minx = np.min(xs)
	maxx = np.max(xs)
	if maxx == minx:
		xnorm = np.array([0.5] * len(xs))
	else:
		xco = 0.98 / (maxx - minx)
		xnorm = np.multiply(np.subtract(xs, [minx] * len(xs)), xco)
	miny = np.min(ys)
	maxy = np.max(ys)
	if miny == maxy:
		ynorm = np.array([0.5] * len(ys))
	else:
		yco = 0.98 / (maxy - miny)
		ynorm = np.multiply(np.subtract(ys, [miny] * len(ys)), yco)
	return dict(zip(ks, zip(map(float, xnorm), map(float, ynorm))))


def format_call_sig(func: Callable | str, *args, **kwargs):
	if not isinstance(func, (str, bytes)):
		func = func.__name__
	if not kwargs:
		return str(func) + pformat(args)
	return (
		f"{func}({', '.join(map(pformat, args))}"
		f"{', ' if args and kwargs else ''}"
		f"{', '.join(f'{arg}={pformat(item)}' for (arg, item) in kwargs.items())})"
	)


def print_call_sig(
	func: Callable | str, *args, file=sys.stdout, end="\n", **kwargs
):
	print(format_call_sig(func, *args, **kwargs), file=file, end=end)


@contextmanager
def _garbage_ctx(collect=True):
	"""Context manager to disable the garbage collector

	:param collect: Whether to immediately collect garbage upon context exit

	"""
	gc_was_active = gc.isenabled()
	if gc_was_active:
		gc.disable()
	yield
	if gc_was_active:
		gc.enable()
	if collect:
		gc.collect()


def _garbage_dec(fn: Callable, collect: bool = True) -> Callable:
	"""Decorator to disable the garbage collector for a function

	:param collect: Whether to immediately collect garbage when the function returns

	"""

	@wraps(fn)
	def garbage(*args, **kwargs):
		with _garbage_ctx(collect=collect):
			return fn(*args, **kwargs)

	return garbage


def garbage(arg: Callable | None = None, collect: bool = False):
	"""Disable the garbage collector, then re-enable it when done.

	May be used as a context manager or a decorator.

	:param arg: A function to call without garbage collection; if ``None``
		(the default), we're a context manager.
	:param collect: Whether to immediately run a collection after re-enabling
		the garbage collector. Default ``False``.

	"""

	if arg is None:
		return _garbage_ctx(collect=collect)
	else:
		return _garbage_dec(arg, collect=collect)


_U = TypeVar("_U")
_V = TypeVar("_V")
_W = TypeVar("_W")


class Lockable[_U, _V, _W](Protocol):
	def __call__(self, *_U, **_V) -> _W: ...


def world_locked(fn: Lockable[_U, _V, _W]) -> Lockable[_U, _V, _W]:
	"""Decorator for functions that alter the world state

	They will hold a reentrant lock, preventing more than one function
	from mutating the world at a time.

	"""

	@wraps(fn)
	def lockedy(*args, **kwargs):
		with args[0].world_lock:
			return fn(*args, **kwargs)

	return lockedy


def unwrap(v: _W) -> _W:
	if hasattr(v, "unwrap"):
		return v.unwrap()
	return v


ILLEGAL_CHARACTER_NAMES = {
	"graphs",
	"universal",
	"universals",
	"rulebooks",
	"rule_triggers",
	"rule_prereqs",
	"rule_actions",
	"rule_neighborhood",
	"rule_big",
	"character_rules_handled",
	"unit_rules_handled",
	"character_thing_rules_handled",
	"character_place_rules_handled",
	"character_portal_rules_handled",
	"node_rules_handled",
	"portal_rules_handled",
}


def msgpack_array_header(n: int) -> bytes:
	if n <= 15:
		return (0x90 + n).to_bytes(1, signed=False)
	elif n <= 0xFFFF:
		return (0xDC).to_bytes(1, signed=False) + n.to_bytes(2, signed=False)
	elif n <= 0xFFFFFFFF:
		return (0xDD).to_bytes(1, signed=False) + n.to_bytes(4, signed=False)
	else:
		raise ValueError("tuple is too large")


def msgpack_map_header(n: int) -> bytes:
	if n <= 15:
		return (0x80 + n).to_bytes(1, signed=False)
	elif n <= 0xFFFF:
		return (0xDE).to_bytes(1, signed=False) + n.to_bytes(2, signed=False)
	elif n <= 0xFFFFFFFF:
		return (0xDF).to_bytes(1, signed=False) + n.to_bytes(4, signed=False)
	else:
		raise ValueError("dict is too large")


def getatt(attribute_name: str) -> property:
	"""An easy way to make an alias"""
	from operator import attrgetter

	ret = property(attrgetter(attribute_name))
	ret.__doc__ = "Alias to `{}`".format(attribute_name)
	return ret
