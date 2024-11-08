# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
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
from kivy.clock import Clock
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from functools import partial


class SelectableRecycleBoxLayout(
	FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout
):
	pass


class trigger:
	"""Make a trigger from a method.

	Decorate a method with this and it will become a trigger. Supply a
	numeric parameter to set a timeout.

	Not suitable for methods that expect any arguments other than
	``dt``. However, you should make your method accept ``*args`` for
	compatibility.

	"""

	def __init__(self, func_or_timeout):
		if callable(func_or_timeout):
			self.func = func_or_timeout
			self.timeout = 0
		else:
			self.func = None
			self.timeout = func_or_timeout

	def __call__(self, func):
		self.func = func
		self.__doc__ = func.__doc__
		return self

	def __get__(self, instance, owner=None):
		if instance is None:
			# EventDispatcher iterates over its attributes before it
			# instantiates.  Don't try making any trigger in that
			# case.
			return
		retval = Clock.create_trigger(
			partial(self.func, instance), self.timeout
		)
		setattr(instance, self.func.__name__, retval)
		return retval


def dummynum(character, name):
	"""Count how many nodes there already are in the character whose name
	starts the same.

	"""
	num = 0
	for nodename in character.node:
		nodename = str(nodename)
		if not nodename.startswith(name):
			continue
		try:
			nodenum = int(nodename.lstrip(name))
		except ValueError:
			continue
		num = max((nodenum, num))
	return num
