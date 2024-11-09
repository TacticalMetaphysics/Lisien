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
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior


class SelectableRecycleBoxLayout(
	FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout
):
	pass


def dummynum(character, name):
	"""Count how many nodes there already are in the character whose name
	starts the same.

	"""
	num = 0
	for nodename in character.node:
		nodename = str(nodename)
		if nodename[: len(name)] != name:
			continue
		try:
			nodenum = int(nodename.lstrip(name))
		except ValueError:
			continue
		num = max((nodenum, num))
	return num
