# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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
from collections import defaultdict
from threading import Thread

from kivy.app import App
from kivy.clock import triggered
from kivy.graphics import Color, Line
from kivy.logger import Logger
from kivy.properties import (
	BooleanProperty,
	NumericProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.uix.button import Button
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget

from .util import load_string_once


def trigger(func):
	return triggered()(func)


class ThornyRectangle(Button):
	left_margin = NumericProperty(10)
	right_margin = NumericProperty(10)
	top_margin = NumericProperty(10)
	bottom_margin = NumericProperty(10)

	draw_left = BooleanProperty(False)
	draw_right = BooleanProperty(False)
	draw_up = BooleanProperty(False)
	draw_down = BooleanProperty(False)

	branch = StringProperty()
	turn = NumericProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bind(
			pos=self._trigger_redraw,
			size=self._trigger_redraw,
			left_margin=self._trigger_redraw,
			right_margin=self._trigger_redraw,
			top_margin=self._trigger_redraw,
			bottom_margin=self._trigger_redraw,
			draw_left=self._trigger_redraw,
			draw_right=self._trigger_redraw,
			draw_up=self._trigger_redraw,
			draw_down=self._trigger_redraw,
		)
		self._trigger_redraw()

	def collide_point(self, x, y):
		return (
			self.x + self.left_margin < x < self.right - self.right_margin
			and self.y + self.bottom_margin < y < self.top - self.top_margin
		)

	def _redraw_line(self, enabled, name, point_lambda):
		if enabled:
			points = point_lambda()
			if hasattr(self, name):
				the_line = getattr(self, name)
				the_line.points = points
			else:
				the_line = Line(points=points)
			if the_line not in self.canvas.children:
				self.canvas.add(the_line)
			setattr(self, name, the_line)
		elif hasattr(self, name):
			self.canvas.remove(getattr(self, name))
			delattr(self, name)

	def _get_left_line_points(self):
		return [
			self.x,
			self.center_y,
			self.x + self.left_margin,
			self.center_y,
		]

	def _get_right_line_points(self):
		return [
			self.right - self.right_margin,
			self.center_y,
			self.right,
			self.center_y,
		]

	def _get_top_line_points(self):
		return [
			self.center_x,
			self.top,
			self.center_x,
			self.top - self.top_margin,
		]

	def _get_bottom_line_points(self):
		return [
			self.center_x,
			self.y,
			self.center_x,
			self.y + self.bottom_margin,
		]

	def _redraw(self, *_):
		self._color = Color(rgba=[1, 1, 1, 1])
		if self._color not in self.canvas.children:
			self.canvas.add(self._color)
		rectpoints = [
			self.x + self.left_margin,
			self.y + self.bottom_margin,
			self.right - self.right_margin,
			self.y + self.bottom_margin,
			self.right - self.right_margin,
			self.top - self.top_margin,
			self.x + self.left_margin,
			self.top - self.top_margin,
			self.x + self.left_margin,
			self.y + self.bottom_margin,
		]
		if hasattr(self, "_rect"):
			self._rect.points = rectpoints
		else:
			self._rect = Line(points=rectpoints)
			self.canvas.add(self._rect)
		self._redraw_line(
			self.draw_left, "_left_line", self._get_left_line_points
		)
		self._redraw_line(
			self.draw_right, "_right_line", self._get_right_line_points
		)
		self._redraw_line(self.draw_up, "_top_line", self._get_top_line_points)
		self._redraw_line(
			self.draw_down, "_bot_line", self._get_bottom_line_points
		)
		self.canvas.ask_update()

	_trigger_redraw = trigger(_redraw)

	def on_release(self):
		if self.branch is None or self.turn is None:
			return
		app = App.get_running_app()
		app.mainscreen.toggle_timestream()
		self._push_time()

	@triggered(timeout=0.1)
	def _push_time(self, *_):
		app = App.get_running_app()
		app.branch, app.turn = self.branch, self.turn


class Cross(Widget):
	draw_left = BooleanProperty(True)
	draw_right = BooleanProperty(True)
	draw_up = BooleanProperty(True)
	draw_down = BooleanProperty(True)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bind(
			draw_left=self._trigger_redraw,
			draw_right=self._trigger_redraw,
			draw_up=self._trigger_redraw,
			draw_down=self._trigger_redraw,
			size=self._trigger_redraw,
			pos=self._trigger_redraw,
		)

	def _draw_line(self, enabled, name, get_points):
		if enabled:
			points = get_points()
			if hasattr(self, name):
				getattr(self, name).points = points
			else:
				the_line = Line(points=points)
				setattr(self, name, the_line)
				self.canvas.add(the_line)
		elif hasattr(self, name):
			the_line = getattr(self, name)
			if the_line in self.canvas.children:
				self.canvas.remove(the_line)
			delattr(self, name)

	def _get_left_points(self):
		return [self.x, self.center_y, self.center_x, self.center_y]

	def _get_right_points(self):
		return [self.center_x, self.center_y, self.right, self.center_y]

	def _get_up_points(self):
		return [self.center_x, self.center_y, self.center_x, self.top]

	def _get_down_points(self):
		return [self.center_x, self.center_y, self.center_x, self.y]

	def _redraw(self, *_):
		self._draw_line(self.draw_left, "_left_line", self._get_left_points)
		self._draw_line(self.draw_right, "_right_line", self._get_right_points)
		self._draw_line(self.draw_up, "_up_line", self._get_up_points)
		self._draw_line(self.draw_down, "_down_line", self._get_down_points)
		self.canvas.ask_update()

	_trigger_redraw = trigger(_redraw)


class Timestream(RecycleView):
	cols = NumericProperty(
		1
	)  # should be equal to the number of turns on which branches were created + 1


def _data_and_cols_from_branches(branches):
	start_turn_branches = defaultdict(set)
	end_turn_branches = defaultdict(set)
	branch_split_turns_todo = defaultdict(set)
	branch_split_turns_done = defaultdict(set)
	for branch, (
		parent,
		parent_turn,
		parent_tick,
		end_turn,
		end_tick,
	) in branches.items():
		start_turn_branches[parent_turn].add(branch)
		end_turn_branches[end_turn].add(branch)
		branch_split_turns_todo[parent].add(parent_turn)
	branch_split_turns_todo["trunk"].add(0)
	col2turn = sorted(start_turn_branches.keys() | end_turn_branches.keys())
	if not col2turn:
		return [], 1
	Logger.debug("Timestream: read branch lineage, processing...")
	data = []
	trunk_lineage = branches.pop("trunk")
	sorted_branches = [("trunk", (None, 0, 0, 0, 0))] + sorted(
		branches.items(), key=lambda x: x[1][1]
	)
	branches["trunk"] = trunk_lineage
	for row, (branch, _) in enumerate(sorted_branches):
		for turn in col2turn:
			if branch == "trunk" and turn == 0:
				data.append(
					{
						"widget": "ThornyRectangle",
						"branch": "trunk",
						"turn": 0,
						"draw_left": False,
						"draw_up": False,
						"draw_down": len(start_turn_branches[turn]) > 1,
						"draw_right": bool(branch_split_turns_todo[branch]),
					}
				)
			elif branch in start_turn_branches[turn]:
				here_branches = [
					(branches[b][1], b) for b in start_turn_branches[turn]
				]
				if branch == min(here_branches)[1]:
					data.append(
						{
							"widget": "ThornyRectangle",
							"branch": branch,
							"turn": turn,
							"draw_left": False,
							"draw_up": turn == branches[branch][1],
							"draw_down": len(start_turn_branches[turn]) > 1,
							"draw_right": branches[branch][3] > turn,
						}
					)
				elif branch == max(here_branches)[1]:
					data.append(
						{
							"widget": "ThornyRectangle",
							"branch": branch,
							"turn": turn,
							"draw_left": branches[branch][1] > turn,
							"draw_up": False,
							"draw_down": len(start_turn_branches[turn]) > 1,
							"draw_right": False,
						}
					)
				else:
					data.append(
						{
							"widget": "Cross",
							"draw_left": False,
							"draw_up": True,
							"draw_down": len(start_turn_branches[turn]) > 1,
							"draw_right": branches[branch][3] > turn,
						}
					)
			elif branch in end_turn_branches[turn]:
				data.append(
					{
						"widget": "ThornyRectangle",
						"branch": branch,
						"turn": turn,
						"draw_left": True,
						"draw_up": row > 0 and bool(start_turn_branches[turn]),
						"draw_down": bool(start_turn_branches[turn]),
						"draw_right": False,
					}
				)
			elif branches[branch][1] <= turn:
				here_branches = [
					(branches[b][1], b)
					for b in start_turn_branches[turn]
					if b != branch
				]
				data.append(
					{
						"widget": "Cross",
						"draw_left": turn < branches[branch][3],
						"draw_right": turn < branches[branch][3],
						"draw_up": branch in branches
						and branches[branch][0] in branches
						and branches[branches[branch][0]][3] >= turn,
						"draw_down": bool(here_branches),
					}
				)
			else:
				data.append({"widget": "Widget"})
			branch_split_turns_todo[branch].discard(turn)
			start_turn_branches[turn].discard(branch)
			if turn in end_turn_branches:
				end_turn_branches[turn].discard(branch)
			branch_split_turns_done[branch].add(turn)
		Logger.debug(f"Timestream: processed branch {branch}")
	return data, len(col2turn)


class TimestreamScreen(Screen):
	toggle = ObjectProperty()
	timestream = ObjectProperty()
	_thread: Thread

	def on_pre_enter(self, *_):
		self.timestream.disabled = True
		self._thread = Thread(target=self._get_data)
		self._thread.start()

	def _get_data(self, *_):
		Logger.debug("Timestream: getting branches")
		engine = App.get_running_app().engine
		data, cols = _data_and_cols_from_branches(engine.handle("branches"))
		self.timestream.cols = cols
		self.timestream.data = data
		Logger.debug("Timestream: loaded!")
		self.timestream.disabled = False


load_string_once(r"""
<ThornyRectangle>:
	text: f"{self.branch}\n{int(self.turn)}"
<Timestream>:
	key_viewclass: 'widget'
	effect_cls: 'ScrollEffect'
	RecycleGridLayout:
		cols: root.cols
		default_width: 100
		default_height: 100
		default_size_hint: None, None
		height: self.minimum_height
		width: self.minimum_width
		size_hint: None, None
<TimestreamScreen>:
	name: 'timestream'
	timestream: timestream
	BoxLayout:
		orientation: 'vertical'
		Timestream:
			id: timestream
			size_hint_y: 0.95
		BoxLayout:
			size_hint_y: 0.05
			Button:
				text: 'Cancel'
				on_press: root.toggle()
""")

if __name__ == "__main__":
	from kivy.base import runTouchApp

	branches = {
		"trunk": (None, 0, 0, 1, 27),
		"trunk1": ("trunk", 0, 2494, 0, 2494),
		"trunk2": ("trunk1", 0, 2494, 1, 27),
		"trunk3": ("trunk", 1, 27, 1, 31),
	}
	data, cols = _data_and_cols_from_branches(branches)
	timestream = Timestream(cols=cols)
	timestream.data = data
	runTouchApp(timestream)
