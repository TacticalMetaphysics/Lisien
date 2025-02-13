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
from kivy.clock import mainthread
from kivy.graphics import Color, Line
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView

from .util import load_string_once


class RuleStepper(RecycleView):
	name = StringProperty()

	def from_rules_handled_turn(self, rules_handled_turn):
		data = [
			{
				"widget": "RuleStepperRuleButton",
				"name": "start",
				"end_tick": 0,
				"height": 40,
			}
		]
		all_rules = []
		for rbtyp, rules in rules_handled_turn.items():
			for tick, rule in rules.items():
				all_rules.append((tick, rbtyp, rule))
		all_rules.sort()
		prev_tick = 0
		last_entity = None
		last_rulebook = None
		lasttyp = None
		for tick, rbtyp, (entity, rulebook, rule) in all_rules:
			if tick == prev_tick:
				continue  # Rules that aren't triggered are still "handled". Ignore them.
			if lasttyp != rbtyp:
				data.append({"widget": "RulebookTypeLabel", "name": rbtyp})
			lasttyp = rbtyp
			rulebook_per_entity = rbtyp in {"thing", "place", "portal"}
			if not rulebook_per_entity:
				if rulebook != last_rulebook:
					last_rulebook = rulebook
					data.append({"widget": "RulebookLabel", "name": rulebook})
			if entity != last_entity:
				last_entity = entity
				data.append({"widget": "EntityLabel", "name": entity})
			if rulebook_per_entity:
				if rulebook != last_rulebook:
					rulebook = last_rulebook
					data.append({"widget": "RulebookLabel", "name": rulebook})
			data.append(
				{
					"widget": "RuleStepperRuleButton",
					"name": rule,
					"start_tick": prev_tick,
					"end_tick": tick,
					"height": 40,
				}
			)
			prev_tick = tick
		self.data = data


class RuleStepperRuleButton(Button):
	name = StringProperty()
	start_tick = NumericProperty()
	end_tick = NumericProperty()
	tick = NumericProperty()
	set_tick = ObjectProperty()

	def __init__(self, **kwargs):
		super(RuleStepperRuleButton, self).__init__(**kwargs)
		self.bind(pos=self.upd_line, size=self.upd_line, tick=self.upd_line)

	def on_release(self, *args):
		self.set_tick(self.end_tick)
		self.tick = self.end_tick

	@mainthread
	def upd_line(self, *_):
		if hasattr(self, "color_inst"):
			if self.tick == self.end_tick:
				self.color_inst.rgba = [1, 0, 0, 1]
				self.line.points = [self.x, self.y, self.right, self.y]
			else:
				self.color_inst.rgba = [0, 0, 0, 0]
		else:
			with self.canvas:
				self.color_inst = Color(
					rgba=(
						[1, 0, 0, 1]
						if self.tick in (self.start_tick, self.end_tick)
						else [0, 0, 0, 0]
					)
				)
				self.line = Line(
					points=[self.x, self.top, self.right, self.top]
				)


class EntityLabel(Label):
	name = ObjectProperty()


class RulebookLabel(Label):
	name = ObjectProperty()  # rulebooks may have tuples for names


class RulebookTypeLabel(Label):
	name = StringProperty()


load_string_once("""
#:import ScrollEffect kivy.effects.scroll.ScrollEffect
<RuleStepper>:
	key_viewclass: 'widget'
	effect_cls: ScrollEffect
	RecycleGridLayout:
		cols: 1
		size_hint_y: None
		default_size_hint: 1, None
		default_height: 20
		height: self.minimum_height
<RuleStepperRuleButton>:
	text: '\\n'.join((self.name, str(self.end_tick)))
	font_size: 14
	text_size: self.width, None
	halign: 'center'
	tick: app.tick
	set_tick: app.time_travel_to_tick
<EntityLabel>:
	multiline: True
	text: str(self.name)
	text_size: self.width, None
	size: self.texture_size
	font_size: 14
	padding_x: 8
<RulebookLabel>:
	text: str(self.name)
	text_size: self.width, None
	size: self.texture_size
	font_size: 14
	bold: True
	padding_x: 4
<RulebookTypeLabel>:
	text: self.name
	text_size: self.width, None
	font_size: 16
	bold: True
	size: self.texture_size
""")
