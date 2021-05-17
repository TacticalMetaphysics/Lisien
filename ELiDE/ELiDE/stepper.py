from kivy.app import App
from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.lang import Builder
from kivy.graphics import Color, Line


class RuleStepper(RecycleView):
    name = StringProperty()

    def from_rules_handled_turn(self, start_tick, rules_handled_turn):
        data = []
        for rbtyp, rules in rules_handled_turn.items():
            if not rules:
                continue
            data.append({
                'widget': 'RulebookTypeLabel',
                'name': rbtyp
            })
            last_entity = None
            last_rulebook = None
            # rules is a WindowDict, guaranteed to be sorted
            prev_tick = start_tick
            for tick, (entity, rulebook, rule) in rules.items():
                rulebook_per_entity = rbtyp in {'thing', 'place', 'portal'}
                if not rulebook_per_entity:
                    if rulebook != last_rulebook:
                        last_rulebook = rulebook
                        data.append({
                            'widget': 'RulebookLabel',
                            'name': rulebook
                        })
                if entity != last_entity:
                    last_entity = entity
                    data.append({
                        'widget': 'EntityLabel',
                        'name': entity
                    })
                if rulebook_per_entity:
                    if rulebook != last_rulebook:
                        rulebook = last_rulebook
                        data.append({
                            'widget': 'RulebookLabel',
                            'name': rulebook
                        })
                data.append({
                    'widget': 'RuleStepperRuleButton',
                    'name': rule,
                    'start_tick': prev_tick,
                    'end_tick': tick,
                    'height': 40
                })
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
        tick = App.get_running_app().tick
        if tick == self.end_tick:
            tick = self.start_tick
        else:
            tick = self.end_tick
        self.set_tick(tick)
        self.tick = tick

    def upd_line(self, *args):
        if hasattr(self, 'color_inst'):
            if self.tick == self.end_tick:
                self.color_inst.rgba = [1, 0, 0, 1]
                self.line.points = [self.x, self.y, self.right, self.y]
            else:
                self.color_inst.rgba = [0, 0, 0, 0]
        else:
            with self.canvas:
                self.color_inst = Color(
                    rgba=([1, 0, 0, 1]
                          if self.tick in (self.start_tick, self.end_tick)
                          else [0, 0, 0, 0]))
                self.line = Line(
                    points=[self.x, self.top, self.right, self.top])


class EntityLabel(Label):
    name = ObjectProperty()


class RulebookLabel(Label):
    name = ObjectProperty()  # rulebooks may have tuples for names


class RulebookTypeLabel(Label):
    name = StringProperty()


Builder.load_string("""
<RuleStepper>:
    key_viewclass: 'widget'
    RecycleGridLayout:
        cols: 1
        size_hint_y: None
        default_size_hint: 1, None
        height: self.minimum_height
<RuleStepperRuleButton>:
    text: '\\n'.join((str(self.start_tick), self.name, str(self.end_tick)))
    font_size: 14
    text_size: self.width, None
    halign: 'center'
    tick: app.tick
    set_tick: app.set_tick
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
