from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.lang import Builder


class RuleStepper(RecycleView):
    name = StringProperty()

    def from_rules_handled_turn(self, rules_handled_turn):
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
            for tick, (entity, rulebook, rule) in rules.items():
                if data:
                    prev = data[-1]
                    if prev['widget'] == 'RuleStepperRuleButton':
                        assert prev['end_tick'] is None
                        prev['end_tick'] = tick - 1
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
                    'start_tick': tick,
                    'end_tick': None,
                    'height': 28
                })
            if data:
                prev = data[-1]
                assert prev['widget'] == 'RuleStepperRuleButton'
                assert prev['end_tick'] is None
                prev['end_tick'] = tick - 1
        self.data = data


class RuleStepperRuleButton(Button):
    name = StringProperty()
    start_tick = NumericProperty()
    end_tick = NumericProperty()
    tick = NumericProperty()
    set_tick = ObjectProperty()

    def on_release(self, *args):
        if self.tick == self.end_tick:
            self.set_tick(self.start_tick)
        else:
            self.set_tick(self.end_tick)


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
    text: self.name
    font_size: 14
    text_size: self.width, None
    halign: 'center'
    tick: app.tick
    set_tick: app.set_tick
    canvas:
        Color:
            rgba: (1, 0, 0, 1) if self.tick in (self.start_tick, self.end_tick) else (0, 0, 0, 0)
        Line:
            points: [self.x, self.y + self.height, self.x + self.width, self.top] if app.tick == self.start_tick else [self.x, self.y, self.x + self.width, self.y]
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
