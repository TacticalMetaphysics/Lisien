from kivy.animation import AnimationTransition
from kivy.effects.kinetic import KineticEffect
from kivy.properties import (
    ObjectProperty,
    NumericProperty)


class StiffScrollEffect(KineticEffect):
    drag_threshold = NumericProperty('20sp')
    '''Minimum distance to travel before the movement is considered as a
    drag.'''
    min = NumericProperty(0)
    """Minimum boundary to stop the scrolling at."""
    max = NumericProperty(0)
    """Maximum boundary to stop the scrolling at."""
    body = NumericProperty(0.9)
    """Proportion of the range in which you can scroll unimpeded."""
    scroll = NumericProperty(0.)
    """Computed value for scrolling"""
    transition_min = ObjectProperty(AnimationTransition.out_expo)
    transition_max = ObjectProperty(AnimationTransition.out_expo)
    target_widget = ObjectProperty(None, allownone=True, baseclass=Widget)
    displacement = NumericProperty(0)
    scroll = NumericProperty(0.)

    def __init__(self, **kwargs):
        super(StiffScrollEffect, self).__init__(**kwargs)
        self.base_friction = self.friction

    def update_velocity(self, dt):
        hard_min = self.min
        hard_max = self.max
        if hard_min > hard_max:
            hard_min, hard_max = hard_max, hard_min
        margin = (1. - self.body) * (hard_max - hard_min)
        soft_min = self.min + margin
        soft_max = self.max - margin

        if self.value < soft_min and self.value < self.history[-1][1]:
            try:
                prop = (soft_min - self.value) / (soft_min - hard_min)
                self.friction = self.base_friction + (
                    1.0 - self.base_friction) * self.transition_min(prop)
            except ZeroDivisionError:
                pass
        elif self.value > soft_max and self.value > self.history[-1][1]:
            try:
                # normalize how far past soft_max I've gone as a
                # proportion of the distance between soft_max and hard_max
                prop = (self.value - soft_max) / (hard_max - soft_max)
                self.friction = self.base_friction + (
                    1.0 - self.base_friction) * self.transition_max(prop)
            except ZeroDivisionError:
                pass
        else:
            self.friction = self.base_friction
        return super(StiffScrollEffect, self).update_velocity(dt)

    def on_value(self, *args):
        if self.value < self.min:
            self.velocity = 0
            self.scroll = self.min
        elif self.value > self.max:
            self.velocity = 0
            self.scroll = self.max
        else:
            self.scroll = self.value

    def start(self, val, t=None):
        self.is_manual = True
        self.displacement = 0
        return super(StiffScrollEffect, self).start(val, t)

    def update(self, val, t=None):
        self.trigger_velocity_update()
        self.displacement += abs(val - self.history[-1][1])
        return super(StiffScrollEffect, self).update(val, t)

    def stop(self, val, t=None):
        self.is_manual = False
        self.displacement += abs(val - self.history[-1][1])
        if self.displacement <= self.drag_threshold:
            self.velocity = 0
            return
        return super(StiffScrollEffect, self).stop(val, t)
