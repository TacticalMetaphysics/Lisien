from kivy.clock import Clock
from kivy.properties import (
    NumericProperty,
    ObjectProperty
)
from .util import trigger


class PawnBehavior:
    """Mix-in class for things in places represented graphically"""
    loc_name = ObjectProperty()
    default_image_paths = ['atlas://rltiles/base.atlas/unseen']
    priority = NumericProperty()
    board = ObjectProperty()

    def __init__(self, **kwargs):
        if 'thing' in kwargs:
            kwargs['proxy'] = kwargs['thing']
            del kwargs['thing']
        if 'proxy' in kwargs:
            kwargs['loc_name'] = kwargs['proxy']['location']
        super().__init__(**kwargs)
        self.register_event_type('on_drop')

    def on_proxy(self, *args):
        self.loc_name = self.proxy['location']

    def on_parent(self, *args):
        if not self.parent:
            Clock.schedule_once(self.on_parent, 0)
            return
        self.board = self.parent.board
        self.bind(
            loc_name=self._trigger_relocate
        )
        if self.proxy:
            self._trigger_relocate()

    def finalize(self, initial=True):
        if initial:
            self.loc_name = self.proxy['location']
            self.priority = self.proxy.get('_priority', 0.0)
        self.bind(
            loc_name=self._trigger_push_location
        )
        super().finalize(initial)

    def unfinalize(self):
        self.unbind(
            loc_name=self._trigger_push_location
        )
        super().unfinalize()

    def pull_from_proxy(self, *args):
        super().pull_from_proxy(*args)
        relocate = False
        if self.loc_name != self.proxy['location']:
            self.loc_name = self.proxy['location']
            relocate = True
        if '_priority' in self.proxy:
            self.priority = self.proxy['_priority']
        if relocate:
            self.relocate()

    def relocate(self, *args):
        if not getattr(self, '_finalized', False) or not self.parent or not self.proxy or not self.proxy.exists:
            return
        try:
            location = self._get_location_wid()
        except KeyError:
            return
        if location != self.parent:
            if self.parent:
                self.parent.remove_widget(self)
            location.add_widget(self)
    _trigger_relocate = trigger(relocate)

    def on_priority(self, *args):
        if self.proxy['_priority'] != self.priority:
            self.proxy['_priority'] = self.priority
        self.parent.restack()

    def push_location(self, *args):
        if self.proxy['location'] != self.loc_name:
            self.proxy['location'] = self.loc_name
    _trigger_push_location = trigger(push_location)

    def _get_location_wid(self):
        return self.board.spot[self.loc_name]

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        for spot in self.board.spot.values():
            if self.collide_widget(spot) and spot.name != self.loc_name:
                new_spot = spot
                break
        else:
            new_spot = None

        self.dispatch('on_drop', new_spot)
        touch.ungrab(self)
        return True

    def on_drop(self, spot):
        parent = self.parent
        if spot:
            self.loc_name = self.proxy['location'] = spot.name
            parent.remove_widget(self)
            spot.add_widget(self)
        else:
            x, y = getattr(self, 'rel_pos', (0, 0))
            self.pos = parent.x + x, parent.y + y