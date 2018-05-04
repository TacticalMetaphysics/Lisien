# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty
)
from .pawnspot import PawnSpot
from ..util import trigger


class Pawn(PawnSpot):
    """A token to represent a :class:`Thing`.

    :class:`Thing` is the LiSE class to represent items that are
    located in some :class:`Place` or other. Accordingly,
    :class:`Pawn`'s coordinates are never set directly; they are
    instead derived from the location of the :class:`Thing`
    represented. That means a :class:`Pawn` will appear next to the
    :class:`Spot` representing the :class:`Place` that its
    :class:`Thing` is in. The exception is if the :class:`Thing` is
    currently moving from its current :class:`Place` to another one,
    in which case the :class:`Pawn` will appear some distance along
    the :class:`Arrow` that represents the :class:`Portal` it's moving
    through.

    """
    _touch_ox_diff = NumericProperty()
    _touch_oy_diff = NumericProperty()
    _touch_opos_diff = ReferenceListProperty(_touch_ox_diff, _touch_oy_diff)
    _touch = ObjectProperty(None, allownone=True)
    loc_name = ObjectProperty()
    next_loc_name = ObjectProperty(None, allownone=True)
    thing = AliasProperty(
        lambda self: self.proxy,
        lambda self, v: self.proxy.setter()(v),
        bind=('proxy',)
    )
    default_image_paths = ['atlas://rltiles/base.atlas/unseen']
    priority = NumericProperty()

    def __init__(self, **kwargs):
        if 'thing' in kwargs:
            kwargs['proxy'] = kwargs['thing']
            del kwargs['thing']
        super().__init__(**kwargs)
        self.register_event_type('on_drop')

    def on_parent(self, *args):
        if self.parent:
            self.board = self.parent.board
            self.bind(
                loc_name=self._trigger_relocate,
                next_loc_name=self._trigger_relocate
            )
            if self.proxy:
                self._trigger_relocate()

    def relocate(self, *args):
        if not self.proxy.exists:
            return
        location = self.board.spot[self.loc_name]
        if location != self.parent:
            if self.parent:
                self.parent.remove_widget(self)
            location.add_widget(self)
    _trigger_relocate = trigger(relocate)

    def finalize(self, initial=True):
        if initial:
            self.loc_name = self.proxy['location']
            self.next_loc_name = self.proxy.get('next_location', None)
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
            self.loc_name = self.proxy['location']  # aliasing? could be trouble
            relocate = True
        if self.next_loc_name != self.proxy['next_location']:
            self.next_loc_name = self.proxy['next_location']
            relocate = True
        if '_priority' in self.proxy and self.priority != self.proxy['_priority']:
            self.priority = self.proxy['_priority']
        if relocate:
            self.relocate()

    def on_priority(self, *args):
        if self.proxy['_priority'] != self.priority:
            self.proxy['_priority'] = self.priority
        self.parent.restack()

    def push_location(self, *args):
        if self.proxy['location'] != self.loc_name:
            self.proxy['location'] = self.loc_name
    _trigger_push_location = trigger(push_location)

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
        if hasattr(self, '_start'):
            del self._start
        return True

    def on_drop(self, spot):
        parent = self.parent
        if spot:
            self.loc_name = spot.name
            parent.remove_widget(self)
            spot.add_widget(self)
        else:
            self.pos = parent.positions[self.uid]

    def get_layout_canvas(self):
        return self.board.pawnlayout.canvas

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '{}-in-{}'.format(
            self.name,
            self.loc_name
        )
