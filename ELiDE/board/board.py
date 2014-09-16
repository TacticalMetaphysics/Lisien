# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    DictProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty,
    BooleanProperty
)
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from .spot import Spot
from .arrow import Arrow
from .pawn import Pawn


class BoardLayout(FloatLayout):
    """Simplistic handling of touch events, just loop over the children"""
    def on_touch_down(self, touch):
        """Check each child in turn and if one of them catches the touch,
        return it

        """
        for child in self.children:
            if child.on_touch_down(touch):
                return child

    def on_touch_move(self, touch):
        """Move all children"""
        for child in self.children:
            child.on_touch_move(touch)

    def on_touch_up(self, touch):
        """Return the first child that handles the touch"""
        for child in self.children:
            if child.on_touch_up(touch):
                return child


class Board(RelativeLayout):
    """A graphical view onto a facade, resembling a game board."""
    layout = ObjectProperty()
    character = ObjectProperty()
    spot = DictProperty({})
    pawn = DictProperty({})
    arrow = DictProperty({})
    arrow_bg = ListProperty()
    arrow_fg = ListProperty()
    arrow_width = NumericProperty()
    arrowhead_size = NumericProperty()
    arrowlayout = ObjectProperty()
    spotlayout = ObjectProperty()
    app = ObjectProperty()
    engine = ObjectProperty()
    redatad = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Make a trigger for ``_redata`` and run it"""
        self._trigger_redata = Clock.create_trigger(self._redata)
        super().__init__(**kwargs)

    def _make_pawn(self, thing):
        """Make a :class:`Pawn` to represent a :class:`Thing`"""
        if thing["name"] in self.pawn:
            raise KeyError("Already have a Pawn for this Thing")
        r = Pawn(
            board=self,
            thing=thing,
        )
        self.pawn[thing["name"]] = r
        return r

    def _make_spot(self, place):
        """Make a :class:`Spot` to represent a :class:`Place`"""
        if place["name"] in self.spot:
            raise KeyError("Already have a Spot for this Place")
        r = Spot(
            board=self,
            place=place
        )
        self.spot[place["name"]] = r
        return r

    def _make_arrow(self, portal):
        """Make an :class:`Arrow` to represent a :class:`Portal`"""
        if (
                portal["origin"] not in self.spot or
                portal["destination"] not in self.spot
        ):
            raise ValueError(
                "An :class:`Arrow` should only be made after "
                "the :class:`Spot`s it connects"
            )
        if (
                portal["origin"] in self.arrow and
                portal["destination"] in self.arrow[portal["origin"]]
        ):
            raise KeyError("Already have an Arrow for this Portal")
        r = Arrow(
            board=self,
            engine=self.engine,
            portal=portal
        )
        if portal["origin"] not in self.arrow:
            self.arrow[portal["origin"]] = {}
        self.arrow[portal["origin"]][portal["destination"]] = r
        return r

    def on_character(self, *args):
        """Arrange to save my scroll state in my character, and to get updated
        whenever my character is

        """
        if self.character is None or self.engine is None:
            Clock.schedule_once(self.on_character, 0)
            return

        def updscrollx(*args):
            self.character.stat['_scroll_x'] = self.parent.scroll_x
        trigger_updscrollx = Clock.create_trigger(updscrollx)

        def updscrolly(*args):
            self.character.stat['_scroll_y'] = self.parent.scroll_y
        trigger_updscrolly = Clock.create_trigger(updscrolly)

        for prop in '_scroll_x', '_scroll_y':
            if (
                    prop not in self.character.stat or
                    self.character.stat[prop] is None
            ):
                self.character.stat[prop] = 0.0

        self.parent.scroll_x = self.character.stat['_scroll_x']
        self.parent.scroll_y = self.character.stat['_scroll_y']
        self.parent.bind(scroll_x=trigger_updscrollx)
        self.parent.bind(scroll_y=trigger_updscrolly)

        @self.engine.on_time
        def ontime(*args):
            self._trigger_update()

        self._trigger_update()

    def _rmpawn(self, name):
        """Remove the :class:`Pawn` by the given name"""
        if name not in self.pawn:
            raise KeyError("No Pawn")
        pwn = self.pawn[name]
        pwn.parent.remove(pwn)
        del self.pawn[name]

    def _rmspot(self, name):
        """Remove the :class:`Spot` by the given name"""
        if name not in self.spot:
            raise KeyError("No Spot")
        self.spotlayout.remove_widget(self.pawn[name])
        del self.spot[name]

    def _rmarrow(self, orig, dest):
        """Remove the :class:`Arrow` that goes from ``orig`` to ``dest``"""
        if (
                orig not in self.arrow or
                dest not in self.arrow[orig]
        ):
            raise KeyError("No Arrow")
        self.spotlayout.remove_widget(self.arrow[orig][dest])
        del self.arrow[orig][dest]

    def _redata(self, *args):
        """Refresh myself from the database"""
        # remove widgets that don't represent anything anymore
        for pawn_name in self.pawn:
            if pawn_name not in self.character.thing:
                self._rmpawn(pawn_name)
        for spot_name in self.spot:
            if spot_name not in self.character.place:
                self._rmspot(spot_name)
        for arrow_origin in self.arrow:
            for arrow_destination in self.arrow[arrow_origin]:
                if (
                        arrow_origin not in self.character.portal or
                        arrow_destination not in
                        self.character.portal[arrow_origin]
                ):
                    self._rmarrow(arrow_origin, arrow_destination)
        # add widgets to represent new stuff
        for place_name in self.character.place:
            if place_name not in self.spot:
                self.spotlayout.add_widget(
                    self._make_spot(self.character.place[place_name])
                )
        for arrow_orig in self.character.portal:
            for arrow_dest in self.character.portal[arrow_orig]:
                if (
                        arrow_orig not in self.arrow or
                        arrow_dest not in self.arrow[arrow_orig]
                ):
                    self.arrowlayout.add_widget(
                        self._make_arrow(
                            self.character.portal[arrow_orig][arrow_dest]
                        )
                    )
        for thing_name in self.character.thing:
            if thing_name not in self.pawn:
                pwn = self._make_pawn(self.character.thing[thing_name])
                try:
                    whereat = self.arrow[
                        pwn.thing['location']
                    ][
                        pwn.thing['next_location']
                    ]
                except KeyError:
                    whereat = self.spot[pwn.thing['location']]
                whereat.add_widget(pwn)
                self.pawn[thing_name] = pwn
        self.redatad = True

    def _trigger_update(self, *args):
        """Make sure that _redata happens before _update, otherwise just a
        trigger.

        """
        self.redatad = False
        self._trigger_redata()
        Clock.schedule_once(self._update, 0)

    def _update(self, *args):
        """trigger all entities to refresh themselves"""
        if not self.redatad:
            Clock.schedule_once(self._update, 0)
            return
        for spot in self.spot.values():
            spot._update()
        for pawn in self.pawn.values():
            pawn._update()

    def __repr__(self):
        """Look like a :class:`Character` wrapped in ``Board(...)```"""
        return "Board({})".format(repr(self.character))

    def on_touch_down(self, touch):
        """Check pawns, ``spotlayout``, and ``arrowlayout`` in turn,
        stopping and returning the first true result I get.

        Assign the result to my layout's ``grabbed`` attribute.

        """
        for pawn in self.pawn.values():
            if pawn.dispatch('on_touch_down', touch):
                return pawn
        r = self.spotlayout.on_touch_down(touch)
        if r:
            self.layout.grabbed = r
            return r
        r = self.arrowlayout.on_touch_down(touch)
        if r:
            self.layout.grabbed = r
            return r
        return super().on_touch_down(touch)
