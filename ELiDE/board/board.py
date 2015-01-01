# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    StringProperty,
    ReferenceListProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from .spot import Spot
from .arrow import Arrow
from .pawn import Pawn


class Board(RelativeLayout):
    """A graphical view onto a facade, resembling a game board."""
    character = ObjectProperty()
    spot = DictProperty({})
    pawn = DictProperty({})
    arrow = DictProperty({})
    arrowlayout = ObjectProperty()
    spotlayout = ObjectProperty()
    pawnlayout = ObjectProperty()
    spots_unposd = ListProperty([])
    layout_tries = NumericProperty(5)
    new_spots = ListProperty([])
    selection = ObjectProperty(None, allownone=True)
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)

    def __init__(self, **kwargs):
        """Make a trigger for my ``update`` method."""
        self._trigger_update = Clock.create_trigger(self.update)
        super().__init__(**kwargs)

    def make_pawn(self, thing):
        """Make a :class:`Pawn` to represent a :class:`Thing`, store it, and
        return it.

        """
        if thing["name"] in self.pawn:
            raise KeyError("Already have a Pawn for this Thing")
        r = Pawn(
            board=self,
            thing=thing
        )
        self.pawn[thing["name"]] = r
        return r

    def make_spot(self, place):
        """Make a :class:`Spot` to represent a :class:`Place`, store it, and
        return it.

        """
        if place["name"] in self.spot:
            raise KeyError("Already have a Spot for this Place")
        r = Spot(
            board=self,
            place=place
        )
        self.spot[place["name"]] = r
        return r

    def make_arrow(self, portal):
        """Make an :class:`Arrow` to represent a :class:`Portal`, store it,
        and return it.

        """
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
            portal=portal
        )
        if portal["origin"] not in self.arrow:
            self.arrow[portal["origin"]] = {}
        self.arrow[portal["origin"]][portal["destination"]] = r
        return r

    def on_character(self, *args):
        """Arrange to save my scroll state in my character, and to get updated
        whenever my character is.

        """
        if self.character is None:
            Clock.schedule_once(self.on_character, 0)
            return

        for prop in '_scroll_x', '_scroll_y':
            if (
                    prop not in self.character.stat or
                    self.character.stat[prop] is None
            ):
                self.character.stat[prop] = 0.0

        self.parent.scroll_x = self.character.stat['_scroll_x']
        self.parent.scroll_y = self.character.stat['_scroll_y']

        self.track_xvel = False
        self.track_yvel = False
        self.parent.effect_x.bind(velocity=self.track_x_vel)
        self.parent.effect_y.bind(velocity=self.track_y_vel)

        self._trigger_update()

    def upd_x_when_scrolling_stops(self, *args):
        """Wait for the scroll to stop, then store where it ended."""
        if self.parent.effect_x.velocity < self.parent.effect_x.min_velocity:
            self.character.stat['_scroll_x'] = self.parent.scroll_x
            self.track_xvel = False
            return
        Clock.schedule_once(self.upd_x_when_scrolling_stops, 0.001)

    def track_x_vel(self, *args):
        """Track scrolling once it starts, so that we can tell when it
        stops.

        """
        if (
                not self.track_xvel and
                self.parent.effect_x.velocity >
                self.parent.effect_x.min_velocity
        ):
            self.upd_x_when_scrolling_stops()
            self.track_xvel = True

    def upd_y_when_scrolling_stops(self, *args):
        """Wait for the scroll to stop, then store where it ended."""
        if self.parent.effect_y.velocity < self.parent.effect_y.min_velocity:
            self.character.stat['_scroll_y'] = self.parent.scroll_y
            self.track_yvel = False
            return
        Clock.schedule_once(self.upd_y_when_scrolling_stops, 0.001)

    def track_y_vel(self, *args):
        """Track scrolling once it starts, so that we can tell when it
        stops.

        """
        if (
                not self.track_yvel and
                self.parent.effect_y.velocity >
                self.parent.effect_y.min_velocity
        ):
            self.upd_y_when_scrolling_stops()
            self.track_yvel = True

    def _rm_arrows_to_and_from(self, name):
        if name in self.arrow:
            l = list(self.arrow[name].keys())
            Logger.debug(
                "Board: removing arrows from {} to: {}".format(
                    name,
                    l
                )
            )
            for dest in l:
                self._rmarrow(name, dest)
        l = []
        for orig in list(self.arrow.keys()):
            if name in self.arrow[orig]:
                l.append(orig)
                self._rmarrow(orig, name)
        Logger.debug(
            "Board: removed arrows to {} from: {}".format(
                name,
                l
            )
        )

    def _rmpawn(self, name):
        """Remove the :class:`Pawn` by the given name"""
        if name not in self.pawn:
            raise KeyError("No Pawn named {}".format(name))
        # Currently there's no way to connect Pawns with Arrows but I
        # think there will be, so, insurance
        self._rm_arrows_to_and_from(name)
        pwn = self.pawn[name]
        pwn.parent.remove_widget(pwn)
        del self.pawn[name]

    def _rmspot(self, name):
        """Remove the :class:`Spot` by the given name"""
        if name not in self.spot:
            raise KeyError("No Spot named {}".format(name))
        self._rm_arrows_to_and_from(name)
        self.spotlayout.remove_widget(self.spot[name])
        del self.spot[name]

    def _rmarrow(self, orig, dest):
        """Remove the :class:`Arrow` that goes from ``orig`` to ``dest``"""
        if (
                orig not in self.arrow or
                dest not in self.arrow[orig]
        ):
            raise KeyError("No Arrow from {} to {}".format(orig, dest))
        self.arrowlayout.remove_widget(self.arrow[orig][dest])
        del self.arrow[orig][dest]

    def grid_layout(self, graph):
        from networkx import spectral_layout
        return spectral_layout(graph)

    def update(self, *args):
        """Refresh myself from the database"""
        # remove widgets that don't represent anything anymore
        pawns_removed = []
        for pawn_name in list(self.pawn.keys()):
            if pawn_name not in self.character.thing:
                pawns_removed.append(pawn_name)
                self._rmpawn(pawn_name)
        Logger.debug(
            "Board: removed {} pawns from {}'s board".format(
                len(pawns_removed),
                self.character.name
            )
        )
        spots_removed = []
        for spot_name in list(self.spot.keys()):
            if spot_name not in self.character.place:
                spots_removed.append(spot_name)
                self._rmspot(spot_name)
        Logger.debug(
            "Board: removed {} spots from {}'s board".format(
                len(spots_removed),
                self.character.name
            )
        )
        arrows_removed = []
        for arrow_origin in list(self.arrow.keys()):
            for arrow_destination in list(self.arrow[arrow_origin].keys()):
                if (
                        arrow_origin not in self.character.portal or
                        arrow_destination not in
                        self.character.portal[arrow_origin]
                ):
                    arrows_removed.append((arrow_origin, arrow_destination))
                    self._rmarrow(arrow_origin, arrow_destination)
        Logger.debug(
            "Board: removed {} arrows from {}'s board".format(
                len(arrows_removed),
                self.character.name
            )
        )
        # add widgets to represent new stuff
        self.spots_unposd = []
        spots_added = []
        for place_name in self.character.place:
            if place_name not in self.spot:
                spot = self.make_spot(self.character.place[place_name])
                self.spotlayout.add_widget(spot)
                spots_added.append(spot)
        Logger.debug(
            "Board: added {} spots to {}'s board".format(
                len(spots_added),
                self.character.name
            )
        )
        self.new_spots = spots_added
        arrows_added = []
        for arrow_orig in self.character.portal:
            for arrow_dest in self.character.portal[arrow_orig]:
                if (
                        arrow_orig not in self.arrow or
                        arrow_dest not in self.arrow[arrow_orig]
                ):
                    arrows_added.append(
                        (
                            arrow_orig,
                            arrow_dest
                        )
                    )
                    self.arrowlayout.add_widget(
                        self.make_arrow(
                            self.character.portal[arrow_orig][arrow_dest]
                        )
                    )
        Logger.debug(
            "Board: added {} arrows to {}'s board".format(
                len(arrows_added),
                self.character.name
            )
        )
        pawns_added = []
        for thing_name in self.character.thing:
            if thing_name not in self.pawn:
                pawns_added.append(thing_name)
                pwn = self.make_pawn(self.character.thing[thing_name])
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
        Logger.debug(
            "Board: added {} pawns to {}'s board".format(
                len(pawns_added),
                self.character.name
            )
        )

    def on_spots_unposd(self, *args):
        if len(self.spots_unposd) != len(self.new_spots):
            return
        for spot in self.new_spots:
            if spot not in self.spots_unposd:
                self.new_spots = self.spots_unposd = []
                return
        # No spots have positions;
        # do a layout.
        Clock.schedule_once(self.nx_layout, 0)

    def nx_layout(self, *args):
        """Use my ``grid_layout`` method to decide where all my spots should
        go, and move them there.

        """
        from functools import partial
        spots_only = self.character.facade()
        for thing in list(spots_only.thing.keys()):
            del spots_only.thing[thing]
        l = self.grid_layout(spots_only)

        def position_spot(spot, x, y, *args):
            if not (spot.name and spot.remote and spot.mirror):
                Clock.schedule_once(partial(position_spot, spot, x, y), 0)
                return
            spot.mirror['_x'] = spot.remote['_x'] = x
            spot.mirror['_y'] = spot.remote['_y'] = y
            spot.pos = (
                int(x * self.width),
                int(y * self.height)
            )
        for spot in self.new_spots:
            position_spot(spot, *l[spot.name])
        self.new_spots = self.spots_unposd = []

    def __repr__(self):
        """Look like a :class:`Character` wrapped in ``Board(...)```"""
        return "Board({})".format(repr(self.character))

    def arrows(self):
        for o in self.arrow:
            for arro in self.arrow[o].values():
                yield arro

    def pawns_at(self, x, y):
        for pawn in self.pawn.values():
            if pawn.collide_point(x, y):
                yield pawn

    def spots_at(self, x, y):
        for spot in self.spot.values():
            if spot.collide_point(x, y):
                yield spot

    def arrows_at(self, x, y):
        for arrow in self.arrows():
            if arrow.collide_point(x, y):
                yield arrow
