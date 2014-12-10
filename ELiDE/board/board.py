# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    DictProperty,
    ObjectProperty,
    NumericProperty
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from .spot import Spot
from .arrow import Arrow
from .pawn import Pawn


class Board(RelativeLayout):
    """A graphical view onto a facade, resembling a game board."""
    layout = ObjectProperty()
    character = ObjectProperty()
    spot = DictProperty({})
    pawn = DictProperty({})
    arrow = DictProperty({})
    arrowlayout = ObjectProperty()
    spotlayout = ObjectProperty()
    pawnlayout = ObjectProperty()
    app = ObjectProperty()
    engine = ObjectProperty()
    spots_unposd = NumericProperty(0)
    selection = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        """Make a trigger for ``_redata`` and run it"""
        self._trigger_update = Clock.create_trigger(self._update)
        super().__init__(**kwargs)

    def make_pawn(self, thing):
        """Make a :class:`Pawn` to represent a :class:`Thing`"""
        if thing["name"] in self.pawn:
            raise KeyError("Already have a Pawn for this Thing")
        r = Pawn(
            board=self,
            thing=thing
        )
        self.pawn[thing["name"]] = r
        return r

    def make_spot(self, place):
        """Make a :class:`Spot` to represent a :class:`Place`"""
        if place["name"] in self.spot:
            raise KeyError("Already have a Spot for this Place")
        r = Spot(
            board=self,
            place=place
        )
        self.spot[place["name"]] = r
        return r

    def make_arrow(self, portal):
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

        @self.engine.on_time
        def ontime(*args):
            self._trigger_update()

        self._trigger_update()

    def upd_x_when_scrolling_stops(self, *args):
        if self.parent.effect_x.velocity < self.parent.effect_x.min_velocity:
            self.character.stat['_scroll_x'] = self.parent.scroll_x
            self.track_xvel = False
            return
        Clock.schedule_once(self.upd_x_when_scrolling_stops, 0.001)

    def track_x_vel(self, *args):
        if (
                not self.track_xvel and
                self.parent.effect_x.velocity >
                self.parent.effect_x.min_velocity
        ):
            self.upd_x_when_scrolling_stops()
            self.track_xvel = True

    def upd_y_when_scrolling_stops(self, *args):
        if self.parent.effect_y.velocity < self.parent.effect_y.min_velocity:
            self.character.stat['_scroll_y'] = self.parent.scroll_y
            self.track_yvel = False
            return
        Clock.schedule_once(self.upd_y_when_scrolling_stops, 0.001)

    def track_y_vel(self, *args):
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

    def _update(self, *args):
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
        spots_added = []
        for place_name in self.character.place:
            if place_name not in self.spot:
                spots_added.append(place_name)
                self.spotlayout.add_widget(
                    self.make_spot(self.character.place[place_name])
                )
        Logger.debug(
            "Board: added {} spots to {}'s board".format(
                len(spots_added),
                self.character.name
            )
        )
        self._new_spots = spots_added
        self._layout_tries = 5
        Clock.schedule_once(self.maybe_layout, 0)
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

    def maybe_layout(self, *args):
        if self._layout_tries <= 0:
            return
        if self.spots_unposd == 0:
            return
        if self.spots_unposd < len(self._new_spots):
            Logger.debug(
                'Board: {} spots of {} unpositioned, no layout'.format(
                    self.spots_unposd,
                    self._new_spots
                )
            )
            if self._layout_tries > 0:
                self._layout_tries = self._layout_tries - 1
                Clock.schedule_once(self.maybe_layout, 0)
            return
        # No spots have positions;
        # do a layout.
        Logger.debug('Board: layout!')
        from functools import partial
        spots_only = self.character.facade()
        for thing in list(spots_only.thing.keys()):
            del spots_only.thing[thing]
        l = self.grid_layout(spots_only)

        def position_spot(spot, x, y, *args):
            if not (spot.name and spot.remote and spot.mirror):
                Clock.schedule_once(partial(position_spot, spot, x, y), 0)
                return
            spot.pos = (
                int(x * self.width),
                int(y * self.height)
            )
            spot._trigger_upd_to_remote_pos()
        for spot in self.spot.values():
            position_spot(spot, *l[spot.name])
        Logger.debug(
            "Board: auto layout of spots"
        )
        self.spots_unposd = 0

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
