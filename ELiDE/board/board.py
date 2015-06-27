# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from functools import partial
from kivy.properties import (
    BooleanProperty,
    StringProperty,
    ReferenceListProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty
)
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from .spot import Spot
from .arrow import Arrow
from .pawn import Pawn


class Board(RelativeLayout):
    """A graphical view onto a :class:`LiSE.Character`, resembling a game
    board.

    """
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
    tracking_vel = BooleanProperty(False)

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
        self.rebind()
        self._old_character = self.character

        self.parent.scroll_x = self.character.stat.get('_scroll_x', 0.0)
        self.parent.scroll_y = self.character.stat.get('_scroll_y', 0.0)
        self.parent.effect_x.bind(velocity=self.track_vel)
        self.parent.effect_y.bind(velocity=self.track_vel)

        self._trigger_update()

    def rebind(self, *args):
        """Bind my listeners to the new character, unbinding from the old
        character first if needed.

        """
        if hasattr(self, '_old_character'):
            self._old_character.thing.unlisten(
                self.char_thing_listener
            )
            self._old_character.place.unlisten(
                self.char_place_listener
            )
            self._old_character.portal.unlisten(
                self.char_portal_listener
            )
            del self._old_character
        self.character.thing.listener(self.char_thing_listener)
        self.character.place.listener(self.char_place_listener)
        self.character.portal.listener(self.char_portal_listener)

    def char_place_listener(self, branch, tick, mapping, place, extant):
        if extant:
            self._trigger_add_spot(place)
        else:
            self._trigger_discard_spot(place)

    def char_thing_listener(self, branch, tick, mapping, thing, extant):
        if extant:
            self._trigger_add_pawn(thing)
        else:
            self._trigger_discard_pawn(thing)

    def char_portal_listener(self, branch, tick, mapping, orig, dest, extant):
        if extant:
            self._trigger_add_arrow(orig, dest)
        else:
            self._trigger_discard_arrow(orig, dest)

    def track_vel(self, *args):
        """Track scrolling once it starts, so that we can tell when it
        stops.

        """
        if (
                not self.tracking_vel and (
                    self.parent.effect_x.velocity > 0 or
                    self.parent.effect_y.velocity > 0
                )
        ):
            self.upd_pos_when_scrolling_stops()
            self.tracking_vel = True

    def upd_pos_when_scrolling_stops(self, *args):
        """Wait for the scroll to stop, then store where it ended."""
        if self.parent.effect_x.velocity \
           == self.parent.effect_y.velocity == 0:
            self.character.stat['_scroll_x'] = self.parent.scroll_x
            self.character.stat['_scroll_y'] = self.parent.scroll_y
            self.tracking_vel = False
            return
        Clock.schedule_once(self.upd_pos_when_scrolling_stops, 0.001)

    def rm_arrows_to_and_from(self, name):
        origs = list(self.arrow.keys())
        if name in origs:
            origs.remove(name)
            for dest in list(self.arrow[name].keys()):
                self.rm_arrow(name, dest)
        for orig in origs:
            if name in self.arrow[orig]:
                self.rm_arrow(orig, name)

    def rm_pawn(self, name):
        """Remove the :class:`Pawn` by the given name."""
        if name not in self.pawn:
            raise KeyError("No Pawn named {}".format(name))
        # Currently there's no way to connect Pawns with Arrows but I
        # think there will be, so, insurance
        self.rm_arrows_to_and_from(name)
        pwn = self.pawn[name]
        pwn.parent.remove_widget(pwn)
        for canvas in (
                self.pawnlayout.canvas.after,
                self.pawnlayout.canvas.before,
                self.pawnlayout.canvas
        ):
            if pwn.group in canvas.children:
                canvas.remove(pwn.group)
        pwn.canvas.clear()
        del self.pawn[name]

    def _trigger_rm_pawn(self, name):
        Clock.schedule_once(partial(self.rm_pawn, name), 0)

    def rm_spot(self, name):
        """Remove the :class:`Spot` by the given name."""
        if name not in self.spot:
            raise KeyError("No Spot named {}".format(name))
        spot = self.spot[name]
        pawns_here = list(spot.children)
        self.rm_arrows_to_and_from(name)
        self.spotlayout.remove_widget(spot)
        spot.canvas.clear()
        del self.spot[name]
        for pawn in pawns_here:
            self.rm_pawn(pawn.name)

    def _trigger_rm_spot(self, name):
        Clock.schedule_once(partial(self.rm_spot, name), 0)

    def rm_arrow(self, orig, dest):
        """Remove the :class:`Arrow` that goes from ``orig`` to ``dest``."""
        if (
                orig not in self.arrow or
                dest not in self.arrow[orig]
        ):
            raise KeyError("No Arrow from {} to {}".format(orig, dest))
        self.arrowlayout.remove_widget(self.arrow[orig][dest])
        del self.arrow[orig][dest]

    def _trigger_rm_arrow(self, orig, dest):
        Clock.schedule_once(partial(self.rm_arrow, orig, dest), 0)

    def grid_layout(self, graph):
        from networkx import spectral_layout
        return spectral_layout(graph)

    def discard_pawn(self, thingn, *args):
        if thingn in self.pawn:
            self.rm_pawn(thingn)

    def _trigger_discard_pawn(self, thing):
        Clock.schedule_once(partial(self.discard_pawn, thing), 0)

    def remove_absent_pawns(self, *args):
        Logger.debug(
            "Board: removing pawns absent from {}".format(
                self.character.name
            )
        )
        for pawn_name in list(self.pawn.keys()):
            if pawn_name not in self.character.thing:
                self.rm_pawn(pawn_name)

    def discard_spot(self, placen, *args):
        if placen in self.spot:
            self.rm_spot(placen)

    def _trigger_discard_spot(self, place):
        Clock.schedule_once(partial(self.discard_spot, place), 0)

    def remove_absent_spots(self, *args):
        Logger.debug(
            "Board: removing spots absent from {}".format(
                self.character.name
            )
        )
        for spot_name in list(self.spot.keys()):
            if spot_name not in self.character.place:
                self.rm_spot(spot_name)

    def discard_arrow(self, orign, destn, *args):
        if (
            orign in self.arrow and
            destn in self.arrow[orign]
        ):
            self.rm_arrow(orign, destn)

    def _trigger_discard_arrow(self, orig, dest):
        Clock.schedule_once(partial(self.discard_arrow, orig, dest), 0)

    def remove_absent_arrows(self, *args):
        Logger.debug(
            "Board: removing arrows absent from {}".format(
                self.character.name
            )
        )
        for arrow_origin in list(self.arrow.keys()):
            for arrow_destination in list(self.arrow[arrow_origin].keys()):
                if (
                        arrow_origin not in self.character.portal or
                        arrow_destination not in
                        self.character.portal[arrow_origin]
                ):
                    self.rm_arrow(arrow_origin, arrow_destination)

    def add_spot(self, placen, *args):
        if (
            placen in self.character.place and
            placen not in self.spot
        ):
            self.spotlayout.add_widget(
                self.make_spot(self.character.place[placen])
            )

    def _trigger_add_spot(self, placen):
        Clock.schedule_once(partial(self.add_spot, placen), 0)

    def add_new_spots(self, *args):
        Logger.debug(
            "Board: adding new spots to {}".format(
                self.character.name
            )
        )
        spots_added = []
        for place_name in self.character.place:
            if place_name not in self.spot:
                spot = self.make_spot(self.character.place[place_name])
                self.spotlayout.add_widget(spot)
                spots_added.append(spot)
        self.new_spots = spots_added

    def add_arrow(self, orign, destn, *args):
        if not (
                orign in self.arrow and
                destn in self.arrow[orign]
        ):
            self.arrowlayout.add_widget(
                self.make_arrow(
                    self.character.portal[orign][destn]
                )
            )

    def _trigger_add_arrow(self, orign, destn):
        Clock.schedule_once(partial(self.add_arrow, orign, destn), 0)

    def add_new_arrows(self, *args):
        Logger.debug(
            "Board: adding new arrows to {}".format(
                self.character.name
            )
        )
        for arrow_orig in self.character.portal:
            for arrow_dest in self.character.portal[arrow_orig]:
                if (
                        arrow_orig not in self.arrow or
                        arrow_dest not in self.arrow[arrow_orig]
                ):
                    self.arrowlayout.add_widget(
                        self.make_arrow(
                            self.character.portal[arrow_orig][arrow_dest]
                        )
                    )

    def add_pawn(self, thingn, *args):
        if (
            thingn in self.character.thing and
            thingn not in self.pawn
        ):
            pwn = self.make_pawn(self.character.thing[thingn])
            locn = pwn.thing['location']
            nextlocn = pwn.thing['next_location']
            if nextlocn is None:
                self.add_spot(nextlocn)
                whereat = self.spot[nextlocn]
            else:
                self.add_arrow(locn, nextlocn)
                whereat = self.arrow[locn][nextlocn]
            whereat.add_widget(pwn)
            self.pawn[thingn] = pwn

    def _trigger_add_pawn(self, thingn):
        Clock.schedule_once(partial(self.add_pawn, thingn), 0)

    def add_new_pawns(self, *args):
        Logger.debug(
            "Board: adding new pawns to {}".format(
                self.character.name
            )
        )
        for thing_name in self.character.thing:
            if thing_name not in self.pawn:
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

    def update(self, *args):
        """Force an update to match the current state of my character.

        This polls every element of the character, and therefore
        causes me to sync with the LiSE core for a long time. Avoid
        when possible.

        """

        # remove widgets that don't represent anything anymore
        Logger.debug("Board: updating")
        self.remove_absent_pawns()
        self.remove_absent_spots()
        self.remove_absent_arrows()
        # add widgets to represent new stuff
        self.add_new_spots()
        self.add_new_arrows()
        self.add_new_pawns()
        self.spots_unposd = [
            spot for spot in self.spot.values()
            if not ('_x' in spot.mirror and '_y' in spot.mirror)
        ]

    def on_spots_unposd(self, *args):
        # TODO: If only some spots are unpositioned, and they remain
        # that way for several frames, put them somewhere that the
        # user will be able to find.
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
        """Iterate over all my arrows."""
        for o in self.arrow.values():
            for arro in o.values():
                yield arro

    def pawns_at(self, x, y):
        """Iterate over pawns that collide the given point."""
        for pawn in self.pawn.values():
            if pawn.collide_point(x, y):
                yield pawn

    def spots_at(self, x, y):
        """Iterate over spots that collide the given point."""
        for spot in self.spot.values():
            if spot.collide_point(x, y):
                yield spot

    def arrows_at(self, x, y):
        """Iterate over arrows that collide the given point."""
        for arrow in self.arrows():
            if arrow.collide_point(x, y):
                yield arrow


Builder.load_string(
    """
<Board>:
    size_hint: (None, None)
    size: wallpaper.size
    arrowlayout: arrowlayout
    spotlayout: spotlayout
    pawnlayout: pawnlayout
    Image:
        id: wallpaper
        source: resource_find(root.character.stat['_wallpaper']) \
        if root.character is not None and \
        '_wallpaper' in root.character.stat else \
        resource_find('wallpape.jpg')
        size_hint: (None, None)
        size: self.texture_size
    FloatLayout:
        id: arrowlayout
    FloatLayout:
        id: spotlayout
    Widget:
        id: pawnlayout
"""
)
