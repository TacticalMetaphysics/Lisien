# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The big widget that shows the graph of the selected Character."""
from functools import partial
from kivy.properties import (
    BooleanProperty,
    ReferenceListProperty,
    DictProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from .spot import Spot
from .arrow import Arrow, ArrowWidget
from .pawn import Pawn
from ..dummy import Dummy
from ..util import trigger


def normalize_layout(l):
    import numpy as np
    xs = []
    ys = []
    ks = []
    for (k, (x, y)) in l.items():
        xs.append(x)
        ys.append(y)
        ks.append(k)
    minx = np.min(xs)
    maxx = np.max(xs)
    try:
        xco = 0.98 / (maxx - minx)
        xnorm = np.multiply(xs, xco)
    except ZeroDivisionError:
        xnorm = np.array(xs)
    miny = np.min(ys)
    maxy = np.max(ys)
    try:
        yco = 0.98 / (maxy - miny)
        ynorm = np.multiply(ys, yco)
    except ZeroDivisionError:
        ynorm = np.array(ys)
    o = {}
    for i in range(len(ks)):
        o[ks[i]] = (xnorm[i], ynorm[i])
    return o


class KvLayoutBack(FloatLayout):
    """What to show behind the graph.

    By default, shows a static image.

    """
    character = ObjectProperty()


class KvLayoutFront(FloatLayout):
    """What to show in front of the graph.

    By default, shows nothing.

    """
    pass


class Board(RelativeLayout):
    """A graphical view onto a :class:`LiSE.Character`, resembling a game
    board.

    """
    engine = ObjectProperty()
    character = ObjectProperty()
    spot = DictProperty({})
    pawn = DictProperty({})
    arrow = DictProperty({})
    kvlayoutback = ObjectProperty()
    arrowlayout = ObjectProperty()
    spotlayout = ObjectProperty()
    pawnlayout = ObjectProperty()
    kvlayoutfront = ObjectProperty()
    wids = ReferenceListProperty(
        kvlayoutback,
        arrowlayout,
        spotlayout,
        pawnlayout,
        kvlayoutfront
    )
    spots_unposd = ListProperty([])
    layout_tries = NumericProperty(5)
    new_spots = ListProperty([])
    tracking_vel = BooleanProperty(False)
    branch = StringProperty('master')
    tick = NumericProperty(0)
    selection_candidates = ListProperty([])
    selection = ObjectProperty(allownone=True)
    keep_selection = ObjectProperty(False)
    adding_portal = BooleanProperty(False)
    reciprocal_portal = BooleanProperty(False)
    grabbing = BooleanProperty(True)
    grabbed = ObjectProperty(None, allownone=True)

    @property
    def widkwargs(self):
        return {
            'size_hint': (None, None),
            'size': self.size,
            'pos': (0, 0)
        }

    def on_touch_down(self, touch):
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        if not self.collide_point(*touch.pos):
            return
        if self.selection:
            self.selection.hit = self.selection.collide_point(*touch.pos)
            if self.selection.hit:
                Logger.debug("Board: hit selection")
                touch.grab(self.selection)
        pawns = list(self.pawns_at(*touch.pos))
        if pawns:
            Logger.debug("Board: hit {} pawns".format(len(pawns)))
            self.selection_candidates = pawns
            if self.selection in self.selection_candidates:
                self.selection_candidates.remove(self.selection)
            return True
        spots = list(self.spots_at(*touch.pos))
        if spots:
            Logger.debug("Board: hit {} spots".format(len(spots)))
            self.selection_candidates = spots
            if self.adding_portal:
                self.origspot = self.selection_candidates.pop(0)
                self.protodest = Dummy(
                    name='protodest',
                    pos=touch.pos,
                    size=(0, 0)
                )
                self.add_widget(self.protodest)
                self.protodest.on_touch_down(touch)
                self.protoportal = ArrowWidget(
                    origin=self.origspot,
                    destination=self.protodest
                )
                self.add_widget(self.protoportal)
                if self.reciprocal_portal:
                    self.protoportal2 = ArrowWidget(
                        destination=self.origspot,
                        origin=self.protodest
                    )
                    self.add_widget(self.protoportal2)
            return True
        if not self.selection_candidates:
            arrows = list(self.arrows_at(*touch.pos))
            if arrows:
                Logger.debug("Board: hit {} arrows".format(len(arrows)))
                self.selection_candidates = arrows
                return True

    def on_touch_move(self, touch):
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        if self.selection in self.selection_candidates:
            self.selection_candidates.remove(self.selection)
        if self.selection:
            if not self.selection_candidates:
                self.keep_selection = True
            return self.selection.dispatch('on_touch_move', touch)
        elif self.selection_candidates:
            for cand in self.selection_candidates:
                if cand.collide_point(*touch.pos):
                    if hasattr(self.selection, 'selected'):
                        self.selection.selected = False
                    if hasattr(self.selection, 'hit'):
                        self.selection.hit = False
                    self.selection = cand
                    cand.hit = cand.selected = True
                    touch.grab(cand)
                    return cand.dispatch('on_touch_move', touch)

    def portal_touch_up(self, touch):
        try:
            # If the touch ended upon a spot, and there isn't
            # already a portal between the origin and this
            # destination, create one.
            destspot = next(self.spots_at(*touch.pos))
            orig = self.origspot.remote
            dest = destspot.remote
            if not(
                orig.name in self.character.portal and
                dest.name in self.character.portal[orig.name]
            ):
                port = self.character.new_portal(
                    orig.name,
                    dest.name
                )
                self.arrowlayout.add_widget(
                    self.make_arrow(port)
                )
            # And another in the opposite direction if needed
                if (
                    hasattr(self, 'protoportal2') and not(
                            orig.name in self.character.preportal and
                            dest.name in self.character.preportal[orig.name]
                        )
                ):
                    deport = self.character.new_portal(
                        dest.name,
                        orig.name
                    )
                    self.arrowlayout.add_widget(
                        self.make_arrow(deport)
                    )
        except StopIteration:
            pass
        self.remove_widget(self.protoportal)
        if hasattr(self, 'protoportal2'):
            self.remove_widget(self.protoportal2)
            del self.protoportal2
        self.remove_widget(self.protodest)
        del self.protoportal
        del self.protodest

    def on_touch_up(self, touch):
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        self._lasttouch = touch
        if hasattr(self, 'protodest'):
            Logger.debug("Board: on_touch_up making a portal")
            touch.ungrab(self)
            return self.portal_touch_up(touch)
        if hasattr(self.selection, 'on_touch_up'):
            self.selection.dispatch('on_touch_up', touch)
        while self.selection_candidates:
            candidate = self.selection_candidates.pop(0)
            if candidate.collide_point(*touch.pos):
                Logger.debug("Board: selecting " + repr(candidate))
                if hasattr(self.selection, 'selected'):
                    self.selection.selected = False
                if hasattr(self.selection, 'hit'):
                    self.selection.hit = False
                if hasattr(self.selection, '_start'):
                    self.selection.pos = self.selection._start
                    del self.selection._start
                self.selection = candidate
                if hasattr(self.selection, 'selected'):
                    self.selection.selected = True
                if hasattr(self.selection, 'hit'):
                    self.selection.hit = True
                if (
                    hasattr(self.selection, 'thing') and not
                    hasattr(self.selection, '_start')
                ):
                    self.selection._start = tuple(self.selection.pos)
                self.keep_selection = True
        if not self.keep_selection:
            Logger.debug("Board: deselecting " + repr(self.selection))
            if hasattr(self.selection, 'selected'):
                self.selection.selected = False
            if hasattr(self.selection, 'hit'):
                self.selection.hit = False
            self.selection = None
        self.keep_selection = False
        touch.ungrab(self)
        return

    def on_parent(self, *args):
        if not self.parent or hasattr(self, '_parented'):
            return
        self._parented = True
        self.kvlayoutback = KvLayoutBack(
            character=self.character,
            pos=(0, 0)
        )
        self.bind(character=self.kvlayoutback.setter('character'))
        self.size = self.kvlayoutback.size
        self.kvlayoutback.bind(size=self.setter('size'))
        self.arrowlayout = FloatLayout(**self.widkwargs)
        self.spotlayout = FloatLayout(**self.widkwargs)
        self.pawnlayout = FloatLayout(**self.widkwargs)
        self.kvlayoutfront = KvLayoutFront(**self.widkwargs)
        for wid in self.wids:
            if wid != self.kvlayoutback:
                self.bind(size=wid.setter('size'))
            self.add_widget(wid)
        if hasattr(self.parent, 'effect_x'):
            self.parent.effect_x.bind(velocity=self.track_vel)
        if hasattr(self.parent, 'effect_y'):
            self.parent.effect_y.bind(velocity=self.track_vel)
        self.trigger_update()

    def on_character(self, *args):
        if self.character is None:
            return
        if self.parent is None:
            Clock.schedule_once(self.on_character, 0)
            return

        currently = tuple(self.engine.time)
        self.engine.time = (self.branch, self.tick)
        if hasattr(self.parent, 'scroll_x'):
            self.parent.scroll_x = self.character.stat.get('_scroll_x', 0.0)
        if hasattr(self.parent, 'scroll_y'):
            self.parent.scroll_y = self.character.stat.get('_scroll_y', 0.0)
        self.engine.time = currently

    @trigger
    def kv_updated(self, *args):
        for wid in self.wids:
            self.remove_widget(wid)
        self.kvlayoutback = KvLayoutBack(pos=(0, 0))
        self.kvlayoutfront = KvLayoutFront(**self.widkwargs)
        self.size = self.kvlayoutback.size
        self.kvlayoutback.bind(size=self.setter('size'))
        for wid in self.wids:
            self.add_widget(wid)

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
        if '_x' in place and '_y' in place:
            r.pos = (
                self.width * place['_x'],
                self.height * place['_y']
            )
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
            currently = tuple(self.engine.time)
            self.engine.time = (self.branch, self.tick)
            self.character.stat['_scroll_x'] = self.parent.scroll_x
            self.character.stat['_scroll_y'] = self.parent.scroll_y
            self.engine.time = currently
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

    def rm_pawn(self, name, *args):
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

    def rm_spot(self, name, *args):
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
        part = partial(self.rm_spot, name)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def rm_arrow(self, orig, dest, *args):
        """Remove the :class:`Arrow` that goes from ``orig`` to ``dest``."""
        if (
                orig not in self.arrow or
                dest not in self.arrow[orig]
        ):
            raise KeyError("No Arrow from {} to {}".format(orig, dest))
        self.arrowlayout.remove_widget(self.arrow[orig][dest])
        del self.arrow[orig][dest]

    def _trigger_rm_arrow(self, orig, dest):
        part = partial(self.rm_arrow, orig, dest)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def graph_layout(self, graph):
        from networkx.drawing.layout import spring_layout
        return normalize_layout(spring_layout(graph))

    def discard_pawn(self, thingn, *args):
        if thingn in self.pawn:
            self.rm_pawn(thingn)

    def _trigger_discard_pawn(self, thing):
        part = partial(self.discard_pawn, thing)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

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
        nodes_patch = {}
        for place_name in self.character.place:
            if place_name not in self.spot:
                place = self.character.place[place_name]
                spot = self.make_spot(place)
                patch = {}
                if '_image_paths' in place:
                    zeroes = [0] * len(place['_image_paths'])
                else:
                    patch['_image_paths'] = spot.default_image_paths
                    zeroes = [0]
                if '_offxs' not in place:
                    patch['_offxs'] = zeroes
                if '_offys' not in place:
                    patch['_offys'] = zeroes
                if '_stackhs' not in place:
                    patch['_stackhs'] = zeroes
                nodes_patch[place_name] = patch
                self.spotlayout.add_widget(spot)
                spots_added.append(spot)
        self.engine.handle(
            'update_nodes',
            char=self.character.name, 
            patch=nodes_patch
        )
        for spot in spots_added:
            spot.finalize()
        self.new_spots = spots_added

    def add_arrow(self, orign, destn, *args):
        if not (
            orign in self.character.portal and
            destn in self.character.portal[orign]
        ):
            raise ValueError("No portal for arrow {}->{}".format(orign, destn))
        if not (
                orign in self.arrow and
                destn in self.arrow[orign]
        ):
            self.arrowlayout.add_widget(
                self.make_arrow(
                    self.character.portal[orign][destn]
                )
            )
        assert self.arrow[orign][destn] in self.arrowlayout.children

    def _trigger_add_arrow(self, orign, destn):
        part = partial(self.add_arrow, orign, destn)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def add_new_arrows(self, *args):
        Logger.debug(
            "Board: adding new arrows to {}".format(
                self.character.name
            )
        )
        self._new_arrows_todo = set()
        for arrow_orig in self.character.portal:
            for arrow_dest in self.character.portal[arrow_orig]:
                if (
                        arrow_orig not in self.arrow or
                        arrow_dest not in self.arrow[arrow_orig]
                ):
                    self._new_arrows_todo.add((arrow_orig, arrow_dest))
        self._pop_new_arrow()

    def _pop_new_arrow(self, *args):
        notyet = set()
        try:
            (orig, dest) = self._new_arrows_todo.pop()
        except KeyError:
            return
        while not (orig in self.spot and dest in self.spot):
            notyet.add((orig, dest))
            try:
                (orig, dest) = self._new_arrows_todo.pop()
            except KeyError:
                self._new_arrows_todo.update(notyet)
                return
        if not (orig in self.spot and dest in self.spot) or (
                orig in self.arrow and dest in self.arrow[orig]
        ):
            self._new_arrows_todo.update(notyet)
            return
        self.arrowlayout.add_widget(
            self.make_arrow(
                self.character.portal[orig][dest]
            )
        )
        self._new_arrows_todo.update(notyet)
        if self._new_arrows_todo:
            Clock.schedule_once(self._pop_new_arrow, 0)

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
        part = partial(self.add_pawn, thingn)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def add_new_pawns(self, *args):
        Logger.debug(
            "Board: adding new pawns to {}".format(
                self.character.name
            )
        )
        nodes_patch = {}
        pawns_added = []
        for (thing_name, thing) in self.character.thing.items():
            if thing_name not in self.pawn:
                pwn = self.make_pawn(thing)
                pawns_added.append(pwn)
                patch = {}
                if '_image_paths' in thing:
                    zeroes = [0] * len(thing['_image_paths'])
                else:
                    patch['_image_paths'] = Pawn.default_image_paths
                    zeroes = [0] * len(Pawn.default_image_paths)
                if '_offxs' not in thing:
                    patch['_offxs'] = zeroes
                if '_offys' not in thing:
                    patch['_offys'] = zeroes
                if '_stackhs' not in thing:
                    patch['_stackhs'] = zeroes
                nodes_patch[thing_name] = patch
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
        self.engine.handle(
            'update_nodes',
            char=self.character.name,
            patch=nodes_patch
        )
        for pwn in pawns_added:
            pwn.finalize()

    @trigger
    def trigger_update(self, *args):
        """Force an update to match the current state of my character.

        This polls every element of the character, and therefore
        causes me to sync with the LiSE core for a long time. Avoid
        when possible.

        """

        # remove widgets that don't represent anything anymore
        Logger.debug("Board: updating")
        currently = tuple(self.engine.time)
        self.engine.time = (self.branch, self.tick)
        self.remove_absent_pawns()
        self.remove_absent_spots()
        self.remove_absent_arrows()
        # add widgets to represent new stuff
        self.add_new_spots()
        self.add_new_arrows()
        self.add_new_pawns()
        self.spots_unposd = [
            spot for spot in self.spot.values()
            if not ('_x' in spot.remote and '_y' in spot.remote)
        ]
        self.engine.time = currently

    def update_from_diff(self, chardiff, *args):
        """Apply the changes described in the dict ``chardiff``."""
        for (place, extant) in chardiff['places'].items():
            if extant and place not in self.spot:
                self.add_spot(place)
                spot = self.spot[place]
                if '_x' not in spot.place or '_y' not in spot.place:
                    self.new_spots.append(spot)
                    self.spots_unposd.append(spot)
            elif not extant and place in self.spot:
                self.rm_spot(place)
        for (thing, extant) in chardiff['things'].items():
            if extant and thing not in self.pawn:
                self.add_pawn(thing)
            elif not extant and thing in self.pawn:
                self.rm_pawn(thing)
        for (node, stats) in chardiff['node_stat'].items():
            if node in self.spot:
                spot = self.spot[node]
                if '_x' in stats:
                    spot.x = stats['_x'] * self.width
                if '_y' in stats:
                    spot.y = stats['_y'] * self.height
                if '_image_paths' in stats:
                    spot.paths = stats['_image_paths']
            elif node in self.pawn:
                pawn = self.pawn[node]
                if 'location' in stats:
                    pawn.loc_name = stats['location']
                if 'next_location' in stats:
                    pawn.next_loc_name = stats['next_location']
                if '_image_paths' in stats:
                    pawn.paths = stats['_image_paths']
            else:
                raise ValueError(
                    "Diff tried to change stats of "
                    "nonexistent node {}".format(node)
                )
        for ((orig, dest), extant) in chardiff['portals'].items():
            if extant and (orig not in self.arrow or dest not in self.arrow[orig]):
                self.add_arrow(orig, dest)
            elif not extant and orig in self.arrow and dest in self.arrow[orig]:
                self.rm_arrow(orig, dest)

    def trigger_update_from_diff(self, chardiff, *args):
        part = partial(self.update_from_diff, chardiff)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

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
        for spot in self.new_spots:
            if not (spot.name and spot.remote):
                Clock.schedule_once(self.nx_layout, 0)
                return
        spots_only = self.character.facade()
        for thing in list(spots_only.thing.keys()):
            del spots_only.thing[thing]
        l = self.graph_layout(spots_only)

        node_upd = {}

        for spot in self.new_spots:
            (x, y) = l[spot.name]
            assert 0 <= x <= 0.98
            assert 0 <= y <= 0.98
            node_upd[spot.remote.name] = {
                '_x': x,
                '_y': y
            }
            spot.pos = (
                int(x * self.width),
                int(y * self.height)
            )
        self.engine.handle(
            'update_nodes',
            char=self.character.name,
            patch=node_upd
        )
        self.new_spots = self.spots_unposd = []

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


class BoardView(ScrollView):
    """A ScrollView that contains the Board for the character being
    viewed.

    """
    screen = ObjectProperty()
    engine = ObjectProperty()
    character = ObjectProperty()
    board = ObjectProperty()
    branch = StringProperty('master')
    tick = NumericProperty(0)
    selection_candidates = ListProperty([])
    selection = ObjectProperty(allownone=True)
    keep_selection = BooleanProperty(False)
    adding_portal = BooleanProperty(False)
    reciprocal_portal = BooleanProperty(False)

    def on_touch_down(self, touch):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if self.board and self.board.dispatch('on_touch_down', touch):
            touch.pop()
            return True
        touch.pop()
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if self.board and self.board.dispatch('on_touch_move', touch):
            touch.pop()
            return True
        touch.pop()
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if self.board and self.board.dispatch('on_touch_up', touch):
            touch.pop()
            return True
        touch.pop()
        return super().on_touch_up(touch)

    def spot_from_dummy(self, dummy):
        """Create a new :class:`board.Spot` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        position, and imagery of the provided dummy.

        """
        (x, y) = self.to_local(*dummy.pos_up)
        x /= self.board.width
        y /= self.board.height
        self.board.spotlayout.add_widget(
            self.board.make_spot(
                self.board.character.new_place(
                    dummy.name,
                    _x=x,
                    _y=y,
                    _image_paths=list(dummy.paths)
                )
            )
        )
        dummy.num += 1

    def pawn_from_dummy(self, dummy):
        """Create a new :class:`board.Pawn` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        location, and imagery of the provided dummy.

        """
        dummy.pos = self.to_local(*dummy.pos)
        for spot in self.board.spotlayout.children:
            if spot.collide_widget(dummy):
                whereat = spot
                break
        else:
            return
        whereat.add_widget(
            self.board.make_pawn(
                self.board.character.new_thing(
                    dummy.name,
                    whereat.place.name,
                    _image_paths=list(dummy.paths)
                )
            )
        )
        dummy.num += 1

    def arrow_from_wid(self, wid):
        """When the user has released touch after dragging to make an arrow,
        check whether they've drawn a valid one, and if so, make it.

        This doesn't handle touch events. It takes a widget as its
        argument: the one the user has been dragging to indicate where
        they want the arrow to go. Said widget ought to be invisible.

        """
        for spot in self.board.spotlayout.children:
            if spot.collide_widget(wid):
                whereto = spot
                break
        else:
            return
        self.board.arrowlayout.add_widget(
            self.board.make_arrow(
                self.board.character.new_portal(
                    self.board.grabbed.place.name,
                    whereto.place.name,
                    reciprocal=self.reciprocal_portal
                )
            )
        )

Builder.load_string(
    """
#: import StiffScrollEffect ELiDE.kivygarden.stiffscroll.StiffScrollEffect
#: import resource_find kivy.resources.resource_find
<KvLayoutBack>:
    size: wallpaper.size
    size_hint: (None, None)
    Image:
        id: wallpaper
        source: resource_find(root.character.stat.get('wallpaper', 'wallpape.jpg')) if root.character else ''
        size_hint: (None, None)
        size: self.texture.size if self.texture else (1, 1)
        pos: root.pos
<BoardView>:
    effect_cls: StiffScrollEffect
    board: board
    selection_candidates: board.selection_candidates
    selection: board.selection
    keep_selection: board.keep_selection
    adding_portal: board.adding_portal
    reciprocal_portal: board.reciprocal_portal
    Board:
        size_hint: None, None
        id: board
        branch: root.branch
        tick: root.tick
        engine: root.engine
        character: root.character
        keep_selection: root.keep_selection
        adding_portal: root.adding_portal
        reciprocal_portal: root.reciprocal_portal
"""
)
