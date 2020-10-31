# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import ScatterPlane
from kivy.graphics.transformation import Matrix
from kivy.vector import Vector

from .spot import GraphSpot
from .arrow import GraphArrow, GraphArrowWidget, ArrowLayout, get_points_multi
from .pawn import Pawn
from ..dummy import Dummy
from ..util import trigger
from ..boardview import BoardView
import numpy as np


def normalize_layout(l):
    """Make sure all the spots in a layout are where you can click.

    Returns a copy of the layout with all spot coordinates are
    normalized to within (0.0, 0.98).

    """
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
        xnorm = np.multiply(np.subtract(xs, [minx] * len(xs)), xco)
    except ZeroDivisionError:
        xnorm = np.array([0.5] * len(xs))
    miny = np.min(ys)
    maxy = np.max(ys)
    try:
        yco = 0.98 / (maxy - miny)
        ynorm = np.multiply(np.subtract(ys, [miny] * len(ys)), yco)
    except ZeroDivisionError:
        ynorm = np.array([0.5] * len(ys))
    return dict(zip(ks, zip(map(float, xnorm), map(float, ynorm))))


class KvLayoutBack(FloatLayout):
    """What to show behind the graph.

    By default, shows nothing.

    """


class KvLayoutFront(FloatLayout):
    """What to show in front of the graph.

    By default, shows nothing.

    """
    pass


class FinalLayout(FloatLayout):
    def finalize_all(self, *args):
        for child in self.children:
            child.finalize()
        self.bind(children=self._trigger_finalize_all)
    
    _trigger_finalize_all = trigger(finalize_all)


class GraphBoard(RelativeLayout):
    """A graphical view onto a :class:`LiSE.Character`, resembling a game
    graph.

    """
    app = ObjectProperty()
    character = ObjectProperty()
    wallpaper_path = StringProperty()
    spot = DictProperty({})
    pawn = DictProperty({})
    arrow = DictProperty({})
    wallpaper = ObjectProperty()
    kvlayoutback = ObjectProperty()
    arrowlayout = ObjectProperty()
    spotlayout = ObjectProperty()
    kvlayoutfront = ObjectProperty()
    wids = ReferenceListProperty(
        wallpaper,
        kvlayoutback,
        arrowlayout,
        spotlayout,
        kvlayoutfront
    )
    spots_unposd = ListProperty([])
    layout_tries = NumericProperty(5)
    tracking_vel = BooleanProperty(False)
    selection_candidates = ListProperty([])
    selection = ObjectProperty(allownone=True)
    keep_selection = ObjectProperty(False)
    adding_portal = BooleanProperty(False)
    reciprocal_portal = BooleanProperty(False)
    grabbing = BooleanProperty(True)
    grabbed = ObjectProperty(None, allownone=True)
    spot_cls = ObjectProperty(GraphSpot)
    pawn_cls = ObjectProperty(Pawn)
    arrow_cls = ObjectProperty(GraphArrow, allownone=True)
    proto_arrow_cls = ObjectProperty(GraphArrowWidget)
    _scheduled_rm_spot = DictProperty()
    _scheduled_rm_arrow = DictProperty()
    _scheduled_discard_pawn = DictProperty()
    _scheduled_add_pawn = DictProperty()

    @property
    def widkwargs(self):
        return {
            'size_hint': (None, None),
            'size': self.size,
            'pos': (0, 0)
        }

    def on_touch_down(self, touch):
        """Check for collisions and select an appropriate entity."""
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        if not self.collide_point(*touch.pos):
            return
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if self.app.selection:
            if self.app.selection.collide_point(*touch.pos):
                Logger.debug("Board: hit selection")
                touch.grab(self.app.selection)
        pawns = list(self.pawns_at(*touch.pos))
        if pawns:
            Logger.debug("Board: hit {} pawns".format(len(pawns)))
            self.selection_candidates = pawns
            if self.app.selection in self.selection_candidates:
                self.selection_candidates.remove(self.app.selection)
            touch.pop()
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
                self.protoportal = self.proto_arrow_cls(
                    origin=self.origspot,
                    destination=self.protodest
                )
                self.add_widget(self.protoportal)
                if self.reciprocal_portal:
                    self.protoportal2 = self.proto_arrow_cls(
                        destination=self.origspot,
                        origin=self.protodest
                    )
                    self.add_widget(self.protoportal2)
            touch.pop()
            return True
        arrows = list(self.arrows_at(*touch.pos))
        if arrows:
            Logger.debug("Board: hit {} arrows".format(len(arrows)))
            self.selection_candidates = arrows
            if self.app.selection in self.selection_candidates:
                self.selection_candidates.remove(self.app.selection)
            if isinstance(self.app.selection, GraphArrow) and self.app.selection.reciprocal in self.selection_candidates:
                self.selection_candidates.remove(self.app.selection.reciprocal)
            touch.pop()
            return True
        touch.pop()

    def on_touch_move(self, touch):
        """If an entity is selected, drag it."""
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        if self.app.selection in self.selection_candidates:
            self.selection_candidates.remove(self.app.selection)
        if self.app.selection:
            if not self.selection_candidates:
                self.keep_selection = True
            ret = super().on_touch_move(touch)
            return ret
        elif self.selection_candidates:
            for cand in self.selection_candidates:
                if cand.collide_point(*touch.pos):
                    self.app.selection = cand
                    cand.selected = True
                    touch.grab(cand)
                    ret = super().on_touch_move(touch)
                    return ret

    def portal_touch_up(self, touch):
        """Try to create a portal between the spots the user chose."""
        try:
            # If the touch ended upon a spot, and there isn't
            # already a portal between the origin and this
            # destination, create one.
            destspot = next(self.spots_at(*touch.pos))
            orig = self.origspot.proxy
            dest = destspot.proxy
            if not(
                orig.name in self.character.portal and
                dest.name in self.character.portal[orig.name]
            ):
                symmetrical = hasattr(self, 'protoportal2') and not (
                        orig.name in self.character.preportal and
                        dest.name in self.character.preportal[orig.name]
                )
                port = self.character.new_portal(
                    orig.name,
                    dest.name,
                    symmetrical=symmetrical
                )
                self.arrowlayout.add_widget(
                    self.make_arrow(port)
                )
                if symmetrical:
                    self.arrowlayout.add_widget(
                        self.make_arrow(self.character.portal[dest.name][orig.name])
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
        """Delegate touch handling if possible, else select something."""
        if hasattr(self, '_lasttouch') and self._lasttouch == touch:
            return
        self._lasttouch = touch
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if hasattr(self, 'protodest'):
            Logger.debug("Board: on_touch_up making a portal")
            touch.ungrab(self)
            ret = self.portal_touch_up(touch)
            touch.pop()
            return ret
        if self.app.selection and hasattr(self.app.selection, 'on_touch_up'):
            self.app.selection.dispatch('on_touch_up', touch)
        for candidate in self.selection_candidates:
            if candidate.collide_point(*touch.pos):
                if hasattr(candidate, 'selected'):
                    if isinstance(candidate, GraphArrow) and candidate.reciprocal:
                        # mirror arrows can't be selected directly, you have to work with the one they mirror
                        if candidate.portal.get('is_mirror', False):
                            candidate.selected = True
                            candidate = candidate.reciprocal
                        elif candidate.reciprocal and candidate.reciprocal.portal.get('is_mirror', False):
                            candidate.reciprocal.selected = True
                    candidate.selected = True
                if hasattr(self.app.selection, 'selected'):
                    self.app.selection.selected = False
                    if isinstance(self.app.selection, GraphArrow) and self.app.selection.reciprocal\
                            and candidate is not self.app.selection.reciprocal:
                        self.app.selection.reciprocal.selected = False
                self.app.selection = candidate
                self.keep_selection = True
                parent = candidate.parent
                parent.remove_widget(candidate)
                parent.add_widget(candidate)
                break
        if not self.keep_selection:
            Logger.debug("Board: deselecting " + repr(self.app.selection))
            if hasattr(self.app.selection, 'selected'):
                self.app.selection.selected = False
                if isinstance(self.app.selection, GraphArrow) and self.app.selection.reciprocal:
                    self.app.selection.reciprocal.selected = False
            self.app.selection = None
        self.keep_selection = False
        touch.ungrab(self)
        touch.pop()
        return

    def _pull_size(self, *args):
        if self.wallpaper.texture is None:
            Clock.schedule_once(self._pull_size, 0.001)
            return
        self.size = self.wallpaper.size = self.wallpaper.texture.size

    def _pull_image(self, *args):
        self.wallpaper.source = self.wallpaper_path
        self._pull_size()

    def on_parent(self, *args):
        """Create some subwidgets and trigger the first update."""
        if not self.parent or hasattr(self, '_parented'):
            return
        if not self.wallpaper_path:
            Logger.debug("Board: waiting for wallpaper_path")
            Clock.schedule_once(self.on_parent, 0)
            return
        self._parented = True
        self.wallpaper = Image(source=self.wallpaper_path)
        self.bind(wallpaper_path=self._pull_image)
        self._pull_size()
        self.kvlayoutback = KvLayoutBack(**self.widkwargs)
        self.arrowlayout = ArrowLayout(**self.widkwargs)
        self.spotlayout = FinalLayout(**self.widkwargs)
        self.kvlayoutfront = KvLayoutFront(**self.widkwargs)
        for wid in self.wids:
            self.add_widget(wid)
            wid.pos = 0, 0
            wid.size = self.size
            if wid is not self.wallpaper:
                self.bind(size=wid.setter('size'))
        self.update()

    def on_character(self, *args):
        if self.character is None:
            return
        if self.parent is None:
            Clock.schedule_once(self.on_character, 0)
            return

        self.engine = getattr(self.character, 'engine', None)
        self.wallpaper_path = self.character.stat.setdefault('wallpaper', 'wallpape.jpg')
        if '_control' not in self.character.stat or 'wallpaper' not in self.character.stat['_control']:
            control = self.character.stat.setdefault('_control', {})
            control['wallpaper'] = 'textinput'
            self.character.stat['_control'] = control
        self.character.stat.connect(self._trigger_pull_wallpaper)
        self.trigger_update()

    def pull_wallpaper(self, *args):
        self.wallpaper_path = self.character.stat.setdefault('wallpaper', 'wallpape.jpg')

    def _trigger_pull_wallpaper(self, *args, **kwargs):
        if kwargs['key'] != 'wallpaper':
            return
        if hasattr(self, '_scheduled_pull_wallpaper'):
            Clock.unschedule(self._scheduled_pull_wallpaper)
        self._scheduled_pull_wallpaper = Clock.schedule_once(self.pull_wallpaper, 0)

    @trigger
    def kv_updated(self, *args):
        self.unbind(wallpaper_path=self.kvlayoutback.setter('wallpaper_path'))
        for wid in self.wids:
            self.remove_widget(wid)
        self.kvlayoutback = KvLayoutBack(pos=(0, 0), wallpaper_path=self.wallpaper_path)
        self.bind(wallpaper_path=self.kvlayoutback.setter('wallpaper_path'))
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
        r = self.pawn_cls(
            board=self,
            proxy=thing
        )
        self.pawn[thing["name"]] = r
        return r

    def make_spot(self, place):
        """Make a :class:`Spot` to represent a :class:`Place`, store it, and
        return it.

        """
        if place["name"] in self.spot:
            raise KeyError("Already have a Spot for this Place")
        r = self.spot_cls(
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
        return self._core_make_arrow(portal, self.spot[portal['origin']], self.spot[portal['destination']], self.arrow)

    def _core_make_arrow(self, portal, origspot, destspot, arrowmap, points=None):
        if points is None:
            r = self.arrow_cls(
                board=self,
                portal=portal,
                origspot=origspot,
                destspot=destspot,
            )
            r._trigger_repoint()
        else:
            r = self.arrow_cls(
                board=self,
                portal=portal,
                origspot=origspot,
                destspot=destspot,
                points=points
            )
        orign = portal["origin"]
        if orign not in arrowmap:
            arrowmap[orign] = {}
        arrowmap[orign][portal["destination"]] = r
        return r

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
        pwn = self.pawn.pop(name)
        if pwn in self.selection_candidates:
            self.selection_candidates.remove(pwn)
        pwn.parent.remove_widget(pwn)

    def _trigger_rm_pawn(self, name):
        Clock.schedule_once(partial(self.rm_pawn, name), 0)

    def rm_spot(self, name, *args):
        """Remove the :class:`Spot` by the given name."""
        if name not in self.spot:
            raise KeyError("No Spot named {}".format(name))
        spot = self.spot.pop(name)
        if spot in self.selection_candidates:
            self.selection_candidates.remove(spot)
        pawns_here = list(spot.children)
        self.rm_arrows_to_and_from(name)
        self.spotlayout.remove_widget(spot)
        spot.canvas.clear()
        for pawn in pawns_here:
            self.rm_pawn(pawn.name)
        if name in self._scheduled_rm_spot:
            del self._scheduled_rm_spot[name]

    def _trigger_rm_spot(self, name):
        part = partial(self.rm_spot, name)
        if name in self._scheduled_rm_spot:
            Clock.unschedule(self._scheduled_rm_spot[name])
        self._scheduled_rm_spot[name] = Clock.schedule_once(part, 0)

    def rm_arrow(self, orig, dest, *args):
        """Remove the :class:`Arrow` that goes from ``orig`` to ``dest``."""
        if (
                orig not in self.arrow or
                dest not in self.arrow[orig]
        ):
            raise KeyError("No Arrow from {} to {}".format(orig, dest))
        arr = self.arrow[orig].pop(dest)
        if arr in self.selection_candidates:
            self.selection_candidates.remove(arr)
        self.arrowlayout.remove_widget(arr)
        if (orig, dest) in self._scheduled_rm_arrow:
            del self._scheduled_rm_arrow[orig, dest]

    def _trigger_rm_arrow(self, orig, dest):
        part = partial(self.rm_arrow, orig, dest)
        if (orig, dest) in self._scheduled_rm_arrow:
            Clock.unschedule(self._scheduled_rm_arrow[orig, dest])
        self._scheduled_rm_arrow[orig, dest] = Clock.schedule_once(part, 0)

    def graph_layout(self, graph):
        from networkx.drawing.layout import spring_layout
        return normalize_layout(spring_layout(graph))

    def discard_pawn(self, thingn, *args):
        if thingn in self.pawn:
            self.rm_pawn(thingn)
        if thingn in self._scheduled_discard_pawn:
            del self._scheduled_discard_pawn[thingn]

    def _trigger_discard_pawn(self, thing):
        part = partial(self.discard_pawn, thing)
        if thing in self._scheduled_discard_pawn:
            Clock.unschedule(self._scheduled_discard_pawn[thing])
        self._scheduled_discard_pawn[thing] = Clock.schedule_once(part, 0)

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
        places2add = []
        spots_unposd = []
        nodes_patch = {}
        placemap = self.character.place
        spotmap = self.spot
        default_image_paths = GraphSpot.default_image_paths
        for place_name in placemap:
            if place_name not in spotmap:
                place = placemap[place_name]
                places2add.append(place)
                patch = {}
                if '_image_paths' in place:
                    zeroes = [0] * len(place['_image_paths'])
                else:
                    patch['_image_paths'] = default_image_paths
                    zeroes = [0]
                if '_offxs' not in place:
                    patch['_offxs'] = zeroes
                if '_offys' not in place:
                    patch['_offys'] = zeroes
                if patch:
                    nodes_patch[place_name] = patch
        if nodes_patch:
            self.character.node.patch(nodes_patch)
        make_spot = self.make_spot
        spotlayout = self.spotlayout
        add_widget_to_spotlayout = spotlayout.add_widget
        for place in places2add:
            spot = make_spot(place)
            add_widget_to_spotlayout(spot)
            if '_x' not in place or '_y' not in place:
                spots_unposd.append(spot)
        self.spots_unposd = spots_unposd

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

    def add_new_arrows(self, *args):
        Logger.debug(
            "Board: adding new arrows to {}".format(
                self.character.name
            )
        )
        portmap = self.character.portal
        arrowmap = self.arrow
        spotmap = self.spot
        arrowlayout = self.arrowlayout
        add_widget_to_arrowlayout = arrowlayout.add_widget
        core_make_arrow = self._core_make_arrow
        todo = []
        for arrow_orig, arrow_dests in portmap.items():
            for arrow_dest, portal in arrow_dests.items():
                if (
                        arrow_orig not in arrowmap or
                        arrow_dest not in arrowmap[arrow_orig]
                ):
                    todo.append((portal, spotmap[arrow_orig], spotmap[arrow_dest]))
        points = get_points_multi((origspot, destspot, 10) for (portal, origspot, destspot) in todo)
        for portal, origspot, destspot in todo:
            add_widget_to_arrowlayout(
                core_make_arrow(portal, origspot, destspot, arrowmap, points[origspot, destspot])
            )

    def add_pawn(self, thingn, *args):
        if (
            thingn in self.character.thing and
            thingn not in self.pawn
        ):
            pwn = self.make_pawn(self.character.thing[thingn])
            whereat = self.spot[pwn.loc_name]
            whereat.add_widget(pwn)
            self.pawn[thingn] = pwn
        if thingn in self._scheduled_add_pawn:
            del self._scheduled_add_pawn[thingn]

    def _trigger_add_pawn(self, thingn):
        part = partial(self.add_pawn, thingn)
        if thingn in self._scheduled_add_pawn:
            Clock.unschedule(self._scheduled_add_pawn[thingn])
        self._scheduled_add_pawn[thingn] = Clock.schedule_once(part, 0)

    def add_new_pawns(self, *args):
        Logger.debug(
            "Board: adding new pawns to {}".format(
                self.character.name
            )
        )
        nodes_patch = {}
        things2add = []
        pawns_added = []
        pawnmap = self.pawn
        for (thing_name, thing) in self.character.thing.items():
            if thing_name not in pawnmap:
                things2add.append(thing)
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
                if patch:
                    nodes_patch[thing_name] = patch
        if nodes_patch:
            self.character.node.patch(nodes_patch)
        make_pawn = self.make_pawn
        spotmap = self.spot
        for thing in things2add:
            pwn = make_pawn(thing)
            pawns_added.append(pwn)
            whereat = spotmap[thing['location']]
            whereat.add_widget(pwn)

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
        if self.arrow_cls:
            self.add_new_arrows()
        self.add_new_pawns()
        self.spotlayout.finalize_all()
        Logger.debug("Board: updated")
    trigger_update = trigger(update)

    def update_from_delta(self, delta, *args):
        """Apply the changes described in the dict ``delta``."""
        for (node, extant) in delta.get('nodes', {}).items():
            if extant:
                if node in delta.get('node_val', {}) \
                        and 'location' in delta['node_val'][node]\
                        and node not in self.pawn:
                    self.add_pawn(node)
                elif node not in self.spot:
                    self.add_spot(node)
                    spot = self.spot[node]
                    if '_x' not in spot.place or '_y' not in spot.place:
                        self.spots_unposd.append(spot)
            else:
                if node in self.pawn:
                    self.rm_pawn(node)
                if node in self.spot:
                    self.rm_spot(node)
        for (node, stats) in delta.get('node_val', {}).items():
            if node in self.spot:
                spot = self.spot[node]
                x = stats.get('_x')
                y = stats.get('_y')
                if x is not None:
                    spot.x = x * self.width
                if y is not None:
                    spot.y = y * self.height
                if '_image_paths' in stats:
                    spot.paths = stats['_image_paths'] or spot.default_image_paths
            elif node in self.pawn:
                pawn = self.pawn[node]
                if 'location' in stats:
                    pawn.loc_name = stats['location']
                if '_image_paths' in stats:
                    pawn.paths = stats['_image_paths'] or pawn.default_image_paths
            else:
                Logger.warning(
                    "Board: diff tried to change stats of node {} "
                    "but I don't have a widget for it".format(node)
                )
        for (orig, dests) in delta.get('edges', {}).items():
            for (dest, extant) in dests.items():
                if extant and (orig not in self.arrow or dest not in self.arrow[orig]):
                    self.add_arrow(orig, dest)
                elif not extant and orig in self.arrow and dest in self.arrow[orig]:
                    self.rm_arrow(orig, dest)

    def trigger_update_from_delta(self, delta, *args):
        part = partial(self.update_from_delta, delta)
        if hasattr(self, '_scheduled_update_from_delta'):
            Clock.unschedule(self._scheduled_update_from_delta)
        self._scheduled_update_from_delta = Clock.schedule_once(part, 0)

    def on_spots_unposd(self, *args):
        # TODO: If only some spots are unpositioned, and they remain
        # that way for several frames, put them somewhere that the
        # user will be able to find.
        if not self.spots_unposd:
            return
        try:
            self.grid_layout()
        except (TypeError, ValueError):
            self.nx_layout()

    def _apply_node_layout(self, l, *args):
        if self.width == 1 or self.height == 1:
            Clock.schedule_once(partial(self._apply_node_layout, l), 0.01)
            return
        node_upd = {}
        for spot in self.spots_unposd:
            (x, y) = l[spot.name]
            assert 0 <= x <= 0.99, "{} has invalid x: {}".format(spot.name, x)
            assert 0 <= y <= 0.99, "{} has invalid y: {}".format(spot.name, y)
            assert spot in self.spotlayout.children
            assert self.spotlayout.width == self.width
            assert self.spotlayout.height == self.height
            node_upd[spot.name] = {
                '_x': x,
                '_y': y
            }
            spot.pos = (
                x * self.width,
                y * self.height
            )
        if node_upd:
            self.character.node.patch(node_upd)
        self.spots_unposd = []

    def grid_layout(self, *args):
        self._apply_node_layout(
            normalize_layout(
                {spot.name: spot.name for spot in self.spots_unposd}
            )
        )

    def nx_layout(self, *args):
        for spot in self.spots_unposd:
            if not (spot.name and spot.proxy):
                Clock.schedule_once(self.nx_layout, 0)
                return
        spots_only = self.character.facade()
        for thing in list(spots_only.thing.keys()):
            del spots_only.thing[thing]
        self._apply_node_layout(self.graph_layout(spots_only))

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


class BoardScatterPlane(ScatterPlane):
    selection_candidates = ListProperty([])
    selection = ObjectProperty(allownone=True)
    keep_selection = BooleanProperty(False)
    board = ObjectProperty()
    adding_portal = BooleanProperty(False)
    reciprocal_portal = BooleanProperty()

    def spot_from_dummy(self, dummy):
        """Make a real place and its spot from a dummy spot.

        Create a new :class:`graph.Spot` instance, along with the
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
        """Make a real thing and its pawn from a dummy pawn.

        Create a new :class:`graph.Pawn` instance, along with the
        underlying :class:`LiSE.Thing` instance, and give it the name,
        location, and imagery of the provided dummy.

        """
        candidates = []
        dummy_center = self.to_local(*dummy.center)
        dummy.pos = self.to_local(*dummy.pos)
        for spot in self.board.spot.values():
            if spot.collide_widget(dummy):
                candidates.append(spot)
        if not candidates:
            return
        whereat = candidates.pop()
        if candidates:
            dist = Vector(*whereat.center).distance(dummy_center)
            while candidates:
                thereat = candidates.pop()
                thereto = Vector(*thereat.center).distance(dummy_center)
                if thereto < dist:
                    whereat, dist = thereat, thereto
        whereat.add_widget(
            self.board.make_pawn(
                self.board.character.new_thing(
                    dummy.name,
                    whereat.proxy.name,
                    _image_paths=list(dummy.paths)
                )
            )
        )
        dummy.num += 1

    def on_board(self, *args):
        if hasattr(self, '_oldboard'):
            self.unbind(
                adding_portal=self._oldboard.setter('adding_portal'),
                reciprocal_portal=self._oldboard.setter('reciprocal_portal')
            )
        self.clear_widgets()
        self.add_widget(self.board)
        self.board.adding_portal = self.adding_portal
        self.board.reciprocal_portal = self.reciprocal_portal
        self.bind(
            adding_portal=self.board.setter('adding_portal'),
            reciprocal_portal=self.board.setter('reciprocal_portal')
        )
        self._oldboard = self.board

    def on_touch_down(self, touch):
        if touch.is_mouse_scrolling:
            scale = self.scale + (0.05 if touch.button == 'scrolldown' else -0.05)
            if (self.scale_min and scale < self.scale_min) \
                    or (self.scale_max and scale > self.scale_max):
                return
            rescale = scale * 1.0 / self.scale
            self.apply_transform(Matrix().scale(rescale, rescale, rescale),
                                 post_multiply=True,
                                 anchor=self.to_local(*touch.pos))
            return self.dispatch('on_transform_with_touch', touch)
        return super().on_touch_down(touch)

    def apply_transform(self, trans, post_multiply=False, anchor=(0, 0)):
        super().apply_transform(trans, post_multiply=post_multiply, anchor=anchor)
        self._last_transform = trans, post_multiply, anchor

    def on_transform_with_touch(self, touch):
        x, y = self.pos
        w = self.board.width * self.scale
        h = self.board.height * self.scale
        if hasattr(self, '_last_transform') and (w < self.parent.width or h < self.parent.height):
            trans, post_multiply, anchor = self._last_transform
            super().apply_transform(trans.inverse(), post_multiply, anchor)
            return
        if x > self.parent.x:
            self.x = self.parent.x
        if y > self.parent.y:
            self.y = self.parent.y
        if x + w < self.parent.right:
            self.x = self.parent.right - w
        if y + h < self.parent.top:
            self.y = self.parent.top - h


class GraphBoardView(BoardView):
    adding_portal = BooleanProperty(False)
    reciprocal_portal = BooleanProperty(True)


Builder.load_string("""
<GraphBoard>:
    app: app
    size_hint: None, None
<GraphBoardView>:
    plane: boardplane
    BoardScatterPlane:
        id: boardplane
        board: root.board
        adding_portal: root.adding_portal
        reciprocal_portal: root.reciprocal_portal
        scale_min: root.scale_min
        scale_max: root.scale_max
""")