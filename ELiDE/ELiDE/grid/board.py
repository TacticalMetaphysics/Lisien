from functools import partial

from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty
)
from kivy.uix.relativelayout import RelativeLayout
from .spot import Spot
from .pawn import Pawn


class Grid(RelativeLayout):
    character = ObjectProperty()
    rows = NumericProperty()
    cols = NumericProperty()
    tile_width = NumericProperty()
    tile_height = NumericProperty()
    tile_size = ReferenceListProperty(tile_width, tile_height)
    pawn = DictProperty({})
    spot = DictProperty({})

    def do_layout(self, *args):
        colw = self.width / self.cols
        rowh = self.height / self.rows
        for name, tile in self.spot.items():
            gx, gy = name
            tile.pos = gx * colw, gy * rowh

    def add_tile(self, placen, *args):
        if (
            placen in self.character.place and
            placen not in self.spot
        ):
            self.add_widget(self.make_tile(self.character.place[placen]))

    def _trigger_add_tile(self, placen):
        part = partial(self.add_tile, placen)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def add_new_tiles(self, *args):
        placemap = self.character.place
        tilemap = self.spot
        default_image_paths = Spot.default_image_paths
        default_zeroes = [0] * len(default_image_paths)
        places2add = []
        nodes_patch = {}
        for place_name, place in placemap.items():
            if place_name not in tilemap:
                places2add.append(place)
                patch = {}
                if '_image_paths' in place:
                    zeroes = [0] * len(place['_image_paths'])
                else:
                    patch['_image_paths'] = default_image_paths
                    zeroes = default_zeroes
                if '_offxs' not in place:
                    patch['_offxs'] = zeroes
                if '_offys' not in place:
                    patch['_offys'] = zeroes
                if patch:
                    nodes_patch[place_name] = patch
        if nodes_patch:
            self.character.node.patch(nodes_patch)
        make_tile = self.make_tile
        add_widget = self.add_widget()
        for place in places2add:
            add_widget(make_tile(place))

    def add_pawn(self, thingn, *args):
        if (
            thingn in self.character.thing and
            thingn not in self.pawn
        ):
            pwn = self.make_pawn(self.character.thing[thingn])
            whereat = self.spot[pwn.thing['location']]
            whereat.add_widget(pwn)
            self.pawn[thingn] = pwn

    def _trigger_add_pawn(self, thingn):
        part = partial(self.add_pawn, thingn)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def add_new_pawns(self, *args):
        nodes_patch = {}
        things2add = []
        pawns_added = []
        pawnmap = self.pawn
        default_image_paths = Pawn.default_image_paths
        default_zeroes = [0] * len(default_image_paths)
        for thingn, thing in self.character.thing.items():
            if thingn not in pawnmap:
                things2add.append(thing)
                patch = {}
                if '_image_paths' in thing:
                    zeroes = [0] * len(thing['_image_paths'])
                else:
                    patch['_image_paths'] = default_image_paths
                    zeroes = default_zeroes
                if '_offxs' not in thing:
                    patch['_offxs'] = zeroes
                if '_offys' not in thing:
                    patch['_offys'] = zeroes
                if patch:
                    nodes_patch[thingn] = patch
        if nodes_patch:
            self.character.node.patch(nodes_patch)
        make_pawn = self.make_pawn
        tilemap = self.spot
        for thing in things2add:
            pwn = make_pawn(thing)
            pawns_added.append(pwn)
            whereat = tilemap[thing['location']]
            whereat.add_widget(pwn)

    def update(self, *args):
        pass

    def update_from_delta(self, delta, *args):
        pass

    def _trigger_update_from_delta(self, delta, *args):
        part = partial(self.update_from_delta, delta)
        Clock.unschedule(part)
        Clock.schedule_once(part, 0)

    def pawns_at(self, x, y):
        for pawn in self.pawn.values():
            if pawn.collide_point(x, y):
                yield pawn

    def tiles_at(self, x, y):
        for tile in self.spot.values():
            if tile.collide_point(x, y):
                yield tile