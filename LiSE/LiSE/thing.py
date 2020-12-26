# This file is part of LiSE, a framework for life simulation games.
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
"""The sort of node that is ultimately located in a Place.

Things may be located in other Things as well, but eventually must be
recursively located in a Place.

There's a subtle distinction between "location" and "containment": a
Thing may be contained by a Portal, but cannot be located there --
only in one of the Portal's endpoints. Things are both located in and
contained by Places, or possibly other Things.

"""
import networkx as nx
from .node import Node
from .exc import TravelException
from .allegedb import HistoryError


def roerror(*args, **kwargs):
    raise ValueError("Read-only")


class Thing(Node):

    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    __slots__ = ('graph', 'db', 'node', '_rulebook')

    extrakeys = {
        'name',
        'location'
    }

    def _getname(self):
        return self.name

    def _getloc(self):
        return self.engine._things_cache.retrieve(
            self.character.name, self.name, *self.engine._btt()
        )

    def _get_arrival_time(self):
        charn = self.character.name
        n = self.name
        thingcache = self.engine._things_cache
        for b, trn, tck in self.engine._iter_parent_btt():
            try:
                v = thingcache.turn_before(charn, n, b, trn)
            except KeyError:
                v = thingcache.turn_after(charn, n, b, trn)
            if v is not None:
                return v
        else:
            raise ValueError("Couldn't find arrival time")

    def _set_loc(self, loc):
        self.engine._set_thing_loc(
            self.character.name,
            self.name,
            loc
        )
        self.send(self, key='location', val=loc)

    _getitem_dispatch = {
        'name': _getname,
        'location': _getloc
    }

    _setitem_dispatch = {
        'name': roerror,
        'location': _set_loc
    }

    def __getitem__(self, key):
        """Return one of my stats stored in the database, or a few
        special cases:

        ``name``: return the name that uniquely identifies me within
        my Character

        ``character``: return the name of my character

        ``location``: return the name of my location

        """
        try:
            return self._getitem_dispatch[key](self)
        except KeyError:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``=``value`` for the present game-time."""
        try:
            self._setitem_dispatch[key](self, value)
        except HistoryError as ex:
            raise ex
        except KeyError:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        """As of now, this key isn't mine."""
        if key in self.extrakeys:
            raise ValueError("Can't delete {}".format(key))
        super().__delitem__(key)

    def __repr__(self):
        return "{}.character['{}'].thing['{}']".format(
            self.engine,
            self.character.name,
            self.name
        )

    def delete(self):
        super().delete()
        self._set_loc(None)
        self.character.thing.send(self.character.thing, key=self.name, val=None)

    def clear(self):
        """Unset everything."""
        for k in list(self.keys()):
            if k not in self.extrakeys:
                del self[k]

    @property
    def location(self):
        """The ``Thing`` or ``Place`` I'm in."""
        try:
            locn = self['location']
        except KeyError:
            return None
        return self.engine._get_node(self.character, locn)

    @location.setter
    def location(self, v):
        if hasattr(v, 'name'):
            v = v.name
        self['location'] = v

    @property
    def next_location(self):
        branch = self.engine.branch
        turn = self.engine._things_cache.turn_after(self.character.name, self.name, *self.engine.time)
        if turn is None:
            return None
        return self.engine._get_node(self.character, self.engine._things_cache.retrieve(
            self.character.name, self.name, branch, turn, self.engine._turn_end_plan[branch, turn]
        ))

    def go_to_place(self, place, weight=''):
        """Assuming I'm in a :class:`Place` that has a :class:`Portal` direct
        to the given :class:`Place`, schedule myself to travel to the
        given :class:`Place`, taking an amount of time indicated by
        the ``weight`` stat on the :class:`Portal`, if given; else 1
        turn.

        Return the number of turns the travel will take.

        """
        if hasattr(place, 'name'):
            placen = place.name
        else:
            placen = place
        curloc = self["location"]
        orm = self.character.engine
        turns = self.engine._portal_objs[
            (self.character.name, curloc, place)].get(weight, 1)
        with self.engine.plan():
            orm.turn += turns
            self['location'] = placen
        return turns

    def follow_path(self, path, weight=None):
        """Go to several :class:`Place`s in succession, deciding how long to
        spend in each by consulting the ``weight`` stat of the
        :class:`Portal` connecting the one :class:`Place` to the next.

        Return the total number of turns the travel will take. Raise
        :class:`TravelException` if I can't follow the whole path,
        either because some of its nodes don't exist, or because I'm
        scheduled to be somewhere else.

        """
        if len(path) < 2:
            raise ValueError("Paths need at least 2 nodes")
        eng = self.character.engine
        turn_now, tick_now = eng.time
        with eng.plan():
            prevplace = path.pop(0)
            if prevplace != self['location']:
                raise ValueError("Path does not start at my present location")
            subpath = [prevplace]
            for place in path:
                if (
                        prevplace not in self.character.portal or
                        place not in self.character.portal[prevplace]
                ):
                    raise TravelException(
                        "Couldn't follow portal from {} to {}".format(
                            prevplace,
                            place
                        ),
                        path=subpath,
                        traveller=self
                    )
                subpath.append(place)
                prevplace = place
            turns_total = 0
            prevsubplace = subpath.pop(0)
            subsubpath = [prevsubplace]
            for subplace in subpath:
                portal = self.character.portal[prevsubplace][subplace]
                turn_inc = portal.get(weight, 1)
                eng.turn += turn_inc
                self.location = subplace
                turns_total += turn_inc
                subsubpath.append(subplace)
                prevsubplace = subplace
            self.location = subplace
            eng.time = turn_now, tick_now
        return turns_total

    def travel_to(self, dest, weight=None, graph=None):
        """Find the shortest path to the given :class:`Place` from where I am
        now, and follow it.

        If supplied, the ``weight`` stat of the :class:`Portal`s along
        the path will be used in pathfinding, and for deciding how
        long to stay in each Place along the way.

        The ``graph`` argument may be any NetworkX-style graph. It
        will be used for pathfinding if supplied, otherwise I'll use
        my :class:`Character`. In either case, however, I will attempt
        to actually follow the path using my :class:`Character`, which
        might not be possible if the supplied ``graph`` and my
        :class:`Character` are too different. If it's not possible,
        I'll raise a :class:`TravelException`, whose ``subpath``
        attribute holds the part of the path that I *can* follow. To
        make me follow it, pass it to my ``follow_path`` method.

        Return value is the number of turns the travel will take.

        """
        destn = dest.name if hasattr(dest, 'name') else dest
        if destn == self.location.name:
            raise ValueError("I'm already at {}".format(destn))
        graph = self.character if graph is None else graph
        path = nx.shortest_path(graph, self["location"], destn, weight)
        return self.follow_path(path, weight)