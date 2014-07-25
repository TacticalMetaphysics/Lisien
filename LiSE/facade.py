
# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Ways of viewing Character that simulate imperfect
knowledge. Suitable for eg. deciding what to show the player.

"""
from collections import Mapping
from LiSE.graph import (
    Thing,
    Place,
    Portal
)
from LiSE.mapping import (
    CharacterThingMapping,
    CharacterPlaceMapping,
    CharacterPortalSuccessorsMapping,
    CharacterPortalPredecessorsMapping
)
from LiSE.funlist import FunList


class Munger(object):
    def __init__(self, worldview, name, omitters=[], distorters=[]):
        self.worldview = worldview
        self.name = name
        self.worldview.cursor.execute(
            "SELECT COUNT(*) FROM mungers WHERE munger=?;",
            (self.name,)
        )
        (ct,) = self.worldview.cursor.fetchone()
        if ct == 0:
            self.worldview.cursor.execute(
                "INSERT INTO mungers (munger) VALUES (?);",
                (self.name,)
            )
        self.omitters = FunList(self.worldview, 'mungers', ['munger'], [self.name], 'omitters', omitters)
        self.distorters = FunList(self.worldview, 'mungers', ['munger'], [self.name], 'distorters', distorters)

    def omitted(self, engine, facade, entity, *data):
        curtime = engine.time
        for omitter in self.omitters:
            if omitter(engine, facade, entity, *data):
                return True
            engine.time = curtime
        return False

    def distorted(self, engine, facade, entity, data):
        for distorter in self.distorters:
            data = distorter(engine, facade, entity, data)
        return data


class MungerList(FunList):
    def __init__(self, worldview, table, preset_fields, preset_values, field):
        self.worldview = worldview
        self.table = table
        self.preset_fields = tuple(preset_fields)
        self.preset_values = tuple(preset_values)
        self.field = field

    def __iter__(self):
        for munn in self._getlist():
            yield Munger(self.worldview, munn)

    def __getitem__(self, i):
        return Munger(self.worldview, self._getlist()[i])

    def __setitem__(self, i, v):
        if isinstance(v, str) or isinstance(v, str):
            munn = v
        else:
            munn = v.name
        l = self._getlist()
        l[i] = munn
        self._setlist(l)


def get_mungers(facade, table_name, field="mungers"):
    (branch, tick) = facade.engine.time
    for (branch, tick) in facade.engine.orm._active_branches():
        facade.engine.orm.cursor.execute(
            "SELECT branch, MAX(tick) FROM {tabn} WHERE branch=? AND tick<=?;",
            (branch, tick)
        )
        try:
            (branch, tick) = facade.engine.orm.cursor.fetchone()
        except TypeError:
            if branch == 'master':
                raise ValueError("No mungers")
            else:
                continue
    return MungerList(
        facade.engine.worldview,
        table_name,
        ["observer_char", "observed_char", "facade", "branch", "tick"], 
        [facade.observer.name, facade.observed.name, facade.name, branch, tick],
        field
    )


class AbstractFacadeMapping(Mapping):
    def _omitted(self, k):
        return any(
            munger.omitted(self.facade, self, k)
            for munger in self.mungers
        )

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __setitem__(self, k, v):
        raise TypeError("Read only")

    def __delitem__(self, k):
        raise TypeError("Read only")


class FacadeThing(Thing, AbstractFacadeMapping):
    def __init__(self, facade, name):
        self.facade = facade
        self.character = self.facade.observed
        self.name = name

    @property
    def thing(self):
        return self.character.thing[self.name]

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_thing_stats",
        )

    def __iter__(self):
        for k in super(FacadeThing, self):
            if not self._omitted(k):
                yield k

    def __getitem__(self, k):
        mungers = self.mungers
        for munger in mungers:
            if munger.omitted(self.facade.engine, self.facade, self, k):
                raise KeyError("Omitted")
        r = super(FacadeThing, self)[k]
        for munger in mungers:
            r = munger.distorted(self.facade.engine, self.facade, self, r)
        return r


class FacadeThingMapping(CharacterThingMapping):
    def __init__(self, facade):
        self.facade = facade
        self.character = self.facade.observed
        self.worldview = self.character.worldview
        self.name = self.character.name

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_things"
        )

    def __iter__(self):
        for k in super(FacadeThingMapping, self):
            if not self._omitted(k):
                yield k

    def __getitem__(self, k):
        if self._omitted(k):
            raise KeyError("Omitted")
        return FacadeThing(self.facade, k)


class FacadePlace(Place, AbstractFacadeMapping):
    def __init__(self, facade, name):
        self.facade = facade
        self.character = self.facade.observed
        self.worldview = self.character.worldview
        self.name = name

    @property
    def place(self):
        return self.character.place[self.name]

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_place_stats"
        )

    def __iter__(self):
        for k in super(FacadePlace, self):
            if not self._omitted(k):
                yield k

    def __getitem__(self, k):
        mungers = self.mungers
        for munger in mungers:
            if munger.omitted(self.facade, self, k):
                raise KeyError("Omitted")
        r = super(FacadePlace, self)[k]
        for munger in mungers:
            r = munger.distorted(self.facade, self, r)
        return r


class FacadePlaceMapping(CharacterPlaceMapping, AbstractFacadeMapping):
    def __init__(self, facade):
        self.facade = facade
        self.character = self.facade.observed
        self.worldview = self.character.worldview
        self.name = self.character.name

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_places"
        )

    def __iter__(self):
        for k in super(FacadePlaceMapping, self):
            if not self._omitted(k):
                yield k

    def __getitem__(self, k):
        if self._omitted(k):
            raise KeyError("Omitted")
        return FacadePlace(self.facade, k)


class FacadePortal(Portal, AbstractFacadeMapping):
    def __init__(self, facade, origin, destination):
        self.facade = facade
        self.character = self.facade.observed
        self.worldview = self.character.worldview
        self.origin = origin
        self.destination = destination

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_portal_stats"
        )

    def __iter__(self):
        for k in super(FacadePortal, self):
            if not self._omitted(k):
                yield k

    def __getitem__(self, k):
        mungers = self.mungers
        for munger in mungers:
            if munger.omitted(self.facade, self, k):
                raise KeyError("Omitted")
        r = super(FacadePortal, self)[k]
        for munger in mungers:
            r = munger.distorted(self.facade, self, r)
        return


class FacadeSuccessors(CharacterPortalSuccessorsMapping.Successors, AbstractFacadeMapping):
    def __init__(self, outer, nodeA):
        self.outer = outer
        self.facade = outer.facade
        self.nodeA = nodeA

    @property
    def mungers(self):
        return self.outer.mungers

    def _omitted(self, b):
        return self.outer._omitted(self.nodeA, b)

    def _getsub(self, nodeB):
        return FacadePortal(self.facade, self.nodeA, nodeB)

    def __iter__(self):
        for nodeB in super(FacadeSuccessors, self):
            if not self._omitted(nodeB):
                yield nodeB

    def __getitem__(self, nodeB):
        super(FacadeSuccessors, self)[nodeB]
        # I don't want the Edge object, just to throw KeyError if
        # there isn't one
        if self._omitted(nodeB):
            raise KeyError("Omitted")
        return self._getsub(nodeB)


class FacadePortalSuccessorsMapping(CharacterPortalSuccessorsMapping, AbstractFacadeMapping):
    def __init__(self, facade):
        self.facade = facade
        self.character = self.facade.observed
        self.graph = self.character
        self.worldview = self.character.worldview
        self.gorm = self.worldview.gorm

    @property
    def mungers(self):
        return get_mungers(
            self.facade,
            "facade_portals"
        )

    def _omitted(self, a, b):
        return any(munger.omitter(self.facade, self, a, b) for munger in self.mungers)

    def __contains__(self, a):
        if a in self.character.portal:
            for b in self.character.portal[a]:
                if not self._omitted(a, b):
                    return True
        return False

    def __iter__(self):
        for a in self.character.portal:
            for b in self.character.portal[a]:
                if not self._omitted(a, b):
                    yield a
                    break

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No apparent Portals off of {}".format(k))
        return FacadeSuccessors(self, k)


class FacadePredecessors(CharacterPortalPredecessorsMapping.Predecessors, AbstractFacadeMapping):
    def __init__(self, outer, nodeB):
        self.outer = outer
        self.facade = outer.facade
        self.nodeB = nodeB

    @property
    def mungers(self):
        return self.outer.mungers

    def _omitted(self, a):
        return self.outer._omitted(a, self.nodeB)

    def _getsub(self, nodeA):
        return FacadePortal(self.facade, nodeA, self.nodeB)

    def __iter__(self):
        for nodeA in super(FacadePredecessors, self):
            if not self._omitted(nodeA):
                yield nodeA

    def __getitem__(self, nodeA):
        super(FacadeSuccessors, self)[nodeA]
        if self._omitted(nodeA):
            raise KeyError("Omitted")
        return self._getsub(nodeA)


class FacadePortalPredecessorsMapping(FacadePortalSuccessorsMapping):
    def __contains__(self, b):
        if b in self.character.preportal:
            for a in self.character.preportal[b]:
                if not self._omitted(a, b):
                    return True
        return False

    def __iter__(self):
        for b in self.character.preportal:
            for a in self.character.preportal[b]:
                if not self._omitted(a, b):
                    yield b
                    break

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No apparent Portals leading to {}".format(k))
        return FacadePredecessors(self, k)


class Facade(object):
    def __init__(self, engine, observer, observed, name='perception'):
        self.engine = engine
        self.observer = observer
        self.observed = observed
        self.name = name
        self.engine.cursor.execute(
            "SELECT COUNT(*) FROM facades WHERE "
            "observer_char=? AND "
            "observed_char=? AND "
            "facade=?;",
            (
                self.observer.name,
                self.observed.name,
                self.name
            )
        )
        (ct,) = self.engine.cursor.fetchone()
        if ct == 0:
            (branch, tick) = self.engine.time
            self.engine.cursor.execute(
                "INSERT INTO facades ("
                "observer_char, "
                "observed_char, "
                "facade, "
                "branch, "
                "tick, "
                "extant"
                ") VALUES (?, ?, ?, ?, ?, ?);",
                (
                    self.observer.name,
                    self.observed.name,
                    self.name,
                    branch,
                    tick,
                    True
                )
            )
        self.thing = FacadeThingMapping(self)
        self.place = FacadePlaceMapping(self)
        self.portal = FacadePortalSuccessorsMapping(self)
        self.preportal = FacadePortalPredecessorsMapping(self)

    def __getitem__(self, name):
        """For when there was actually only one Facade between the selected
        pair of characters, but the user supplied a name anyway.

        """
        if name == self.name:
            return self
        raise KeyError("No such Facade")
