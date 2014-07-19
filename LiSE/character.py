# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    Mapping,
    Callable
)
from gorm.graph import DiGraph
from LiSE.rule import Rule
from LiSE.graph import (
    Thing,
    Place,
)
from LiSE.mapping import (
    CharacterThingMapping,
    CharacterPlaceMapping,
    CharacterPortalSuccessorsMapping,
    CharacterPortalPredecessorsMapping
)
from LiSE.facade import Facade


class CharRules(Mapping):
    """Maps rule names to rules the Character is following, and is also a
    decorator to create said rules from action functions.

    Decorating a function with this turns the function into the
    first action of a new rule of the same name, and applies the
    rule to the character. Add more actions with the @rule.action
    decorator, and add prerequisites with @rule.prereq

    """
    def __init__(self, orm, char):
        """Store the character"""
        self.orm = orm
        self.character = char
        self.name = char.name

    def __call__(self, v):
        """If passed a Rule, activate it. If passed a string, get the rule by
        that name and activate it. If passed a function (probably
        because I've been used as a decorator), make a rule with the
        same name as the function, with the function itself being the
        first action of the rule, and activate that rule.

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            # create a new rule performing the action v
            vname = self.orm.function(v)
            self._activate_rule(
                Rule(
                    self.orm,
                    vname,
                    actions=[vname]
                )
            )
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._activate_rule(Rule(self.orm, v))

    def __iter__(self):
        """Iterate over all rules presently in effect"""
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
                "SELECT char_rules.rule, char_rules.active "
                "FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "character=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.character=hitick.character "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;"
                (
                    self.character.name,
                    branch,
                    tick
                )
            )
            for (rule, active) in self.orm.cursor.fetchall():
                if active and rule not in seen:
                    yield rule
                seen.add(rule)

    def __len__(self):
        """Count the rules presently in effect"""
        n = 0
        for rule in self:
            n += 1
        return n

    def __getitem__(self, rulen):
        """Get the rule by the given name, if it is in effect"""
        # make sure the rule is active at the moment
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
                "SELECT char_rules.active "
                "FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "character=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (
                    self.character.name,
                    rulen,
                    branch,
                    tick
                )
            )
            data = self.orm.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in char_rules table")
            else:
                (active,) = data[0]
                if not active:
                    raise KeyError("No such rule at the moment")
                return Rule(self.orm, rulen)
        raise KeyError("No such rule, ever")

    def _activate_rule(self, rule):
        """Indicate that the rule is active and should be followed. Add the
        given arguments to whatever's there.

        """
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM char_rules WHERE "
            "character=? AND "
            "rule=? AND "
            "branch=? AND "
            "tick=?;",
            (
                self.character.name,
                rule.name,
                branch,
                tick
            )
        )
        self.orm.cursor.execute(
            "INSERT INTO char_rules "
            "(character, rule, branch, tick, active) "
            "VALUES (?, ?, ?, ?, ?);",
            (
                self.character.name,
                rule.name,
                branch,
                tick,
                True
            )
        )

    def __delitem__(self, rule):
        """Deactivate the rule"""
        if isinstance(rule, Rule):
            rulen = rule.name
        else:
            rulen = self.orm.function(rule)
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM char_rules WHERE "
            "character=? AND "
            "rule=? AND "
            "branch=? AND "
            "tick=?;",
            (
                self.name,
                rulen,
                branch,
                tick
            )
        )
        self.orm.cursor.execute(
            "INSERT INTO char_rules "
            "(character, rule, branch, tick, active) "
            "VALUES (?, ?, ?, ?, ?);",
            (
                self.name,
                rulen,
                branch,
                tick,
                False
            )
        )


class CharObservedFacades(Mapping):
    def __init__(self, engine, observer, observed):
        self.engine = engine
        self.observer = observer
        self.observed = observed

    def __iter__(self):
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.orm.cursor.execute(
                "SELECT facades.facade, facades.active "
                "FROM facades JOIN ("
                "SELECT observer_char, observed_char, facade, branch, MAX(tick) AS tick "
                "FROM facades WHERE "
                "observer_char=? AND "
                "observed_char=? AND "
                "branch=? AND "
                "tick<=? GROUP BY observer_char, observed_char, facade, branch) AS hitick "
                "ON facades.observer_char=hitick.observer_char "
                "AND facades.observed_char=hitick.observed_char "
                "AND facades.facade=hitick.facade "
                "AND facades.branch=hitick.branch "
                "AND facades.tick=hitick.tick;",
                (
                    self.observer.name,
                    self.observed.name,
                    branch,
                    tick
                )
            )
            for (facade, active) in self.engine.orm.cursor.fetchall():
                if active and facade not in seen:
                    yield facade
                seen.add(facade)

    def __len__(self):
        n = 0
        for facade in self:
            n += 1
        return n

    def __getitem__(self, facn):
        for (branch, tick) in self.engine._active_branches():
            self.engine.orm.cursor.execute(
                "SELECT facades.active "
                "FROM facades JOIN ("
                "SELECT observer_char, observed_char, facade, branch, MAX(tick) AS tick "
                "FROM facades WHERE "
                "observer_char=? AND "
                "observed_char=? AND "
                "branch=? AND "
                "tick<=? GROUP BY observer_char, observed_char, facade, branch) AS hitick "
                "ON facades.observer_char=hitick.observer_char "
                "AND facades.observed_char=hitick.observed_char "
                "AND facades.facade=hitick.facade "
                "AND facades.branch=hitick.branch "
                "AND facades.tick=hitick.tick;",
                (
                    self.observer.name,
                    self.observed.name,
                    branch,
                    tick
                )
            )
            data = self.engine.orm.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in facades table")
            else:
                (active,) = data[0]
                if not active:
                    raise KeyError("Facade has been deleted")
                return Facade(self.engine, self.observer, self.observed, facn)


class CharObserved(Mapping):
    def __init__(self, engine, char):
        self.engine = engine
        self.character = char
        self.name = char.name

    def __iter__(self):
        seen = set()
        yielded = set()
        for (branch, tick) in self.engine.orm._active_branches():
            self.engine.orm.cursor.execute(
                "SELECT facades.observed_char, facades.facade, facades.active "
                "FROM facades JOIN ("
                "SELECT observer_char, observed_char, facade, branch, MAX(tick) AS tick "
                "FROM facades WHERE "
                "observer_char=? AND "
                "branch=? AND "
                "tick<=? GROUP BY observer_char, observed_char, facade, branch) AS hitick"
                "ON facades.observer_char=hitick.observer_char "
                "AND facades.observed_char=hitick.observed_char "
                "AND facades.facade=hitick.facade "
                "AND facades.branch=hitick.branch "
                "AND facades.tick=hitick.tick;",
                (
                    self.character.name,
                    branch,
                    tick
                )
            )
            for (observed, facade, active) in self.engine.orm.cursor.fetchall():
                if active and (observed, facade) not in seen and observed not in yielded:
                    yield observed
                    yielded.add(observed)
                seen.add((observed, facade))

    def __len__(self):
        n = 0
        for obsrvd in self:
            n += 1
        return n

    def __getitem__(self, observed):
        if isinstance(observed, str) or isinstance(observed, str):
            char = self.engine.orm.get_character(observed)
        else:
            char = observed
        r = CharObservedFacades(self.engine, self.character, char)
        lr = len(r)
        if lr == 0:
            raise KeyError("No facades onto {}".format(char.name))
        elif lr == 1:
            # When there is only one facade, there's no sense in
            # letting the user specify which one they want, so just
            # return the facade itself.
            return r[next(r)]
        else:
            return r


class CharacterAvatarGraphMapping(Mapping):
    def __init__(self, char):
        """Remember my character"""
        self.char = char
        self.worldview = char.worldview
        self.name = char.name

    def __call__(self, av):
        """Add the avatar. It must be an instance of Place or Thing."""
        if av.__class__ not in (Place, Thing):
            raise TypeError("Only Things and Places may be avatars")
        self.char.add_avatar(av.name, av.character.name)

    def _datadict(self):
        """Get avatar-ness data and return it"""
        d = {}
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
                "SELECT "
                "avatars.avatar_graph, "
                "avatars.avatar_node, "
                "avatars.is_avatar FROM avatars "
                "JOIN ("
                "SELECT character_graph, avatar_graph, avatar_node, "
                "branch, MAX(tick) AS tick FROM avatars WHERE "
                "character_graph=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character_graph, avatar_graph, "
                "avatar_node, branch) AS hitick ON "
                "avatars.character_graph=hitick.character_graph AND "
                "avatars.avatar_graph=hitick.avatar_graph AND "
                "avatars.avatar_node=hitick.avatar_node AND "
                "avatars.branch=hitick.branch AND "
                "avatars.tick=hitick.tick;",
                (
                    self.name,
                    branch,
                    rev
                )
            )
            for (graph, node, avatar) in self.worldview.cursor.fetchall():
                is_avatar = bool(avatar)
                if graph not in d:
                    d[graph] = {}
                if node not in d[graph]:
                    d[graph][node] = is_avatar
        return d

    def __iter__(self):
        """Iterate over every avatar graph that has at least one avatar node
        in it presently

        """
        d = self._datadict()
        for graph in d:
            for node in d[graph]:
                if d[graph][node]:
                    yield graph
                    break

    def __len__(self):
        """Number of graphs in which I have an avatar"""
        n = 0
        for g in self:
            n += 1
        return n

    def __getitem__(self, g):
        """Get the CharacterAvatarMapping for the given graph, if I have any
        avatars in it. Otherwise raise KeyError.

        """
        d = self._datadict()[g]
        for node in d:
            if d[node]:
                return self.CharacterAvatarMapping(self, g)
        raise KeyError("No avatars in {}".format(g))

    def __repr__(self):
        d = {}
        for k in self:
            d[k] = dict(self[k])
        return repr(d)


    class CharacterAvatarMapping(Mapping):
        """Mapping of avatars of one Character in another Character."""
        def __init__(self, outer, graphn):
            """Store the character and the name of the "graph", ie. the other
            character.

            """
            self.char = outer.char
            self.worldview = outer.worldview
            self.name = outer.name
            self.graph = graphn

        def __getattr__(self, attrn):
            """If I don't have such an attribute, but I contain exactly one
            avatar, and *it* has the attribute, return the
            avatar's attribute.

            """
            if len(self) == 1:
                return getattr(self[next(iter(self))], attrn)
            return super(Character.CharacterAvatarMapping, self).__getattr__(attrn)

        def __iter__(self):
            """Iterate over the names of all the presently existing nodes in the
            graph that are avatars of the character

            """
            seen = set()
            for (branch, rev) in self.worldview._active_branches():
                self.worldview.cursor.execute(
                    "SELECT "
                    "avatars.avatar_node, "
                    "avatars.is_avatar FROM avatars JOIN ("
                    "SELECT character_graph, avatar_graph, avatar_node, "
                    "branch, MAX(tick) AS tick FROM avatars "
                    "WHERE character_graph=? "
                    "AND avatar_graph=? "
                    "AND branch=? "
                    "AND tick<=? GROUP BY "
                    "character_graph, avatar_graph, avatar_node, branch"
                    ") AS hitick ON "
                    "avatars.character_graph=hitick.character_graph "
                    "AND avatars.avatar_graph=hitick.avatar_graph "
                    "AND avatars.avatar_node=hitick.avatar_node "
                    "AND avatars.branch=hitick.branch "
                    "AND avatars.tick=hitick.tick;",
                    (
                        self.name,
                        self.graph,
                        branch,
                        rev
                    )
                )
                for (node, extant) in self.worldview.cursor.fetchall():
                    if extant and node not in seen:
                        yield node
                    seen.add(node)

        def __contains__(self, av):
            for (branch, rev) in self.worldview._active_branches():
                self.worldview.cursor.execute(
                    "SELECT avatars.is_avatar FROM avatars JOIN ("
                    "SELECT character_graph, avatar_graph, avatar_node, "
                    "branch, MAX(tick) AS tick FROM avatars "
                    "WHERE character_graph=? "
                    "AND avatar_graph=? "
                    "AND avatar_node=? "
                    "AND branch=? "
                    "AND tick<=? GROUP BY "
                    "character_graph, avatar_graph, avatar_node, "
                    "branch) AS hitick ON "
                    "avatars.character_graph=hitick.character_graph "
                    "AND avatars.avatar_graph=hitick.avatar_graph "
                    "AND avatars.avatar_node=hitick.avatar_node "
                    "AND avatars.branch=hitick.branch "
                    "AND avatars.tick=hitick.tick;",
                    (
                        self.name,
                        self.graph,
                        av,
                        branch,
                        rev
                    )
                )
                try:
                    return bool(self.worldview.cursor.fetchone()[0])
                except (TypeError, IndexError):
                    continue
            return False

        def __len__(self):
            """Number of presently existing nodes in the graph that are avatars of
            the character"""
            n = 0
            for a in self:
                n += 1
            return n

        def __getitem__(self, av):
            """Return the Place or Thing by the given name in the graph, if it's
            my avatar and it exists.

            If I contain exactly *one* Place or Thing, and you're
            not trying to get it by its name, delegate to its
            __getitem__. It's common for one Character to have
            exactly one avatar in another Character, and when that
            happens, it's nice not to have to specify the avatar's
            name.

            """
            if av in self:
                if self.worldview._is_thing(self.graph, av):
                    return Thing(
                        self.worldview.get_character(self.graph),
                        av
                    )
                else:
                    return Place(
                        self.worldview.character[self.graph],
                        av
                    )
            if len(self) == 1:
                return self[next(iter(self))][av]
            raise KeyError("No such avatar")

        def __repr__(self):
            d = {}
            for k in self:
                d[k] = dict(self[k])
            return repr(d)





class Character(DiGraph):
    """A graph that follows game rules and has a containment hierarchy.

    Nodes in a Character are subcategorized into Things and
    Places. Things have locations, and those locations may be Places
    or other Things. A Thing might also travel, in which case, though
    it will spend its travel time located in its origin node, it may
    spend some time contained by a Portal (i.e. an edge specialized
    for Character). If a Thing is not contained by a Portal, it's
    contained by whatever it's located in.

    """
    def __init__(self, worldview, name):
        """Store worldview and name, and set up mappings for Thing, Place, and
        Portal

        """
        super(Character, self).__init__(worldview.gorm, name)
        self.worldview = worldview
        self.thing = CharacterThingMapping(self)
        self.place = CharacterPlaceMapping(self)
        self.portal = CharacterPortalSuccessorsMapping(self)
        self.preportal = CharacterPortalPredecessorsMapping(self)
        self.avatar = CharacterAvatarGraphMapping(self)

    def add_place(self, name, **kwargs):
        """Create a new Place by the given name, and set its initial
        attributes based on the keyword arguments (if any).

        """
        super(Character, self).add_node(name, **kwargs)

    def add_thing(self, name, location, next_location=None, **kwargs):
        """Create a Thing, set its location and next_location (if provided),
        and set its initial attributes from the keyword arguments (if
        any).

        """
        super(Character, self).add_node(name, **kwargs)
        self.place2thing(name, location)

    def place2thing(self, name, location, next_location=None):
        """Turn a Place into a Thing with the given location and (if provided)
        next_location. It will keep all its attached Portals.

        """
        (branch, tick) = self.worldview.time
        self.worldview.cursor.execute(
            "INSERT INTO things ("
            "graph, node, branch, tick, location, next_location"
            ") VALUES ("
            "?, ?, ?, ?, ?"
            ");",
            (
                self.name,
                name,
                branch,
                tick,
                location,
                next_location
            )
        )

    def thing2place(self, name):
        """Unset a Thing's location, and thus turn it into a Place."""
        self.place2thing(name, None)

    def add_portal(self, origin, destination, **kwargs):
        """Connect the origin to the destination with a Portal. Keyword
        arguments are the Portal's attributes.

        """
        if origin.__class__ in (Place, Thing):
            origin = origin.name
        if destination.__class__ in (Place, Thing):
            destination = destination.name
        super(Character, self).add_edge(origin, destination, **kwargs)

    def add_avatar(self, name, host, location=None, next_location=None):
        (branch, tick) = self.worldview.time
        if isinstance(host, Character):
            host = host.name
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        self.worldview.cursor.execute(
            "INSERT INTO nodes (graph, node, branch, rev, extant) "
            "VALUES (?, ?, ?, ?, ?);",
            (
                host,
                name,
                branch,
                tick,
                True
            )
        )
        if location:
            # This will convert the node into a Thing if it isn't already
            self.worldview.cursor.execute(
                "INSERT INTO things ("
                "character, thing, branch, tick, location, next_location"
                ") VALUES (?, ?, ?, ?, ?, ?);",
                (
                    host,
                    name,
                    branch,
                    tick,
                    location,
                    next_location
                )
            )
        # Declare that the node is my avatar
        self.worldview.cursor.execute(
            "INSERT INTO avatars ("
            "character_graph, avatar_graph, avatar_node, "
            "branch, tick, is_avatar"
            ") VALUES (?, ?, ?, ?, ?, ?);",
            (
                self.name,
                host,
                name,
                branch,
                tick,
                True
            )
        )
