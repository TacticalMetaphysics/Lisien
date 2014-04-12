# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Graphical view upon a Facade.

"""
from kivy.clock import Clock
from kivy.uix.scrollview import ScrollView
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    StringProperty,
    OptionProperty,
    BoundedNumericProperty,
    ReferenceListProperty,
    ListProperty,
    NumericProperty,
    DictProperty)
from kivy.graphics import (
    Color,
    Line,
    Triangle)

from LiSE.gui.kivybits import (
    ClosetLabel,
    SaveableWidgetMetaclass)


class TimelineCell(ClosetLabel):
    """A block representing the span of time for which the variable had a
    particular value, shown by my text.

    """
    timeline = ObjectProperty()
    stringname = StringProperty()
    orientation = OptionProperty(
        'vertical', options=['horizontal', 'vertical'])
    bg_r = BoundedNumericProperty(0., min=0., max=1.)
    bg_g = BoundedNumericProperty(0., min=0., max=1.)
    bg_b = BoundedNumericProperty(0., min=0., max=1.)
    bg_a = BoundedNumericProperty(1., min=0., max=1.)
    bg_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    tick_from = NumericProperty()
    tick_to = NumericProperty()


class TimelineCursor(Widget):
    """Line representing the present moment."""
    timeline = ObjectProperty()
    color = ListProperty()
    orientation = OptionProperty(
        'vertical', options=['horizontal', 'vertical'])

    def __init__(self, **kwargs):
        """Prepare instructions in a list before adding them to the canvas.

        This is to make it easier to find the instructions and modify
        them elsewhere.

        """
        super(TimelineCursor, self).__init__(**kwargs)
        self.instructions = [
            Color(*self.color),
            Line(points=self.get_line_points()),
            Triangle(points=self.get_triangle_points())]
        for instr in self.instructions:
            self.canvas.add(instr)

    def get_line_points(self):
        """Return a list of points to make a :class:`Line` with."""
        if self.orientation == 'vertical':
            return [self.x, self.y, self.x, self.y+self.timeline.height]
        else:
            return [self.x, self.y, self.x+self.timeline.width, self.y]

    def get_triangle_points(self):
        """Return a list of points to make a :class:`Triangle` with."""
        if self.orientation == 'vertical':
            return [self.x-8, self.y,
                    self.x+8, self.y,
                    self.x, self.y-16]
        else:
            return [self.x, self.y+8,
                    self.x+16, self.y,
                    self.x, self.y-8]

    def on_color(self, *args):
        """Set the ``rgba`` property of my :class:`Color` instruction to match
        ``self.color``."""
        if len(self.instructions) < 1:
            return
        self.instructions[0].rgba = self.color

    def on_pos(self, *args):
        """Set the ``points`` property on each of my :class:`Line` and
        :class:`Triangle` instructions so they are in the correct
        positions relative to my ``pos`` property.

        """
        if len(self.instructions) < 2:
            return
        self.instructions[1].points = [
            self.x, self.y, self.x, self.y+self.timeline.height]
        if len(self.instructions) < 3:
            return
        self.instructions[2].points = [
            self.x-8, self.y+self.timeline.height,
            self.x+8, self.y+self.timeline.height,
            self.x, self.y+self.timeline.height-16]


class Timeline(StackLayout):
    """Blocks of time proceeding either left to right or top to bottom."""
    charsheet_item = ObjectProperty()
    branch = NumericProperty(None, allownone=True)
    data_iter = ObjectProperty()
    ticks_wide = BoundedNumericProperty(75, min=2)
    cursor_tick = NumericProperty()

    def __init__(self, **kwargs):
        self._trigger_refresh = Clock.create_trigger(self.refresh)
        super(Timeline, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not self.charsheet_item and self.data_getter:
            Clock.schedule_once(self.finalize, 0)
            return
        self.refresh()

    def refresh(self, *args):
        self.clear_widgets()
        for cell in self.iter_cells():
            self.add_widget(cell)

    def iter_cells(self):
        closet = self.charsheet_item.charsheet.observed.closet
        branch = self.branch if self.branch else closet.branch
        earliest = self.ticks_wide / 2
        remainder = self.ticks_wide % 2
        latest = self.ticks_wide + remainder
        for (tick_from, tick_to, text) in self.data_iter(
                branch, earliest, latest):
            yield TimelineCell(
                timeline=self,
                stringname=text,
                tick_from=tick_from,
                tick_to=tick_to)


class TimelineView(ScrollView):
    """ScrollView with a Timeline and its cursor."""
    charsheet_item = ObjectProperty()
    data_iter = ObjectProperty()
    orientation = OptionProperty(
        'lr-tb',
        options=['lr-tb', 'tb-lr', 'rl-tb', 'tb-rl',
                 'lr-bt', 'bt-lr', 'rl-bt', 'bt-rl'])
    branch = NumericProperty(None, allownone=True)
    curs_r = BoundedNumericProperty(1., min=0., max=1.)
    curs_g = BoundedNumericProperty(0., min=0., max=1.)
    curs_b = BoundedNumericProperty(0., min=0., max=1.)
    curs_a = BoundedNumericProperty(1., min=0., max=1.)
    cursor_color = ReferenceListProperty(curs_r, curs_g, curs_b, curs_a)
    cursor_tick = NumericProperty()


class NameAndValue(BoxLayout):
    """Display the name of one of the facade's attributes, as well as
    its value at the present moment.

    The attribute in question could be

    - a Thing's location
    - a Portal's origin or destination
    - a Stat (possibly that of a Thing, or a Portal, or the Character itself)

    """
    charsheet = ObjectProperty()
    """The CharSheet I'm in"""
    referent_name = StringProperty()
    """The name to use as a key for looking up the referent."""
    referent_type = OptionProperty(
        'character',
        options=['thing', 'place', 'portal', 'character'])
    """What sort of item the referent is. This determines how to look up
    the value of the variable.

    """
    variable_name = StringProperty()
    """The name of the variable whose value will be displayed."""
    display_string_name = StringProperty()
    """String to put next to the value, so the user can tell what it is."""

    def _sanity_check(self):
        """Raise ValueError when missing charsheet or referent_name"""
        if not self.charsheet:
            raise ValueError("I have no charsheet")
        if not self.referent_name:
            raise ValueError("I don't know what my referent is named")

    def get_name(self):
        """Get the text to display next to the value."""
        closet = self.charsheet.facade.observed.closet
        return closet.get_text(self.display_string_name)

    def get_value(self, branch=None, tick=None):
        """Get the value in a Unicode string."""
        def _get_thing_loc(branch=None, tick=None):
            if self.referent_type != 'thing':
                raise TypeError("I am not for thing")
            if self.variable_name != 'location':
                raise TypeError("I am not for location")
            self.sanity_check()
            return unicode(self.charsheet.facade.get_thing_location(
                self.referent_name, branch, tick))

        def _get_portal_bone(branch=None, tick=None):
            if self.referent_type != 'portal':
                raise TypeError("I am not for portal")
            if self.variable_name not in ("origin", "destination"):
                raise TypeError("Portal doesn't have that variable_name")
            self.sanity_check()
            return self.charsheet.facade.get_portal_loc_bone(
                self.referent_name, branch, tick)

        def _get_portal_orig(branch=None, tick=None):
            return unicode(_get_portal_bone(branch, tick).origin)

        def _get_portal_dest(branch=None, tick=None):
            return unicode(_get_portal_bone(branch, tick).destination)

        def _get_thing_stat(branch=None, tick=None):
            if self.referent_type != 'thing':
                raise TypeError("I am not for thing")
            self.sanity_check()
            return unicode(self.charsheet.facade.get_thing_stat(
                self.referent_name, branch, tick))

        def _get_place_stat(branch=None, tick=None):
            if self.referent_type != 'place':
                raise TypeError("I am not for place")
            self.sanity_check()
            return unicode(self.charsheet.facade.get_place_stat(
                self.referent_name, branch, tick))

        def _get_portal_stat(branch=None, tick=None):
            if self.referent_type != 'portal':
                raise TypeError("I am not for portal")
            self.sanity_check()
            return unicode(self.charsheet.facade.get_portal_stat(
                self.referent_name, branch, tick))

        def _get_character_stat(branch=None, tick=None):
            if self.referent_type != 'character':
                raise TypeError("I am not for character")
            self.sanity_check()
            return unicode(self.charsheet.facade.get_character_stat(
                self.referent_name, branch, tick))

        if self.referent_type == 'thing':
            if self.variable_name == 'location':
                return _get_thing_loc(self.referent_name, branch, tick)
            else:
                return _get_thing_stat(
                    self.referent_name, self.variable_name, branch, tick)
        if self.referent_type == 'place':
            return _get_place_stat(self.referent_name, branch, tick)
        if self.referent_type == 'portal':
            if self.variable_name == 'origin':
                return _get_portal_orig(self.referent_name, branch, tick)
            elif self.variable_name == 'destination':
                return _get_portal_dest(self.referent_name, branch, tick)
            else:
                return _get_portal_stat(
                    self.referent_name, self.variable_name, branch, tick)
        if self.referent_type == 'character':
            return _get_character_stat(
                self.referent_name, self.variable_name, branch, tick)
        raise ValueError(
            "Unknown referent type: {}".format(self.referent_type))


class CharSheetItem(BoxLayout):
    """Either a NameAndValue or a Timeline, and a toggle button to switch
    between them."""
    charsheet = ObjectProperty()
    item_type = OptionProperty(
        'character_stat',
        options=['thing_loc', 'thing_stat', 'place_stat',
                 'portal_orig', 'portal_dest', 'portal_stat',
                 'character_stat'])
    referent = StringProperty()
    stat_value = StringProperty()
    stat_name = StringProperty()
    is_timeline = BooleanProperty()

    def __init__(self, **kwargs):
        super(CharSheetItem, self).__init__(**kwargs)
        self._finalize()

    def _finalize(self, *args):
        if not (self.charsheet and self.idx):
            Clock.schedule_once(self.finalize, 0)
            return
        if self.is_timeline:
            self.add_widget(self._mk_timeline())
        else:
            self.add_name_and_value()
        self.add_toggle()

    def _thing_loc_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_thing_loc_bones(
            thing=self.referent, branch=branch,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            # from, to, value
            yield (prev_bone.tick, bone.tick, prev_bone.location)

    def _thing_stat_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_thing_stat_bones(
            thing=self.referent, stat=self.stat_name,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick,
                   getattr(prev_bone, self.stat_name))

    def _place_stat_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_place_stat_bones(
            place=self.referent, stat=self.stat_name,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick,
                   getattr(prev_bone, self.stat_name))

    def _portal_orig_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_portal_loc_bones(
            portal=self.referent,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick, prev_bone.origin)

    def _portal_dest_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_portal_loc_bones(
            portal=self.referent,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick, prev_bone.destination)

    def _portal_stat_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_portal_stat_bones(
            portal=self.referent, stat=self.stat_name,
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick,
                   getattr(prev_bone, self.stat_name))

    def _character_stat_data_iter(self, branch, min_tick, max_tick):
        innit = self.charsheet.facade.iter_stat_bones(
            branch_from=branch, branch_to=branch,
            tick_from=min_tick, tick_to=max_tick)
        prev_bone = next(innit)
        for bone in innit:
            yield (prev_bone.tick, bone.tick,
                   getattr(prev_bone, self.stat_name))

    def _mk_timeline(self):
        return TimelineView(
            charsheet_item=self,
            data_iter={
                'thing_loc': self._thing_loc_data_iter,
                'thing_stat': self._thing_stat_data_iter,
                'place_stat': self._place_stat_data_iter,
                'portal_orig': self._portal_orig_data_iter,
                'portal_dest': self._portal_dest_data_iter,
                'portal_stat': self._portal_stat_data_iter,
                'character_stat': self._character_stat_data_iter
            }[self.item_type])


class CharSheet(StackLayout):
    """The visual representation of a Facade.

    Consists of a list of key-value pairs displayed in CharSheetItems,
    which may look like cells in a table, or may look like timelines
    showing the values for the key in the past and future.

    """
    facade = ObjectProperty()
    _character_data = ListProperty()
    _thing_data = ListProperty()
    _place_data = ListProperty()
    _portal_data = ListProperty()
    timelineness = DictProperty(
        {"thing_loc": set(),
         "thing_stat": set(),
         "place_stat": set(),
         "portal_orig": set(),
         "portal_dest": set(),
         "portal_stat": set(),
         "character_stat": set()})

    def __init__(self, **kwargs):
        self._trigger_refresh = Clock.create_trigger(self.refresh)
        super(CharSheet, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not self.facade:
            Clock.schedule_once(self.finalize, 0)
            return
        self._trigger_refresh()

    def refresh(self, *args):
        old_data = [list(self._thing_data),
                    list(self._place_data),
                    list(self._portal_data)]
        self._redata()
        if [self._thing_data, self._place_data, self._portal_data] != old_data:
            self.clear_widgets()
            _ = lambda x: x
            if len(self._character_data) > 0:
                for (key, value) in self._character_data:
                    self.add_widget(self._mk_cs_item(
                        "character_stat", self.facade, key, value))
            if len(self._thing_data) > 0:
                self.add_widget(ClosetLabel(
                    closet=self.facade.closet,
                    stringname=_("Things:")))
                for (thing, data) in self._thing_data:
                    # these should really have different format than
                    # the headers
                    self.add_widget(ClosetLabel(
                        closet=self.facade.closet,
                        stringname=unicode(thing)))
                    for (key, value) in data:
                        self.add_widget(self._mk_cs_item(
                            "thing_loc" if key == "location" else "thing_stat",
                            thing, key, value))
            if len(self._place_data) > 0:
                self.add_widget(ClosetLabel(
                    closet=self.facade.closet,
                    stringname=_("Places")))
                for (place, data) in self._place_data:
                    self.add_widget(ClosetLabel(
                        closet=self.facade.closet,
                        stringname=unicode(place)))
                    for (key, value) in data:
                        self.add_widget(self._mk_cs_item(
                            "place_stat", place, key, value))
            if len(self._portal_data) > 0:
                self.add_widget(ClosetLabel(
                    closet=self.facade.closet,
                    stringname=_("Portals")))
                for (portal, data) in self._portal_data:
                    self.add_widget(ClosetLabel(
                        closet=self.facade.closet,
                        stringname=unicode(portal)))
                    for (key, value) in data:
                        if key == "origin":
                            a = "portal_orig"
                        elif key == "destination":
                            a = "portal_dest"
                        else:
                            a = "portal_stat"
                        self.add_widget(self._mk_cs_item(
                            a, portal, key, value))

    def _redata(self):
        _ = lambda x: x
        (branch, tick) = self.facade.closet.time

        def iter_thing_stat_bones(thing):
            for stat_bone in self.facade.iter_thing_stat_bones(
                    unicode(thing),
                    branch_from=branch, branch_to=branch,
                    tick_from=tick, tick_to=tick):
                yield stat_bone

        def mk_thing_data(thing):
            return (
                unicode(thing),
                [(_("location"), self.facade.get_thing_location(
                    thing, branch, tick))] + [
                    (bone.key, bone.value) for bone in
                    iter_thing_stat_bones(thing)])

        self._thing_data = [
            mk_thing_data(thing) for thing in
            self.facade.iter_things(branch, tick)]

        def iter_place_stat_bones(place):
            for stat_bone in self.facade.iter_place_stat_bones(
                    unicode(place),
                    branch_from=branch,
                    branch_to=branch,
                    tick_from=tick,
                    tick_to=tick):
                yield stat_bone

        def mk_place_data(place):
            return (unicode(place), [(bone.key, bone.value)
                                     for bone in iter_place_stat_bones])

        self._character_data = [
            (bone.key, bone.value) for bone in
            self.facade.iter_stat_bones(
                branch_from=branch,
                branch_to=branch,
                tick_from=tick,
                tick_to=tick)]

        self._place_data = [
            mk_place_data(place)
            for place in self.facade.iter_places(branch, tick)]

        def iter_portal_stat_bones(portal):
            for stat_bone in self.facade.iter_portal_stat_bones(
                    unicode(portal),
                    branch_from=branch,
                    branch_to=branch,
                    tick_from=tick,
                    tick_to=tick):
                yield stat_bone

        def mk_portal_data(portal):
            locb = self.facade.get_portal_loc_bone(portal, branch, tick)
            return (
                unicode(portal),
                [(_("origin"), locb.origin),
                 (_("destination"), locb.destination)] + [
                    (bone.key, bone.value) for bone in
                    iter_portal_stat_bones(portal)])

        self._portal_data = [
            mk_portal_data(portal) for portal in
            self.facade.iter_portals(branch, tick)]

    def _mk_cs_item(self, item_type, item_name, key, value):
        assert(item_type in (
            "thing_loc", "thing_stat", "place_stat", "portal_orig",
            "portal_dest", "portal_stat", "character_stat"))
        is_timeline = item_name in self.timelineness[item_type]
        return CharSheetItem(
            charsheet=self,
            item_type=item_type,
            referent=item_name,
            is_timeline=is_timeline,
            stat_name=key,
            stat_value=value)
