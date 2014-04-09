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
    ObjectProperty,
    StringProperty,
    OptionProperty,
    BoundedNumericProperty,
    ReferenceListProperty,
    ListProperty,
    NumericProperty)
from kivy.graphics import (
    Color,
    Line,
    Triangle)

from LiSE.gui.kivybits import ClosetLabel


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
    charsheet = ObjectProperty()
    """The CharSheet I am in"""
    keys = ListProperty()
    """Keys into the skeleton. These take me almost all the way down to
    the lowest level of the skeleton; they lack only the present
    branch.

    """
    data = ObjectProperty()
    """Technically this contains a :class:`Skeleton`, but only the lowest
    level of it, where it's a wrapper around an :class:`Array`.

    """
    field = StringProperty('value')
    """What property of the bones of the variable represented here
    contains the variable's present value?

    Defaults to "value". This should become an OptionProperty once
    I've settled on what field names to use for special data like
    location.

    """
    branch = BoundedNumericProperty(None, min=0, allownone=True)
    """If set, the timeline will only display data in this
    branch. Otherwise it will display data from the present branch.

    """
    min_tick = BoundedNumericProperty(0, min=0)
    """The first tick to show."""
    max_tick = BoundedNumericProperty(100, min=0)
    """The last tick to show."""

    def __init__(self, **kwargs):
        """Set up a trigger for self.redata, then call self.finalize()"""
        super(Timeline, self).__init__(**kwargs)
        self._trigger_redata = Clock.create_trigger(self.redata)
        self.finalize()

    def finalize(self, *args):
        """Arrange to refresh my data whenever it changes, and whenever the
        branch changes.

        """
        if not self.charsheet and self.keys:
            Clock.schedule_once(self.finalize, 0)
            return
        ptr = self.charsheet.closet.skeleton
        for key in self.keys:
            ptr = ptr[key]
        ptr.register_listener(lambda child, k, v: self._trigger_redata)
        self.charsheet.closet.timestream.register_branch_listener(
            self._trigger_redata)
        self.bind(branch=self._trigger_redata)
        self.cursor = TimelineCursor(
            timeline=self,
            color=self.cursor_color)
        self._trigger_redata()

    def redata(self, *args):
        """Replace my data with the latest from the closet."""
        ptr = self.charsheet.closet.skeleton
        for key in self.keys:
            ptr = ptr[key]
        self.data = ptr[self.branch if self.branch
                        else self.charsheet.character.closet.branch]

    def iter_cells(self):
        """Iterate over the bones in self.data, and yield TimelineCell
        instances representing them."""
        boneiter = self.data.iterbones()
        bone = next(boneiter)
        prev_tick = bone.tick
        prev_value = getattr(bone, self.field)
        while prev_tick < self.min_tick:
            bone = next(boneiter)
            prev_tick = bone.tick
            prev_value = getattr(bone, self.field)
        while bone.tick < self.max_tick:
            bone = next(boneiter)
            yield TimelineCell(
                timeline=self,
                stringname=prev_value,
                tick_from=prev_tick,
                tick_to=bone.tick)
            prev_tick = bone.tick
            prev_value = getattr(bone, self.field)

    def on_data(self, *args):
        """Clear old cells and add new ones."""
        self.clear_widgets()
        for cell in self.iter_cells():
            self.add_widget(cell)


class TimelineView(ScrollView):
    """ScrollView with a Timeline and its cursor."""
    charsheet = ObjectProperty()
    keys = ListProperty()
    field = StringProperty('value')
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
