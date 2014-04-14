# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets displaying information about "characters," which are
collections of simulated entities and facts.

"""
from calendar import CalendarView
from collections import defaultdict

from kivy.adapters.listadapter import ListAdapter
from kivy.adapters.models import SelectableDataItem
from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    NumericProperty,
    BooleanProperty,
    OptionProperty,
    AliasProperty,
    ListProperty,
    ReferenceListProperty,
    ObjectProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.listview import ListView, SelectableView
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
from kivy.uix.stacklayout import StackLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget

from kivy.logger import Logger

from LiSE.gui.kivybits import (
    SaveableWidgetMetaclass,
    ClosetButton,
    ClosetToggleButton
)
from LiSE.util import (
    SHEET_ITEM_TYPES,
    CALENDAR_TYPES
)

from table import TableView

csitem_type_table_d = defaultdict(set)
csitem_type_table_d['char_cal'].add('char_cal')
_ = lambda x: x


class ListItemToggle(SelectableView, ToggleButton):
    """ToggleButton workalike for a ListView."""
    noun = ObjectProperty(allownone=True)


class NounItem(SelectableDataItem):
    def __init__(self, noun, **kwargs):
        """Remember the noun. Start unselected."""
        super(NounItem, self).__init__(**kwargs)
        self.noun = noun
        self.is_selected = False


class NounListView(StackLayout):
    """Container for a ListView that offers a selection of nouns."""
    charsheet = ObjectProperty()
    """The charsheet I am in."""
    selection_mode = OptionProperty(
        'multiple', options=['none', 'single', 'multiple'])
    """To be passed to the internal ListView."""
    selection = ListProperty([])
    """Nouns selected here."""
    allow_empty_selection = BooleanProperty(False)
    """To be passed to the internal ListView."""

    def __init__(self, **kwargs):
        """Call self.finalize() ASAP."""
        super(NounListView, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        """Put together a ListView out of the nouns I get from self.getiter"""
        if self.charsheet is None:
            Clock.schedule_once(self.finalize, 0)
            return
        nouniter = self.getiter(self.charsheet.facade)
        adapter = ListAdapter(
            data=[NounItem(noun) for noun in nouniter],
            selection_mode=self.selection_mode,
            args_converter=lambda k, v: {
                'noun': v.noun,
                'text': unicode(v.noun),
                'size_hint_y': None,
                'height': 25},
            allow_empty_selection=self.allow_empty_selection,
            cls=ListItemToggle)
        adapter.bind(selection=self.setter('selection'))
        listview = ListView(adapter=adapter)
        self.add_widget(listview)
        self.finalized = True


class ThingListView(NounListView):
    """NounListView for Things"""
    @staticmethod
    def getiter(facade, branch=None, tick=None):
        return facade.iter_things(*facade.sanetime(branch, tick))


class PlaceListView(NounListView):
    """NounListView for Places"""
    @staticmethod
    def getiter(facade, branch=None, tick=None):
        return facade.iter_places()


class PortalListView(NounListView):
    """NounListView for Portals"""
    @staticmethod
    def getiter(facade, branch=None, tick=None):
        return facade.iter_portals(*facade.sanetime(branch, tick))


class StatItem(SelectableDataItem):
    """SelectableDataItem for Stats"""
    def __init__(self, name, **kwargs):
        super(StatItem, self).__init__(**kwargs)
        self.name = name


class SpecialItem(SelectableDataItem):
    """SelectableDataItem for something unusual, like location"""
    def __init__(self, name, **kwargs):
        super(SpecialItem, self).__init__(**kwargs)
        self.name = name


class NounStatListView(StackLayout):
    """Container for a ListView that shows selectable stats had by nouns
    (eg. places, portals, things).

    """
    charsheet = ObjectProperty()
    selection = AliasProperty(
        lambda self: list(self.iter_selection()),
        lambda self, v: None)
    specialitems = ListProperty([])
    nounitems = ListProperty()
    selection_mode = OptionProperty(
        'multiple', options=['none', 'single', 'multiple'])
    allow_empty_selection = BooleanProperty(False)
    finalized = BooleanProperty(False)

    def iter_selection(self):
        for child in self.children:
            if child.state == 'down':
                yield child

    def add_stat(self, stat):
        self.specialitems.append(stat)
        self.on_nounitems()

    def on_nounitems(self, *args):
        def iteritems():
            for special in self.specialitems:
                yield special
            for nounitem in self.nounitems:
                for key in nounitem.noun.iter_stat_keys():
                    yield key
        self.clear_widgets()
        for item in iteritems():
            self.add_widget(ClosetToggleButton(
                closet=self.charsheet.character.closet,
                stringname=item,
                size_hint_y=None,
                height=25))


class StatListView(Widget):
    """Container for a ListView for stats"""
    charsheet = ObjectProperty()
    selection = ListProperty([])
    specialitems = ListProperty([])
    selection_mode = OptionProperty(
        'single', options=['none', 'single', 'multiple'])
    allow_empty_selection = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(StatListView, self).__init__(**kwargs)
        self.finalize()

    def add_stat(self, stat):
        self.specialitems.append(stat)
        self.clear_widgets()
        self.finalize()

    def finalize(self, *args):
        if self.charsheet is None:
            Clock.schedule_once(self.finalize, 0)
            return
        adapter = ListAdapter(
            data=[StatItem(name=key) for key in
                  self.charsheet.character.iter_stat_keys()],
            args_converter=lambda k, v: {
                'name': v.name,
                'size_hint_y': None,
                'height': 25},
            selection_mode=self.selection_mode,
            allow_empty_selection=self.allow_empty_selection,
            cls=ListItemToggle)
        listview = ListView(adapter=adapter)
        self.add_widget(listview)


class CharSheetAdder(ModalView):
    """A dialog in which you pick something to add to the CharSheet."""
    charsheet = ObjectProperty()
    character = AliasProperty(
        lambda self: self.charsheet.character,
        lambda self, v: None,
        bind=('charsheet',))
    closet = AliasProperty(
        lambda self: self.charsheet.character.closet,
        lambda self, v: None,
        bind=('charsheet',))
    get_text = AliasProperty(
        lambda self: self.charsheet.character.closet.get_text,
        lambda self, v: None,
        bind=('charsheet',))
    insertion_point = NumericProperty(0)

    def confirm(self):
        """The user pressed the button to corfirm adding something to the
        charsheet. Handle it.

        """
        csitskel = self.closet.skeleton[u"character_sheet_item_type"]
        if unicode(self.character) not in csitskel:
            csitskel[unicode(self.charsheet.character)] = {}
        # TODO change over to size_hint_y for every charsheet item
        r = self.new_bones()
        if r:
            self.charsheet.insert_bones(r)
            self.charsheet._trigger_repop()
            self.dismiss()

    def new_bones(self):
        """Return a tuple of bones for the new charsheet item.

        The first bone is always for character_sheet_item_type. The
        ones beyond that are specific to one type or another.

        """
        charname = unicode(self.charsheet.character)
        idx = self.insertion_point
        protobone = CharSheet.bonetype(
            character=charname,
            idx=idx,
            height=min(self.charsheet.height / 2,
                       self.charsheet.height - sum(
                           getattr(csitem, 'height') for csitem in
                           self.charsheet.csitems)))
        if self.ids.panel.current_tab == self.ids.calendars:
            tab = self.ids.calendars_panel
            if tab.current_tab == self.ids.place_cal:
                if len(self.ids.place_cal_place.selection) != 1:
                    return False
                if len(self.ids.place_cal_stat.selection) != 1:
                    return False
                placen = self.ids.place_cal_place.selection[0].text
                statn = self.ids.place_cal_stat.selection[0].text
                return (
                    protobone._replace(type="place_cal"),
                    CharSheet.bonetypes["place_cal"](
                        character=charname,
                        place=unicode(placen),
                        stat=unicode(statn),
                        idx=idx,
                        type='place_cal'))
            elif tab.current_tab == self.ids.portal_cal:
                if len(self.ids.portal_cal_portal.selection) != 1:
                    return
                if len(self.ids.portal_cal_portal.selection) != 1:
                    return
                portn = self.ids.portal_cal_portal.selection[0].text
                statn = self.ids.portal_cal_stat.selection[0].text
                return (
                    protobone._replace(type="portal_cal"),
                    CharSheet.bonetypes["portal_cal"](
                        character=charname,
                        portal=unicode(portn),
                        stat=unicode(statn),
                        idx=idx,
                        type='portal_cal'))
            elif tab.current_tab == self.ids.char_cal:
                if len(self.ids.char_cal_stat.selection) != 1:
                    return
                statn = self.ids.char_cal_stat.selection[0].text
                return (
                    protobone._replace(type="char_cal"),
                    CharSheet.bonetypes["char_cal"](
                        character=charname,
                        stat=unicode(statn),
                        idx=idx,
                        type='char_cal'))
            else:
                if len(self.ids.thing_cal_thing.selection) != 1:
                    return
                if len(self.ids.thing_cal_stat.selection) != 1:
                    return
                thingn = self.ids.thing_cal_thing.selection[0].text
                statn = self.ids.thing_cal_stat.selection[0].text
                return (
                    protobone._replace(type="thing_cal"),
                    CharSheet.bonetypes["thing_cal"](
                        character=charname,
                        thing=unicode(thingn),
                        stat=unicode(statn),
                        idx=idx,
                        type='thing_cal'))
        else:
            tab = self.ids.tables_panel
            if tab.current_tab == self.ids.place_tab:
                place_tab_places = [
                    CharSheet.bonetypes["place_tab_place"](
                        character=charname,
                        idx=idx,
                        place=unicode(nounitem.text),
                        type='place_tab')
                    for nounitem in self.ids.place_tab_place.selection]
                if len(place_tab_places) < 1:
                    return
                place_tab_stats = [
                    CharSheet.bonetypes["place_tab_stat"](
                        character=charname,
                        idx=idx,
                        place=unicode(statitem.text),
                        type='place_tab')
                    for statitem in self.ids.place_tab_stat.selection]
                if len(place_tab_stats) < 1:
                    return
                return (
                    protobone._replace(type="place_tab"),
                    place_tab_places,
                    place_tab_stats)
            elif tab.current_tab == self.ids.portal_tab:
                portal_tab_portals = [
                    CharSheet.bonetypes["portal_tab_portal"](
                        character=charname,
                        idx=idx,
                        portal=unicode(nounitem.text),
                        type='portal_tab')
                    for nounitem in self.ids.portal_tab_portal.selection]
                if len(portal_tab_portals) < 1:
                    return
                portal_tab_stats = [
                    CharSheet.bonetypes["portal_tab_stats"](
                        character=charname,
                        idx=idx,
                        stat=unicode(statitem.text),
                        type='portal_tab')
                    for statitem in self.ids.portal_tab_stat.selection]
                if len(portal_tab_stats) < 1:
                    return
                return (
                    protobone._replace(type="portal_tab"),
                    portal_tab_portals, portal_tab_stats)
            elif tab.current_tab == self.ids.char_tab:
                char_tab_stats = [
                    CharSheet.bonetypes["char_tab_stat"](
                        character=charname,
                        idx=idx,
                        stat=unicode(statitem.text),
                        type='char_tab')
                    for statitem in self.ids.char_tab_stat.selection]
                if len(char_tab_stats) < 1:
                    return
                return (
                    protobone._replace(type="char_tab"),
                    char_tab_stats)
            else:
                thing_tab_things = [
                    CharSheet.bonetypes["thing_tab_thing"](
                        character=charname,
                        idx=idx,
                        thing=unicode(nounitem.text),
                        type='thing_tab')
                    for nounitem in self.ids.thing_tab_thing.selection]
                if len(thing_tab_things) < 1:
                    return
                thing_tab_stats = [
                    CharSheet.bonetypes["thing_tab_stat"](
                        character=charname,
                        idx=idx,
                        stat=unicode(statitem.text),
                        type='thing_tab')
                    for statitem in self.ids.thing_tab_stat.selection]
                if len(thing_tab_stats) < 1:
                    return
                return (
                    protobone._replace(type="thing_tab"),
                    thing_tab_things,
                    thing_tab_stats)


class CharSheetItem(BoxLayout):
    """Container for either a Calendar or a Table.

    In either case, it has some buttons on the right for deleting it
    or moving it up or down, and some buttons on the bottom for adding
    a new one below or resizing.

    """
    csbone = ObjectProperty()
    content = ObjectProperty()
    sizer = ObjectProperty(None, allownone=True)
    adder = ObjectProperty(None, allownone=True)
    asbox = ObjectProperty(None, allownone=True)
    buttons = ListProperty()
    middle = ObjectProperty()
    item_class = ObjectProperty()
    item_kwargs = DictProperty()
    widspec = ReferenceListProperty(item_class, item_kwargs)
    charsheet = ObjectProperty()
    mybone = ObjectProperty()
    i = AliasProperty(
        lambda self: self.csbone.idx if self.csbone else -1,
        lambda self, v: None,
        bind=('csbone',))

    def __init__(self, **kwargs):
        self._trigger_set_bone = Clock.create_trigger(self.set_bone)
        kwargs['orientation'] = 'vertical'
        kwargs['size_hint_y'] = None
        super(CharSheetItem, self).__init__(**kwargs)
        self.finalize()

    def on_item_kwargs(self, *args):
        if not self.item_kwargs:
            return
        self.charsheet = self.item_kwargs['charsheet']
        if 'mybone' in self.item_kwargs:
            self.mybone = self.item_kwargs['mybone']

    def on_touch_down(self, touch):
        if self.sizer.collide_point(*touch.pos):
            touch.ud['sizer'] = self.sizer
            return True
        return super(CharSheetItem, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if not ('sizer' in touch.ud and touch.ud['sizer'] is self.sizer):
            return
        touch.push()
        b = touch.y - self.sizer.height / 2
        h = self.top - b
        self.y = b
        self.height = h
        touch.pop()

    def on_touch_up(self, touch):
        if 'sizer' in touch.ud:
            del touch.ud['sizer']

    def set_bone(self, *args):
        if self.csbone:
            self.charsheet.character.closet.set_bone(self.csbone)

    def finalize(self, *args):
        _ = lambda x: x
        closet = self.charsheet.character.closet
        if not (self.item_class and self.item_kwargs):
            Clock.schedule_once(self.finalize, 0)
            return
        self.middle = BoxLayout()
        self.content = self.item_class(**self.item_kwargs)
        self.buttonbox = BoxLayout(
            orientation='vertical',
            size_hint_x=0.2)
        self.buttons = [ClosetButton(
            closet=closet,
            symbolic=True,
            stringname=_('@del'),
            fun=self.charsheet.del_item,
            arg=self.i)]
        if self.i > 0:
            self.buttons.insert(0, ClosetButton(
                closet=closet,
                symbolic=True,
                stringname=_('@up'),
                fun=self.charsheet.move_it_up,
                arg=self.i,
                size_hint_y=0.1))
            if self.i+1 < len(self.charsheet.csitems):
                self.buttons.append(ClosetButton(
                    closet=closet,
                    symbolic=True,
                    stringname=_('@down'),
                    fun=self.charsheet.move_it_down,
                    arg=self.i,
                    size_hint_y=0.1))
        for button in self.buttons:
            self.buttonbox.add_widget(button)
        self.middle.add_widget(self.content)
        self.middle.add_widget(self.buttonbox)
        self.add_widget(self.middle)
        self.sizer = ClosetButton(
            closet=self.charsheet.character.closet,
            symbolic=True,
            stringname=_('@ud'),
            size_hint_x=0.2)
        self.adder = ClosetButton(
            closet=self.charsheet.character.closet,
            symbolic=True,
            stringname=_('@add'),
            fun=self.charsheet.add_item,
            arg=self.i+1,
            size_hint_x=0.8)
        self.asbox = BoxLayout(
            size_hint_y=None,
            height=40)
        self.asbox.add_widget(self.sizer)
        self.asbox.add_widget(self.adder)
        self.add_widget(self.asbox)
        self.height = self.csbone.height
        self.content.height = self.buttonbox.height = (
            self.height - self.asbox.height)
        self.charsheet.i2wid[self.i] = self


class CharSheet(StackLayout):
    """A display of some or all of the information making up a Character.

    CharSheet contains a ListView of CharSheetItems.

    """
    __metaclass__ = SaveableWidgetMetaclass
    demands = ["character"]

    def _calendar_decl(
            table_name, col_x, typ, foreign_key=(None, None)):
        """Generate a tuple to describe one of the database tables that the
        Calendar widget uses.

        The form of the tuple is that used by LiSE.orm.SaveableMetaclass

        """
        csitem_type_table_d[typ].add(table_name)
        r = (
            table_name,
            {"columns":
             {"character": "TEXT NOT NULL",
              "idx": "INTEGER NOT NULL",
              col_x: "TEXT NOT NULL",
              "stat": "TEXT NOT NULL",
              "type": "TEXT DEFAULT {}".format(typ)},
             "primary_key":
             ("character", "idx"),
             "foreign_keys":
             {"character, idx, type":
              ("character_sheet_item_type", "character, idx, type")},
             "checks": ["type='{}'".format(typ)]})
        if None not in foreign_key:
            (foreign_key_tab, foreign_key_key) = foreign_key
            r[1]["foreign_keys"].update(
                {"character, {}".format(col_x):
                 (foreign_key_tab, "character, {}".format(foreign_key_key))})
        return r

    def _table_decl(
            table_name, final_pkey, typ, foreign_key=(None, None)):
        """Generate a tuple to describe one of CharSheet's database tables.

        The form of the tuple is that used by
        LiSE.orm.SaveableMetaclass

        """
        csitem_type_table_d[typ].add(table_name)
        r = (
            table_name,
            {"columns":
             {"character": "TEXT NOT NULL",
              "observer": "TEXT NOT NULL",
              "idx": "INTEGER NOT NULL",
              final_pkey: "TEXT NOT NULL",
              "type": "TEXT NOT NULL DEFAULT {}".format(typ)},
             "primary_key":
             ("character", "idx", final_pkey),
             "foreign_keys":
             {"character, idx, type":
              ("character_sheet_item_type", "character, idx, type")},
             "checks": ["type='{}'".format(typ)]})
        if None not in foreign_key:
            (foreign_key_tab, foreign_key_key) = foreign_key
            r[1]["foreign_keys"].update(
                {"character, {}".format(final_pkey):
                 (foreign_key_tab, "character, {}".format(foreign_key_key))})
        return r

    tables = [
        ("character_sheet_item_type",
         {"columns":
          {"character": "TEXT NOT NULL",
           "observer": "TEXT NOT NULL",
           "idx": "INTEGER NOT NULL",
           "type": "TEXT NOT NULL",
           "height": "INTEGER NOT NULL"},
          "primary_key":
          ("character", "idx"),
          "checks":
          ["type IN ({})".format(
              ", ".join(
                  ["'{}'".format(typ)
                   for typ in SHEET_ITEM_TYPES]))]}),
        _table_decl(
            "thing_tab_thing",
            "thing",
            "thing_tab",
            foreign_key=("thing", "name")),
        _table_decl(
            "thing_tab_stat",
            "stat",
            "thing_tab"),
        _table_decl(
            "place_tab_place",
            "place",
            "place_tab",
            foreign_key=("place", "place")),
        _table_decl(
            "place_tab_stat",
            "stat",
            "place_tab"),
        _table_decl(
            "portal_tab_portal",
            "portal",
            "portal_tab",
            foreign_key=("portal", "name")),
        _table_decl(
            "portal_tab_stat",
            "stat",
            "portal_tab"),
        _table_decl(
            "char_tab_stat",
            "stat",
            "char_tab"),
        _calendar_decl(
            "thing_cal",
            "thing",
            "thing_cal",
            foreign_key=("thing", "name")),
        _calendar_decl(
            "place_cal",
            "place",
            "place_cal",
            foreign_key=("place", "place")),
        _calendar_decl(
            "portal_cal",
            "portal",
            "portal_cal",
            foreign_key=("portal", "name")),
        ("char_cal",
         {"columns":
          {"character": "TEXT NOT NULL",
           "idx": "INTEGER NOT NULL",
           "stat": "TEXT NOT NULL",
           "type": "TEXT DEFAULT 'char_cal'",
           "height": "INTEGER NOT NULL DEFAULT 100"},
          "primary_key":
          ("character", "idx"),
          "foreign_keys":
          {"character, idx, type":
           ("character_sheet_item_type", "character, idx, type")},
          "checks": ["type='char_cal'",
                     "height>=50"]})]
    character = ObjectProperty()
    """The character this sheet is about."""
    observer = ObjectProperty()
    """The character whose view on this one we'll show.

    Defaults to Omniscient, a character with no distinguishing
    characteristics save knowing everything about everything with
    perfect accuracy.

    """
    facade = ObjectProperty()
    """The facade from which I will get all the information I'll display.

    It will in turn get the information from the character, though it
    may be distorted.

    """
    csitems = ListProperty([])
    """Bones from character_sheet_item_type"""
    i2wid = DictProperty()
    view = ObjectProperty()

    def __init__(self, **kwargs):
        self._trigger_repop = Clock.create_trigger(self.repop)
        kwargs['size_hint'] = (1, None)
        super(CharSheet, self).__init__(**kwargs)
        sv = ScrollView(
            do_scroll_x=False,
            pos=self.pos,
            size=self.size)
        self.bind(pos=sv.setter('pos'),
                  size=sv.setter('size'))
        self.add_widget(sv)
        self.view = StackLayout()
        sv.add_widget(self.view)
        self.repop()

    def on_character(self, *args):
        if not self.observer:
            self.observer = self.character.closet.character_d['Omniscient']
        self.facade = self.character.get_facade(self.observer)
        self.finalize()

    def add_item(self, i):
        self.parent.handle_adbut(self, i)

    def del_item(self, i):
        pass

    def finalize(self, *args):
        """If I do not yet contain any items, show a button to add
        one. Otherwise, fill myself with the widgets for the items."""
        closet = self.character.closet
        obsrvr = unicode(self.observer)
        char = unicode(self.character)
        skel = closet.skeleton[u'character_sheet_item_type']
        if obsrvr not in skel:
            skel[obsrvr] = {}
        if char not in skel[obsrvr]:
            skel[obsrvr][char] = {}
        myskel = skel[obsrvr][char]
        myskel.register_set_listener(self._trigger_repop)
        myskel.register_del_listener(self._trigger_repop)
        closet.register_time_listener(self._trigger_repop)
        self.repop()
        self.add_widget(ClosetButton(
            closet=closet,
            symbolic=True,
            stringname=_("@add"),
            fun=self.add_item,
            arg=0,
            size_hint_y=None,
            height=50))

    def bone2widspec(self, bone):
        if bone is None:
            return {'csbone': bone,
                    'item_class': Widget,
                    'item_kwargs': {}}
        if bone.type == 'thing_tab':
            headers = ["thing"]
            fieldnames = ["name"]
            stats = []
            for subbone in self.iter_tab_i_bones("thing_tab_stat", bone.idx):
                if subbone.stat == "location":
                    headers.append("location")
                    fieldnames.append("location")
                else:
                    stats.append(subbone.stat)
            return {
                'csbone': bone,
                'item_class': TableView,
                'item_kwargs': {
                    'charsheet': self,
                    'headers': fieldnames,
                    'fieldnames': fieldnames,
                    'items': [
                        self.character.get_thing(subsubbone.thing)
                        for subsubbone in self.iter_tab_i_bones(
                            "thing_tab_thing", bone.idx)]}}
        elif bone.type == 'place_tab':
            return {
                'csbone': bone,
                'item_class': TableView,
                'item_kwargs': {
                    'charsheet': self,
                    'headers': ['place'],
                    'fieldnames': ['name'],
                    'items': [
                        self.character.get_place(subbone.place)
                        for subbone in self.iter_tab_i_bones(
                            "thing_tab_thing", bone.idx)]}}
        elif bone.type == 'portal_tab':
            headers = ["portal"]
            fieldnames = ["name"]
            stats = []
            for subbone in self.iter_tab_i_bones("portal_tab_stat", bone.idx):
                if subbone.stat in ("origin", "destination"):
                    headers.append(subbone.stat)
                    fieldnames.append(subbone.stat)
                else:
                    stats.append(subbone.stat)
            return {
                'csbone': bone,
                'item_class': TableView,
                'item_kwargs': {
                    'charsheet': self,
                    'headers': headers,
                    'fieldnames': fieldnames,
                    'stats': stats,
                    'items': [
                        self.character.get_portal(subbone.portal)
                        for subbone in self.iter_tab_i_bones(
                            "portal_tab_portal", bone.idx)]}}
        elif bone.type in CALENDAR_TYPES:
            keyns = {
                'thing_cal': ("thing", "stat"),
                'place_cal': ("place", "stat"),
                'portal_cal': ("portal", "stat"),
                'char_cal': ("stat",)
            }[bone.type]
            subbone = self.character.closet.skeleton[bone.type][
                unicode(self.character)][bone.idx]
            return {
                'csbone': bone,
                'item_class': CalendarView,
                'item_kwargs': {
                    'charsheet': self,
                    'boneatt': keyns[-1],
                    'key': getattr(subbone, keyns[0]),
                    'stat': getattr(subbone, keyns[1]),
                    'mybone': subbone}}
        else:
            raise ValueError("Unknown CharSheet item type: {}".format(
                bone.type))

    def repop(self, *args):
        if not self.character:
            Clock.schedule_once(self.repop, 0)
            return
        Logger.debug("CharSheet: about to repopulate")
        myskel = self.character.closet.skeleton[
            u'character_sheet_item_type'][
            unicode(self.observer)][
            unicode(self.character)]
        # This used to be a ListView. It may yet be again. I changed
        # it to make the debugger easier to use.
        self.csitems = []
        for bone in myskel.iterbones():
            widspec = self.bone2widspec(bone)
            wid = CharSheetItem(**widspec)
            self.csitems.append(wid)
        self.view.clear_widgets()
        for item in self.csitems:
            self.view.add_widget(item)
        if len(self.csitems) == 0:
            self.view.add_widget(ClosetButton(
                closet=self.character.closet,
                symbolic=True,
                stringname=_("@add"),
                fun=self.add_item,
                arg=0,
                size_hint_y=None,
                height=50,
                top=self.top))

    def iter_tab_i_bones(self, tab, i):
        for bone in self.facade.closet.skeleton[tab][
                unicode(self.character)][i].iterbones():
            yield bone

    def on_touch_down(self, touch):
        """If one of my children catches the touch, nobody else ought to, so
        return True in that case.

        """
        for child in self.children:
            if child.on_touch_down(touch):
                touch.ud['charsheet'] = self
                return True

    # I think "dispatch" actually has some special meaning
    def on_touch_move(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_move(touch)

    def on_touch_up(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_up(touch)

    def _localbones(self):
        closet = self.character.closet
        r = {}
        for tab in self.tablenames:
            r[tab] = list(closet.skeleton[tab][
                unicode(self.observer)][
                unicode(self.character)])
        return r

    def _writebones(self, bone_d):
        closet = self.character.closet
        for bone_l in bone_d.itervalues():
            for old_bone in bone_l:
                # correct the idx
                new_bone = old_bone._replace(idx=bone_l.index(old_bone))
                closet.del_bone(old_bone)
                closet.set_bone(new_bone)

    def insert_bones(self, bones):
        """Move extant items downward to make room for each new bone as
        needed. Then set as normal.

        """
        # For now, this works by making a local copy of the whole
        # charsheet, modifying that, and clobbering the original. This
        # causes more disk activity than necessary. Probably not much
        # disk activity in the absolute.
        localbones = self._localbones()
        for bone in bones:
            localbones[bone._name].insert(bone.idx, bone)
        self._writebones(localbones)

    def move_items(self, begin, end, n):
        """Move items ``begin`` through ``end`` forward``n`` places.

        ``n`` may be negative.

        """
        if begin == end:
            self.move_item(begin, n)
            return
        if n == 0:
            return
        localbones = self._localbones()
        for tab in self.tablenames:
            mobile_bones = localbones[tab][begin:end]
            del localbones[tab][begin:end]
            new_beginning = begin + n
            localbones[tab][new_beginning:new_beginning] = mobile_bones
        self._writebones(localbones)
        self._trigger_repop()

    def move_item(self, i, n):
        """Get item at index ``i``, and move it ``n`` places forward.

        ``n`` may be negative.

        """
        if n == 0:
            return
        localbones = self._localbones()
        for tab in self.tablenames:
            bone = localbones[tab][i]
            del localbones[tab][i]
            localbones[tab].insert(
                i+n,
                bone._replace(idx=i+n))
        self._writebones(localbones)
        self._trigger_repop()

    def move_it_up(self, i, n):
        self.move_item(i, -n)

    def move_it_down(self, i, n):
        self.move_item(i, n)
