# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets displaying information about "characters," which are
collections of simulated entities and facts.

"""
from calendar import CalendarView
from table import (
    TableView,
    CharStatTableView
)
from item import CharSheetItem
from LiSE.gui.kivybits import (
    SaveableWidgetMetaclass,
    ClosetButton)
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
from kivy.adapters.listadapter import ListAdapter
from kivy.adapters.models import SelectableDataItem
from kivy.uix.listview import ListView, SelectableView
from kivy.properties import (
    NumericProperty,
    BooleanProperty,
    OptionProperty,
    AliasProperty,
    ListProperty,
    ObjectProperty)
from kivy.clock import Clock
from LiSE.util import (
    THING_TAB,
    PLACE_TAB,
    PORTAL_TAB,
    CHAR_TAB,
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL,
    SHEET_ITEM_TYPES,
    CALENDAR_TYPES)


class ListItemToggle(SelectableView, ToggleButton):
    noun = ObjectProperty(allownone=True)


class CharListAdapter(ListAdapter):
    character = ObjectProperty()

    def __init__(self, **kwargs):
        super(CharListAdapter, self).__init__(**kwargs)
        self.trigger_redata = Clock.create_trigger(self.redata)

    def on_character(self, *args):
        self.trigger_redata()


def args_converter(idx, item):
    return {
        'noun': item.noun,
        'text': unicode(item.noun),
        'size_hint_y': None,
        'height': 25}


class NounItem(SelectableDataItem):
    def __init__(self, noun, **kwargs):
        super(NounItem, self).__init__(**kwargs)
        self.noun = noun
        self.is_selected = False


class NounListView(StackLayout):
    charsheet = ObjectProperty()
    selection_mode = OptionProperty('multiple',
                                    options=['none', 'single', 'multiple'])
    selection = ListProperty([])
    allow_empty_selection = BooleanProperty(False)
    finalized = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(NounListView, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if self.charsheet is None:
            Clock.schedule_once(self.finalize, 0)
            return
        if self.finalized:
            raise Exception("It seems the charsheet has been set twice")
        else:
            nouniter = self.getiter(self.charsheet.character)
            adapter = ListAdapter(
                data=[NounItem(noun) for noun in nouniter],
                selection_mode=self.selection_mode,
                args_converter=args_converter,
                allow_empty_selection=self.allow_empty_selection,
                cls=ListItemToggle)
            adapter.bind(selection=self.setter('selection'))
            listview = ListView(adapter=adapter)
            self.add_widget(listview)
            self.finalized = True


class ThingListView(NounListView):
    @staticmethod
    def getiter(character):
        (branch, tick) = character.sanetime()
        return character.iter_things(branch, tick)


class PlaceListView(NounListView):
    @staticmethod
    def getiter(character):
        return character.iter_places()


class PortalListView(NounListView):
    @staticmethod
    def getiter(character):
        (branch, tick) = character.sanetime()
        return character.iter_portals(branch, tick)


class StatItem(SelectableDataItem):
    def __init__(self, name, **kwargs):
        super(StatItem, self).__init__(**kwargs)
        self.name = name


class SpecialItem(SelectableDataItem):
    def __init__(self, name, **kwargs):
        super(SpecialItem, self).__init__(**kwargs)
        self.name = name


class NounStatListView(StackLayout):
    specialitems = ListProperty([])
    nounitems = ListProperty()
    selection = ListProperty([])
    selection_mode = OptionProperty('multiple',
                                    options=['none', 'single', 'multiple'])
    allow_empty_selection = BooleanProperty(False)
    finalized = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(NounStatListView, self).__init__(**kwargs)
        self.finalize()

    def add_stat(self, stat):
        self.specialitems.append(stat)
        self.clear_widgets()
        self.finalize()

    def on_nounitems(self, *args):
        data2b = [SpecialItem(special) for special in
                  self.specialitems]
        for nounitem in self.nounitems:
            for key in nounitem.noun.iter_stat_keys():
                data2b.append(StatItem(key))
        self.adapter.data = data2b

    def finalize(self, *args):
        self.adapter = ListAdapter(
            data=[],  # will be filled on_nounitems
            selection_mode=self.selection_mode,
            args_converter=lambda k, v: {
                'text': v.name,
                'size_hint_y': None,
                'height': 25},
            allow_empty_selection=self.allow_empty_selection,
            cls=ListItemToggle)
        self.adapter.bind(selection=self.setter('selection'))
        listview = ListView(adapter=self.adapter)
        self.add_widget(listview)
        self.finalized = True


class StatListView(Widget):
    charsheet = ObjectProperty()
    selection = ListProperty([])
    specialitems = ListProperty([])
    selection_mode = OptionProperty('single',
                                    options=['none', 'single', 'multiple'])
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
    charsheet = ObjectProperty()
    closet = AliasProperty(
        lambda self: self.charsheet.character.closet,
        lambda self, v: None,
        bind=('charsheet',))
    get_text = AliasProperty(
        lambda self: self.closet.get_text,
        lambda self, v: None,
        bind=('closet',))
    insertion_point = NumericProperty(0)

    def confirm(self):
        r = self.record()
        if r is not None:  # might be 0
            type_bone = CharSheet.bonetype(
                character=unicode(self.charsheet.character),
                idx=self.insertion_point,
                type=r,
                height=max([100, self.charsheet.height - sum(
                    csitem.height for csitem in self.charsheet.csitems)]))
            itemskel = self.closet.skeleton[u"character_sheet_item_type"][
                unicode(self.charsheet.character)]
            i = max(itemskel.keys())
            if self.insertion_point > i:
                self.charsheet.character.closet.set_bone(type_bone)
                self.charsheet.repop()
                self.dismiss()
                return
            self.charsheet.shift_its(self.insertion_point, 1)
            self.charsheet.character.closet.set_bone(type_bone)
            self.charsheet.repop()
            self.dismiss()

    def record(self):
        character = self.charsheet.character
        if self.ids.panel.current_tab == self.ids.calendars:
            tab = self.ids.calendars_panel
            if tab.current_tab == self.ids.place_cal:
                if len(self.ids.place_cal_place.selection) != 1:
                    return False
                if len(self.ids.place_cal_stat.selection) != 1:
                    return False
                placen = self.ids.place_cal_place.selection[0].text
                statn = self.ids.place_cal_stat.selection[0].text
                self.charsheet.character.closet.set_bone(
                    CharSheet.bonetypes["place_cal"](
                        character=unicode(character),
                        place=unicode(placen),
                        stat=unicode(statn),
                        idx=self.insertion_point,
                        type=PLACE_CAL))
                return PLACE_CAL
            elif tab.current_tab == self.ids.portal_cal:
                if len(self.ids.portal_cal_portal.selection) != 1:
                    return
                if len(self.ids.portal_cal_portal.selection) != 1:
                    return
                portn = self.ids.portal_cal_portal.selection[0].text
                statn = self.ids.portal_cal_stat.selection[0].text
                self.charsheet.character.closet.set_bone(
                    CharSheet.bonetypes["portal_cal"](
                        character=unicode(character),
                        portal=unicode(portn),
                        stat=unicode(statn),
                        idx=self.insertion_point,
                        type=PORTAL_CAL))
                return PORTAL_CAL
            elif tab.current_tab == self.ids.char_cal:
                if len(self.ids.char_cal_stat.selection) != 1:
                    return
                statn = self.ids.char_cal_stat.selection[0].text
                self.charsheet.character.closet.set_bone(
                    CharSheet.bonetypes["char_cal"](
                        character=unicode(character),
                        stat=unicode(statn),
                        idx=self.insertion_point,
                        type=CHAR_CAL))
                return CHAR_CAL
            else:
                if len(self.ids.thing_cal_thing.selection) != 1:
                    return
                if len(self.ids.thing_cal_stat.selection) != 1:
                    return
                thingn = self.ids.thing_cal_thing.selection[0].text
                statn = self.ids.thing_cal_stat.selection[0].text
                self.charsheet.character.closet.set_bone(
                    CharSheet.bonetypes["thing_cal"](
                        character=unicode(character),
                        thing=unicode(thingn),
                        stat=unicode(statn),
                        idx=self.insertion_point,
                        type=THING_CAL))
                return THING_CAL
        else:
            tab = self.ids.tables_panel
            if tab.current_tab == self.ids.place_tab:
                place_tab_places = [
                    CharSheet.bonetypes["place_tab_place"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        place=unicode(nounitem.text),
                        type=PLACE_TAB)
                    for nounitem in self.ids.place_tab_place.selection]
                if len(place_tab_places) < 1:
                    return
                place_tab_stats = [
                    CharSheet.bonetypes["place_tab_stat"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        place=unicode(statitem.text),
                        type=PLACE_TAB)
                    for statitem in self.ids.place_tab_stat.selection]
                if len(place_tab_stats) < 1:
                    return
                for bone in place_tab_places + place_tab_stats:
                    self.charsheet.character.closet.set_bone(bone)
                return PLACE_TAB
            elif tab.current_tab == self.ids.portal_tab:
                portal_tab_portals = [
                    CharSheet.bonetypes["portal_tab_portal"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        portal=unicode(nounitem.text),
                        type=PORTAL_TAB)
                    for nounitem in self.ids.portal_tab_portal.selection]
                if len(portal_tab_portals) < 1:
                    return
                portal_tab_stats = [
                    CharSheet.bonetypes["portal_tab_stats"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        stat=unicode(statitem.text),
                        type=PORTAL_TAB)
                    for statitem in self.ids.portal_tab_stat.selection]
                if len(portal_tab_stats) < 1:
                    return
                for bone in portal_tab_portals + portal_tab_stats:
                    self.charsheet.character.closet.set_bone(bone)
                return PORTAL_TAB
            elif tab.current_tab == self.ids.char_tab:
                char_tab_stats = [
                    CharSheet.bonetypes["char_tab_stat"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        stat=unicode(statitem.text),
                        type=CHAR_TAB)
                    for statitem in self.ids.char_tab_stat.selection]
                if len(char_tab_stats) < 1:
                    return
                for bone in char_tab_stats:
                    self.charsheet.character.closet.set_bone(bone)
                return CHAR_TAB
            else:
                thing_tab_things = [
                    CharSheet.bonetypes["thing_tab_thing"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        thing=unicode(nounitem.text),
                        type=THING_TAB)
                    for nounitem in self.ids.thing_tab_thing.selection]
                if len(thing_tab_things) < 1:
                    return
                thing_tab_stats = [
                    CharSheet.bonetypes["thing_tab_stat"](
                        character=unicode(character),
                        idx=self.insertion_point,
                        stat=unicode(statitem.text),
                        type=THING_TAB)
                    for statitem in self.ids.thing_tab_stat.selection]
                if len(thing_tab_stats) < 1:
                    return
                for bone in thing_tab_things + thing_tab_stats:
                    self.charsheet.character.closet.set_bone(bone)
                return THING_TAB


def char_sheet_table_def(
        table_name, final_pkey, typ, foreign_key=(None, None)):
    r = (
        table_name,
        {"columns":
         {"character": "TEXT NOT NULL",
          "idx": "INTEGER NOT NULL",
          final_pkey: "TEXT NOT NULL",
          "type": "INTEGER NOT NULL DEFAULT {}".format(typ)},
         "primary_key":
         ("character", "idx", final_pkey),
         "foreign_keys":
         {"character, idx, type":
          ("character_sheet_item_type", "character, idx, type")},
         "checks": ["type={}".format(typ)]})
    if None not in foreign_key:
        (foreign_key_tab, foreign_key_key) = foreign_key
        r[1]["foreign_keys"].update(
            {"character, {}".format(final_pkey):
             (foreign_key_tab, "character, {}".format(foreign_key_key))})
    return r


def char_sheet_calendar_def(
        table_name, col_x, typ, foreign_key=(None, None)):
    r = (
        table_name,
        {"columns":
         {"character": "TEXT NOT NULL",
          "idx": "INTEGER NOT NULL",
          col_x: "TEXT NOT NULL",
          "stat": "TEXT NOT NULL",
          "type": "INTEGER DEFAULT {}".format(typ)},
         "primary_key":
         ("character", "idx"),
         "foreign_keys":
         {"character, idx, type":
          ("character_sheet_item_type", "character, idx, type")},
         "checks": ["type={}".format(typ)]})
    if None not in foreign_key:
        (foreign_key_tab, foreign_key_key) = foreign_key
        r[1]["foreign_keys"].update(
            {"character, {}".format(col_x):
             (foreign_key_tab, "character, {}".format(foreign_key_key))})
    return r


class Sizer(ClosetButton):
    charsheet = ObjectProperty()
    i = NumericProperty()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud['charsheet'] = self.charsheet
            touch.ud['sizer'] = self
            touch.grab(self)
            self.prior_y = self.y
            self.state = 'down'
            touch.ud['sizer_i'] = self.i
            return True

    def on_touch_move(self, touch):
        if 'sizer' not in touch.ud or touch.ud['sizer'] is not self:
            touch.ungrab(self)
            self.state = 'normal'
            return
        self.parent.center_y = touch.y
        wid_before = self.charsheet.csitems[self.i]
        wid_after = self.charsheet.csitems[self.i+1]
        wid_before.y = self.top
        wid_before.height = wid_before.top - self.top
        wid_after.height = self.y - wid_after.y
        return True

    def on_touch_up(self, touch):
        if 'sizer' not in touch.ud or touch.ud['sizer'] is not self:
            touch.ungrab(self)
            self.state = 'normal'
            return
        self.state = 'normal'
        return True


class CharSheet(StackLayout):
    """A display of some or all of the information making up a Character.

A CharSheet is a layout of vertically stacked widgets that are not
usable outside of a CharSheet. Those widgets are instances or
subclasses of Table, Calendar, or Image. In developer mode, each
widget has a toggle next to it that will enable editing the data
therein. The new data will be applied at the current branch and
tick.

    """
    __metaclass__ = SaveableWidgetMetaclass
    demands = ["character"]
    tables = [
        ("character_sheet_item_type",
         {"columns":
          {"character": "TEXT NOT NULL",
           "idx": "INTEGER NOT NULL",
           "type": "INTEGER NOT NULL",
           "height": "INTEGER NOT NULL"},
          "primary_key":
          ("character", "idx"),
          "checks":
          ["type IN ({})".format(
              ", ".join([str(typ) for typ in SHEET_ITEM_TYPES]))]}),
        char_sheet_table_def(
            "thing_tab_thing",
            "thing",
            THING_TAB,
            foreign_key=("thing", "name")),
        char_sheet_table_def(
            "thing_tab_stat",
            "stat",
            THING_TAB),
        char_sheet_table_def(
            "place_tab_place",
            "place",
            PLACE_TAB,
            foreign_key=("place", "place")),
        char_sheet_table_def(
            "place_tab_stat",
            "stat",
            PLACE_TAB),
        char_sheet_table_def(
            "portal_tab_portal",
            "portal",
            PORTAL_TAB,
            foreign_key=("portal", "name")),
        char_sheet_table_def(
            "portal_tab_stat",
            "stat",
            PORTAL_TAB),
        char_sheet_table_def(
            "char_tab_stat",
            "stat",
            CHAR_TAB),
        char_sheet_calendar_def(
            "thing_cal",
            "thing",
            THING_CAL,
            foreign_key=("thing", "name")),
        char_sheet_calendar_def(
            "place_cal",
            "place",
            PLACE_CAL,
            foreign_key=("place", "place")),
        char_sheet_calendar_def(
            "portal_cal",
            "portal",
            PORTAL_CAL,
            foreign_key=("portal", "name")),
        ("char_cal",
         {"columns":
          {"character": "TEXT NOT NULL",
           "idx": "INTEGER NOT NULL",
           "stat": "TEXT NOT NULL",
           "type": "INTEGER DEFAULT {}".format(CHAR_CAL),
           "height": "INTEGER NOT NULL DEFAULT 100"},
          "primary_key":
          ("character", "idx"),
          "foreign_keys":
          {"character, idx, type":
           ("character_sheet_item_type", "character, idx, type")},
          "checks": ["type={}".format(CHAR_CAL),
                     "height>=50"]})]
    character = ObjectProperty()
    csitems = ListProperty()
    boxeditems = ListProperty()

    def _get_bones(self):
        superskel = self.character.closet.skeleton
        r = {}
        for tab in self.tablenames:
            if unicode(self.character) not in superskel[tab]:
                superskel[tab][unicode(self.character)] = []
            r[tab] = []
            for (i, v) in self.character.closet.skeleton[tab][
                    unicode(self.character)].iteritems():
                while len(r[tab]) <= i:
                    r[tab].append(None)
                r[tab][i] = v
        return r

    def _set_bones(self, v):
        def setter(v):
            assert(v.character is not None)
            self.character.closet.set_bone(v)
        for tab in self.tablenames:
            self.character.closet.skeleton[tab][
                unicode(self.character)] = []
            i = 0
            for w in v[tab]:
                if w is None:
                    continue
                elif hasattr(w, 'itervalues'):
                    for bone in w.itervalues():
                        if bone is not None:
                            setter(bone)
                else:
                    setter(w)
                i += 1

    def __init__(self, **kwargs):
        super(CharSheet, self).__init__(**kwargs)
        self.repop()

    def add_item(self, i):
        # I need the layout, proper
        layout = self.parent.parent
        layout.handle_adbut(self, i)

    outerbone_ksd = {
        THING_TAB: [u'thing_tab_thing', u'thing_tab_stat'],
        PLACE_TAB: [u'place_tab_place', u'place_tab_stat'],
        PORTAL_TAB: [u'portal_tab_portal', u'portal_tab_stat'],
        CHAR_TAB: [u'char_tab_stat'],
        THING_CAL: [u'thing_cal'],
        PLACE_CAL: [u'place_cal'],
        PORTAL_CAL: [u'portal_cal'],
        CHAR_CAL: [u'char_cal']
    }

    def move_it(self, i, d):
        superskel = self.character.closet.skeleton
        mainbone = superskel[u'character_sheet_item_type'][
            unicode(self.character)].pop(i)
        belowbone = superskel[u'character_sheet_item_type'][
            unicode(self.character)].pop(i+d)
        abovebones = set([mainbone._replace(idx=i+d)])
        for ks in self.outerbone_ksd[mainbone.type]:
            for k in ks:
                try:
                    abovebones.add(
                        superskel[k][unicode(self.character)].pop(
                            i)._replace(idx=i+d))
                except KeyError:
                    pass
        belowbones = set([belowbone._replace(idx=i-d)])
        for ks in self.outerbone_ksd[belowbone.type]:
            for k in ks:
                try:
                    belowbones.add(superskel[k][unicode(self.charater)].pop(
                        i)._replace(idx=i-d))
                except KeyError:
                    pass
        for bone in abovebones.union(belowbones):
            self.character.closet.set_bone(bone)

    def move_it_up(self, i):
        self.move_it(i, -1)

    def move_it_down(self, i):
        self.move_it(i, 1)

    def shift_its(self, pivot, d):
        if d == 0:
            return
        r = {}
        for (tab, bones) in self._get_bones().iteritems():
            r[tab] = set()

            def bitters():
                for v in bones:
                    if hasattr(v, 'itervalues'):
                        for bone in v.itervalues():
                            yield bone
                    elif v is None:
                        continue
                    else:
                        yield v
            for bone in bitters():
                if (bone.idx > pivot and d > 0) or (
                        bone.idx < pivot and d > 0):
                    r[tab].add(bone._replace(
                        idx=bone.idx+d))
                else:
                    r[tab].add(bone)

    def del_it(self, i):
        superskel = self.character.closet.skeleton
        mainbone = superskel[u'character_sheet_item_type'][
            unicode(self.character)].pop(i)
        if mainbone is not None:
            for ks in self.outerbone_ksd[mainbone.type]:
                for k in ks:
                    try:
                        del superskel[k][unicode(self.character)][i]
                    except KeyError:
                        pass

    def repop(self, *args):
        """Iterate over the bones under my name, and add widgets appropriate
to each of them.

Each widget gets kwargs character, item_type, and
keys. item_type is an integer, defined in SHEET_ITEM_TYPE,
representing both the widget class and the way it looks up its
data. keys identify a particular entity whose data will be displayed,
but they never include branch or tick--CharSheet will only display
things appropriate to the present, whenever that may be.

        """
        self.size_hint = (1, None)
        for box in self.boxeditems:
            box.clear_widgets()
        self.clear_widgets()
        self.boxeditems = []
        self.csitems = []
        if unicode(self.character) not in self.character.closet.skeleton[
                u"character_sheet_item_type"]:
            self.add_widget(ClosetButton(
                closet=self.character.closet,
                symbolic=True,
                stringname="@add",
                fun=self.add_item,
                arg=0,
                size_hint_y=None,
                height=50))
            return
        _ = self.character.closet.get_text
        i = 0
        for bone in self.character.closet.skeleton[
                u"character_sheet_item_type"][
                unicode(self.character)].iterbones():
            assert(bone.character is not None)
            if bone.type == THING_TAB:
                headers = [_("thing")]
                fieldnames = ["name"]
                stats = []
                for subbone in self.iter_tab_i_bones("thing_tab_stat", i):
                    if subbone.stat == "location":
                        headers.append(_("location"))
                        fieldnames.append("location")
                    else:
                        stats.append(subbone.stat)
                widspec = (TableView, {
                    'headers': headers,
                    'fieldnames': fieldnames,
                    'items': [self.character.get_thing(subbone.thing)
                              for subbone in
                              self.iter_tab_i_bones("thing_tab_thing", i)],
                    'stats': stats})
            elif bone.type == PLACE_TAB:
                widspec = (TableView, {
                    'headers': [_("place")],
                    'fieldnames': ["name"],
                    'items': [
                        self.character.get_place(bone.place)
                        for bone in
                        self.iter_tab_i_bones("place_tab_place", i)],
                    'stats': [
                        bone.stat for bone in
                        self.iter_tab_i_bones("place_tab_stat", i)]})
            elif bone.type == PORTAL_TAB:
                headers = ["portal"]
                fieldnames = ["name"]
                stats = []
                for subbone in self.iter_tab_i_bones("portal_tab_stat", i):
                    if subbone.stat == "origin":
                        headers.append(_("origin"))
                        fieldnames.append("origin")
                    elif subbone.stat == "destination":
                        headers.append(_("destination"))
                        fieldnames.append("destination")
                    else:
                        stats.append(subbone.stat)
                widspec = (TableView, {
                    'headers': headers,
                    'fieldnames': fieldnames,
                    'stats': stats,
                    'items': [
                        self.character.get_portal(bone.portal) for bone in
                        self.iter_tab_i_bones("portal_tab_portal", i)]})
            elif bone.type == CHAR_TAB:
                widspec = (CharStatTableView, {
                    'stats': [
                        bone.stat for bone in
                        self.iter_tab_i_bones("char_tab_stat", i)]})
            elif bone.type in CALENDAR_TYPES:
                (tabn, keyns) = {
                    THING_CAL: ("thing_cal", ["thing", "stat"]),
                    PLACE_CAL: ("place_cal", ["place", "stat"]),
                    PORTAL_CAL: ("portal_cal", ["portal", "stat"]),
                    CHAR_CAL: ("char_cal", ["stat"])
                }[bone.type]
                subbone = self.character.closet.skeleton[tabn][
                    unicode(self.character)][i]
                widspec = (CalendarView, {
                    'cal_type': bone.type,
                    'boneatt': keyns[1],
                    'key': getattr(subbone, keyns[0]),
                    'stat': getattr(subbone, keyns[1])
                    if len(keyns) == 2 else '',
                    'mybone': subbone})
            else:
                raise ValueError("Unknown item type: {}".format(bone.type))

            kwargs = {
                'charsheet': self,
                'character': self.character,
                'csbone': bone,
                'i': i,
                'size_hint_x': 0.8,
                'height': bone.height}
            widspec[1].update(kwargs)
            self.csitems.append(widspec)

            i += 1

        itemct = i
        initial_addbut = AddButton(
            closet=self.character.closet,
            fun=self.add_item,
            arg=0,
            size_hint_y=None,
            height=20)
        final_addbut = AddButton(
            closet=self.character.closet,
            fun=self.add_item,
            arg=itemct,
            size_hint_y=None,
            height=20)
        middle = StackLayout(size_hint_y=0.9)
        for widspec in self.csitems:
            itembox = CharSheetItem(
                charsheet=self,
                closet=self.character.closet,
                widspec=widspec)
            self.boxeditems.append(itembox)
            buttonbox = BoxLayout(
                orientation='vertical',
                size_hint_x=0.2)
            itembox.add_widget(buttonbox)
            if item.i > 0:
                upb = UpButton(
                    closet=self.character.closet,
                    fun=self.move_it_up,
                    arg=item.i,
                    size_hint_y=0.2)
                buttonbox.add_widget(upb)
            delb = DelButton(
                closet=self.character.closet,
                fun=self.del_it,
                arg=item.i,
                size_hint_y=0.6)
            buttonbox.add_widget(delb)
            whereat = item.i + 1
            if whereat < itemct:
                downb = DownButton(
                    closet=self.character.closet,
                    fun=self.move_it_down,
                    arg=item.i,
                    size_hint_y=0.2)
                buttonbox.add_widget(downb)
            middle.add_widget(itembox)
            if whereat < itemct:
                sizeaddbox = BoxLayout(
                    size_hint_y=None,
                    height=20)
                sizeaddbox.add_widget(Sizer(
                    charsheet=self,
                    i=item.i,
                    closet=self.character.closet,
                    size_hint_x=0.2))
                sizeaddbox.add_widget(AddButton(
                    closet=self.character.closet,
                    fun=self.add_item,
                    arg=whereat,
                    size_hint_x=0.8))
                sizeaddbox.bind(top=itembox.setter('y'))
                middle.add_widget(sizeaddbox)

        self.add_widget(initial_addbut)
        self.add_widget(middle)
        self.add_widget(final_addbut)

    def iter_tab_i_bones(self, tab, i):
        for bone in self.character.closet.skeleton[tab][
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

    def on_touch_move(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_move(touch)

    def on_touch_up(self, touch):
        """Dispatch this touch to all my children."""
        for child in self.children:
            child.on_touch_up(touch)

    def push_down(self, i, n):
        """Move every item after i forward n spaces."""
        for (table, decl) in self.tables:
            skel = self.character.closet.skeleton[table][
                unicode(self.character)]
            for j in xrange(i, i+n):
                if j in skel:
                    leks = skel[j]
                    del skel[j]
                    for bone in leks.iterbones():
                        self.character.closet.set_bone(
                            bone._replace(idx=bone.idx+n))


class CharSheetView(ScrollView):
    charsheet = ObjectProperty()

    def on_charsheet(self, *args):
        if self.charsheet is None:
            return
        self.add_widget(self.charsheet)
        self.bind(pos=self.charsheet.setter('pos'),
                  size=self.charsheet.setter('size'))

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.ud["charsheet"] = self.children[0]
        return super(CharSheetView, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if "charsheet" in touch.ud:
            del touch.ud["charsheet"]
        return super(CharSheetView, self).on_touch_up(touch)
