# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets displaying information about "characters," which are
collections of simulated entities and facts.

"""
from calendar import CalendarView
from table import (
    TableView,
    CharStatTableView)
from LiSE.gui.kivybits import (
    SaveableWidgetMetaclass,
    ClosetButton)
from kivy.logger import Logger
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.uix.modalview import ModalView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
from kivy.adapters.listadapter import ListAdapter
from kivy.adapters.models import SelectableDataItem
from kivy.uix.listview import ListView, SelectableView
from kivy.properties import (
    DictProperty,
    NumericProperty,
    BooleanProperty,
    OptionProperty,
    AliasProperty,
    ListProperty,
    ObjectProperty)
from kivy.clock import Clock
from LiSE.data import (
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
    pass


class CharListAdapter(ListAdapter):
    character = ObjectProperty()

    def __init__(self, **kwargs):
        super(CharListAdapter, self).__init__(**kwargs)
        self.trigger_redata = Clock.create_trigger(self.redata)

    def on_character(self, *args):
        self.trigger_redata()


def args_converter(idx, item):
    return {
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
        if not self.nounitems:
            Clock.schedule_once(self.finalize, 0)
            return
        inidata = [SpecialItem(special) for special in
                   self.specialitems]
        for noun in self.nounitems:
            print(noun)
            inidata.extend([StatItem(key) for key in noun.iter_stat_keys()])
        adapter = ListAdapter(
            data=inidata,
            selection_mode=self.selection_mode,
            args_converter=lambda k, v: {
                'text': v.name,
                'size_hint_y': None,
                'height': 25},
            allow_empty_selection=self.allow_empty_selection,
            cls=ListItemToggle)
        adapter.bind(selection=self.setter('selection'))
        listview = ListView(adapter=adapter)
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
        if r is not None:
            type_bone = CharSheet.bonetype(
                character=unicode(self.charsheet.character),
                idx=self.insertion_point,
                type=r)
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
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
                self.charsheet.repop()
                return THING_TAB


def char_sheet_table_def(
        table_name, final_pkey, typ, foreign_key=(None, None)):
    r = (
        table_name,
        {"columns":
         {"character": "TEXT NOT NULL",
          "idx": "INTEGER NOT NULL",
          final_pkey: "TEXT NOT NULL",
          "type": "INTEGER NOT NULL DEFAULT {}".format(typ),
          "height": "INTEGER NOT NULL DEFAULT 100"},
         "primary_key":
         ("character", "idx", final_pkey),
         "foreign_keys":
         {"character, idx, type":
          ("character_sheet_item_type", "character, idx, type")},
         "checks": ["type={}".format(typ),
                    "height>=50"]})
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
          "type": "INTEGER DEFAULT {}".format(typ),
          "height": "INTEGER NOT NULL DEFAULT 100"},
         "primary_key":
         ("character", "idx"),
         "foreign_keys":
         {"character, idx, type":
          ("character_sheet_item_type", "character, idx, type")},
         "checks": ["type={}".format(typ),
                    "height>=50"]})
    if None not in foreign_key:
        (foreign_key_tab, foreign_key_key) = foreign_key
        r[1]["foreign_keys"].update(
            {"character, {}".format(col_x):
             (foreign_key_tab, "character, {}".format(foreign_key_key))})
    return r


class Sizer(ClosetButton):
    pass


class AddButton(ClosetButton):
    pass


class EditButton(ToggleButton):
    imgd = DictProperty({})


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
           "type": "INTEGER NOT NULL"},
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

    def __init__(self, **kwargs):
        super(CharSheet, self).__init__(**kwargs)
        self.bind(character=self.repop)

    def add_item(self, i):
        # I need the layout, proper
        layout = self.parent.parent
        layout.handle_adbut(self, i)

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
        def make_calendar(i, bone, edbut):
            typ = bone.type
            (tabn, keyns) = {
                THING_CAL: ("thing_cal", ["thing", "stat"]),
                PLACE_CAL: ("place_cal", ["place", "stat"]),
                PORTAL_CAL: ("portal_cal", ["portal", "stat"]),
                CHAR_CAL: ("char_cal", ["stat"])
            }[typ]
            bone = self.character.closet.skeleton[tabn][
                unicode(self.character)][i]
            return CalendarView(
                charsheet=self,
                cal_type=typ,
                boneatt=keyns[1],
                key=getattr(bone, keyns[0]),
                stat=getattr(bone, keyns[1]) if len(keyns) == 2 else '',
                edbut=edbut,
                size_hint=(0.8, None),
                height=bone.height)
        self.size_hint = (1, None)
        self.clear_widgets()
        _ = self.character.closet.get_text
        i = 0
        self.add_widget(
            AddButton(
                closet=self.character.closet,
                fun=self.add_item,
                arg=0,
                size_hint_y=None,
                height=20))
        if unicode(self.character) not in self.character.closet.skeleton[
                u"character_sheet_item_type"]:
            return
        edimgd = self.character.closet.get_imgs(
            ("locked", "unlocked"))
        edimgd = {'normal': edimgd['locked'],
                  'down': edimgd['unlocked']}
        for bone in self.character.closet.skeleton[
                u"character_sheet_item_type"][
                unicode(self.character)].iterbones():
            edbut = EditButton(imgd=edimgd)
            if bone.type == THING_TAB:
                headers = [_("thing")]
                fieldnames = ["name"]
                stats = []
                for subbone in self.iter_tab_i_bones("thing_tab_stat", i):
                    if subbone.stat == "location":
                        headers.append("location")
                        fieldnames.append("location")
                    else:
                        stats.append(subbone.stat)
                cwid = TableView(
                    character=self.character,
                    headers=headers,
                    fieldnames=fieldnames,
                    items=[self.character.get_thing(bone.thing) for bone in
                           self.iter_tab_i_bones("thing_tab_thing", i)],
                    stats=stats,
                    edbut=edbut,
                    size_hint=(0.8, None),
                    height=bone.height)
            elif bone.type == PLACE_TAB:
                cwid = TableView(
                    character=self.character,
                    headers=[_("place")],
                    fieldnames=["name"],
                    items=[self.character.get_place(bone.place) for bone in
                           self.iter_tab_i_bones("place_tab_place", i)],
                    stats=[bone.stat for bone in
                           self.iter_tab_i_bones("place_tab_stat", i)],
                    edbut=edbut,
                    size_hint=(0.8, None),
                    height=bone.height)
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
                cwid = TableView(
                    character=self.character,
                    headers=headers,
                    fieldnames=fieldnames,
                    stats=stats,
                    items=[self.character.get_portal(bone.portal) for bone in
                           self.iter_tab_i_bones("portal_tab_portal", i)],
                    edbut=edbut,
                    size_hint=(0.8, None),
                    height=bone.height)
            elif bone.type == CHAR_TAB:
                cwid = CharStatTableView(
                    character=self.character,
                    stats=[bone.stat for bone in
                           self.iter_tab_i_bones("char_tab_stat", i)],
                    edbut=edbut,
                    size_hint=(0.8, None),
                    height=bone.height)
            elif bone.type in CALENDAR_TYPES:
                cwid = make_calendar(i, bone, edbut)
            else:
                raise ValueError("Unknown item type: {}".format(bone.type))

            entry = GridLayout(cols=2, size_hint_y=None)
            buttons = StackLayout(size_hint_x=0.2)
            if i > 0:
                sizer = Sizer(
                    closet=self.character.closet,
                    height=10)
                buttons.add_widget(sizer)
                addbut = AddButton(
                    closet=self.character.closet,
                    fun=self.add_item,
                    arg=i,
                    size_hint_y=None,
                    height=20)
                buttons.add_widget(addbut)
            buttons.add_widget(edbut)
            entry.add_widget(cwid)
            entry.add_widget(buttons)
            self.csitems.append(entry)
            self.add_widget(entry)

            i += 1

        # Supposing I've got to the final entry without running out of
        # space, let the final entry fill all available space;
        # otherwise leave it be.
        def adj_final_entry(*args):
            final_entry = self.csitems[-1]
            bot = self.y + 20
            if final_entry.y > bot:
                final_entry.size_hint_y = 1.
            else:
                final_entry.size_hint_y = None
        if hasattr(self, 'adj_final_entry'):
            self.unbind(y=self.adj_final_entry)
        self.adj_final_entry = adj_final_entry
        self.bind(y=self.adj_final_entry)

        final_addbut = AddButton(
            closet=self.character.closet,
            fun=self.add_item,
            arg=i,
            size_hint_y=None,
            height=20)
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
    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.ud["charsheet"] = self.children[0]
        return super(CharSheetView, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if "charsheet" in touch.ud:
            del touch.ud["charsheet"]
        return super(CharSheetView, self).on_touch_up(touch)
