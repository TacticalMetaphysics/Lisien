from kivy.properties import (
    ObjectProperty,
    ReferenceListProperty,
    StringProperty)
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label


class Table(GridLayout):
    key0 = StringProperty()
    key1 = StringProperty(None, allownone=True)
    key2 = StringProperty(None, allownone=True)
    keys = ReferenceListProperty(key0, key1, key2)
    charsheet = ObjectProperty()

    @property
    def skel(self):
        if self.keys[0] is None:
            return self.character_skel
        elif self.keys[1] is None:
            return self.character_skel[self.keys[0]]
        elif self.keys[2] is None:
            return self.character_skel[self.keys[0]][self.keys[1]]
        else:
            return self.character_skel[
                self.keys[0]][self.keys[1]][self.keys[2]]

    def on_parent(self, *args):
        self.cols = len(self.colkeys)
        self.row_default_height = (self.charsheet.style.fontsize
                                   + self.charsheet.style.spacing)
        self.row_force_default = True

        for key in self.colkeys:
            self.add_widget(Label(
                text=key,
                font_name=self.charsheet.style.fontface,
                font_size=self.charsheet.style.fontsize,
                color=self.charsheet.style.textcolor.rgba))
        for rd in self.iter_skeleton():
            for key in self.colkeys:
                self.add_widget(Label(
                    text=rd[key],
                    font_name=self.charsheet.style.fontface,
                    font_size=self.charsheet.style.fontsize,
                    color=self.charsheet.style.textcolor.rgba))

    def iter_skeleton(self, branch=None, tick=None):
        if branch is None:
            branch = self.charsheet.character.closet.branch
        if tick is None:
            tick = self.charsheet.character.closet.tick
        for rd in self.character_skel.iterrows():
            if (
                    rd["branch"] == branch and
                    rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick)):
                yield rd

    def iterrows(self, branch=None, tick=None):
        closet = self.charsheet.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        for rd in self.iter_skeleton(branch, tick):
            yield [rd[key] for key in self.colkeys]


class ThingTable(Table):
    colkeys = ["dimension", "thing", "location"]

    @property
    def character_skel(self):
        return self.charsheet.character.thingdict

    def get_branch_rd_iter(self, branch):
        thingdict = self.charsheet.character.thingdict
        if self.keys[0] is None:
            for dimension in thingdict:
                for thing in thingdict[dimension]:
                    for rd in thingdict[dimension][thing][
                            branch].iterrows():
                        yield rd
        elif self.keys[1] is None:
            dimension = self.keys[0]
            for thing in thingdict[dimension]:
                for rd in thingdict[
                        dimension][thing][branch].iterrows():
                    yield rd
        else:
            dimension = self.keys[0]
            thing = self.keys[1]
            for rd in thingdict[
                    dimension][thing][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        closet = self.charsheet.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        covered = set()
        for rd in self.get_branch_rd_iter(branch):
            if (rd["dimension"], rd["thing"]) in covered:
                continue
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick):
                thing = closet.get_thing(
                    rd["dimension"], rd["thing"])
                rd2 = thing.locations[branch]
                prev = None
                r = None
                for (tick_from, rd3) in rd2.items():
                    if tick_from > tick:
                        if prev is not None:
                            r = {
                                "dimension": rd["dimension"],
                                "thing": rd["thing"],
                                "location": prev["location"]}
                        break
                    prev = rd3
                if r is None:
                    if prev is None:
                        r = {
                            "dimension": rd["dimension"],
                            "thing": rd["thing"],
                            "location": "nowhere"}
                    else:
                        r = {
                            "dimension": rd["dimension"],
                            "thing": rd["thing"],
                            "location": prev["location"]}
                covered.add((rd["dimension"], rd["thing"]))
                yield r


class PlaceTable(Table):
    colkeys = ["dimension", "place"]

    @property
    def character_skel(self):
        return self.charsheet.character.placedict


class PortalTable(Table):
    colkeys = ["dimension", "origin", "destination"]

    @property
    def character_skel(self):
        return self.charsheet.character.portaldict


class StatTable(Table):
    colkeys = ["stat", "value"]

    @property
    def character_skel(self):
        return self.charsheet.character.statdict

    def get_branch_rd_iter(self, branch):
        statdict = self.charsheet.character.statdict
        if self.keys[0] is None:
            for stat in statdict:
                for rd in statdict[
                        stat][branch].iterrows():
                    yield rd
        else:
            stat = self.keys[0]
            for rd in statdict[
                    stat][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        closet = self.charsheet.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        covered = set()
        prev = None
        for rd in self.get_branch_rd_iter(branch):
            if rd["stat"] in covered:
                continue
            elif rd["tick_from"] == tick:
                covered.add(rd["stat"])
                prev = None
                yield rd
            elif rd["tick_from"] > tick:
                covered.add(rd["stat"])
                r = prev
                prev = None
                yield r
            prev = rd


class SkillTable(Table):
    colkeys = ["skill", "deck"]

    @property
    def character_skel(self):
        return self.charsheet.character.skilldict

    def get_branch_rd_iter(self, branch):
        skilldict = self.charsheet.character.skilldict
        if self.keys[0] is None:
            for skill in skilldict:
                for rd in skilldict[
                        skill][branch].iterrows():
                    yield rd
        else:
            skill = self.keys[0]
            for rd in skilldict[
                    skill][branch].iterrows():
                yield rd

    def iter_skeleton(self, branch=None, tick=None):
        closet = self.charsheet.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        covered = set()
        prev = None
        for rd in self.get_branch_rd_iter(branch):
            if rd["skill"] in covered:
                continue
            elif rd["tick_from"] == tick:
                covered.add(rd["skill"])
                prev = None
                yield rd
            elif rd["tick_from"] > tick:
                covered.add(rd["skill"])
                r = prev
                prev = None
                yield r
            prev = rd


class TableView(RelativeLayout):
    table = ObjectProperty()

    def __init__(self, **kwargs):
        super(TableView, self).__init__(**kwargs)
        closet = self.table.charsheet.character.closet
        self.edit_button = ToggleButton(
            pos_hint={'x': 0, 'top': 1},
            text=closet.get_text('edit'))
        self.add_widget(self.edit_button)
        self.add_widget(self.table)
