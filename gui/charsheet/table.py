from kivy.properties import (
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from itemview import ItemView


def iter_skeleton(self, branch=None, tick=None):
    if branch is None:
        branch = self.character.closet.branch
    if tick is None:
        tick = self.character.closet.tick
    for rd in self.character_skel.iterrows():
        if (
                rd["branch"] == branch and
                rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick)):
            yield rd


def get_branch_rd_iter_thing(self, branch):
    thingdict = self.character.thingdict
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


def iter_skeleton_thing(self, branch=None, tick=None):
    closet = self.character.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    for rd in get_branch_rd_iter_thing(self, branch):
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


def get_branch_rd_iter_stat(self, branch):
    statdict = self.character.statdict
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


def iter_skeleton_stat(self, branch=None, tick=None):
    closet = self.character.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    prev = None
    for rd in get_branch_rd_iter_stat(self, branch):
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


def get_branch_rd_iter_skill(self, branch):
    skilldict = self.character.skilldict
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


def iter_skeleton_skill(self, branch=None, tick=None):
    closet = self.character.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    prev = None
    for rd in get_branch_rd_iter_skill(self, branch):
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


class Table(GridLayout):
    keys = ListProperty()
    character = ObjectProperty()
    bg_color_inactive = ListProperty()
    text_color_inactive = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    colkeys = ListProperty()
    skeliter = ObjectProperty()
    charatt = StringProperty()
    completedness = NumericProperty(0)

    @property
    def character_skel(self):
        getattr(self.character, self.charatt)

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

    def iter_skeleton(self):
        return self.skeliter(self, self.character.closet.branch)

    def on_completedness(self, i, v):
        if v == 5:
            self.completed()

    def on_text_color_inactive(self, *args):
        self.completedness += 1

    def on_bg_color_inactive(self, *args):
        self.completedness += 1

    def on_colkeys(self, *args):
        self.completedness += 1

    def on_font_name(self, *args):
        self.completedness += 1

    def on_font_size(self, *args):
        self.completedness += 1

    def completed(self):
        for key in self.colkeys:
            self.add_widget(Label(
                text=key,
                font_name=self.font_name,
                font_size=self.font_size,
                color=self.text_color_inactive))
        for rd in self.iter_skeleton():
            for key in self.colkeys:
                self.add_widget(Label(
                    text=rd[key],
                    font_name=self.font_name,
                    font_size=self.font_size,
                    color=self.text_color_inactive))

    def iterrows(self, branch=None, tick=None):
        closet = self.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        for rd in self.iter_skeleton(branch, tick):
            yield [rd[key] for key in self.colkeys]


class TableView(ItemView):
    colkeys = ListProperty()
    chartab = StringProperty()
    colkey_dict = {
        0: ["dimension", "thing", "location"],
        1: ["dimension", "place"],
        2: ["dimension", "origin", "destination"],
        3: ["stat", "value"],
        4: ["skill", "deck"]}

    chartab_dict = {
        0: 'thingdict',
        1: 'placedict',
        2: 'portaldict',
        3: 'statdict',
        4: 'skilldict'}

    iterskel_dict = {
        0: iter_skeleton_thing,
        1: iter_skeleton,
        2: iter_skeleton,
        3: iter_skeleton_stat,
        4: iter_skeleton_skill}
