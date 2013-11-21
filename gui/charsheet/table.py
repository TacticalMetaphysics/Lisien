from kivy.properties import (
    BooleanProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from itemview import ItemView
from util import placex, portex
from re import match


def character_bone(self, keys, character_skel):
    if keys[0] is None:
        return character_skel
    elif keys[1] is None:
        return character_skel[keys[0]]
    elif keys[2] is None:
        return character_skel[keys[0]][keys[1]]
    else:
        return character_skel[
            keys[0]][keys[1]][keys[2]]


def iter_skeleton(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    for rd in skel.iterrows():
        if (
                rd["branch"] == branch and
                rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or
                    rd["tick_to"] >= tick)):
            yield rd


def mk_iter_skeleton(keys, char, skel):
    def inner_iter_skeleton(branch=None, tick=None):
        for it in iter_skeleton(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton


def get_branch_rd_iter_thing(keys, character, branch):
    thingdict = character.thingdict
    if keys[0] is None:
        for dimension in thingdict:
            for thing in thingdict[dimension]:
                for rd in thingdict[dimension][thing][
                        branch].iterrows():
                    yield rd
    elif keys[1] is None:
        dimension = keys[0]
        for thing in thingdict[dimension]:
            for rd in thingdict[
                    dimension][thing][branch].iterrows():
                yield rd
    else:
        dimension = keys[0]
        thing = keys[1]
        for rd in thingdict[
                dimension][thing][branch].iterrows():
            yield rd


def iter_skeleton_thing(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    for rd in get_branch_rd_iter_thing(keys, char, branch):
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


def mk_iter_skeleton_thing(keys, char, skel):
    def inner_iter_skeleton_thing(branch=None, tick=None):
        for it in iter_skeleton_thing(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_thing


def get_branch_rd_iter_stat(keys, skel, branch):
    statdict = skel
    if keys[0] is None:
        for stat in statdict:
            for rd in statdict[
                    stat][branch].iterrows():
                yield rd
    else:
        stat = keys[0]
        for rd in statdict[
                stat][branch].iterrows():
            yield rd


def iter_skeleton_stat(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    prev = None
    for rd in get_branch_rd_iter_stat(keys, skel, branch):
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


def mk_iter_skeleton_stat(keys, char, skel):
    def inner_iter_skeleton_stat(branch=None, tick=None):
        for it in iter_skeleton_stat(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_stat


def get_branch_rd_iter_skill(keys, skel, branch):
    skilldict = skel
    if keys[0] is None:
        for skill in skilldict:
            for rd in skilldict[
                    skill][branch].iterrows():
                yield rd
    else:
        skill = keys[0]
        for rd in skilldict[
                skill][branch].iterrows():
            yield rd


def iter_skeleton_skill(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    covered = set()
    prev = None
    for rd in get_branch_rd_iter_skill(keys, skel, branch):
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


def mk_iter_skeleton_skill(keys, char, skel):
    def inner_iter_skeleton_skill(branch=None, tick=None):
        for it in iter_skeleton_skill(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_skill


class TableTextInput(TextInput):
    table = ObjectProperty()
    rd = ObjectProperty()
    key = StringProperty()

    def on_text_validate(self):
        ittyp = self.table.parent.item_type
        character = self.table.parent.character
        colkeys = self.table.parent.colkey_dict[ittyp][:-1]
        save = False
        if self.key in colkeys:
            pass
        elif ittyp == 0:
            skel = character.closet.skeleton["thing_location"][
                self.rd["dimension"]][self.rd["thing"]]
            m = match(placex, self.text)
            if m is not None:
                save = True
            else:
                m = match(portex, self.text)
                if m is not None:
                    save = True
        elif ittyp == 1:
            return
        elif ittyp == 2:
            return
        elif ittyp == 3:
            return
        elif ittyp == 4:
            skel = character.closet.skeleton["character_stats"][
                unicode(self.character)][self.rd["stat"]]
            # there'll be type checking eventually I guess
            save = True
        else:
            skel = character.closet.skeleton["character_skills"][
                unicode(self.character)][self.rd["skill"]]
            # and check that the Cause exists
            save = True
        branch = character.closet.branch
        tick = character.closet.tick
        if save:
            skel[branch][tick][self.key] = type(self.rd[self.key])(self.text)
        else:
            self.text = skel[branch][tick][self.key]


class TableHeader(BoxLayout):
    table = ObjectProperty()
    text = StringProperty()


class Table(GridLayout):
    bg_color_active = ListProperty()
    bg_color_inactive = ListProperty()
    fg_color_active = ListProperty()
    fg_color_inactive = ListProperty()
    text_color_active = ListProperty()
    text_color_inactive = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    completedness = NumericProperty(0)
    content_children = ListProperty()
    editing = BooleanProperty()
    colkeys = ListProperty()
    skel = ObjectProperty()
    iter_skeleton = ObjectProperty()

    def toggle_inputs(self, i, v):
        for child in self.children:
            if hasattr(child, 'disabled'):
                child.disabled = not v

    def on_completedness(self, i, v):
        if v == 4:
            self.completed()

    def on_text_color_inactive(self, *args):
        self.completedness += 1

    def on_bg_color_inactive(self, *args):
        self.completedness += 1

    def on_colkeys(self, *args):
        self.completedness += 1

    def on_iter_skeleton(self, *args):
        self.completedness += 1

    def completed(self):
        for key in self.colkeys:
            self.add_widget(TableHeader(
                table=self,
                text=key))

        for rd in self.iter_skeleton():
            for key in self.colkeys:
                child = TableTextInput(
                    table=self,
                    key=key,
                    rd=rd)
                print("Assigned rd {} to {}".format(rd, child))
                self.add_widget(child)

    def iterrows(self, branch=None, tick=None):
        closet = self.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        for rd in self.iter_skeleton(branch, tick):
            yield [rd[key] for key in self.colkeys]


class TableView(ItemView):
    character_skel = ObjectProperty()
    colkey_dict = {
        0: ["dimension", "thing", "location"],
        1: ["dimension", "place"],
        2: ["dimension", "origin", "destination"],
        3: ["stat", "value"],
        4: ["skill", "deck"]}

    chardictd = {
        0: 'thingdict',
        1: 'placedict',
        2: 'portaldict',
        3: 'statdict',
        4: 'skilldict'}

    iterskel_dict = {
        0: mk_iter_skeleton_thing,
        1: mk_iter_skeleton,
        2: mk_iter_skeleton,
        3: mk_iter_skeleton_stat,
        4: mk_iter_skeleton_skill}
