from kivy.properties import (
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.stencilview import StencilView
from kivy.graphics import (
    Callback,
    Color,
    Rectangle)
from util import placex, portex, get_bone_during
from re import match
from itemlayout import ItemLayout


def character_bone(self, keys, character_skel):
    """Look up the keys in the skel.

It's supposed to be one of the skels that a Character keeps its members in."""
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
    """Iterate over the bones in only the subset of the skeleton that
pertains to the keys.

You need to pass a character to this thing, even though it only uses
the closet in that character, because the other iterators need the
whole character, and I need a consistent API here.

    """
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    for bone in skel.iterbones():
        if (
                bone.branch == branch and
                bone.tick_from <= tick and (
                    bone.tick_to is None or
                    bone.tick_to >= tick)):
            yield rd


def mk_iter_skeleton(keys, char, skel):
    """Make a function that makes a boneiter over the skeleton for the
item specified by the keys.

    """
    def inner_iter_skeleton(branch=None, tick=None):
        for it in iter_skeleton(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton


def get_branch_bone_iter_thing(keys, character, branch, tick):
    """Generate strings representing the locations of all Things in the
character that match the keys."""
    thingdict = character.thingdict
    skeleton = character.closet.skeleton
    bone_during_thing = lambda dimension, thing: get_bone_during(
        skeleton["thing_location"][dimension][thing],
        branch, tick)

    if keys[0] is None:
        for dimension in thingdict:
            for thing in thingdict[dimension]:
                yield bone_during_thing(dimension, thing)
    elif keys[1] is None:
        dimension = keys[0]
        for thing in thingdict[dimension]:
            yield bone_during_thing(dimension, thing)

    else:
        dimension = keys[0]
        thing = keys[1]
        yield bone_during_thing(dimension, thing)


def iter_skeleton_thing(keys, char, skel, branch=None, tick=None):
    """Get the locations of the things in the character matching the keys,
and yield dicts representing rows in a table of locations."""
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    for bone in get_branch_bone_iter_thing(keys, char, branch, tick):
        yield bone


def mk_iter_skeleton_thing(keys, char, skel):
    """Make a generator for the bones of the thing specified."""
    def inner_iter_skeleton_thing(branch=None, tick=None):
        for it in iter_skeleton_thing(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_thing


def get_branch_bone_iter_stat(keys, skel, branch, tick):
    """Generate the bones for the generator of values for the stat
identified in the keys."""
    statdict = skel
    if keys[0] is None:
        for stat in statdict:
            yield get_bone_during(statdict[stat], branch, tick)
    else:
        stat = keys[0]
        yield get_bone_during(statdict[stat], branch, tick)


def iter_skeleton_stat(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    for bone in get_branch_bone_iter_stat(keys, skel, branch, tick):
        yield {"stat": bone["stat"],
               "value": bone["value"]}


def mk_iter_skeleton_stat(keys, char, skel):
    def inner_iter_skeleton_stat(branch=None, tick=None):
        for it in iter_skeleton_stat(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_stat


def get_branch_bone_iter_skill(keys, skel, branch, tick):
    skilldict = skel
    if keys[0] is None:
        for skill in skilldict:
            yield get_bone_during(skilldict[skill], branch, tick)
    else:
        skill = keys[0]
        yield get_bone_during(skilldict[skill], branch, tick)


def iter_skeleton_skill(keys, char, skel, branch=None, tick=None):
    closet = char.closet
    if branch is None:
        branch = closet.branch
    if tick is None:
        tick = closet.tick
    for bone in get_branch_bone_iter_skill(keys, skel, branch, tick):
        yield {"skill": bone["skill"]}


def mk_iter_skeleton_skill(keys, char, skel):
    def inner_iter_skeleton_skill(branch=None, tick=None):
        for it in iter_skeleton_skill(keys, char, skel, branch, tick):
            yield it
    return inner_iter_skeleton_skill


class TableTextInput(TextInput):
    table = ObjectProperty()
    bone = ObjectProperty(None)
    key = StringProperty()
    finality = NumericProperty(0)

    def on_bone(self, i, v):
        if v is None:
            return
        if self.bone is not None and self.bone_listener in self.bone.listeners:
            self.bone.listeners.remove(self.bone_listener)
        v.listeners.append(self.bone_listener)

    def bone_listener(self, skel, k, v):
        if k == u"location":
            self.text = unicode(v)

    def time_listener(self, closet, branch, tick):
        ittyp = self.table.parent.item_type
        character = self.table.parent.character
        colkeys = self.table.parent.colkey_dict[ittyp][:-1]
        if self.key in colkeys:
            pass
        elif ittyp == 0:
            skel = closet.skeleton[u"thing_location"][
                self.bone["dimension"]][self.bone["thing"]][
                branch]
            # Sometimes I get called during the creation of a new
            # branch, before the new branch has anything in it. Wait
            # it out.
            if len(skel) == 0:
                return
            tick_from = skel.key_or_key_before(tick)
            self.text = skel[tick_from]["location"]
        elif ittyp == 1:
            pass
        elif ittyp == 2:
            pass
        elif ittyp == 3:
            pass
        elif ittyp == 4:
            self.text = unicode(
                closet.skeleton["character_stats"][
                    unicode(character)][self.bone["stat"]][
                    branch][tick])
        else:
            self.text = unicode(
                closet.skeleton["character_skills"][
                    unicode(character)][self.bone["stat"]][
                    branch][tick])

    def on_text_validate(self):
        ittyp = self.table.parent.item_type
        character = self.table.parent.character
        colkeys = self.table.parent.colkey_dict[ittyp][:-1]
        save = False
        if self.key in colkeys:
            pass
        elif ittyp == 0:
            skel = character.closet.skeleton["thing_location"][
                self.bone["dimension"]][self.bone["thing"]]
            dimension = character.closet.get_dimension(self.bone["dimension"])
            thing = character.closet.get_thing(
                self.bone["dimension"], self.bone["thing"])
            if "->" in self.text:
                (orign, destn) = self.text.split("->")
                ovid = dimension.graph.vs.find(name=orign)
                dvid = dimension.graph.vs.find(name=destn)
                try:
                    eid = dimension.graph.get_eid(ovid, dvid)
                    portal = dimension.graph.es[eid]["portal"]
                    thing.set_location(portal)
                    save = True
                except Exception as e:
                    print e
                    pass
            else:
                try:
                    v = dimension.graph.vs.find(name=self.text)
                    thing.set_location(v["place"])
                    save = True
                except ValueError:
                    pass
        elif ittyp == 1:
            return
        elif ittyp == 2:
            return
        elif ittyp == 3:
            return
        elif ittyp == 4:
            skel = character.closet.skeleton["character_stats"][
                unicode(character)][self.bone["stat"]]
            # there'll be type checking eventually I guess
            save = True
        else:
            skel = character.closet.skeleton["character_skills"][
                unicode(character)][self.bone["skill"]]
            # and check that the Cause exists
            save = True
        if not save:
            branch = character.closet.branch
            tick = skel[branch].key_or_key_before(character.closet.tick)
            self.text = skel[branch][tick][self.key]

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            touch.grab(self)
            super(TableTextInput, self).on_touch_down(touch)
            return True

    def edbut_listener(self, i, v):
        if v == 'normal':
            self.focus = False


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
    headers = ListProperty()
    content_children = ListProperty()
    colkeys = ListProperty()
    skel = ObjectProperty(None)
    iter_skeleton = ObjectProperty()
    edbut = ObjectProperty()
    xmov = NumericProperty()

    def on_completedness(self, i, v):
        if v == 6:
            self.complete()

    def on_text_color_inactive(self, *args):
        self.completedness += 1

    def on_bg_color_inactive(self, *args):
        self.completedness += 1

    def on_colkeys(self, *args):
        self.completedness += 1

    def on_iter_skeleton(self, *args):
        self.completedness += 1

    def on_parent(self, *args):
        self.completedness += 1

    def on_character_skel(self, *args):
        self.completedness += 1

    def capitate(self):
        for key in self.colkeys:
            head = TableHeader(
                table=self,
                text=key)
            self.headers.append(head)
            self.add_widget(head)

    def corpitate(self):
        for bone in self.iter_skeleton():
            for key in self.colkeys:
                child = TableTextInput(
                    table=self,
                    key=key,
                    bone=bone)
                self.add_widget(child)
                self.edbut.extra_listeners.append(child.edbut_listener)
                self.parent.character.closet.time_listeners.append(
                    child.time_listener)

    def complete(self):
        self.skel = character_bone(self, self.parent.keys, self.character_skel)
        self.capitate()
        self.corpitate()

    def iterbones(self, branch=None, tick=None):
        closet = self.character.closet
        if branch is None:
            branch = closet.branch
        if tick is None:
            tick = closet.tick
        for bone in self.iter_skeleton(branch, tick):
            yield [bone[key] for key in self.colkeys]

    def on_touch_down(self, touch):
        for child in self.children:
            if not child.disabled and child.on_touch_down(touch):
                return True
        touch.grab(self)
        return True

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return
        self.xmov += touch.dx
        return True

    def on_touch_up(self, touch):
        self.xmov = 0
        return super(Table, self).on_touch_up(touch)


class TableLayout(ItemLayout, StencilView):
    character_skel = ObjectProperty()
    edbut = ObjectProperty()
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

    def __init__(self, **kwargs):
        super(TableLayout, self).__init__(**kwargs)
