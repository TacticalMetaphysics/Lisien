# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com

"""The big layout that you view all of ELiDE through.

Handles touch, selection, and time control. Contains a board, a stat
grid, the time control panel, and the menu.

"""
from functools import partial
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    BoundedNumericProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty
)
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.logger import Logger

from .dummy import Dummy
from .spritebuilder import PawnConfigScreen, SpotConfigScreen
from .charmenu import CharMenu
from .board.arrow import ArrowWidget
from .util import dummynum, trigger


Factory.register('CharMenu', cls=CharMenu)


class KvLayout(FloatLayout):
    pass

class BoardView(ScrollView):
    """A ScrollView that contains the Board for the character being
    viewed.

    """
    selection = ObjectProperty(None, allownone=True)
    branch = StringProperty('master')
    tick = NumericProperty(0)
    character = ObjectProperty()
    board = ObjectProperty()


class StatListPanel(BoxLayout):
    """A panel that displays a simple two-column grid showing the stats of
    the selected entity, defaulting to those of the character being
    viewed.

    Has a 'cfg' button on the bottom to open the StatWindow in which
    to add and delete stats, or to change the way they are displayed
    in the StatListPanel.

    """
    branch = StringProperty()
    tick = NumericProperty()
    time = ReferenceListProperty(branch, tick)
    selected_remote = ObjectProperty()
    selection_name = StringProperty()
    button_text = StringProperty('cfg')
    set_value = ObjectProperty()
    cfgstatbut = ObjectProperty()
    toggle_stat_cfg = ObjectProperty()
    stat_list = ObjectProperty()


class TimePanel(BoxLayout):
    """A panel that lets you to start and stop the game, or browse through
    its history.

    There's a "play" button, which is toggleable. When toggled on, the
    simulation will continue to run until it's toggled off
    again. Below this is a "Next tick" button, which will simulate
    exactly one tick and stop. And there are two text fields in which
    you can manually enter a Branch and Tick to go to. Moving through
    time this way doesn't simulate anything--you'll only see what
    happened as a result of "play," "next tick," or some other input
    that's been made to call the ``advance`` method of the LiSE core.

    """
    next_tick = ObjectProperty()
    branch = StringProperty()
    branch_setter = ObjectProperty()
    tick = NumericProperty()
    tick_setter = ObjectProperty()
    playbut = ObjectProperty()
    time = ReferenceListProperty(branch, tick)

    def set_branch(self, *args):
        branch = self.ids.branchfield.text
        self.ids.branchfield.text = ''
        self.branch_setter(branch)

    def set_tick(self, *args):
        tick = int(self.ids.tickfield.text)
        self.ids.tickfield.text = ''
        self.tick_setter(tick)


class MainScreen(Screen):
    """A master layout that contains one board and some menus.

    This contains three elements: a scrollview (containing the board),
    a menu, and the time control panel. This class has some support methods
    for handling interactions with the menu and the character sheet,
    but if neither of those happen, the scrollview handles touches on its
    own.

    """
    character = ObjectProperty()
    character_name = StringProperty()
    board = ObjectProperty()
    kv = StringProperty()
    use_kv = BooleanProperty()
    play_speed = NumericProperty()
    playbut = ObjectProperty()
    portaladdbut = ObjectProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    dummies = ReferenceListProperty(dummyplace, dummything)
    visible = AliasProperty(
        lambda self: self.current == self,
        lambda self, v: None,
        bind=('current',)
    )
    engine = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    grabbing = BooleanProperty(True)
    reciprocal_portal = BooleanProperty(False)
    grabbed = ObjectProperty(None, allownone=True)
    selection = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    selected_remote = ObjectProperty()
    keep_selection = BooleanProperty(False)
    rules_per_frame = BoundedNumericProperty(10, min=1)
    current = StringProperty()
    branch = StringProperty()
    tick = NumericProperty()
    time = ReferenceListProperty(branch, tick)
    set_branch = ObjectProperty()
    set_tick = ObjectProperty()
    set_time = ObjectProperty()

    select_character = ObjectProperty()
    pawn_cfg = ObjectProperty()
    spot_cfg = ObjectProperty()
    stat_cfg = ObjectProperty()
    stat_list = ObjectProperty()
    rules = ObjectProperty()
    chars = ObjectProperty()
    strings = ObjectProperty()
    funcs = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(selection=self._trigger_reremote)
        self._trigger_reremote()

    def on_stat_list(self, *args):
        if not self.stat_cfg:
            Clock.schedule_once(self.on_stat_list, 0)
            return
        self.stat_cfg.stat_list = self.stat_list

    def on_play_speed(self, *args):
        """Change the interval at which ``self.play`` is called to match my
        current ``play_speed``.

        """
        Clock.unschedule(self.play)
        Clock.schedule_interval(self.play, 1.0 / self.play_speed)

    def on_character(self, *args):
        """Arrange to remake the customizable widgets when the character's
        stats change.

        Make them the first time, too, based on the current value of
        the relevant stat.

        """
        if not self.canvas:
            Clock.schedule_once(self.on_character, 0)
            return
        stats = (
            'kv',
            'font_name',
            'font_size',
            'halign',
            'valign',
            'line_height'
        )
        if hasattr(self, '_old_character'):
            for stat in stats:
                self._old_character.stat.unlisten(
                    stat='_'+stat, fun=getattr(self, '_set_'+stat)
                )
            if self.character is None:
                del self._old_character
                return
        else:
            self.remake_display()
            self.bind(kv=self._trigger_remake_display)
        self._old_character = self.character
        for stat in stats:
            funn = '_set_' + stat
            setattr(self, funn, partial(self._set_stat, stat))
            fun = getattr(self, funn)
            fun()
            self.character.stat.listener(
                stat='_'+stat, fun=fun
            )

    def _set_stat(self, stat, *args):
        """When one of the stats that controls my behavior changes on the
        character, update it on me as well.

        """
        if '_' + stat in self.character.stat:
            setattr(self, stat, self.character.stat['_'+stat])
        elif stat == 'kv':
            self.kv = ''

    def remake_display(self, *args):
        """Remake any affected widgets after a change in my ``kv``.

        """
        Builder.load_string(self.kv)
        if hasattr(self, '_kv_layout'):
            self.remove_widget(self._kv_layout)
            del self._kv_layout
        self._kv_layout = KvLayout()
        self.add_widget(self._kv_layout)
    _trigger_remake_display = trigger(remake_display)

    def reremote(self, *args):
        """Arrange to update my ``selected_remote`` with the currently
        selected entity when I can.

        """
        if self.character is None or 'statpanel' not in self.ids:
            Clock.schedule_once(self.reremote, 0)
            return
        try:
            self.selected_remote = self._get_selected_remote()
        except ValueError:
            return
    _trigger_reremote = trigger(reremote)

    def _get_selected_remote(self):
        """Return the currently selected entity, or ``self.character.stat`` if
        no entity is selected.

        """
        Logger.debug('ELiDELayout: getting remote...')
        if self.selection is None:
            return self.character.stat
        elif hasattr(self.selection, 'remote'):
            return self.selection.remote
        elif (
            hasattr(self.selection, 'portal') and
            self.selection.portal is not None
        ):
            return self.selection.portal
        else:
            raise ValueError(
                "Invalid selection: {}".format(repr(self.selection))
            )

    def on_touch_down(self, touch):
        """Dispatch the touch to the board, then its :class:`ScrollView`, then
        the dummies, then the menus.

        """
        if self.visible:
            touch.grab(self)
        for interceptor in (
            self.ids.timepanel,
            self.ids.charmenu,
            self.ids.statpanel,
            self.dummyplace,
            self.dummything
        ):
            if interceptor.collide_point(*touch.pos):
                interceptor.dispatch('on_touch_down', touch)
                self.keep_selection = True
                return True
        if not self.ids.boardview.collide_point(*touch.pos):
            return
        touch.push()
        touch.apply_transform_2d(self.ids.boardview.to_local)
        if self.selection:
            self.selection.hit = self.selection.collide_point(*touch.pos)
        pawns = list(self.board.pawns_at(*touch.pos))
        if pawns:
            self.selection_candidates = pawns
            if self.selection in self.selection_candidates:
                self.selection_candidates.remove(self.selection)
            return True
        spots = list(self.board.spots_at(*touch.pos))
        if spots:
            self.selection_candidates = spots
            if self.portaladdbut.state == 'down':
                self.origspot = self.selection_candidates.pop(0)
                self.protodest = Dummy(
                    name="protodest",
                    pos=touch.pos,
                    size=(0, 0)
                )
                self.board.add_widget(self.protodest)
                self.selection = self.protodest
                # why do I need this next?
                self.protodest.on_touch_down(touch)
                self.protoportal = ArrowWidget(
                    origin=self.origspot,
                    destination=self.protodest
                )
                self.board.add_widget(self.protoportal)
                if self.reciprocal_portal:
                    self.protoportal2 = ArrowWidget(
                        destination=self.origspot,
                        origin=self.protodest
                    )
                    self.board.add_widget(self.protoportal2)
            return True
        if self.selection_candidates == []:
            arrows = list(self.board.arrows_at(*touch.pos))
            if arrows:
                self.selection_candidates = arrows
                return True
        # the board did not handle the touch, so let the view scroll
        touch.pop()
        return self.ids.boardview.dispatch('on_touch_down', touch)

    def on_touch_move(self, touch):
        """If something's selected, it's on the board, so transform the touch
        to the boardview's space before dispatching it to the
        selection. Otherwise dispatch normally.

        """
        touch.push()
        touch.apply_transform_2d(self.ids.boardview.to_local)
        if self.selection in self.selection_candidates:
            self.selection_candidates.remove(self.selection)
        if self.selection and not self.selection_candidates:
            self.keep_selection = True
            self.selection.dispatch('on_touch_move', touch)
        elif self.selection_candidates:
            for cand in self.selection_candidates:
                if cand.collide_point(*touch.pos):
                    if hasattr(self.selection, 'selected'):
                        self.selection.selected = False
                    self.selection = cand
                    cand.hit = cand.selected = True
                    touch.grab(cand)
                    cand.dispatch('on_touch_move', touch)
        touch.pop()
        return super().on_touch_move(touch)

    def _portal_touch_up(self, touch):
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            try:
                # If the touch ended upon a spot, and there isn't
                # already a portal between the origin and this
                # destination, create one.
                destspot = next(self.board.spots_at(*touch.pos))
                orig = self.origspot.remote
                dest = destspot.remote
                if not (
                    orig.name in self.board.character.portal and
                    dest.name in self.board.character.portal[orig.name]
                ):
                    port = self.board.character.new_portal(
                        orig.name,
                        dest.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}->{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.board.arrowlayout.add_widget(
                        self.board.make_arrow(port)
                    )
                # And another in the opposite direction, if the user
                # asked for that.
                if (
                    hasattr(self, 'protoportal2') and not (
                        orig.name in self.board.character.preportal and
                        dest.name in
                        self.board.character.preportal[orig.name]
                    )
                ):
                    deport = self.board.character.new_portal(
                        dest.name,
                        orig.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}<-{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.board.arrowlayout.add_widget(
                        self.board.make_arrow(deport)
                    )
            except StopIteration:
                pass
            self.board.remove_widget(self.protoportal)
            if hasattr(self, 'protoportal2'):
                self.board.remove_widget(self.protoportal2)
                del self.protoportal2
            self.board.remove_widget(self.protodest)
            del self.protoportal
            del self.protodest
            touch.pop()

    def _selection_touch_up(self, touch):
        touch.push()
        touch.apply_transform_2d(self.ids.boardview.to_local)
        while self.selection_candidates:
            candidate = self.selection_candidates.pop(0)
            if candidate.collide_point(*touch.pos):
                if hasattr(self.selection, 'selected'):
                    self.selection.selected = False
                if hasattr(self.selection, '_start'):
                    self.selection.pos = self.selection._start
                    del self.selection._start
                self.selection = candidate
                self.selection.selected = True
                if (
                        hasattr(self.selection, 'thing') and not
                        hasattr(self.selection, '_start')
                ):
                    self.selection._start = tuple(self.selection.pos)
                touch.pop()
                self.keep_selection = True
                return True
        touch.pop()

    def on_touch_up(self, touch):
        """If there's a selection, dispatch the touch to it. Then, if there
        are selection candidates, select the next one that collides
        the touch. Otherwise, if something is selected, unselect
        it.

        """
        if hasattr(self, 'protodest'):
            # We're finishing the process of drawing an arrow to
            # create a new portal.
            self._portal_touch_up(touch)
            return True
        elif self.ids.timepanel.collide_point(*touch.pos):
            self.ids.timepanel.dispatch('on_touch_up', touch)
            return True
        elif self.ids.charmenu.collide_point(*touch.pos):
            self.ids.charmenu.dispatch('on_touch_up', touch)
            return True
        elif self.ids.statpanel.collide_point(*touch.pos):
            self.ids.statpanel.dispatch('on_touch_up', touch)
            return True
        elif hasattr(self.selection, 'on_touch_up'):
            self.selection.dispatch('on_touch_up', touch)
        # If we're not making a portal, and the touch hasn't landed
        # anywhere that would demand special treatment, but the
        # touch_down hit some selectable items, select the first of
        # those that also collides this touch_up.
        if self.selection_candidates:
            self._selection_touch_up(touch)
        if not self.keep_selection:
            if hasattr(self.selection, 'selected'):
                self.selection.selected = False
            self.selection = None
        self.keep_selection = False

    def on_dummies(self, *args):
        """Give the dummies numbers such that, when appended to their names,
        they give a unique name for the resulting new
        :class:`board.Pawn` or :class:`board.Spot`.

        """
        def renum_dummy(dummy, *args):
            dummy.num = dummynum(self.character, dummy.prefix) + 1

        if self.board is None or self.character is None:
            Clock.schedule_once(self.on_dummies, 0)
            return
        for dummy in self.dummies:
            if dummy is None or hasattr(dummy, '_numbered'):
                continue
            if dummy == self.dummything:
                self.ids.charmenu._pawn_config = self.pawn_cfg
            if dummy == self.dummyplace:
                self.ids.charmenu._spot_config = self.spot_cfg
            dummy.num = dummynum(self.character, dummy.prefix) + 1
            Logger.debug("MainScreen: dummy #{}".format(dummy.num))
            dummy.bind(prefix=partial(renum_dummy, dummy))
            dummy._numbered = True

    def spot_from_dummy(self, dummy):
        """Create a new :class:`board.Spot` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        position, and imagery of the provided dummy.

        """
        Logger.debug(
            "ELiDELayout: Making spot from dummy {} ({} #{})".format(
                dummy.name,
                dummy.prefix,
                dummy.num
            )
        )
        (x, y) = self.ids.boardview.to_local(*dummy.pos_up)
        x /= self.board.width
        y /= self.board.height
        self.board.spotlayout.add_widget(
            self.board.make_spot(
                self.board.character.new_place(
                    dummy.name,
                    _x=x,
                    _y=y,
                    _image_paths=list(dummy.paths)
                )
            )
        )
        dummy.num += 1

    def pawn_from_dummy(self, dummy):
        """Create a new :class:`board.Pawn` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        location, and imagery of the provided dummy.

        """
        dummy.pos = self.ids.boardview.to_local(*dummy.pos)
        for spot in self.board.spotlayout.children:
            if spot.collide_widget(dummy):
                whereat = spot
                break
        else:
            return
        whereat.add_widget(
            self.board.make_pawn(
                self.board.character.new_thing(
                    dummy.name,
                    whereat.place.name,
                    _image_paths=list(dummy.paths)
                )
            )
        )
        dummy.num += 1

    def arrow_from_wid(self, wid):
        """When the user has released touch after dragging to make an arrow,
        check whether they've drawn a valid one, and if so, make it.

        This doesn't handle touch events. It takes a widget as its
        argument: the one the user has been dragging to indicate where
        they want the arrow to go. Said widget ought to be invisible.

        """
        for spot in self.board.spotlayout.children:
            if spot.collide_widget(wid):
                whereto = spot
                break
        else:
            return
        self.board.arrowlayout.add_widget(
            self.board.make_arrow(
                self.board.character.new_portal(
                    self.grabbed.place.name,
                    whereto.place.name,
                    reciprocal=self.reciprocal_portal
                )
            )
        )

    def play(self, *args):
        """If the 'play' button is pressed, advance a tick."""
        if self.playbut.state == 'normal':
            return
        if not hasattr(self, '_old_time'):
            self._old_time = self.time
            self.engine.next_tick()
        elif self._old_time == self.time:
            return
        else:
            del self._old_time


Builder.load_string(
    """
#: import StiffScrollEffect ELiDE.kivygarden.stiffscroll.StiffScrollEffect
#: import resource_find kivy.resources.resource_find
#: import remote_setter ELiDE.util.remote_setter
<BoardView>:
    effect_cls: StiffScrollEffect
    board: board
    Board:
        id: board
        selection: root.selection
        branch: root.branch
        tick: root.tick
        character: root.character
<StatListPanel>:
    orientation: 'vertical'
    cfgstatbut: cfgstatbut
    stat_list: stat_list
    id: statpanel
    Label:
        size_hint_y: 0.05
        text: root.selection_name
    StatListView:
        id: stat_list
        size_hint_y: 0.95
        remote: root.selected_remote
        time: root.time
        set_value: root.set_value
    Button:
        id: cfgstatbut
        size_hint_y: 0.05
        text: root.button_text
        on_press: root.toggle_stat_cfg()
<TimePanel>:
    playbut: playbut
    BoxLayout:
        orientation: 'vertical'
        ToggleButton:
            id: playbut
            font_size: 40
            text: '>'
        Button:
            text: 'Next tick'
            size_hint_y: 0.3
            on_press: root.next_tick()
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Branch'
        MenuTextInput:
            setter: root.set_branch
            hint_text: root.branch
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Tick'
        MenuIntInput:
            id: tickfield
            setter: root.set_tick
            hint_text: str(root.tick)
<MainScreen>:
    name: 'main'
    character: self.engine.character[self.character_name] \
    if self.engine and self.character_name else None
    dummyplace: charmenu.dummyplace
    dummything: charmenu.dummything
    grabbing: self.grabbed is None
    board: boardview.board
    playbut: timepanel.playbut
    portaladdbut: charmenu.portaladdbut
    stat_list: statpanel.stat_list
    BoardView:
        id: boardview
        size_hint: (0.85, 0.9)
        pos_hint: {'x': 0.2, 'top': 1}
        selection: root.selection
        branch: root.branch
        tick: root.tick
        character: root.character
    StatListPanel:
        id: statpanel
        pos_hint: {'left': 0, 'top': 1}
        size_hint: (0.2, 0.9)
        time: root.time
        selected_remote: root.selected_remote
        selection_name: str(root.character_name) \
        if root.selection is None else str(root.selection.name)
        set_value: remote_setter(root.selected_remote)
        toggle_stat_cfg: charmenu.toggle_stat_cfg
    TimePanel:
        id: timepanel
        pos_hint: {'bot': 0}
        size_hint: (0.85, 0.1)
        branch: root.branch
        tick: root.tick
        branch_setter: root.set_branch
        tick_setter: root.set_tick
        next_tick: root.engine.next_tick
    CharMenu:
        id: charmenu
        pos_hint: {'right': 1, 'top': 1}
        size_hint: (0.1, 0.9)
        engine: root.engine
        board: root.board
        selection: root.selection
        character: root.character
        character_name: root.character_name
        selected_remote: root.selected_remote
        spot_from_dummy: root.spot_from_dummy
        pawn_from_dummy: root.pawn_from_dummy
        select_character: root.select_character
        pawn_cfg: root.pawn_cfg
        spot_cfg: root.spot_cfg
        stat_cfg: root.stat_cfg
        rules: root.rules
        chars: root.chars
        strings: root.strings
        funcs: root.funcs
"""
)
