# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com

"""The big layout that you view all of ELiDE through.

Handles touch, selection, and time control. Contains a board, a stat
grid, the time control panel, and the menu.

"""
from functools import partial

from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.logger import Logger

from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    BoundedNumericProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty
)
from .dummy import Dummy
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
    screen = ObjectProperty()
    board = ObjectProperty()


class StatListPanel(BoxLayout):
    """A panel that displays a simple two-column grid showing the stats of
    the selected entity, defaulting to those of the character being
    viewed.

    Has a 'cfg' button on the bottom to open the StatWindow in which
    to add and delete stats, or to change the way they are displayed
    in the StatListPanel.

    """
    app = ObjectProperty()
    selection_name = StringProperty()
    button_text = StringProperty('cfg')
    cfgstatbut = ObjectProperty()
    stat_list = ObjectProperty()

    def on_app(self, *args):
        self.app.bind(selected_remote=self.pull_selection_name)
        self.pull_selection_name()

    def pull_selection_name(self, *args):
        if hasattr(self.app.selected_remote, 'name'):
            self.selection_name = str(self.app.selected_remote.name)

    def set_value(self, k, v):
        if v is None:
            del self.app.selected_remote[k]
        else:
            try:
                vv = self.app.engine.json_load(v)
            except (TypeError, ValueError):
                vv = v
            self.app.selected_remote[k] = vv

    def toggle_stat_cfg(self, *args):
        self.app.statcfg.toggle()


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
    screen = ObjectProperty()

    def set_branch(self, *args):
        branch = self.ids.branchfield.text
        self.ids.branchfield.text = ''
        self.screen.app.set_branch(branch)

    def set_tick(self, *args):
        tick = int(self.ids.tickfield.text)
        self.ids.tickfield.text = ''
        self.screen.app.set_tick(tick)

    def next_tick(self, *args):
        self.screen.app.engine.next_tick(
            char=self.screen.app.character_name,
            cb=self.screen._update_from_chardiff
        )

    def _upd_branch_hint(self, *args):
        self.ids.branchfield.hint_text = self.screen.app.branch

    def _upd_tick_hint(self, *args):
        self.ids.tickfield.hint_text = str(self.screen.app.tick)

    def on_screen(self, *args):
        if 'branchfield' not in self.ids or 'tickfield' not in self.ids:
            Clock.schedule_once(self.on_screen, 0)
            return
        self.ids.branchfield.hint_text = self.screen.app.branch
        self.ids.tickfield.hint_text = str(self.screen.app.tick)
        self.screen.app.bind(
            branch=self._upd_branch_hint,
            tick=self._upd_tick_hint
        )


class MainScreen(Screen):
    """A master layout that contains one board and some menus.

    This contains three elements: a scrollview (containing the board),
    a menu, and the time control panel. This class has some support methods
    for handling interactions with the menu and the character sheet,
    but if neither of those happen, the scrollview handles touches on its
    own.

    """

    board = ObjectProperty()
    kv = StringProperty()
    use_kv = BooleanProperty()
    play_speed = NumericProperty()
    playbut = ObjectProperty()
    portaladdbut = ObjectProperty()
    portaldirbut = ObjectProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    dummies = ReferenceListProperty(dummyplace, dummything)
    visible = BooleanProperty()
    _touch = ObjectProperty(None, allownone=True)
    grabbing = BooleanProperty(True)
    reciprocal_portal = BooleanProperty(False)
    grabbed = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    keep_selection = BooleanProperty(False)
    rules_per_frame = BoundedNumericProperty(10, min=1)
    app = ObjectProperty()

    def pull_visibility(self, *args):
        self.visible = self.app.manager.current == 'main'

    def on_app(self, *args):
        self.pull_visibility()
        self.app.manager.bind(current=self.pull_visibility)

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
        if hasattr(self, '_old_character'):
            self._old_character.stat.unlisten(
                key='_kv', fun=self._pull_kv
            )
        else:
            self.bind(kv=self._trigger_remake_display)
        self._old_character = self.character
        if '_kv' in self.character.stat:
            self.kv = self.character.stat['_kv']
        self.board.trigger_update()

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
        if self.app.selection:
            self.app.selection.hit = self.app.selection.collide_point(*touch.pos)
            if self.app.selection.hit:
                touch.grab(self.app.selection)
        pawns = list(self.board.pawns_at(*touch.pos))
        if pawns:
            self.selection_candidates = pawns
            if self.app.selection in self.selection_candidates:
                self.selection_candidates.remove(self.app.selection)
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
                self.app.selection = self.protodest
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
        if self.app.selection in self.selection_candidates:
            self.selection_candidates.remove(self.app.selection)
        if self.app.selection and not self.selection_candidates:
            self.keep_selection = True
            self.app.selection.dispatch('on_touch_move', touch)
        elif self.selection_candidates:
            for cand in self.selection_candidates:
                if cand.collide_point(*touch.pos):
                    if hasattr(self.app.selection, 'selected'):
                        self.app.selection.selected = False
                    if hasattr(self.app.selection, 'hit'):
                        self.app.selection.hit = False
                    self.app.selection = cand
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
                if hasattr(self.app.selection, 'selected'):
                    self.app.selection.selected = False
                if hasattr(self.app.selection, '_start'):
                    self.app.selection.pos = self.app.selection._start
                    del self.app.selection._start
                self.app.selection = candidate
                self.app.selection.selected = True
                if (
                        hasattr(self.app.selection, 'thing') and not
                        hasattr(self.app.selection, '_start')
                ):
                    self.app.selection._start = tuple(self.app.selection.pos)
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
        elif hasattr(self.app.selection, 'on_touch_up'):
            self.app.selection.dispatch('on_touch_up', touch)
        # If we're not making a portal, and the touch hasn't landed
        # anywhere that would demand special treatment, but the
        # touch_down hit some selectable items, select the first of
        # those that also collides this touch_up.
        if self.selection_candidates:
            self._selection_touch_up(touch)
        if not self.keep_selection:
            if hasattr(self.app.selection, 'selected'):
                self.app.selection.selected = False
            self.app.selection = None
        self.keep_selection = False
        touch.ungrab(self)
        return True

    def on_dummies(self, *args):
        """Give the dummies numbers such that, when appended to their names,
        they give a unique name for the resulting new
        :class:`board.Pawn` or :class:`board.Spot`.

        """
        def renum_dummy(dummy, *args):
            dummy.num = dummynum(self.app.character, dummy.prefix) + 1

        for dummy in self.dummies:
            if dummy is None or hasattr(dummy, '_numbered'):
                continue
            if dummy == self.dummything:
                self.app.pawncfg.bind(imgpaths=self._propagate_thing_paths)
            if dummy == self.dummyplace:
                self.app.spotcfg.bind(imgpaths=self._propagate_place_paths)
            dummy.num = dummynum(self.app.character, dummy.prefix) + 1
            Logger.debug("MainScreen: dummy #{}".format(dummy.num))
            dummy.bind(prefix=partial(renum_dummy, dummy))
            dummy._numbered = True

    def _propagate_thing_paths(self, *args):
        # horrible hack
        self.dummything.paths = self.app.pawncfg.imgpaths

    def _propagate_place_paths(self, *args):
        # horrible hack
        self.dummyplace.paths = self.app.spotcfg.imgpaths

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

    def _update_from_chardiff(self, char, chardiff, **kwargs):
        Logger.debug("{}: updating from diff {}".format(
            char, chardiff
        ))
        self.board.trigger_update_from_diff(chardiff)
        self.ids.statpanel.stat_list.mirror = dict(self.app.selected_remote)
        self.app.pull_time()

    def play(self, *args):
        """If the 'play' button is pressed, advance a tick."""
        if self.playbut.state == 'normal':
            return
        elif not hasattr(self, '_old_time'):
            self._old_time = tuple(self.app.time)
            self.app.engine.next_tick(
                char=self.app.character_name,
                cb=self._update_from_chardiff
            )
        elif self._old_time == self.app.time:
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
        screen: root.screen
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
        app: root.app
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
            id: branchfield
            setter: root.set_branch
            hint_text: root.screen.app.branch if root.screen else ''
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Tick'
        MenuIntInput:
            id: tickfield
            setter: root.set_tick
            hint_text: str(root.screen.app.tick) if root.screen else ''
<MainScreen>:
    name: 'main'
    dummyplace: charmenu.dummyplace
    dummything: charmenu.dummything
    grabbing: self.grabbed is None
    board: boardview.board
    playbut: timepanel.playbut
    portaladdbut: charmenu.portaladdbut
    stat_list: statpanel.stat_list
    BoardView:
        id: boardview
        x: statpanel.right
        y: timepanel.top
        size_hint: (None, None)
        width: charmenu.x - statpanel.right
        height: root.height - timepanel.height
        screen: root
    StatListPanel:
        id: statpanel
        app: root.app
        pos_hint: {'left': 0, 'top': 1}
        size_hint: (0.2, 0.9)
    TimePanel:
        id: timepanel
        screen: root
        pos_hint: {'bot': 0}
        size_hint: (0.85, 0.1)
    CharMenu:
        id: charmenu
        screen: root
        pos_hint: {'right': 1, 'top': 1}
        size_hint: (0.1, 0.9)
"""
)
