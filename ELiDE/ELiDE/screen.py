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
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.properties import (
    BooleanProperty,
    BoundedNumericProperty,
    DictProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty
)
from .charmenu import CharMenu
from .dialog import Dialog
from .util import dummynum, trigger

Factory.register('CharMenu', cls=CharMenu)


class KvLayout(FloatLayout):
    pass


class StatListPanel(BoxLayout):
    """A panel that displays a simple two-column grid showing the stats of
    the selected entity, defaulting to those of the character being
    viewed.

    Has a 'cfg' button on the bottom to open the StatWindow in which
    to add and delete stats, or to change the way they are displayed
    in the StatListPanel.

    """
    selection_name = StringProperty()
    button_text = StringProperty('cfg')
    cfgstatbut = ObjectProperty()
    statlist = ObjectProperty()
    engine = ObjectProperty()
    branch = StringProperty('trunk')
    tick = NumericProperty(0)
    remote = ObjectProperty()
    toggle_stat_cfg = ObjectProperty()

    def on_remote(self, *args):
        if hasattr(self.remote, 'name'):
            self.selection_name = str(self.remote.name)

    def set_value(self, k, v):
        if v is None:
            del self.remote[k]
        else:
            try:
                vv = self.engine.json_load(v)
            except (TypeError, ValueError):
                vv = v
            self.remote[k] = vv


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
        self.screen.app.branch = branch

    def set_tick(self, *args):
        tick = int(self.ids.tickfield.text)
        self.ids.tickfield.text = ''
        self.screen.app.tick = tick

    def next_tick(self, *args):
        self.screen.app.engine.next_tick(
            chars=[self.screen.app.character_name],
            cb=self.screen._update_from_next_tick
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
    manager = ObjectProperty()
    boards = DictProperty()
    boardview = ObjectProperty()
    charmenu = ObjectProperty()
    statlist = ObjectProperty()
    statpanel = ObjectProperty()
    timepanel = ObjectProperty()
    kv = StringProperty()
    use_kv = BooleanProperty()
    play_speed = NumericProperty()
    playbut = ObjectProperty()
    portaladdbut = ObjectProperty()
    portaldirbut = ObjectProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    dummies = ReferenceListProperty(dummyplace, dummything)
    dialoglayout = ObjectProperty()
    visible = BooleanProperty()
    _touch = ObjectProperty(None, allownone=True)
    rules_per_frame = BoundedNumericProperty(10, min=1)
    app = ObjectProperty()

    def on_statpanel(self, *args):
        if not self.app:
            Clock.schedule_once(self.on_statpanel, 0)
            return
        self.app.bind(selected_remote=self.statpanel.setter('remote'))

    def pull_visibility(self, *args):
        self.visible = self.manager.current == 'main'

    def on_manager(self, *args):
        self.pull_visibility()
        self.manager.bind(current=self.pull_visibility)

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
        if self.visible:
            touch.grab(self)
        for interceptor in (
            self.timepanel,
            self.charmenu,
            self.statpanel,
            self.dummyplace,
            self.dummything,
            self.dialoglayout
        ):
            if interceptor.collide_point(*touch.pos):
                interceptor.dispatch('on_touch_down', touch)
                self.boardview.keep_selection = True
                return True
        return self.boardview.dispatch('on_touch_down', touch)

    def on_touch_up(self, touch):
        if self.timepanel.collide_point(*touch.pos):
            self.timepanel.dispatch('on_touch_up', touch)
        elif self.charmenu.collide_point(*touch.pos):
            self.charmenu.dispatch('on_touch_up', touch)
        elif self.statpanel.collide_point(*touch.pos):
            self.statpanel.dispatch('on_touch_up', touch)
        return self.boardview.dispatch('on_touch_up', touch)

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

    def _update_from_next_tick(self, ret):
        self._update_dialog(ret[2])
        self._update_from_chardiff(ret[3])

    def _update_from_chardiff(self, chardiff, **kwargs):
        self.boardview.board.trigger_update_from_diff(
            chardiff.get(self.boardview.board.character.name, {})
        )
        self.statpanel.statlist.mirror = dict(self.app.selected_remote)
        self.app.pull_time()

    def _update_dialog(self, diargs, **kwargs):
        layout = self.ids.dialoglayout
        layout.clear_widgets()
        if diargs is None:
            Logger.debug("Screen: null dialog")
            return
        if not hasattr(self, '_dia'):
            self._dia = Dialog()
        dia = self._dia
        if isinstance(diargs, str):
            dia.message_kwargs = {'text': diargs}
            dia.menu_kwargs = {'options': [('OK', self.clear_dialog)]}
        elif isinstance(diargs, list):
            dia.message_kwargs = {'text': 'Select from the following:'}
            dia.menu_kwargs = {'options': diargs}
        elif isinstance(diargs, tuple):
            if len(diargs) != 2:
                # TODO more informative error
                raise TypeError('Need a tuple of length 2')
            dia.message_kwargs, dia.menu_kwargs = diargs
        else:
            raise TypeError("Don't know how to turn {} into a dialog".format(type(diargs)))
        self.ids.dialoglayout.add_widget(dia)

    @trigger
    def clear_dialog(self, *args):
        self.ids.dialoglayout.clear_widgets()

    def play(self, *args):
        """If the 'play' button is pressed, advance a tick."""
        if self.playbut.state == 'normal':
            return
        elif not hasattr(self, '_old_time'):
            self._old_time = (self.app.branch, self.app.tick)
            self.app.engine.next_tick(
                chars=[self.app.character_name],
                cb=lambda ret: self._update_from_chardiff(ret[3])
            )
        elif self._old_time == (self.app.branch, self.app.tick):
            return
        else:
            del self._old_time


Builder.load_string(
    """
#: import resource_find kivy.resources.resource_find
<StatListPanel>:
    orientation: 'vertical'
    cfgstatbut: cfgstatbut
    statlist: statlist
    id: statpanel
    Label:
        size_hint_y: 0.05
        text: root.selection_name
    StatListView:
        id: statlist
        size_hint_y: 0.95
        engine: root.engine
        branch: root.branch
        tick: root.tick
        remote: root.remote
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
    app: app
    dummyplace: charmenu.dummyplace
    dummything: charmenu.dummything
    boardview: boardview
    playbut: timepanel.playbut
    portaladdbut: charmenu.portaladdbut
    charmenu: charmenu
    statlist: statpanel.statlist
    statpanel: statpanel
    timepanel: timepanel
    dialoglayout: dialoglayout
    BoardView:
        id: boardview
        x: statpanel.right
        y: timepanel.top
        size_hint: (None, None)
        width: charmenu.x - statpanel.right
        height: root.height - timepanel.height
        screen: root
        engine: app.engine
        board: root.boards[app.character_name]
        branch: app.branch
        tick: app.tick
        adding_portal: charmenu.portaladdbut.state == 'down'
    StatListPanel:
        id: statpanel
        engine: app.engine
        branch: app.branch
        tick: app.tick
        toggle_stat_cfg: app.statcfg.toggle
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
    RelativeLayout:
        id: dialoglayout
        size_hint: None, None
        x: statpanel.right
        y: timepanel.top
        width: charmenu.x - statpanel.right
        height: root.height - timepanel.y
"""
)
