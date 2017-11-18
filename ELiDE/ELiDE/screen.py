# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com

"""The big layout that you view all of ELiDE through.

Handles touch, selection, and time control. Contains a board, a stat
grid, the time control panel, and the menu.

"""
from functools import partial
from importlib import import_module
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
    ListProperty,
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

    def set_turn(self, *args):
        turn = int(self.ids.turnfield.text)
        self.ids.turnfield.text = ''
        self.screen.app.turn = turn

    def set_tick(self, *args):
        tick = int(self.ids.tickfield.text)
        self.ids.tickfield.text = ''
        self.screen.app.tick = tick

    def _upd_branch_hint(self, *args):
        self.ids.branchfield.hint_text = self.screen.app.branch

    def _upd_turn_hint(self, *args):
        self.ids.turnfield.hint_text = str(self.screen.app.turn)

    def _upd_tick_hint(self, *args):
        self.ids.tickfield.hint_text = str(self.screen.app.tick)

    def on_screen(self, *args):
        if not all(field in self.ids for field in ('branchfield', 'turnfield', 'tickfield')):
            Clock.schedule_once(self.on_screen, 0)
            return
        self.ids.branchfield.hint_text = self.screen.app.branch
        self.ids.turnfield.hint_text = str(self.screen.app.turn)
        self.ids.tickfield.hint_text = str(self.screen.app.tick)
        self.screen.app.bind(
            branch=self._upd_branch_hint,
            turn=self._upd_turn_hint,
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
    dialog_todo = ListProperty([])
    rules_per_frame = BoundedNumericProperty(10, min=1)
    app = ObjectProperty()
    usermod = StringProperty('user')
    userpkg = StringProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.dialog_todo:
            self._advance_dialog()

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
            self.dummything
        ):
            if interceptor.collide_point(*touch.pos):
                interceptor.dispatch('on_touch_down', touch)
                self.boardview.keep_selection = True
                return True
        if self.dialoglayout.dispatch('on_touch_down', touch):
            return True
        return self.boardview.dispatch('on_touch_down', touch)

    def on_touch_up(self, touch):
        if self.timepanel.collide_point(*touch.pos):
            return self.timepanel.dispatch('on_touch_up', touch)
        elif self.charmenu.collide_point(*touch.pos):
            return self.charmenu.dispatch('on_touch_up', touch)
        elif self.statpanel.collide_point(*touch.pos):
            return self.statpanel.dispatch('on_touch_up', touch)
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

    def _update_from_next_turn(self, cmd, branch, turn, tick, ret):
        self.dialog_todo, chardiffs = ret
        self._update_from_chardiffs(cmd, branch, turn, tick, chardiffs)
        self._advance_dialog()

    def _update_from_chardiffs(self, cmd, branch, turn, tick, received, **kwargs):
        ret, diffs = received
        chardiff = diffs.get(self.boardview.board.character.name, {})
        for unwanted in (
            'character_rulebook',
            'avatar_rulebook',
            'character_thing_rulebook',
            'character_place_rulebook',
            'character_portal_rulebook'
        ):
            if unwanted in chardiff:
                del chardiff[unwanted]
        self.boardview.board.trigger_update_from_diff(chardiff)
        self.statpanel.statlist.mirror = dict(self.app.selected_remote)

    def _advance_dialog(self, *args):
        self.ids.dialoglayout.clear_widgets()
        if not self.dialog_todo:
            return
        self._update_dialog(self.dialog_todo.pop(0))

    def _update_dialog(self, diargs, **kwargs):
        if diargs is None:
            Logger.debug("Screen: null dialog")
            return
        if not hasattr(self, '_dia'):
            self._dia = Dialog()
        dia = self._dia
        # Simple text dialogs just tell the player something and let them click OK
        if isinstance(diargs, str):
            dia.message_kwargs = {'text': diargs}
            dia.menu_kwargs = {'options': [('OK', self._trigger_ok)]}
        # List dialogs are for when you need the player to make a choice and don't care much
        # about presentation
        elif isinstance(diargs, list):
            dia.message_kwargs = {'text': 'Select from the following:'}
            dia.menu_kwargs = {'options': list(map(self._munge_menu_option, diargs))}
        # For real control of the dialog, you need a pair of dicts --
        # the 0th describes the message shown to the player, the 1th
        # describes the menu below
        elif isinstance(diargs, tuple):
            if len(diargs) != 2:
                # TODO more informative error
                raise TypeError('Need a tuple of length 2')
            msgkwargs, mnukwargs = diargs
            if isinstance(msgkwargs, dict):
                dia.message_kwargs = msgkwargs
            elif isinstance(msgkwargs, str):
                dia.message_kwargs['text'] = msgkwargs
            else:
                raise TypeError("Message must be dict or str")
            if isinstance(mnukwargs, dict):
                mnukwargs['options'] = list(map(self._munge_menu_option, mnukwargs['options']))
                dia.menu_kwargs = mnukwargs
            elif isinstance(mnukwargs, list) or isinstance(mnukwargs, tuple):
                dia.menu_kwargs['options'] = list(map(self._munge_menu_option, mnukwargs))
            else:
                raise TypeError("Menu must be dict or list")
        else:
            raise TypeError("Don't know how to turn {} into a dialog".format(type(diargs)))
        self.ids.dialoglayout.add_widget(dia)

    def ok(self, *args, cb=None):
        self.ids.dialoglayout.clear_widgets()
        if cb:
            cb()
        self._advance_dialog()
        self.app.engine.universal['last_result_idx'] += 1

    def _trigger_ok(self, *args, cb=None):
        part = partial(self.ok, cb=cb)
        Clock.unschedule(part)
        Clock.schedule_once(part)

    def _lookup_func(self, funcname):
        if not hasattr(self, '_usermod'):
            self._usermod = import_module(self.usermod, self.userpkg)
        return getattr(self.usermod, funcname)

    def _munge_menu_option(self, option):
        name, func = option
        if func is None:
            return name, self._trigger_ok
        if callable(func):
            return name, partial(self._trigger_ok, cb=func)
        if isinstance(func, tuple):
            fun = func[0]
            if isinstance(fun, str):
                fun = self._lookup_func(fun)
            args = func[1]
            if len(func) == 3:
                kwargs = func[2]
                func = partial(fun, *args, **kwargs)
            else:
                func = partial(fun, *args)
        if isinstance(func, str):
            func = self._lookup_func(func)
        return name, partial(self._trigger_ok, cb=func)

    def play(self, *args):
        """If the 'play' button is pressed, advance a turn.

        If you want to disable this, set ``engine.universal['block'] = True``

        """
        if self.playbut.state == 'normal':
            return
        self.next_turn()

    def next_turn(self, *args):
        """Advance time by one turn, if it's not blocked.

        Block time by setting ``engine.universal['block'] = True``"""
        eng = self.app.engine
        if eng.universal.get('block'):
            Logger.info("MainScreen: next_turn blocked, delete universal['block'] to unblock")
            return
        if self.dialog_todo or eng.universal.get('last_result_idx', 0) < len(eng.universal.get('last_result', [])):
            Logger.info("MainScreen: not advancing time while there's a dialog")
            return
        eng.next_turn(
            cb=self._update_from_next_turn
        )


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
            text: 'Next turn'
            size_hint_y: 0.3
            on_press: root.screen.next_turn()
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
            text: 'Turn'
        MenuIntInput:
            id: turnfield
            setter: root.set_turn
            hint_text: str(root.screen.app.turn) if root.screen else ''
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
    FloatLayout:
        id: dialoglayout
        size_hint: None, None
        x: statpanel.right
        y: timepanel.top
        width: charmenu.x - statpanel.right
        height: root.height - timepanel.top
"""
)
