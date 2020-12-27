# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The big layout that you view all of ELiDE through.

Handles touch, selection, and time control. Contains a graph, a stat
grid, the time control panel, and the menu.

"""
from functools import partial
from ast import literal_eval

from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.modalview import ModalView
from kivy.uix.scrollview import ScrollView
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
from .graph.board import GraphBoardView
from .grid.board import GridBoardView
from .calendar import Agenda
from .util import dummynum, trigger

Factory.register('CharMenu', cls=CharMenu)


class KvLayout(FloatLayout):
    pass


class StatListPanel(BoxLayout):
    """A panel that displays a simple two-column grid showing the stats of
    the selected entity, defaulting to those of the character being
    viewed.

    Has a button on the bottom to open the StatWindow in which
    to add and delete stats, or to change the way they are displayed
    in the StatListPanel.

    """
    selection_name = StringProperty()
    button_text = StringProperty('Configure stats')
    cfgstatbut = ObjectProperty()
    statlist = ObjectProperty()
    engine = ObjectProperty()
    proxy = ObjectProperty()
    toggle_stat_cfg = ObjectProperty()
    toggle_gridview = ObjectProperty()
    toggle_calendar = ObjectProperty()

    def on_proxy(self, *args):
        if hasattr(self.proxy, 'name'):
            self.selection_name = str(self.proxy.name)

    def set_value(self, k, v):
        if v is None:
            del self.proxy[k]
        else:
            try:
                vv = literal_eval(v)
            except (TypeError, ValueError):
                vv = v
            self.proxy[k] = vv


class SimulateButton(ToggleButton):
    play_arrow_left = NumericProperty()
    play_arrow_right = NumericProperty()
    play_arrow_points = ListProperty([0] * 6)
    graphics_top = NumericProperty()
    graphics_bot = NumericProperty()
    graphics_center_y = NumericProperty()


class OneTurnButton(Button):
    graphics_top = NumericProperty()
    graphics_bot = NumericProperty()
    graphics_center_y = NumericProperty()
    step_arrow_left = NumericProperty()
    step_center_x = NumericProperty()
    step_bar_right = NumericProperty()
    step_arrow_points = ListProperty([0] * 6)
    step_rect_points = ListProperty([0] * 8)


class TimePanel(BoxLayout):
    """A panel that lets you to start and stop the game, or browse through
    its history.

    There's a "simulate" button, which is toggleable. When toggled on, the
    simulation will continue to run until it's toggled off
    again. Next to this is a "1 turn" button, which will simulate
    exactly one turn and stop. And there are two text fields in which
    you can manually enter a Branch and Tick to go to. Moving through
    time this way doesn't simulate anything--you'll only see what
    happened as a result of "simulate," "1 turn," or some other way
    the LiSE rules engine has been made to run.

    """
    screen = ObjectProperty()
    buttons_font_size = NumericProperty(18)
    disable_one_turn = BooleanProperty()

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
    """A master layout that contains one graph and some menus.

    This contains three elements: a scrollview (containing the graph),
    a menu, and the time control panel. This class has some support methods
    for handling interactions with the menu and the character sheet,
    but if neither of those happen, the scrollview handles touches on its
    own.

    """
    manager = ObjectProperty()
    graphboards = DictProperty()
    gridboards = DictProperty()
    boardview = ObjectProperty()
    mainview = ObjectProperty()
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
    tmp_block = BooleanProperty(False)

    def on_mainview(self, *args):
        if None in (self.statpanel, self.charmenu, self.app) or None in (self.app.character_name, self.charmenu.portaladdbut):
            Clock.schedule_once(self.on_mainview, 0)
            return
        self.boardview = GraphBoardView(
            scale_min=0.2,
            scale_max=4.0,
            size=self.mainview.size,
            pos=self.mainview.pos,
            board=self.graphboards[self.app.character_name],
            adding_portal=self.charmenu.portaladdbut.state == 'down'
        )
        def update_adding_portal(*args):
            self.boardview.adding_portal = self.charmenu.portaladdbut.state == 'down'
        def update_board(*args):
            self.boardview.board = self.graphboards[self.app.character_name]
        self.mainview.bind(
            size=self.boardview.setter('size'),
            pos=self.boardview.setter('pos')
        )
        self.charmenu.portaladdbut.bind(state=update_adding_portal)
        self.app.bind(character_name=update_board)
        self.calendar = Agenda(
            update_mode='present'
        )
        self.calendar_view = ScrollView(
            size=self.mainview.size,
            pos=self.mainview.pos
        )
        self.gridview = GridBoardView(
            scale_min=0.2,
            scale_max=4.0,
            size=self.mainview.size,
            pos=self.mainview.pos,
            board=self.gridboards[self.app.character_name]
        )
        self.mainview.bind(
            size=self.calendar_view.setter('size'),
            pos=self.calendar_view.setter('pos'),
        )
        self.mainview.bind(
            size=self.gridview.setter('size'),
            pos=self.gridview.setter('pos')
        )
        self.calendar_view.add_widget(self.calendar)
        self.mainview.add_widget(self.boardview)

    def on_statpanel(self, *args):
        if not self.app:
            Clock.schedule_once(self.on_statpanel, 0)
            return
        self._update_statlist()
        self.app.bind(
            selected_proxy=self._update_statlist,
            branch=self._update_statlist,
            turn=self._update_statlist,
            tick=self._update_statlist
        )

    @trigger
    def _update_statlist(self, *args):
        if not self.app.selected_proxy:
            self._update_statlist()
            return
        self.app.update_calendar(self.statpanel.statlist, past_turns=0, future_turns=0)

    def pull_visibility(self, *args):
        self.visible = self.manager.current == 'main'

    def on_manager(self, *args):
        self.pull_visibility()
        self.manager.bind(current=self.pull_visibility)

    def on_play_speed(self, *args):
        """Change the interval at which ``self.play`` is called to match my
        current ``play_speed``.

        """
        if hasattr(self, '_play_scheduled'):
            Clock.unschedule(self._play_scheduled)
        self._play_scheduled = Clock.schedule_interval(self.play, 1.0 / self.play_speed)

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
                self.boardview.keep_selection = \
                    self.gridview.keep_selection = True
                return True
        if self.dialoglayout.dispatch('on_touch_down', touch):
            return True
        return self.mainview.dispatch('on_touch_down', touch)

    def on_touch_up(self, touch):
        if self.timepanel.collide_point(*touch.pos):
            return self.timepanel.dispatch('on_touch_up', touch)
        elif self.charmenu.collide_point(*touch.pos):
            return self.charmenu.dispatch('on_touch_up', touch)
        elif self.statpanel.collide_point(*touch.pos):
            return self.statpanel.dispatch('on_touch_up', touch)
        return self.mainview.dispatch('on_touch_up', touch)

    def on_dummies(self, *args):
        """Give the dummies numbers such that, when appended to their names,
        they give a unique name for the resulting new
        :class:`graph.Pawn` or :class:`graph.Spot`.

        """
        if not self.app.character:
            Clock.schedule_once(self.on_dummies, 0)
            return
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

    def _update_from_time_travel(self, command, branch, turn, tick, result, **kwargs):
        self._update_from_delta(command, branch, turn, tick, result[-1])

    def _update_from_delta(self, cmd, branch, turn, tick, delta, **kwargs):
        self.app.branch = branch
        self.app.turn = turn
        self.app.tick = tick
        chardelta = delta.get(self.boardview.board.character.name, {})
        for unwanted in (
            'character_rulebook',
            'avatar_rulebook',
            'character_thing_rulebook',
            'character_place_rulebook',
            'character_portal_rulebook'
        ):
            if unwanted in chardelta:
                del chardelta[unwanted]
        self.boardview.board.trigger_update_from_delta(chardelta)
        self.gridview.board.trigger_update_from_delta(chardelta)
        self.statpanel.statlist.mirror = dict(self.app.selected_proxy)

    def play(self, *args):
        """If the 'play' button is pressed, advance a turn.

        If you want to disable this, set ``engine.universal['block'] = True``

        """
        if self.playbut.state == 'normal':
            return
        self.next_turn()

    def _update_from_next_turn(self, command, branch, turn, tick, result):
        todo, deltas = result
        if isinstance(todo, list):
            self.dialoglayout.todo = todo
            self.dialoglayout.idx = 0
        self._update_from_delta(command, branch, turn, tick, deltas)
        self.dialoglayout.advance_dialog()
        self.app.bind(
            branch=self.app._push_time,
            turn=self.app._push_time,
            tick=self.app._push_time
        )
        self.tmp_block = False

    def next_turn(self, *args):
        """Advance time by one turn, if it's not blocked.

        Block time by setting ``engine.universal['block'] = True``"""
        if self.tmp_block:
            return
        eng = self.app.engine
        dial = self.dialoglayout
        if eng.universal.get('block'):
            Logger.info("MainScreen: next_turn blocked, delete universal['block'] to unblock")
            return
        if dial.idx < len(dial.todo):
            Logger.info("MainScreen: not advancing time while there's a dialog")
            return
        self.tmp_block = True
        self.app.unbind(
            branch=self.app._push_time,
            turn=self.app._push_time,
            tick=self.app._push_time
        )
        eng.next_turn(cb=self._update_from_next_turn)

    def switch_to_calendar(self, *args):
        self.app.update_calendar(self.calendar)
        self.mainview.clear_widgets()
        self.mainview.add_widget(self.calendar_view)

    def switch_to_boardview(self, *args):
        self.mainview.clear_widgets()
        self.app.engine.handle('apply_choices', choices=[self.calendar.get_track()])
        self.mainview.add_widget(self.boardview)

    def toggle_gridview(self, *args):
        if self.gridview in self.mainview.children:
            self.mainview.clear_widgets()
            self.mainview.add_widget(self.boardview)
        else:
            self.mainview.clear_widgets()
            self.mainview.add_widget(self.gridview)

    def toggle_calendar(self, *args):
        # TODO decide how to handle switching between >2 view types
        if self.boardview in self.mainview.children:
            self.switch_to_calendar()
        else:
            self.switch_to_boardview()


Builder.load_string(
    """
#: import resource_find kivy.resources.resource_find
<StatListPanel>:
    orientation: 'vertical'
    cfgstatbut: cfgstatbut
    statlist: statlist
    id: statpanel
    proxy: app.selected_proxy
    Label:
        size_hint_y: 0.05
        text: root.selection_name
        bold: True
    Calendar:
        id: statlist
        size_hint_y: 0.8
        entity: root.proxy
        update_mode: 'present'
    Button:
        id: gridviewbut
        size_hint_y: 0.05
        text: 'toggle grid'
        on_release: root.toggle_gridview()
    Button:
        id: cfgstatbut
        size_hint_y: 0.05
        text: root.button_text
        on_release: root.toggle_stat_cfg()
<SimulateButton>:
    graphics_top: self.y + self.font_size + (self.height - self.font_size) * (3/4)
    graphics_bot: self.y + self.font_size + 3
    graphics_center_y: self.graphics_bot + (self.graphics_top - self.graphics_bot) / 2
    play_arrow_left: self.center_x - self.width / 6
    play_arrow_right: self.center_x + self.width / 6
    play_arrow_points: self.play_arrow_left, self.graphics_top, self.play_arrow_right, self.graphics_center_y, self.play_arrow_left, self.graphics_bot
    canvas:
        Triangle:
            points: root.play_arrow_points
        SmoothLine:
            points: root.play_arrow_points[:-2] + [root.play_arrow_points[-2]+1, root.play_arrow_points[-1]+1]
    Label:
        id: playlabel
        font_size: root.font_size
        center_x: root.center_x
        y: root.y
        size: self.texture_size
        text: 'Simulate'
<OneTurnButton>:
    graphics_top: self.y + self.font_size + (self.height - self.font_size) * (3/4)
    graphics_bot: self.y + self.font_size + 3
    graphics_center_y: self.graphics_bot + (self.graphics_top - self.graphics_bot) / 2
    step_arrow_left: self.center_x - (self.width / 6)
    step_center_x: self.center_x + self.width / 6
    step_bar_right: self.center_x + self.width / 4
    step_arrow_points: self.step_arrow_left, self.graphics_top, self.step_center_x, self.graphics_center_y, self.step_arrow_left, self.graphics_bot
    step_rect_points: self.step_center_x, self.graphics_top, self.step_bar_right, self.graphics_top, self.step_bar_right, self.graphics_bot, self.step_center_x, self.graphics_bot
    canvas:
        Triangle:
            points: root.step_arrow_points
        Quad:
            points: root.step_rect_points 
        SmoothLine:
            points: root.step_arrow_points
        SmoothLine:
            points: root.step_rect_points 
    Label:
        font_size: root.font_size
        center_x: root.center_x
        y: root.y
        size: self.texture_size
        text: '1 turn'
<TimePanel>:
    orientation: 'vertical'
    playbut: playbut
    BoxLayout:
        size_hint_y: 0.4
        BoxLayout:
            orientation: 'vertical'
            Label:
                size_hint_y: 0.4
                text: 'Branch'
            MenuTextInput:
                id: branchfield
                set_value: root.set_branch
                hint_text: root.screen.app.branch if root.screen else ''
        BoxLayout:
            BoxLayout:
                orientation: 'vertical'
                Label:
                    size_hint_y: 0.4
                    text: 'Turn'
                MenuIntInput:
                    id: turnfield
                    set_value: root.set_turn
                    hint_text: str(root.screen.app.turn) if root.screen else ''
            BoxLayout:
                orientation: 'vertical'
                Label:
                    size_hint_y: 0.4
                    text: 'Tick'
                MenuIntInput:
                    id: tickfield
                    set_value: root.set_tick
                    hint_text: str(root.screen.app.tick) if root.screen else ''
    BoxLayout:
        size_hint_y: 0.6
        SimulateButton:
            id: playbut
            font_size: root.buttons_font_size
        OneTurnButton:
            id: stepbut
            font_size: root.buttons_font_size
            on_release: root.screen.next_turn()
            disabled: root.disable_one_turn
<MainScreen>:
    name: 'main'
    app: app
    dummyplace: charmenu.dummyplace
    dummything: charmenu.dummything
    mainview: mainview
    playbut: timepanel.playbut
    portaladdbut: charmenu.portaladdbut
    charmenu: charmenu
    statlist: statpanel.statlist
    statpanel: statpanel
    timepanel: timepanel
    dialoglayout: dialoglayout
    Widget:
        id: mainview
        x: statpanel.right
        y: 0
        size_hint: (None, None)
        width: charmenu.x - statpanel.right
        height: root.height
    StatListPanel:
        id: statpanel
        engine: app.engine
        toggle_stat_cfg: app.statcfg.toggle
        toggle_calendar: root.toggle_calendar
        toggle_gridview: root.toggle_gridview
        pos_hint: {'left': 0, 'top': 1}
        size_hint: (0.25, 0.8)
        selection_name: app.selected_proxy_name
    TimePanel:
        id: timepanel
        screen: root
        pos_hint: {'bot': 0}
        size_hint: (0.25, 0.2)
        disable_one_turn: root.tmp_block
    CharMenu:
        id: charmenu
        screen: root
        pos_hint: {'right': 1, 'top': 1}
        size_hint: (0.1, 1)
    DialogLayout:
        id: dialoglayout
        engine: app.engine
        size_hint: None, None
        x: statpanel.right
        y: timepanel.top
        width: charmenu.x - statpanel.right
        height: root.height - timepanel.top
"""
)
