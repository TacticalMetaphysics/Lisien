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
    OptionProperty,
    StringProperty
)
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.logger import Logger

from .dummy import Dummy
from .spritebuilder import PawnConfigDialog, SpotConfigDialog
from .charmenu import CharMenu
from .board.arrow import ArrowWidget
from .util import dummynum


Factory.register('CharMenu', cls=CharMenu)


class KvLayoutBack(FloatLayout):
    pass


class KvLayoutFront(FloatLayout):
    pass


class Message(Label):
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
    time = ListProperty()
    selected_remote = ObjectProperty()
    selection_name = StringProperty()
    button_text = StringProperty('cfg')
    set_value = ObjectProperty()
    cfgstatbut = ObjectProperty()
    toggle_stat_cfg = ObjectProperty()


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

    def set_branch(self, *args):
        branch = self.ids.branchfield.text
        self.ids.branchfield.text = ''
        self.branch_setter(branch)

    def set_tick(self, *args):
        tick = int(self.ids.tickfield.text)
        self.ids.tickfield.text = ''
        self.tick_setter(tick)


class ELiDELayout(FloatLayout):
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
    message = StringProperty('')
    use_message = BooleanProperty()
    play_speed = NumericProperty()
    playbut = ObjectProperty()
    portaladdbut = ObjectProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty('15sp')
    halign = OptionProperty(
        'left', options=['left', 'center', 'right', 'justify']
    )
    valign = OptionProperty(
        'bottom', options=['bottom', 'middle', 'top']
    )
    line_height = NumericProperty(1.0)
    engine = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    grabbing = BooleanProperty(True)
    reciprocal_portal = BooleanProperty(False)
    grabbed = ObjectProperty(None, allownone=True)
    selection = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    selected_remote = ObjectProperty()
    keep_selection = BooleanProperty(False)
    branch = AliasProperty(
        lambda self: self.engine.branch,
        lambda self, v: setattr(self.engine, 'branch', v),
        bind=('engine',)
    )
    tick = AliasProperty(
        lambda self: self.engine.tick,
        lambda self, v: setattr(self.engine, 'tick', v),
        bind=('engine',)
    )
    time = AliasProperty(
        lambda self: self.engine.time,
        lambda self, v: setattr(self.engine, 'time', v),
        bind=('engine',)
    )
    rules_per_frame = BoundedNumericProperty(10, min=1)

    def __init__(self, **kwargs):
        self._trigger_remake_display = Clock.create_trigger(
            self.remake_display
        )
        super().__init__(**kwargs)
        self._trigger_reremote = Clock.create_trigger(self.reremote)
        self.bind(selection=self._trigger_reremote)
        self._trigger_reremote()

    def on_play_speed(self, *args):
        """Change the interval at which ``self.play`` is called to match my
        current ``play_speed``.

        """
        Clock.unschedule(self.play)
        Clock.schedule_interval(self.play, 1.0 / self.play_speed)

    def on_board(self, *args):
        """Bind ``self._dispatch_time`` to the engine's time.

        This will make sure that my Kivy.properties ``branch``,
        ``tick``, and ``time`` trigger any functions bound to them
        when they change in response to the engine's time changing.

        """
        if self.engine is None or self.board is None:
            return
        self.engine.time_listener(self._dispatch_time)

    def on_character(self, *args):
        """Arrange to remake the customizable widgets when the character's
        stats change.

        Make them the first time, too, based on the current value of
        the relevant stat.

        """
        stats = (
            'kv',
            'message',
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
        elif stat == 'message':
            self.message = ''

    def remake_display(self, *args):
        """Remake any affected widgets after a change in my ``message`` or
        ``kv``.

        """
        Builder.load_string(self.kv)
        if hasattr(self, '_kv_layout_back'):
            self.remove_widget(self._kv_layout_back)
            del self._kv_layout_back
        if hasattr(self, '_message'):
            self.unbind(
                message=self._message.setter('text'),
                font_name=self._message.setter('font_name'),
                font_size=self._message.setter('font_size'),
                halign=self._message.setter('halign'),
                valign=self._message.setter('valign'),
                line_height=self._message.setter('line_height')
            )
            self.remove_widget(self._message)
            del self._message
        if hasattr(self, '_kv_layout_front'):
            self.remove_widget(self._kv_layout_front)
            del self._kv_layout_front
        self._kv_layout_back = KvLayoutBack()
        self._message = Message(
            text=self.message,
            font_name=self.font_name,
            font_size=self.font_size,
            halign=self.halign,
            valign=self.valign,
            line_height=self.line_height
        )
        self.bind(
            message=self._message.setter('text'),
            font_name=self._message.setter('font_name'),
            font_size=self._message.setter('font_size'),
            halign=self._message.setter('halign'),
            valign=self._message.setter('valign'),
            line_height=self._message.setter('line_height')
        )
        self._kv_layout_front = KvLayoutFront()
        self.add_widget(self._kv_layout_back)
        self.add_widget(self._message)
        self.add_widget(self._kv_layout_front)

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

    def select_character(self, char):
        """Change my ``character`` to the selected character object if they
        aren't the same.

        """
        if char == self.character:
            return
        self.character = char
        self.character_name = str(char.name)

    def on_touch_down(self, touch):
        """Dispatch the touch to the board, then its :class:`ScrollView`, then
        the dummies, then the menus.

        """
        if self.ids.timepanel.collide_point(*touch.pos):
            self.ids.timepanel.dispatch('on_touch_down', touch)
            self.keep_selection = True
            return True
        if self.ids.charmenu.collide_point(*touch.pos):
            self.ids.charmenu.dispatch('on_touch_down', touch)
            self.keep_selection = True
            return True
        if self.ids.statpanel.collide_point(*touch.pos):
            self.ids.statpanel.dispatch('on_touch_down', touch)
            self.keep_selection = True
            return True
        if self.dummyplace.collide_point(*touch.pos):
            self.dummyplace.dispatch('on_touch_down', touch)
            return True
        if self.dummything.collide_point(*touch.pos):
            self.dummything.dispatch('on_touch_down', touch)
            return True
        if self.ids.boardview.collide_point(*touch.pos):
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            pawns = list(self.board.pawns_at(*touch.pos))
            if pawns:
                self.selection_candidates = pawns
                if self.selection in self.selection_candidates:
                    self.selection_candidates.remove(self.selection)
                return True
            spots = list(self.board.spots_at(*touch.pos))
            if spots:
                self.selection_candidates = spots
                if self.selection in self.selection_candidates:
                    self.selection_candidates.remove(self.selection)
                if self.portaladdbut.state == 'down':
                    self.origspot = self.selection_candidates.pop(0)
                    self.protodest = Dummy(
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
        for dummy in self.dummies:
            if dummy.dispatch('on_touch_down', touch):
                return True

    def on_touch_move(self, touch):
        """If something's selected, it's on the board, so transform the touch
        to the boardview's space before dispatching it to the
        selection. Otherwise dispatch normally.

        """
        if self.selection:
            touch.push()
            if hasattr(self.selection, 'use_boardspace'):
                touch.apply_transform_2d(self.ids.boardview.to_local)
            r = self.selection.dispatch('on_touch_move', touch)
            touch.pop()
            if r:
                self.keep_selection = True
                return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """If there's a selection, dispatch the touch to it. Then, if there
        are selection candidates, select the next one that collides
        the touch. Otherwise, if something is selected, unselect
        it.

        """
        if hasattr(self, 'protodest'):
            # We're finishing the process of drawing an arrow to
            # create a new portal.
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
        if hasattr(self.selection, 'on_touch_up'):
            self.selection.dispatch('on_touch_up', touch)
        if self.ids.timepanel.collide_point(*touch.pos):
            self.ids.timepanel.dispatch('on_touch_up', touch)
            return True
        if self.ids.charmenu.collide_point(*touch.pos):
            self.ids.charmenu.dispatch('on_touch_up', touch)
            return True
        if self.ids.statpanel.collide_point(*touch.pos):
            self.ids.statpanel.dispatch('on_touch_up', touch)
            return True
        # If we're not making a portal, and the touch hasn't landed
        # anywhere that would demand special treatment, but the
        # touch_down hit some selectable items, select the first of
        # those that also collides this touch_up.
        if not self.keep_selection and self.selection_candidates:
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
                    self.keep_selection = True
                    break
            touch.pop()
        if not self.keep_selection and not (
                self.ids.timepanel.collide_point(*touch.pos) or
                self.ids.charmenu.collide_point(*touch.pos) or
                self.ids.statpanel.collide_point(*touch.pos)
        ):
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
            if hasattr(dummy, '_numbered'):
                continue
            if dummy == self.ids.dummything:
                dummy.paths = ['atlas://rltiles/base/unseen']
                self.ids.charmenu._pawn_config = PawnConfigDialog(layout=self)
            if dummy == self.ids.dummyplace:
                dummy.paths = ['orb.png']
                self.ids.charmenu._spot_config = SpotConfigDialog(layout=self)
            dummy.num = dummynum(self.character, dummy.prefix) + 1
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

    def set_branch(self, b):
        """Set my branch to the given value."""
        self.branch = b

    def set_tick(self, t):
        """Set my tick to the given value, cast to an integer."""
        self.tick = int(t)

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

    def _dispatch_time(self, *args):
        """Dispatch my ``branch``, ``tick``, and ``time`` properties."""
        self.property('branch').dispatch(self)
        self.property('tick').dispatch(self)
        self.property('time').dispatch(self)


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
    id: statpanel
    Label:
        size_hint_y: 0.05
        text: root.selection_name
    StatListView:
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
            setter: root.set_tick
            hint_text: str(root.tick)
<ELiDELayout>:
    character: self.engine.character[self.character_name] \
    if self.engine and self.character_name else None
    dummies: charmenu.dummies
    dummyplace: charmenu.dummyplace
    dummything: charmenu.dummything
    grabbing: self.grabbed is None
    board: boardview.board
    playbut: timepanel.playbut
    portaladdbut: charmenu.portaladdbut
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
        size_hint: (0.2, 0.9)
        engine: root.engine
        board: root.board
        selection: root.selection
        character: root.character
        character_name: root.character_name
        selected_remote: root.selected_remote
        spot_from_dummy: root.spot_from_dummy
        pawn_from_dummy: root.pawn_from_dummy
        select_character: root.select_character
"""
)
