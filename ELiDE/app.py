# -*- coding: utf-8 -*-
# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from kivy.logger import Logger
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    NumericProperty,
    BooleanProperty,
    BoundedNumericProperty,
    ObjectProperty,
    StringProperty,
    DictProperty,
    ListProperty,
    ReferenceListProperty
)
from kivy.resources import resource_add_path
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

from kivy.factory import Factory

from .board import Board
from .board.arrow import ArrowWidget

import LiSE
import ELiDE
from ELiDE.kivygarden.texturestack import ImageStack

resource_add_path(ELiDE.__path__[0] + "/assets")


Factory.register('Board', cls=Board)


class MouseFollower(Widget):
    boardview = ObjectProperty()

    def on_touch_move(self, touch):
        self.center = self.boardview.to_local(*touch.pos)
        return True


class Dummy(ImageStack):
    _touch = ObjectProperty(None, allownone=True)
    name = StringProperty()
    prefix = StringProperty()
    num = NumericProperty()
    x_start = NumericProperty()
    y_start = NumericProperty()
    pos_start = ReferenceListProperty(x_start, y_start)
    x_down = NumericProperty()
    y_down = NumericProperty()
    pos_down = ReferenceListProperty(x_down, y_down)
    x_up = NumericProperty()
    y_up = NumericProperty()
    pos_up = ReferenceListProperty(x_up, y_up)
    x_center_up = NumericProperty()
    y_center_up = NumericProperty()
    center_up = ReferenceListProperty(x_center_up, y_center_up)
    right_up = NumericProperty()
    top_up = NumericProperty()

    def on_pos_up(self, *args):
        self.x_center_up = self.x_up + self.width / 2
        self.y_center_up = self.y_up + self.height / 2
        self.right_up = self.x_up + self.width
        self.top_up = self.y_up + self.height

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        self.pos_start = self.pos
        self.pos_down = (
            self.x - touch.x,
            self.y - touch.y
        )
        touch.grab(self)
        self._touch = touch
        return True

    def on_touch_move(self, touch):
        if touch is not self._touch:
            return False
        self.pos = (
            touch.x + self.x_down,
            touch.y + self.y_down
        )
        return True

    def on_touch_up(self, touch):
        if touch is not self._touch:
            return False
        self.pos_up = self.pos
        self.pos = self.pos_start
        self._touch = None
        return True


class ELiDELayout(FloatLayout):
    """A master layout that contains one board and some menus
    and charsheets.

    This contains three elements: a scrollview (containing the board),
    a menu, and the time control panel. This class has some support methods
    for handling interactions with the menu and the character sheet,
    but if neither of those happen, the scrollview handles touches on its
    own.

    """
    app = ObjectProperty()
    board = ObjectProperty()
    dummies = ListProperty()
    _touch = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    grabbing = BooleanProperty()
    reciprocal_portal = BooleanProperty()
    grabbed = ObjectProperty(None, allownone=True)
    selected = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    engine = ObjectProperty()
    tick_results = DictProperty({})
    branch = StringProperty()
    tick = NumericProperty()
    rules_per_frame = BoundedNumericProperty(10, min=1)

    def on_dummies(self, *args):
        if self.board is None or self.board.character is None:
            Clock.schedule_once(self.on_dummies, 0)
            return
        for dummy in self.dummies:
            if hasattr(dummy, '_numbered'):
                continue
            num = 0
            for nodename in self.board.character.node:
                nodename = str(nodename)
                if not nodename.startswith(dummy.prefix):
                    continue
                try:
                    nodenum = int(nodename.lstrip(dummy.prefix))
                except ValueError:
                    continue
                num = max((nodenum, num))
            dummy.num = num + 1
            dummy._numbered = True

    def spot_from_dummy(self, dummy):
        (x, y) = self.ids.boardview.to_local(*dummy.pos_up)
        x /= self.board.width
        y /= self.board.height
        self.board.spotlayout.add_widget(
            self.board.make_spot(
                self.board.character.new_place(
                    dummy.name,
                    _x=x,
                    _y=y,
                    _image_paths=dummy.paths
                )
            )
        )
        dummy.num += 1

    def pawn_from_dummy(self, dummy):
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
                    _image_paths=dummy.paths
                )
            )
        )
        dummy.num += 1

    def arrow_from_wid(self, wid):
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

    def on_engine(self, *args):
        """Set my branch and tick to that of my engine, and bind them so that
        when you change my branch or tick, you also change my
        engine's.

        """
        if self.engine is None:
            return
        self.branch = self.engine.branch
        self.tick = self.engine.tick
        self.bind(
            branch=self.branchupd,
            tick=self.tickupd
        )

        @self.engine.on_time
        def pulltime(e, b, t):
            self.unbind(
                branch=self.branchupd,
                tick=self.tickupd
            )
            self.branch = b
            self.tick = t
            self.bind(
                branch=self.branchupd,
                tick=self.tickupd
            )
            self.ids.board._trigger_update()

    def branchupd(self, *args):
        """Inform my engine of the new branch, and update the board widget."""
        if self.engine.branch != self.branch:
            self.engine.branch = self.branch
        self.ids.board._trigger_update()

    def set_branch(self, b):
        """``self.branch = b``"""
        self.branch = b

    def tickupd(self, *args):
        """Inform my engine of the new tick, and update the board widget."""
        if self.engine.tick != self.tick:
            self.engine.tick = self.tick
        self.ids.board._trigger_update()

    def set_tick(self, t):
        """``self.tick = t``"""
        self.tick = t

    def advance(self):
        """Resolve one rule and store the results in a list at
        ``self.tick_results[self.branch][self.tick]```.

        """
        if self.branch not in self.tick_results:
            self.tick_results[self.branch] = {}
        if self.tick not in self.tick_results[self.branch]:
            self.tick_results[self.branch][self.tick] = []
        r = self.tick_results[self.branch][self.tick]
        try:
            r.append(next(self.engine._rules_iter))
        except StopIteration:
            self.tick += 1
            self.engine.universal['rando_state'] = (
                self.engine.rando.getstate()
            )
            if (
                    self.engine.commit_modulus and
                    self.tick % self.engine.commit_modulus == 0
            ):
                self.engine.worlddb.commit()
            self.engine._rules_iter = self.engine._follow_rules()
            self.ids.board._trigger_update()

    def next_tick(self, *args):
        """Call ``self.advance()``, and if the tick hasn't changed, schedule
        it to happen again.

        """
        curtick = self.tick
        n = 0
        while (
                curtick == self.tick and
                n < self.rules_per_frame
        ):
            self.advance()
            n += 1
        if self.tick == curtick:
            Clock.schedule_once(self.next_tick, 0)
        else:
            Logger.info(
                "Followed {n} rules on tick {ct}:\n{r}".format(
                    n=n,
                    ct=curtick,
                    r="\n".join(
                        str(tup) for tup in
                        self.tick_results[self.branch][curtick]
                    )
                )
            )

    def on_touch_down(self, touch):
        """Delegate first to the menu, then to the charsheet, then to the
        board, then to the boardview.

        """
        if self.ids.charmenu.dispatch('on_touch_down', touch):
            return True
        if self.ids.timemenu.dispatch('on_touch_down', touch):
            return True
        if self.grabbing:
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            if not self.selection_candidates:
                self.selection_candidates = list(
                    self.board.pawns_at(*touch.pos)
                ) + list(
                    self.board.spots_at(*touch.pos)
                ) + list(
                    self.board.arrows_at(*touch.pos)
                )
            touch.pop()
            if self.ids.boardview.dispatch('on_touch_down', touch):
                self.scrolling = True
                return True
            else:
                return False
        if self.grabbed is not None:
            if (
                    hasattr(self.grabbed, 'place') and
                    self.ids.portaladdbut.state == 'down'
            ):
                self.mouse_follower = Widget(
                    size_hint=(None, None),
                    size=(1, 1),
                    pos=self.ids.boardview.to_local(*touch.pos)
                )
                self.board.add_widget(self.mouse_follower)
                self.dummy_arrow = ArrowWidget(
                    board=self.board,
                    origin=self.grabbed,
                    destination=self.mouse_follower
                )
                self.board.add_widget(self.dummy_arrow)
            return True
        return False

    def on_touch_move(self, touch):
        """If something's been grabbed, transform the touch to the boardview's
        space and then delegate there.

        """
        if self.grabbed or self.selection_candidates:
            if hasattr(self, 'scrolling'):
                del self.scrolling
        if hasattr(self, 'scrolling'):
            return self.ids.boardview.dispatch('on_touch_move', touch)
        if hasattr(self, 'mouse_follower'):
            self.mouse_follower.pos = self.ids.boardview.to_local(
                *touch.pos
            )
        if self.grabbed:
            if hasattr(self.grabbed, 'use_boardspace'):
                touch.push()
                touch.apply_transform_2d(self.ids.boardview.to_local)
            r = self.grabbed.dispatch('on_touch_move', touch)
            if hasattr(self.grabbed, 'use_boardspace'):
                touch.pop()
            return r

    def on_touch_up(self, touch):
        """Dispatch everywhere, and set my ``grabbed`` to ``None``"""
        if hasattr(self, 'mouse_follower'):
            self.arrow_from_wid(self.mouse_follower)
            self.board.remove_widget(self.dummy_arrow)
            self.board.remove_widget(self.mouse_follower)
            del self.dummy_arrow
            del self.mouse_follower
        self.ids.charmenu.dispatch('on_touch_up', touch)
        self.ids.timemenu.dispatch('on_touch_up', touch)
        self.ids.boardview.dispatch('on_touch_up', touch)
        if hasattr(self, 'scrolling'):
            del self.scrolling
            return True
        elif self.selection_candidates:
            if hasattr(self.grabbed, 'selected'):
                self.grabbed.selected = False
            self.grabbed = self.selection_candidates.pop(0)
            if hasattr(self.grabbed, 'selected'):
                self.grabbed.selected = True
        else:
            if hasattr(self.grabbed, 'selected'):
                self.grabbed.selected = False
            self.grabbed = None
            self.selection_candidates = []
        return True


Factory.register('ELiDELayout', cls=ELiDELayout)


class MenuTextInput(TextInput):
    """Special text input for setting the branch"""
    setter = ObjectProperty()

    def __init__(self, **kwargs):
        """Disable multiline, and bind ``on_text_validate`` to ``on_enter``"""
        kwargs['multiline'] = False
        super().__init__(**kwargs)
        self.bind(on_text_validate=self.on_enter)

    def on_enter(self, *args):
        """Call the setter and blank myself out so that my hint text shows
        up. It will be the same you just entered if everything's
        working.

        """
        self.setter(self.text)
        self.text = ''
        self.focus = False


Factory.register('MenuTextInput', cls=MenuTextInput)


class MenuIntInput(MenuTextInput):
    """Special text input for setting the tick"""
    def insert_text(self, s, from_undo=False):
        """Natural numbers only."""
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )


Factory.register('MenuIntInput', cls=MenuIntInput)


class ELiDEApp(App):
    """LiSE, run as a standalone application, and not a library.

    As it's a Kivy app, this implements the things required of the App
    class. I also keep \"globals\" here.

    """
    engine = ObjectProperty()
    cli_args = DictProperty({})

    def build_config(self, config):
        """Set config defaults"""
        for sec in 'LiSE', 'ELiDE':
            config.adddefaultsection(sec)
        config.setdefaults(
            'LiSE',
            {
                'world': 'sqlite:///LiSEworld.db',
                'code': 'LiSEcode.db',
                'language': 'en'
            }
        )
        config.setdefaults(
            'ELiDE',
            {
                'wallpaper': "wallpape.jpg",
                'boardchar': 'physical',
                'sheetchar': 'player'
            }
        )
        for sec in self.cli_args:
            for (k, v) in self.cli_args[sec].items():
                config[sec][k] = v
        config.write()

    def build(self):
        """Make sure I can use the database, create the tables as needed, and
        return the root widget.

        """
        config = self.config
        self.engine = LiSE.Engine(
            config['LiSE']['world'],
            config['LiSE']['code']
        )
        for char in config['ELiDE']['boardchar'], config['ELiDE']['sheetchar']:
            if char not in self.engine.character:
                print("adding character: {}".format(char))
                self.engine.add_character(char)
        l = ELiDELayout(app=self)
        from kivy.core.window import Window
        from kivy.modules import inspector
        inspector.create_inspector(Window, l)
        return l

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.engine.commit()

    def stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.engine.commit()
        self.engine.close()
        super().stop(*largs)
