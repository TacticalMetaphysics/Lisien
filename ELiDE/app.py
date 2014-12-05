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
from kivy.lang import Builder
from kivy.resources import resource_add_path
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

import LiSE
import ELiDE
from ELiDE.kivygarden.texturestack import ImageStack

resource_add_path(ELiDE.__path__[0] + "/assets")


class Dummy(ImageStack):
    """A widget that looks like the ones on the board, which, when dragged
    onto the board, creates one of them.

    """
    _touch = ObjectProperty(None, allownone=True)
    name = StringProperty()
    prefix = StringProperty()
    num = NumericProperty()
    x_start = NumericProperty(0)
    y_start = NumericProperty(0)
    pos_start = ReferenceListProperty(x_start, y_start)
    x_down = NumericProperty(0)
    y_down = NumericProperty(0)
    pos_down = ReferenceListProperty(x_down, y_down)
    x_up = NumericProperty(0)
    y_up = NumericProperty(0)
    pos_up = ReferenceListProperty(x_up, y_up)
    x_center_up = NumericProperty(0)
    y_center_up = NumericProperty(0)
    center_up = ReferenceListProperty(x_center_up, y_center_up)
    right_up = NumericProperty(0)
    top_up = NumericProperty(0)

    def on_touch_down(self, touch):
        """If hit, record my starting position, that I may return to it in
        ``on_touch_up`` after creating a real :class:`board.Spot` or
        :class:`board.Pawn` instance.

        """
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
        """Follow the touch"""
        if touch is not self._touch:
            return False
        self.pos = (
            touch.x + self.x_down,
            touch.y + self.y_down
        )
        return True

    def on_touch_up(self, touch):
        """Return to ``pos_start``, but first, save my current ``pos`` into
        ``pos_up``, so that the layout knows where to put the real
        :class:`board.Spot` or :class:`board.Pawn` instance.

        """
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
    grabbing = BooleanProperty(True)
    reciprocal_portal = BooleanProperty()
    grabbed = ObjectProperty(None, allownone=True)
    selection = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    keep_selection = BooleanProperty(False)
    engine = ObjectProperty()
    tick_results = DictProperty({})
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)
    rules_per_frame = BoundedNumericProperty(10, min=1)

    def on_touch_down(self, touch):
        """Dispatch the touch to the board, then its :class:`ScrollView`, then
        the dummies, then the menus.

        """
        # the menu widgets can handle things themselves
        if self.ids.timemenu.dispatch('on_touch_down', touch):
            return True
        if self.ids.charmenu.dispatch('on_touch_down', touch):
            return True
        if self.ids.charsheet.dispatch('on_touch_down', touch):
            return True
        if (
                self.ids.boardview.collide_point(*touch.pos)
                and not self.selection_candidates
        ):
            # if the board itself handles the touch, let it be
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            pawns = list(self.board.pawns_at(*touch.pos))
            if pawns:
                self.selection_candidates = pawns
                return True
            spots = list(self.board.spots_at(*touch.pos))
            if spots:
                self.selection_candidates = spots
                return True
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
        # For now, there's no way to select things by dragging.
        # However, if something's already selected, you can move it.
        if self.selection:
            touch.push()
            if hasattr(self.selection, 'use_boardspace'):
                touch.apply_transform_2d(self.ids.boardview.to_local)
            r = self.selection.dispatch('on_touch_move', touch)
            touch.pop()
            self.keep_selection = True
            return r
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """If there's a selection, dispatch the touch to it. Then, if there
        are selection candidates, select the next one that collides
        the touch. Otherwise, if something is selected, unselect
        it.

        """
        if hasattr(self.selection, 'on_touch_up'):
            self.selection.dispatch('on_touch_up', touch)
        if self.ids.timemenu.dispatch('on_touch_up', touch):
            return True
        if self.ids.charmenu.dispatch('on_touch_up', touch):
            return True
        if self.ids.charsheet.dispatch('on_touch_up', touch):
            return True
        if self.selection_candidates:
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            while self.selection_candidates:
                candidate = self.selection_candidates.pop(0)
                if candidate.collide_point(*touch.pos):
                    if hasattr(self.selection, 'selected'):
                        self.selection.selected = False
                    if hasattr(self.selection, '_start'):
                        Logger.debug(
                            "selection: moving {} back to {} from {}".format(
                                self.selection,
                                self.selection._start,
                                self.selection.pos
                            )
                        )
                        self.selection.pos = self.selection._start
                        del self.selection._start
                    self.selection = candidate
                    self.selection.selected = True
                    if (
                            hasattr(self.selection, 'thing')
                            and not hasattr(self.selection, '_start')
                    ):
                        self.selection._start = tuple(self.selection.pos)
                    self.keep_selection = True
                    break
            touch.pop()
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
        """Create a new :class:`board.Spot` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        position, and imagery of the provided dummy.

        """
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
            branch=self.timeupd,
            tick=self.timeupd
        )

    def timeupd(self, *args):
        if self.engine.branch != self.branch:
            self.engine.branch = self.branch
        if self.engine.tick != self.tick:
            self.engine.tick = self.tick
        self.ids.board._trigger_update()

    def set_branch(self, b):
        """``self.branch = b``"""
        self.branch = b

    def set_tick(self, t):
        """``self.tick = int(t)``"""
        self.tick = int(t)

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

        This is sort of a hack to fake parallel programming. Until I
        work out how to pass messages between an ELiDE process and a
        LiSE-core process, I'll just assume that each individual rule
        will be quick enough to resolve that the UI won't appear to
        lock up.

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
        if self.text == '':
            return
        self.setter(self.text)
        self.text = ''
        self.focus = False

    def on_focus(self, *args):
        if not self.focus:
            self.on_enter(*args)

    def on_text_validate(self, *args):
        self.on_enter()


class MenuIntInput(MenuTextInput):
    """Special text input for setting the tick"""
    def insert_text(self, s, from_undo=False):
        """Natural numbers only."""
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )


debug = False


class ELiDEApp(App):
    """Extensible LiSE Development Environment.

    As it's a Kivy app, this implements the things required of the App
    class. I also keep \"globals\" here.

    """
    engine = ObjectProperty()

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
                'sheetchar': 'player',
                'debug': 'no'
            }
        )
        config.write()

    def build(self):
        """Make sure I can use the database, create the tables as needed, and
        return the root widget.

        """
        config = self.config
        Logger.debug(
            "ELiDEApp: starting with world {}, code {}, path {}".format(
                config['LiSE']['world'],
                config['LiSE']['code'],
                LiSE.__path__[-1]
            )
        )
        if config['ELiDE']['debug'] == 'yes':
            import pdb
            pdb.set_trace()

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


kv = """
<Dummy>:
    name: "".join((self.prefix, str(self.num)))
    x_center_up: self.x_up + self.width / 2
    y_center_up: self.y_up + self.height / 2
    right_up: self.x_up + self.width
    top_up: self.y_up + self.height
<SymbolLabel@Label>:
    font_name: "Symbola.ttf"
    font_size: 50
<SymbolButton@Button>:
    font_name: "Symbola.ttf"
    font_size: 50
"""
Builder.load_string(kv)
