from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty,
    NumericProperty,
    BoundedNumericProperty,
    ObjectProperty,
    StringProperty,
    DictProperty
)
from kivy.resources import resource_add_path
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput

from kivy.factory import Factory

from .charsheet import CharSheet
from .board import Board
from .texturestack import ImageStack

import LiSE
import ELiDE

resource_add_path(ELiDE.__path__[0] + "/assets")

_ = lambda x: x

Factory.register('CharSheet', cls=CharSheet)
Factory.register('Board', cls=Board)


class TouchlessWidget(Widget):
    """Widget made not to interact with any touch"""
    def on_touch_down(self, *args):
        """Nothing"""
        pass

    def on_touch_move(self, *args):
        """Nothing"""
        pass

    def on_touch_up(self, *args):
        """Nothing"""
        pass


class DummySpot(Widget):
    """This is at the end of the arrow that appears when you're drawing a
    new portal. It's invisible, serving only to mark the pixel the
    arrow ends at for the moment.

    """
    def collide_point(self, *args):
        """This should be wherever you point, and therefore, always
        collides."""
        return True

    def on_touch_move(self, touch):
        """Center to touch"""
        self.center = touch.pos


class DummyPawn(ImageStack):
    """Looks like a Pawn, but doesn't have a Thing associated.

    This is meant to be used when the user is presently engaged with
    deciding where a Thing should be. The Thing in question
    doesn't exist yet, but you know what it should look like.

    """
    thing_name = StringProperty()
    board = ObjectProperty()
    callback = ObjectProperty()

    def on_touch_down(self, touch):
        """Grab the touch if it hits me."""
        if self.collide_point(touch.x, touch.y):
            touch.grab(self)
            touch.ud['pawn'] = self
            return True

    def on_touch_move(self, touch):
        """If I've been grabbed, move to the touch."""
        if 'pawn' in touch.ud and touch.ud['pawn'] is self:
            self.center = touch.pos

    def on_touch_up(self, touch):
        """Create a real Pawn on top of the Spot I am likewise on top of,
        along with a Thing for it to represent.

        """
        if 'pawn' not in touch.ud:
            return
            pass  # TODO


class ELiDELayout(FloatLayout):
    """A master layout that contains one board and some menus
    and charsheets.

    This contains three elements: a board, a menu, and a character
    sheet. This class has some support methods for handling
    interactions with the menu and the character sheet, but if neither
    of those happen, the board handles touches on its own.

    """
    app = ObjectProperty()
    """The App instance that is running and thus holds the globals I need."""
    board = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    """The modal view to use for the various menus that aren't visible by
    default."""
    portaling = BoundedNumericProperty(0, min=0, max=2)
    """Count how far along I am in the process of connecting two Places by
    creating a Portal between them."""
    grabbed = ObjectProperty(None, allownone=True)
    """Thing being grabbed"""
    selected = ObjectProperty(None, allownone=True)
    """Thing that's selected and highlighted for some operation"""
    engine = ObjectProperty()
    tick_results = DictProperty({})
    branch = StringProperty()
    tick = NumericProperty()
    rules_per_frame = BoundedNumericProperty(10, min=1)

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

    def tickupd(self, *args):
        """Inform my engine of the new tick, and update the board widget."""
        if self.engine.tick != self.tick:
            self.engine.tick = self.tick
        self.ids.board._trigger_update()

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

    def dispatch2board(self, event, touch):
        """Translate the touch to the boardview space, then dispatch the touch
        event

        """
        touch.push()
        touch.apply_transform_2d(self.ids.boardview.to_local)
        r = self.ids.board.dispatch(event, touch)
        touch.pop()
        return r

    def on_touch_down(self, touch):
        """Delegate first to the menu, then to the charsheet, then to the
        board, then to the boardview.

        """
        self.ids.charmenu.dispatch('on_touch_down', touch)
        self.ids.timemenu.dispatch('on_touch_down', touch)
        if self.grabbed is None:
            self.dispatch2board('on_touch_down', touch)
        if self.grabbed is None:
            return self.ids.boardview.dispatch('on_touch_down', touch)
        else:
            return self.grabbed is not None

    def on_touch_move(self, touch):
        """If something's been grabbed, transform the touch to the boardview's
        space and then delegate there.

        """
        if self.grabbed is None:
            return self.ids.boardview.dispatch('on_touch_move', touch)
        else:
            return self.grabbed.dispatch('on_touch_move', touch)

    def on_touch_up(self, touch):
        """Dispatch everywhere, and set my ``grabbed`` to ``None``"""
        self.ids.charmenu.dispatch('on_touch_up', touch)
        self.ids.timemenu.dispatch('on_touch_up', touch)
        self.ids.boardview.dispatch('on_touch_up', touch)
        self.dispatch2board('on_touch_up', touch)
        self.grabbed = None
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
                'world': 'LiSEworld.db',
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
        self.engine.character['dwarf'].stat._not_null('sight_radius')
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
        self.engine.close()
        super().stop(*largs)
