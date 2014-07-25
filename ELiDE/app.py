from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    ObjectProperty,
    StringProperty,
    DictProperty
)
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image

from kivy.factory import Factory
from kivy.garden.stiffscroll import StiffScrollEffect

from .board import (
    Board,
    Arrow
)
from .board.arrow import get_points
from .charsheet import CharSheet
from .texturestack import ImageStack

import LiSE
import ELiDE
import gettext
import os

_ = lambda x: x


class TouchlessWidget(Widget):
    def on_touch_down(self, *args):
        pass

    def on_touch_move(self, *args):
        pass

    def on_touch_up(self, *args):
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


class LiSELayout(FloatLayout):
    """A very tiny master layout that contains one board and some menus
    and charsheets.

    This contains three elements: a board, a menu, and a character
    sheet. This class has some support methods for handling
    interactions with the menu and the character sheet, but if neither
    of those happen, the board handles touches on its own.

    """
    app = ObjectProperty()
    """The App instance that is running and thus holds the globals I need."""
    board = ObjectProperty()
    """The Board instance that's visible at present."""
    charsheet = ObjectProperty()
    """The CharSheet object to show the stats and what-not for the
    Character being inspected at present."""
    menu = ObjectProperty()
    """The menu on the left side of the screen. Composed only of buttons
    with graphics on."""
    _touch = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    """The modal view to use for the various menus that aren't visible by
    default."""
    portaling = BoundedNumericProperty(0, min=0, max=2)
    """Count how far along I am in the process of connecting two Places by
    creating a Portal between them."""
    playspeed = BoundedNumericProperty(0, min=0)

    def __init__(self, **kwargs):
        """Make a trigger for draw_arrow, then initialize as for
        FloatLayout."""
        self._trigger_draw_arrow = Clock.create_trigger(self._draw_arrow)
        super(LiSELayout, self).__init__(**kwargs)
        self.finalize()

    def _draw_arrow(self, *args):
        """Draw the arrow that you see when you're in the process of placing a
        portal.

        It looks like the arrows that represent portals, but it
        doesn't represent a portal, because you haven't connected both
        ends yet. If you had, real live Arrow object would be used to
        draw the arrow.

        """
        # Sometimes this gets triggered, *just before* getting
        # unbound, and ends up running one last time *just after*
        # self.dummyspot = None
        if self._touch is None:
            return
        ud = self.portal_d
        (ox, oy) = ud['origspot'].center
        (dx, dy) = self.board.parent.to_local(*self._touch.pos)
        points = get_points(ox, 0, oy, 0, dx, 0, dy, 0, 10)
        ud['dummyarrow'].canvas.clear()
        with ud['dummyarrow'].canvas:
            Color(0.25, 0.25, 0.25)
            Line(width=1.4, points=points)
            Color(1, 1, 1)
            Line(width=1, points=points)

    def make_arrow(self, *args):
        """Start the process of connecting Places with a new Portal.

        This will temporarily disable the ability to drag spots around
        and open the place detail view. The next touch will restore
        that ability, or, if it is a touch-and-drag, will connect the
        place where the touch-and-drag started with the one where it
        ends.

        If the touch-and-drag starts on a spot, but does not end on
        one, it does nothing, and the operation is cancelled.

        """
        _ = self.app.engine.get_text
        self.display_prompt(_(
            "Draw a line between the places to connect with a portal."
        ))
        self.portaling = 1

    def on_touch_down(self, touch):
        """If make_arrow has been called (but not cancelled), an arrow has not
        been made yet, and the touch collides with a Spot, start
        drawing an arrow from the Spot to the touch coordinates.

        The arrow will be redrawn until on_touch_up."""
        if self.portaling == 1:
            self.board.on_touch_down(touch)
            if "spot" in touch.ud:
                touch.grab(self)
                touch.ungrab(touch.ud['spot'])
                touch.ud['portaling'] = True
                self.portal_d = {
                    'origspot': touch.ud['spot'],
                    'dummyspot': DummySpot(pos=touch.pos),
                    'dummyarrow': TouchlessWidget()
                }
                self.board.arrowlayout.add_widget(
                    self.portal_d['dummyarrow'])
                self.add_widget(
                    self.portal_d['dummyspot'])
                self._touch = touch
                self.portaling = 2
            else:
                self.portaling = 0
                self.dismiss_prompt()
            return True
        else:
            return super(LiSELayout, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        """If I'm currently in the process of connecting two Places with a
        Portal, draw the arrow between the place of origin and the
        touch's current coordinates.

        """
        if self.portaling == 2:
            self._touch = touch
            self._trigger_draw_arrow()
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """If I'm currently in the process of connecting two Places with a
        Portal, check whether the touch collides a Spot that isn't the
        one I started at. If so, make the Portal.

        """
        if self.portaling == 2:
            self.portaling = 0
            if touch != self._touch:
                return
            self.portal_d['dummyarrow'].canvas.clear()
            self.remove_widget(self.portal_d['dummyspot'])
            self.board.remove_widget(self.portal_d['dummyarrow'])
            self.dismiss_prompt()
            destspot = self.board.on_touch_up(touch)
            if not destspot or 'origspot' not in self.portal_d:
                self.portal_d['dummyarrow'].canvas.clear()
                self.dismiss_prompt()
                return True
            origplace = self.portal_d['origspot'].place
            destplace = destspot.place
            portalname = "{}->{}".format(origplace, destplace)
            portal = self.board.facade.observed.make_portal(
                portalname, origplace, destplace,
                host=self.board.host)
            arrow = Arrow(
                board=self.board, portal=portal)
            self.board.arrowdict[unicode(portal)] = arrow
            self.board.arrowlayout.add_widget(arrow)
        else:
            return super().on_touch_up(touch)

    def display_prompt(self, text):
        """Put the text in the cue card"""
        self.ids.prompt.ids.l.text = text

    def dismiss_prompt(self, *args):
        """Blank out the cue card"""
        self.ids.prompt.text = ''

    def center_of_view_on_board(self):
        """Get the point on the Board that is presently at the center of the
        screen.

        """
        b = self.board
        bv = self.ids.board_view
        # clamp to that part of the board where the view's center might be
        effective_w = b.width - bv.width
        effective_h = b.height - bv.height
        x = b.width / 2 + effective_w * (bv.scroll_x - 0.5)
        y = b.height / 2 + effective_h * (bv.scroll_y - 0.5)
        return (x, y)

    def normal_speed(self):
        """Advance time at a sensible rate.

        """
        self.playspeed = 0.1

    def pause(self):
        """Halt the flow of time.

        """
        if hasattr(self, 'updater'):
            Clock.unschedule(self.updater)

    def update(self, ticks):
        """Advance time if possible. Otherwise pause.

        """
        target_tick = self.engine.tick + ticks
        while self.engine.tick < target_tick:
            self.engine.advance()
        self.pause()

    def on_playspeed(self, *args):
        """Change the interval of updates to match the playspeed.

        """
        self.pause()
        if self.playspeed > 0:
            ticks = 1
            interval = self.playspeed
        else:
            return
        self.updater = lambda dt: self.update(ticks)
        Clock.schedule_interval(self.updater, interval)

    def finalize(self, *args):
        if self.charsheet is None or self.board is None:
            Clock.schedule_once(self.finalize, 0)
            return
        boardview = ScrollView(effect_cls=StiffScrollEffect)
        boardview.add_widget(self.board)
        self.add_widget(boardview)
        self.add_widget(self.charsheet)


Factory.register('LiSELayout', cls=LiSELayout)


class MenuIntInput(TextInput):
    closet = ObjectProperty()
    stringname = StringProperty()
    attrname = StringProperty()

    def __init__(self, **kwargs):
        self._trigger_upd_time = Clock.create_trigger(self.upd_time)
        super(MenuIntInput, self).__init__(**kwargs)

    def insert_text(self, s, from_undo=False):
        """Natural numbers only."""
        return super(self, MenuIntInput).insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )

    def on_closet(self, *args):
        if self.closet:
            self.closet.timestream.register_time_listener(
                self._trigger_upd_time
            )

    def on_text_validate(self, *args):
        setattr(self.closet, self.attrname, int(self.text))

    def upd_time(self, *args):
        self.hint_text = str(getattr(self.closet, self.attrname))
        self.text = ''


Factory.register('MenuIntInput', cls=MenuIntInput)


class ELiDEApp(App):
    """LiSE, run as a standalone application, and not a library.

    As it's a Kivy app, this implements the things required of the App
    class. I also keep \"globals\" here.

    """
    engine = ObjectProperty()
    cli_args = DictProperty({})

    def build_config(self, config):
        for sec in 'LiSE', 'ELiDE':
            config.adddefaultsection(sec)
        config.setdefaults(
            'LiSE',
            {
                'world': 'lise.world',
                'code': 'lise.code',
                'language': 'en'
            }
        )
        config.setdefaults(
            'ELiDE',
            {
                'wallpaper': ELiDE.__path__[0] + "/assets/wallpape.jpg",
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
        return the root widget."""
        config = self.config
        self.engine = LiSE.LiSE(
            world_filename=config['LiSE']['world'],
            code_filename=config['LiSE']['code'],
            gettext=gettext.translation(
                'LiSE',
                os.sep.join([LiSE.__path__[0], 'localedir']),
                [config['LiSE']['language']]
            ).gettext
        )

        for char in 'boardchar', 'sheetchar':
            if config['ELiDE'][char] not in self.engine.character:
                self.engine.add_character(config['ELiDE'][char])
        boardchar = self.engine.character[config['ELiDE']['boardchar']]
        sheetchar = self.engine.character[config['ELiDE']['sheetchar']]
        l = LiSELayout(
            app=self,
            board=Board(
                engine=self.engine,
                character=boardchar,
                wallpaper=Image(source=config['ELiDE']['wallpaper'])
            ),
            charsheet=CharSheet(
                character=sheetchar,
                engine=self.engine
            )
        )
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
