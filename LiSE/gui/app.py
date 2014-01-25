from os import sep

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    ObjectProperty,
    ListProperty,
    StringProperty)
from kivy.factory import Factory
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.logger import Logger

from sqlite3 import connect, OperationalError

from LiSE.gui.board import (
    Board,
    Pawn,
    Spot,
    Arrow,
    BoardView)
from LiSE.gui.board.gamepiece import GamePiece
from LiSE.gui.board.arrow import get_points

from LiSE.gui.kivybits import TouchlessWidget
from LiSE.gui.swatchbox import SwatchBox, TogSwatch
from LiSE.gui.charsheet import CharSheet, CharSheetAdder

from LiSE.util import TimestreamException
from LiSE.model import Thing
from LiSE.orm import SaveableMetaclass, mkdb, load_closet
from LiSE import __path__

Factory.register('BoardView', cls=BoardView)
Factory.register('SwatchBox', cls=SwatchBox)
Factory.register('TogSwatch', cls=TogSwatch)


class DummySpot(Widget):
    def collide_point(self, *args):
        return True

    def on_touch_move(self, touch):
        self.pos = touch.pos


class DummyPawn(GamePiece):
    """Looks like a Pawn, but doesn't have a Thing associated.

    This is meant to be used when the user is presently engaged with
    deciding where a Thing should be, when the Thing in question
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
        """Create a real Pawn on top of the Spot I am on top of, along
        with a Thing for it to represent."""
        if 'pawn' in touch.ud:
            del touch.ud['pawn']
        else:
            return
        closet = self.board.host.closet
        for spot in self.board.spotlayout.children:
            if self.collide_widget(spot):
                obsrvr = unicode(self.board.facade.observer)
                obsrvd = unicode(self.board.facade.observed)
                hostn = unicode(self.board.host)
                placen = unicode(spot.place)
                tinybone = Thing.bonetype(
                    character=obsrvd,
                    name=self.thing_name,
                    host=hostn)
                bigbone = Thing.bonetypes["thing_loc"](
                    character=obsrvd,
                    name=self.thing_name,
                    branch=closet.branch,
                    tick=closet.tick,
                    location=placen)
                pawnbone = Pawn.bonetype(
                    observer=obsrvr,
                    observed=obsrvd,
                    host=hostn,
                    thing=self.thing_name,
                    branch=closet.branch,
                    tick=closet.tick,
                    graphic=self.graphic_name)
                closet.set_bone(tinybone)
                closet.set_bone(bigbone)
                closet.set_bone(pawnbone)
                th = self.board.facade.observed.make_thing(self.thing_name)
                thingn = unicode(th)
                pawn = Pawn(board=self.board, thing=th)
                self.board.pawndict[thingn] = pawn
                self.board.pawnlayout.add_widget(pawn)
                self.callback()
                return True


class SpriteMenuContent(StackLayout):
    """Menu shown when a place or thing is to be created."""
    closet = ObjectProperty()
    """Closet to make things with and get text from."""
    selection = ListProperty([])
    """Swatches go in here. I'll make picker_args from them."""
    picker_args = ListProperty([])
    """Either one or two arguments for the method that creates the
    spot/pawn."""

    def get_text(self, stringn):
        return self.closet.get_text(stringn)

    def upd_selection(self, togswatch, state):
        if state == 'normal':
            while togswatch in self.selection:
                self.selection.remove(togswatch)
        else:
            if togswatch not in self.selection:
                self.selection.append(togswatch)

    def validate_name(self, name):
        """Return True if the name hasn't been used for a Place in this Host
        before, False otherwise."""
        # assume that this is an accurate record of places that exist
        return name not in self.closet.skeleton[u'place']

    def aggregate(self):
        """Collect the place name and graphics set the user has chosen."""
        if len(self.selection) < 1:
            return False
        else:
            assert(len(self.selection) == 1)
        namer = self.ids.namer
        tog = self.selection.pop()
        if self.validate_name(namer.text):
            self.picker_args.append(namer.text)
            if len(tog.tags) > 0:
                self.picker_args.append(tog.tags)
            else:
                self.picker_args.append(tog.img.name)
            return True
        else:
            self.selection.append(tog)
            namer.text = ''
            namer.hint_text = "That name is taken. Try another."
            namer.background_color = [1, 0, 0, 1]
            namer.focus = False

            def unbg(*args):
                namer.background_color = [1, 1, 1, 1]
            Clock.schedule_once(unbg, 0.5)
            return False


class SpotMenuContent(SpriteMenuContent):
    pass


class PawnMenuContent(SpriteMenuContent):
    pass


class LiSELayout(FloatLayout):
    """A very tiny master layout that contains one board and some menus
and charsheets.

    """
    app = ObjectProperty()
    board = ObjectProperty()
    charsheet = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    portaling = BoundedNumericProperty(0, min=0, max=2)
    playspeed = BoundedNumericProperty(0, min=-0.999, max=0.999)

    def handle_adbut(self, charsheet, i):
        adder = CharSheetAdder(charsheet=charsheet, insertion_point=i)
        adder.open()

    def draw_arrow(self, *args):
        # Sometimes this gets triggered, *just before* getting
        # unbound, and ends up running one last time *just after*
        # self.dummyspot = None
        if self._touch is None:
            return
        (ox, oy) = self._touch.ud['spot'].pos
        (dx, dy) = self._touch.ud['portaling']['dummyspot'].pos
        (ow, oh) = self._touch.ud['spot'].size
        orx = ow / 2
        ory = oh / 2
        points = get_points(ox, orx, oy, ory, dx, 0, dy, 0, 10)
        self._touch.ud['portaling']['dummyarrow'].canvas.clear()
        with self._touch.ud['portaling']['dummyarrow'].canvas:
            Color(0.25, 0.25, 0.25)
            Line(width=1.4, points=points)
            Color(1, 1, 1)
            Line(width=1, points=points)

    def make_arrow(self, *args):
        _ = self.app.closet.get_text
        self.display_prompt(_(
            "Draw a line between the spots to connect with a portal."))
        self.portaling = 1

    def on_touch_down(self, touch):
        if self.portaling == 1:
            self.board.on_touch_down(touch)
            if "spot" in touch.ud:
                touch.grab(self)
                ud = {
                    'dummyspot': DummySpot(pos=touch.pos),
                    'dummyarrow': TouchlessWidget()}
                self.board.arrowlayout.add_widget(ud['dummyarrow'])
                self.add_widget(ud['dummyspot'])
                ud["dummyspot"].bind(pos=self.draw_arrow)
                touch.ud['portaling'] = ud
                self._touch = touch
                self.portaling = 2
            else:
                self.portaling = 0
                if (
                        'portaling' in touch.ud and
                        'dummyspot' in touch.ud['portaling']):
                    ud = touch.ud['portaling']
                    ud['dummyspot'].unbind(pos=self.draw_arrow)
                    self.remove_widget(ud['dummyspot'])
                    ud['dummyarrow'].canvas.clear()
                    self.board.arrowlayout.remove_widget(
                        ud['dummyarrow'])
                    del touch.ud['portaling']
                self.dismiss_prompt()
                self.origspot = None
                self.dummyspot = None
                self.dummyarrow = None
        return super(LiSELayout, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if 'portaling' in touch.ud:
            touch.ud['portaling']['dummyspot'].pos = (
                self.board.spotlayout.to_local(*touch.pos))
            return True
        return super(LiSELayout, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.portaling == 2:
            self.portaling = 0
            if touch != self._touch:
                return
            ud = touch.ud['portaling']
            del touch.ud['portaling']
            ud['dummyspot'].unbind(pos=self.draw_arrow)
            ud['dummyarrow'].canvas.clear()
            self.remove_widget(ud['dummyspot'])
            self.board.remove_widget(ud['dummyarrow'])
            self.dismiss_prompt()
            destspot = None
            for spot in self.board.spotlayout.children:
                if spot.collide_point(*self.board.spotlayout.to_local(
                        *touch.pos)):
                    destspot = spot
                    break
            if destspot is None:
                ud['dummyarrow'].canvas.clear()
                self.dismiss_prompt()
                return True
            origplace = touch.ud['spot'].place
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
            return super(LiSELayout, self).on_touch_up(touch)

    def display_prompt(self, text):
        """Put the text in the cue card"""
        self.ids.prompt.ids.l.text = text

    def dismiss_prompt(self, *args):
        """Blank out the cue card"""
        self.ids.prompt.text = ''

    def show_spot_menu(self):
        hostn = unicode(self.board.host)
        if hostn not in self.app.closet.skeleton[u"place"]:
            self.app.closet.skeleton[u"place"][hostn] = {}
        spot_menu_content = SpotMenuContent(
            closet=self.app.closet,
            skel=self.app.closet.skeleton[u"place"][hostn])
        spot_menu = Popup(
            title="Give your place a name and appearance",
            content=spot_menu_content)

        def confirm():
            if spot_menu_content.aggregate():
                spotpicker_args = spot_menu_content.picker_args
                spot_menu_content.selection = []
                spot_menu.dismiss()
                if isinstance(spotpicker_args[-1], list):
                    self.show_spot_picker(*spotpicker_args)
                else:
                    self.new_spot_with_name_and_graphic(*spotpicker_args)
        spot_menu_content.confirm = confirm

        def cancel():
            spot_menu_content.selection = []
            spot_menu.dismiss()
        spot_menu_content.cancel = cancel
        spot_menu.open()

    def show_pawn_menu(self):
        obsrvd = unicode(self.board.facade.observed)
        if obsrvd not in self.app.closet.skeleton[u"thing"]:
            self.app.closet.skeleton[u"thing"][obsrvd] = {}
        if obsrvd not in self.app.closet.skeleton[u"thing_loc"]:
            self.app.closet.skeleton[u"thing_loc"][obsrvd] = {}
        pawn_menu_content = PawnMenuContent(
            closet=self.app.closet,
            skel=self.app.closet.skeleton[u"thing"][obsrvd])
        pawn_menu = Popup(
            title="Give this thing a name and appearance",
            content=pawn_menu_content)

        def confirm():
            if pawn_menu_content.aggregate():
                pawnpicker_args = pawn_menu_content.picker_args
                pawn_menu_content.selection = []
                self.show_pawn_picker(*pawnpicker_args)
                pawn_menu.dismiss()
        pawn_menu_content.confirm = confirm

        def cancel():
            pawn_menu_content.selection = []
            pawn_menu.dismiss()
        pawn_menu_content.cancel = cancel
        pawn_menu.open()

    def center_of_view_on_board(self):
        # get the point on the board that is presently at the center
        # of the screen
        b = self.board
        bv = self.ids.board_view
        # clamp to that part of the board where the view's center might be
        effective_w = b.width - bv.width
        effective_h = b.height - bv.height
        x = b.width / 2 + effective_w * (bv.scroll_x - 0.5)
        y = b.height / 2 + effective_h * (bv.scroll_y - 0.5)
        return (x, y)

    def new_pawn_with_name_and_graphic(self, thing_name, graphic_name):
        _ = self.app.closet.get_text
        self.display_prompt(_(
            "Drag this to a spot"))
        dummy = DummyPawn(
            thing_name=thing_name,
            board=self.board,
            graphic_name=graphic_name)

        def cb():
            self.board.pawnlayout.remove_widget(dummy)
            self.dismiss_prompt()
        dummy.callback = cb
        dummy.pos = self.center_of_view_on_board()
        self.board.pawnlayout.add_widget(dummy)

    def new_spot_with_name_and_graphic(self, place_name, graphic_name):
        place = self.board.host.make_place(place_name)
        (branch, tick) = self.app.closet.time
        obsrvr = unicode(self.board.facade.observer)
        hst = unicode(self.board.host)
        self.app.closet.set_bone(Spot.bonetypes["spot"](
            observer=obsrvr,
            host=hst,
            place=place_name,
            branch=branch,
            tick=tick,
            graphic=graphic_name))
        (x, y) = self.center_of_view_on_board()
        self.app.closet.set_bone(Spot.bonetypes["spot_coords"](
            observer=obsrvr,
            host=hst,
            place=place_name,
            branch=branch,
            tick=tick,
            x=x,
            y=y))
        self.board.spotlayout.add_widget(
            Spot(board=self.board,
                 place=place))

    def show_spot_picker(self, name, imagery):
        def set_graphic(swatches, dialog):
            graphic_name = u"{}_graphic".format(name)
            self.app.closet.set_bone(GamePiece.bonetypes["graphic"](
                name=name))
            for i in xrange(0, len(imagery)):
                self.app.closet.set_bone(GamePiece.bonetypes["graphic_img"](
                    graphic=graphic_name,
                    img=imagery[i].img.name,
                    layer=i))
            self.new_spot_with_name_and_graphic(name, graphic_name)
            dialog.dismiss()
        dialog = PickImgDialog(name=name)
        dialog.set_imgs = lambda swatches: set_graphic(swatches, dialog)
        dialog.ids.picker.closet = self.app.closet
        dialog.ids.picker.categorized_images = [
            (cat, sorted(self.app.closet.imgs_with_tag(cat.strip("?!"))))
            for cat in imagery]
        popup = Popup(
            title="Select graphics",
            content=dialog,
            size_hint=(0.9, 0.9))
        dialog.cancel = lambda: popup.dismiss()
        popup.open()

    def show_pawn_picker(self, name, imagery):
        """Show a SwatchBox for the given tags. The chosen Swatches will be
        used to build a Pawn later.

        """
        if isinstance(imagery, list):
            pickest = Popup(
                title="Select some images",
                size_hint=(0.9, 0.9))

            def set_imgs(swatches):
                # TODO interface for defining appearances independent
                # of pawns per-se, make the user's defined appearances
                # show up on the pawn picker
                #
                # default pawn offsets
                graphic_name = "{}_imagery".format(name)
                graphic_bone = GamePiece.bonetypes["graphic"](
                    name=graphic_name,
                    offset_x=4,
                    offset_y=8)
                img_bones = [
                    GamePiece.bonetypes["graphic_img"](
                        graphic=graphic_name,
                        layer=i,
                        img=imagery[i])
                    for i in xrange(0, len(imagery))]
                for bone in [graphic_bone] + img_bones:
                    self.app.closet.set_bone(bone)
                self.new_pawn_with_name_and_graphic(name, graphic_name)
                pickest.dismiss()

            catimglst = [
                (cat, sorted(self.app.closet.image_tag_d[cat]))
                for cat in imagery]
            dialog = PickImgDialog(
                name=name,
                categorized_images=catimglst,
                set_imgs=set_imgs,
                cancel=pickest.dismiss)
            dialog.ids.picker.closet = self.app.closet
            pickest.content = dialog
            pickest.open()
        else:
            self.new_pawn_with_name_and_graphic(name, imagery)

    def normal_speed(self, forward=True):
        if forward:
            self.playspeed = 0.1
        else:
            self.playspeed = -0.1

    def pause(self):
        if hasattr(self, 'updater'):
            Clock.unschedule(self.updater)

    def update(self, ticks):
        try:
            self.app.closet.time_travel_inc_tick(ticks)
        except TimestreamException:
            self.pause()

    def on_playspeed(self, i, v):
        self.pause()
        if v > 0:
            ticks = 1
            interval = v
        elif v < 0:
            ticks = -1
            interval = -v
        else:
            return
        self.updater = lambda dt: self.update(ticks)
        Clock.schedule_interval(self.updater, interval)

    def go_to_branch(self, bstr):
        self.app.closet.time_travel(int(bstr), self.app.closet.tick)

    def go_to_tick(self, tstr):
        self.app.closet.time_travel(self.app.closet.branch, int(tstr))


class LoadImgDialog(FloatLayout):
    """Dialog for adding img files to the database."""
    load = ObjectProperty()
    cancel = ObjectProperty()


class PickImgDialog(FloatLayout):
    """Dialog for associating imgs with something, perhaps a Pawn.

In lise.kv this is given a SwatchBox with texdict=root.texdict."""
    categorized_images = ObjectProperty()
    set_imgs = ObjectProperty()
    cancel = ObjectProperty()


class LiSEApp(App):
    closet = ObjectProperty(None)
    dbfn = StringProperty(allownone=True)
    lgettext = ObjectProperty(None)
    observer_name = StringProperty()
    observed_name = StringProperty()
    host_name = StringProperty()

    def build(self):
        """Make sure I can use the database, create the tables as needed, and
        return the root widget."""
        if self.dbfn is None:
            self.dbfn = self.user_data_dir + sep + "default.lise"
            print("No database specified; defaulting to {}".format(self.dbfn))
        try:
            conn = connect(self.dbfn)
            for tab in SaveableMetaclass.tabclas.iterkeys():
                conn.execute("SELECT * FROM {};".format(tab))
        except (IOError, OperationalError):
            mkdb(self.dbfn, __path__[-1], True)
        self.closet = load_closet(
            self.dbfn, self.lgettext,
            load_img=True,
            load_img_tags=['base', 'body'],
            load_gfx=True,
            load_characters=[self.observer_name, self.observed_name,
                             self.host_name],
            load_charsheet=self.observed_name,
            load_board=[self.observer_name, self.observed_name,
                        self.host_name])
        observer = self.closet.get_character(self.observer_name)
        observed = self.closet.get_character(self.observed_name)
        host = self.closet.get_character(self.host_name)
        l = LiSELayout(app=self, board=Board(
            app=self, facade=observed.get_facade(observer),
            host=host), charsheet=CharSheet(character=observed))
        from kivy.core.window import Window
        from kivy.modules import inspector
        inspector.create_inspector(Window, l)
        l.board.finalize()
        return l

    def on_pause(self):
        self.closet.save()

    def stop(self, *largs):
        self.closet.save_game()
        self.closet.end_game()
        super(LiSEApp, self).stop(*largs)
