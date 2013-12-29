from os import sep

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    ObjectProperty,
    ListProperty,
    StringProperty)

from kivy.graphics import Line, Color

from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter, ScatterPlane
from kivy.factory import Factory

from sqlite3 import connect, OperationalError

from LiSE.gui.board import (
    Pawn,
    Spot,
    Arrow)
from LiSE.gui.board.arrow import get_points
from LiSE.gui.kivybits import TexPile, TouchlessWidget
from LiSE.gui.swatchbox import SwatchBox, TogSwatch
from LiSE.gui.charsheet import CharSheetAdder
from LiSE.util import TimestreamException
from LiSE import (
    __path__,
    closet,
    util)


Factory.register('SwatchBox', cls=SwatchBox)
Factory.register('TogSwatch', cls=TogSwatch)


class DummyPawn(Scatter):
    """Looks like a Pawn, but doesn't have a Thing associated.

This is meant to be used when the user is presently engaged with
deciding where a Thing should be, when the Thing in question doesn't
exist yet, but you know what it should look like."""
    imgbones = ListProperty()
    board = ObjectProperty()
    callback = ObjectProperty()
    name = StringProperty()

    def __init__(self, **kwargs):
        """Collect images and show them"""
        super(DummyPawn, self).__init__(**kwargs)
        self.pile = TexPile()
        clost = self.board.facade.closet
        for bone in self.imgbones:
            self.pile.append(
                clost.get_texture(bone.name),
                xoff=bone.off_x,
                yoff=bone.off_y,
                stackh=bone.stacking_height)
        self.add_widget(self.pile)

    def on_touch_up(self, touch):
        """Create a real Pawn on top of the Spot I am on top of, along
        with a Thing for it to represent. Then disappear."""
        clost = self.board.facade.closet
        for spot in self.board.spotdict.itervalues():
            if self.collide_widget(spot):
                obsrvr = unicode(self.board.facade.observer)
                obsrvd = unicode(self.board.facade.observed)
                hostn = unicode(self.board.host)
                placen = unicode(spot.place)
                tinybone = closet.Thing.bonetype(
                    character=obsrvd,
                    name=self.name,
                    host=hostn)
                bigbone = closet.Thing.bonetypes["thing_loc"](
                    character=obsrvd,
                    name=self.name,
                    branch=clost.branch,
                    tick=clost.tick,
                    location=placen)
                clost.set_bone(tinybone)
                clost.set_bone(bigbone)
                th = self.board.facade.observed.make_thing(self.name)
                thingn = unicode(th)
                branch = clost.branch
                tick = clost.tick
                for layer in xrange(0, len(self.imgbones)):
                    pawnbone = Pawn.bonetype(
                        observer=obsrvr,
                        observed=obsrvd,
                        host=hostn,
                        thing=thingn,
                        layer=layer,
                        branch=branch,
                        tick=tick,
                        img=self.imgbones[layer].name)
                    # default to being interactive
                    clost.set_bone(pawnbone)
                pawn = Pawn(board=self.board, thing=th)
                self.board.pawndict[thingn] = pawn
                self.board.add_widget(pawn)
                self.clear_widgets()
                self.callback()
                return True


class SpriteMenuContent(StackLayout):
    closet = ObjectProperty()
    skel = ObjectProperty()
    selection = ListProperty([])
    picker_args = ListProperty([])

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
        return name not in self.skel

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
            if len(tog.img_tags) > 0:
                self.picker_args.append(tog.img_tags)
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
    portaling = BoundedNumericProperty(0, min=0, max=2)
    origspot = ObjectProperty(None, allownone=True)
    dummyspot = ObjectProperty(None, allownone=True)
    playspeed = BoundedNumericProperty(0, min=-0.999, max=0.999)

    def __init__(self, **kwargs):
        """Add board first, then menus and charsheets."""
        super(LiSELayout, self).__init__(**kwargs)
        self._popups = []

    def handle_adbut(self, charsheet, i):
        adder = CharSheetAdder(charsheet=charsheet)

        def cancel():
            adder.dismiss()
        adder.cancel = cancel

        def confirm():
            bones = []
            j = i
            for bone in adder.iter_selection():
                bones.append(bone._replace(idx=j))
                j += 1
            try:
                charsheet.push_down(i, len(bones))
            except KeyError:
                # There were no charsheet items to push down.
                # That's fine. We still have room.
                pass
            for bone in bones:
                self.app.closet.set_bone(bone)
            charsheet._trigger_layout()
            adder.dismiss()
        adder.confirm = confirm
        adder.open()

    def draw_arrow(self, *args):
        # Sometimes this gets triggered, *just before* getting
        # unbound, and ends up running one last time *just after*
        # self.dummyspot = None
        if self.dummyspot is None:
            return
        (ox, oy) = self.origspot.pos
        (dx, dy) = self.dummyspot.pos
        (ow, oh) = self.origspot.size
        orx = ow / 2
        ory = oh / 2
        points = get_points(ox, orx, oy, ory, dx, 0, dy, 0, 10)
        self.dummyarrow.canvas.clear()
        with self.dummyarrow.canvas:
            Color(0.25, 0.25, 0.25)
            Line(width=1.4, points=points)
            Color(1, 1, 1)
            Line(width=1, points=points)

    def make_arrow(self, *args):
        _ = self.app.closet.get_text
        self.display_prompt(_(
            "Draw a line between the spots the portal should connect."))
        self.portaling = 1

    def on_touch_down(self, touch):
        _ = self.app.closet.get_text
        # menus appear above the board. you can't portal on them. so
        # interpret touches there as cancelling the portaling action.
        # same comment for charsheets
        if (
                self.ids.menu.collide_point(touch.x, touch.y) or
                self.ids.charsheet.collide_point(touch.x, touch.y)):
            self.portaling = 0
            return super(LiSELayout, self).on_touch_down(touch)
        elif self.portaling == 1:
            for spot in self.ids.board.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    self.origspot = spot
                    break
            if self.origspot is not None:
                assert(self.origspot not in self.children)
                self.dummyspot = ScatterPlane(
                    pos=(touch.x, touch.y), size=(1, 1))
                self.dummyarrow = TouchlessWidget(pos=(0, 0))
                atop = []
                for pawn in self.ids.board.pawndict.itervalues():
                    if pawn.where_upon is self.origspot:
                        atop.append(pawn)
                self.ids.board.children[0].remove_widget(self.origspot)
                for pawn in atop:
                    self.ids.board.children[0].remove_widget(pawn)
                self.ids.board.children[0].add_widget(self.dummyarrow)
                self.ids.board.children[0].add_widget(self.origspot)
                for pawn in atop:
                    self.ids.board.children[0].add_widget(pawn)
                self.add_widget(self.dummyspot)
                self.dummyspot.bind(pos=self.draw_arrow)
                self.display_prompt(_(
                    'Draw a line between the spots where you want a portal.'))
                self.portaling = 2
            else:
                self.portaling = 0
                self.dummyspot.unbind(pos=self.draw_arrow)
                self.dummyarrow.canvas.clear()
                self.remove_widget(self.dummyspot)
                self.ids.board.children[0].remove_widget(self.dummyarrow)
                self.dismiss_prompt()
                self.origspot = None
                self.dummyspot = None
                self.dummyarrow = None
        else:
            assert(self.portaling == 0)
        return super(LiSELayout, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        if self.portaling == 2:
            self.portaling = 0
            self.dummyspot.unbind(pos=self.draw_arrow)
            self.dummyarrow.canvas.clear()
            self.remove_widget(self.dummyspot)
            self.ids.board.children[0].remove_widget(self.dummyarrow)
            self.dismiss_prompt()
            destspot = None
            for spot in self.ids.board.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    destspot = spot
                    break
            if destspot is None:
                self.dummyarrow.canvas.clear()
                self.dismiss_prompt()
                return True
            origplace = self.origspot.place
            destplace = destspot.place
            portalname = "{}->{}".format(origplace, destplace)
            portal = self.ids.board.facade.observed.make_portal(
                portalname, origplace, destplace,
                host=self.ids.board.host)
            arrow = Arrow(
                board=self.ids.board, portal=portal)
            self.ids.board.arrowdict[unicode(portal)] = arrow
            atop = []
            for pawn in self.ids.board.pawndict.itervalues():
                if pawn.where_upon is self.origspot:
                    atop.append(pawn)
            self.ids.board.children[0].remove_widget(self.origspot)
            for pawn in atop:
                self.ids.board.children[0].remove_widget(pawn)
            self.ids.board.children[0].add_widget(arrow)
            self.ids.board.children[0].add_widget(self.origspot)
            for pawn in atop:
                self.ids.board.children[0].add_widget(pawn)
            self.dummyspot = None
            self.dummyarrow = None
        else:
            return super(LiSELayout, self).on_touch_up(touch)

    def display_prompt(self, text):
        """Put the text in the cue card"""
        self.ids.prompt.ids.l.text = text

    def dismiss_prompt(self, *args):
        """Blank out the cue card"""
        self.ids.prompt.text = ''

    def dismiss_popup(self, *args):
        """Destroy the latest popup"""
        self._popups.pop().dismiss()

    def spot_default_x(self):
        return min((
            (0.1 * self.width) + self.ids.board.scroll_x *
            self.ids.board.viewport_size[0],
            self.ids.board.viewport_size[0] - 32))

    def spot_default_y(self):
        return min((
            (0.9 * self.height) + self.ids.board.scroll_y *
            self.ids.board.viewport_size[1],
            self.ids.board.viewport_size[1] - 32))

    def show_spot_menu(self):
        hostn = unicode(self.ids.board.host)
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
                self.show_spot_picker(*spotpicker_args)
        spot_menu_content.confirm = confirm

        def cancel():
            spot_menu_content.selection = []
            self.dismiss_popup()
        spot_menu_content.cancel = cancel
        self._popups.append(spot_menu)
        spot_menu.open()

    def show_pawn_menu(self):
        obsrvd = unicode(self.ids.board.facade.observed)
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
        pawn_menu_content.confirm = confirm

        def cancel():
            pawn_menu_content.selection = []
            self.dismiss_popup()
        pawn_menu_content.cancel = cancel
        self._popups.append(pawn_menu)
        pawn_menu.open()

    def new_spot_with_name_and_imgs(self, name, imgs):
        _ = self.app.closet.get_text
        if len(imgs) < 1:
            return
        self.display_prompt(_('Drag this place where you want it.'))
        Clock.schedule_once(self.dismiss_prompt, 5)
        place = self.ids.board.host.make_place(name)
        branch = self.app.closet.branch
        tick = self.app.closet.tick
        obsrvr = unicode(self.ids.board.facade.observer)
        host = unicode(self.ids.board.host)
        placen = unicode(place)
        i = 0
        for img in imgs:
            bone = Spot.bonetype(
                observer=obsrvr,
                host=host,
                place=placen,
                layer=i,
                branch=branch,
                tick=tick,
                img=img.name)
            self.app.closet.set_bone(bone)
            i += 1
        coord_bone = Spot.bonetypes["spot_coords"](
            observer=obsrvr,
            host=host,
            place=placen,
            branch=branch,
            tick=tick,
            x=self.spot_default_x(),
            y=self.spot_default_y())
        self.app.closet.set_bone(coord_bone)
        assert(self.app.closet.have_place_bone(
            host, placen))
        spot = Spot(board=self.ids.board, place=place)
        self.ids.board.add_widget(spot)

    def new_pawn_with_name_and_imgs(self, name, imgs):
        """Given some iterable of Swatch widgets, make a dummy pawn, prompt
the user to place it, and dismiss the popup."""
        _ = self.app.closet.get_text
        if len(imgs) < 1:
            return
        self.display_prompt(_(
            'Drag this thing to the spot where you want it.'))
        (w, h) = self.get_root_window().size
        dummy = DummyPawn(
            board=self.ids.board,
            name=name,
            imgbones=imgs)

        def cb():
            self.remove_widget(dummy)
            self.dismiss_prompt()
        dummy.callback = cb
        self.add_widget(dummy)
        dummy.pos = (w*0.1, h*0.9)

    def show_spot_picker(self, name, imagery):
        self.dismiss_popup()
        if isinstance(imagery, list):
            def set_imgs(swatches):
                self.new_spot_with_name_and_imgs(name, [
                    swatch.img for swatch in swatches])
                self.dismiss_popup()
            dialog = PickImgDialog(
                name=name,
                set_imgs=set_imgs,
                cancel=self.dismiss_popup)
            cattexlst = [
                (cat, sorted(self.app.closet.textag_d[cat.strip("!?")]))
                for cat in imagery]
            dialog.ids.picker.closet = self.app.closet
            dialog.ids.picker.cattexlst = cattexlst
            self._popups.append(Popup(
                title="Select graphics",
                content=dialog,
                size_hint=(0.9, 0.9)))
            self._popups[-1].open()
        else:
            # imagery is a string name of an image
            img = self.app.closet.skeleton[u"img"][imagery]
            self.new_spot_with_name_and_imgs(name, [img])

    def show_pawn_picker(self, name, imagery):
        """Show a SwatchBox for the given tags. The chosen Swatches will be
        used to build a Pawn later.

        """
        self.dismiss_popup()
        if isinstance(imagery, list):
            def set_imgs(swatches):
                self.new_pawn_with_name_and_imgs(name, [
                    swatch.img for swatch in swatches])
                self.dismiss_popup()
            dialog = PickImgDialog(
                name=name,
                set_imgs=set_imgs,
                cancel=self.dismiss_popup)
            cattexlst = [
                (cat, sorted(self.app.textag_d[cat]))
                for cat in imagery]
            dialog.ids.picker.closet = self.app.closet
            dialog.ids.picker.cattexlst = cattexlst
            self._popups.append(Popup(
                title="Select some images",
                content=dialog,
                size_hint=(0.9, 0.9)))
            self._popups[-1].open()
        else:
            img = self.app.closet.skeleton[u"img"][imagery]
            self.new_pawn_with_name_and_imgs(name, [img])

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
    set_imgs = ObjectProperty()
    cancel = ObjectProperty()


class LiSEApp(App):
    closet = ObjectProperty(None)
    dbfn = StringProperty(allownone=True)
    lgettext = ObjectProperty(None)
    lise_path = StringProperty()
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
            for tab in util.tabclas.iterkeys():
                conn.execute("SELECT * FROM {};".format(tab))
        except (IOError, OperationalError):
            closet.mkdb(self.dbfn, __path__[-1])
        self.closet = closet.load_closet(
            self.dbfn, self.lgettext, True)
        self.closet.load_img_metadata()
        self.closet.load_textures_tagged(['base', 'body'])
        # Currently the decision of when and whether to update things
        # is split between here and the closet. Seems inappropriate.
        self.closet.load_characters([
            self.observer_name,
            self.observed_name,
            self.host_name])
        Clock.schedule_once(lambda dt: self.closet.checkpoint(), 0)
        self.closet.load_board(
            self.observer_name,
            self.observed_name,
            self.host_name)
        self.closet.load_charsheet(self.observed_name)
        return LiSELayout(app=self)

    def on_pause(self):
        self.closet.save()

    def stop(self, *largs):
        self.closet.save_game()
        self.closet.end_game()
        super(LiSEApp, self).stop(*largs)
