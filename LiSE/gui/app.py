from os import sep, remove

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    BooleanProperty,
    ObjectProperty,
    ListProperty,
    StringProperty)

from kivy.graphics import Line, Color

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter, ScatterPlane
from kivy.uix.widget import Widget
from kivy.factory import Factory

from sqlite3 import connect

from LiSE.gui.board import (
    Pawn,
    Spot,
    Arrow)
from LiSE.gui.board.arrow import get_points
from LiSE.gui.kivybits import TexPile
from LiSE.gui.swatchbox import SwatchBox
from LiSE.util import Skeleton
from LiSE import (
    __path__,
    closet)


Factory.register('SwatchBox', cls=SwatchBox)


class TouchlessWidget(Widget):
    def on_touch_down(self, touch):
        return

    def on_touch_move(self, touch):
        return

    def on_touch_up(self, touch):
        return

    def collide_point(self, x, y):
        return

    def collide_widget(self, w):
        return


class CueCard(TouchlessWidget):
    """Widget that looks like TextInput but doesn't take input and can't be
clicked.

This is used to display feedback to the user when it's not serious
enough to get a popup of its own.

    """
    text = StringProperty()


class DummyPawn(Scatter):
    """Looks like a Pawn, but doesn't have a Thing associated.

This is meant to be used when the user is presently engaged with
deciding where a Thing should be, when the Thing in question doesn't
exist yet, but you know what it should look like."""
    imgbones = ListProperty()
    board = ObjectProperty()
    callback = ObjectProperty()

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
                th = clost.make_generic_thing(
                    self.board.facade.observed,
                    self.board.host,
                    placen)
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


class LiSELayout(FloatLayout):
    """A very tiny master layout that contains one board and some menus
and charsheets.

    """
    menus = ListProperty()
    charsheets = ListProperty()
    board = ObjectProperty()
    prompt = ObjectProperty()
    portaling = BoundedNumericProperty(0, min=0, max=2)
    dummyspot = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        """Add board first, then menus and charsheets."""
        super(LiSELayout, self).__init__(**kwargs)
        self._popups = []
        self.add_widget(self.board)
        for menu in self.menus:
            self.add_widget(menu)
        for charsheet in self.charsheets:
            self.add_widget(charsheet)
        self.add_widget(self.prompt)

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

    def on_touch_down(self, touch):
        clost = self.board.facade.closet
        if self.portaling == 1:
            for spot in self.board.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    self.origspot = spot
                    break
            if hasattr(self, 'origspot'):
                self.dummyspot = ScatterPlane(
                    pos=(touch.x, touch.y), size=(1, 1))
                self.dummyarrow = TouchlessWidget(pos=(0, 0))
                atop = []
                for pawn in self.board.pawndict.itervalues():
                    if pawn.where_upon is self.origspot:
                        atop.append(pawn)
                self.board.children[0].remove_widget(self.origspot)
                for pawn in atop:
                    self.board.children[0].remove_widget(pawn)
                self.board.children[0].add_widget(self.dummyarrow)
                self.board.children[0].add_widget(self.origspot)
                for pawn in atop:
                    self.board.children[0].add_widget(pawn)
                self.add_widget(self.dummyspot)
                self.dummyspot.bind(pos=self.draw_arrow)
                self.display_prompt(clost.get_text("@putportalto"))
                self.portaling = 2
            else:
                self.portaling = 0
                self.dummyspot.unbind(pos=self.draw_arrow)
                self.dummyarrow.canvas.clear()
                self.remove_widget(self.dummyspot)
                self.board.children[0].remove_widget(self.dummyarrow)
                self.dismiss_prompt()
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
            self.board.children[0].remove_widget(self.dummyarrow)
            self.dismiss_prompt()
            destspot = None
            for spot in self.board.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    destspot = spot
                    break
            if destspot is None:
                self.dummyarrow.canvas.clear()
                self.dismiss_prompt()
                return True
            origplace = self.origspot.place
            destplace = destspot.place
            portal = self.board.facade.observed.make_portal(
                origplace, destplace, host=self.board.host)
            arrow = Arrow(
                board=self.board, portal=portal)
            self.board.arrowdict[unicode(portal)] = arrow
            atop = []
            for pawn in self.board.pawndict.itervalues():
                if pawn.where_upon is self.origspot:
                    atop.append(pawn)
            self.board.children[0].remove_widget(self.origspot)
            for pawn in atop:
                self.board.children[0].remove_widget(pawn)
            self.board.children[0].add_widget(arrow)
            self.board.children[0].add_widget(self.origspot)
            for pawn in atop:
                self.board.children[0].add_widget(pawn)
            self.dummyspot = None
            self.dummyarrow = None
        else:
            return super(LiSELayout, self).on_touch_up(touch)

    def display_prompt(self, text):
        """Put the text in the cue card"""
        self.prompt.text = text

    def dismiss_prompt(self, *args):
        """Blank out the cue card"""
        self.prompt.text = ''

    def dismiss_popup(self, *args):
        """Destroy the latest popup"""
        self._popups.pop().dismiss()

    def new_spot_with_swatches(self, swatches):
        clost = self.board.facade.closet
        if len(swatches) < 1:
            return
        self.display_prompt(clost.get_text("@putplace"))
        Clock.schedule_once(self.dismiss_prompt, 5)
        place = clost.make_generic_place(self.board.host)
        branch = clost.branch
        tick = clost.tick
        obsrvr = unicode(self.board.facade.observer)
        host = unicode(self.board.host)
        placen = unicode(place)
        i = 0
        for swatch in swatches:
            bone = Spot.bonetype(
                observer=obsrvr,
                host=host,
                place=placen,
                layer=i,
                branch=branch,
                tick=tick,
                img=swatch.text)
            clost.set_bone(bone)
            i += 1
        coord_bone = Spot.bonetypes.spot_coords(
            observer=obsrvr,
            host=host,
            place=place,
            branch=branch,
            tick=tick,
            x=min((
                (0.1 * self.width) + self.board.scroll_x *
                self.board.viewport_size[0],
                self.board.viewport_size[0] - 32)),
            y=min((
                (0.9 * self.height) + self.board.scroll_y *
                self.board.viewport_size[1],
                self.board.viewport_size[1] - 32)))
        clost.set_bone(coord_bone)
        spot = Spot(board=self.board, place=place)
        self.add_widget(spot)
        self.dismiss_popup()

    def new_pawn_with_swatches(self, swatches):
        """Given some iterable of Swatch widgets, make a dummy pawn, prompt
the user to place it, and dismiss the popup."""
        clost = self.board.facade.closet
        self.display_prompt(clost.get_text("@putthing"))
        (w, h) = self.get_root_window().size
        dummy = DummyPawn(
            board=self.board,
            imgbones=[
                clost.skeleton[u'img'][swatch.text]
                for swatch in swatches],
            callback=self.dismiss_prompt)
        self.add_widget(dummy)
        dummy.pos = (w*0.1, h*0.9)
        self.dismiss_popup()

    def show_spot_picker(self, categories):
        """Show a SwatchBox for the given tags. The chosen swatches will be
        used to build a Spot later.

        """
        clost = self.board.facade.closet
        cattexlst = [
            (cat, sorted(clost.textag_d[cat.strip("!?")]))
            for cat in categories]
        dialog = PickImgDialog(
            set_imgs=self.new_spot_with_swatches,
            cancel=self.dismiss_popup)
        dialog.ids.picker.closet = clost
        dialog.ids.picker.cattexlst = cattexlst
        self._popups.append(Popup(
            title="Select graphics",
            content=dialog,
            size_hint=(0.9, 0.9)))
        self._popups[-1].open()

    def show_pawn_picker(self, categories):
        """Show a SwatchBox for the given tags. The chosen Swatches will be
        used to build a Pawn later.

        """
        clost = self.board.facade.closet
        cattexlst = [
            (cat, sorted(clost.textag_d[cat]))
            for cat in categories]
        dialog = PickImgDialog(
            set_imgs=self.new_pawn_with_swatches,
            cancel=self.dismiss_popup)
        dialog.ids.picker.closet = clost
        dialog.ids.picker.cattexlst = cattexlst
        self._popups.append(Popup(
            title="Select some images",
            content=dialog,
            size_hint=(0.9, 0.9)))
        self._popups[-1].open()

    def make_arrow(self, orig=None):
        """Prompt user to click the origin for a new Portal. Then start
drawing the Arrow for it and prompt user to click the
destination. Then make the Portal and its Arrow.

        """
        clost = self.board.facade.closet
        self.display_prompt(clost.get_text("@putportalfrom"))
        self.portaling = 1


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
    lang = StringProperty()
    lise_path = StringProperty()
    menu_name = StringProperty()
    observer_name = StringProperty()
    observed_name = StringProperty()
    host_name = StringProperty()
    charsheet_name = StringProperty()
    debug = BooleanProperty(False)
    logfile = StringProperty('')

    def build(self):
        """Make sure I can use the database, create the tables as needed, and
        return the root widget."""
        if self.dbfn is None:
            self.dbfn = self.user_data_dir + sep + "default.lise"
            print("No database specified; defaulting to {}".format(self.dbfn))
        try:
            if self.debug:
                # always want a fresh db for debug
                remove(self.dbfn)
            conn = connect(self.dbfn)
            i = 0
            for stmt in conn.iterdump():
                i += 1
                if i > 3:
                    break
            if i < 3:
                conn.close()
                closet.mkdb(self.dbfn, __path__[-1])
        except IOError:
            closet.mkdb(self.dbfn, __path__[-1])
        self.closet = closet.load_closet(
            self.dbfn, self.lise_path, self.lang, True)
        self.closet.load_img_metadata()
        self.closet.load_textures_tagged(['base', 'body'])
        self.closet.uptick_skel()
        self.updater = Clock.schedule_interval(self.closet.update, 0.1)
        self.closet.load_characters([
            self.observer_name,
            self.observed_name,
            self.host_name,
            self.charsheet_name])
        menu = self.closet.load_menu(self.menu_name)
        board = self.closet.load_board(
            self.observer_name,
            self.observed_name,
            self.host_name)
        charsheet = self.closet.load_charsheet(self.charsheet_name)
        prompt = CueCard()
        Clock.schedule_once(lambda dt: self.closet.checkpoint(), 0)
        return LiSELayout(
            menus=[menu],
            board=board,
            charsheets=[charsheet],
            prompt=prompt)

    def on_pause(self):
        self.closet.save()

    def stop(self, *largs):
        self.closet.save_game()
        self.closet.end_game()
        super(LiSEApp, self).stop(*largs)
