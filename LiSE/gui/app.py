from os import sep, remove

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    BooleanProperty,
    ObjectProperty,
    ListProperty,
    StringProperty)
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.widget import Widget
from kivy.factory import Factory

from sqlite3 import connect, DatabaseError

from LiSE.gui.board import Pawn
from LiSE.gui.board import Spot
from LiSE.gui.board import Arrow
from LiSE.gui.swatchbox import SwatchBox
from LiSE.model import Portal
from LiSE.util import Skeleton
from LiSE import (
    __path__,
    closet)


Factory.register('SwatchBox', cls=SwatchBox)


class CueCard(Widget):
    """Widget that looks like TextInput but doesn't take input and can't be
clicked.

This is used to display feedback to the user when it's not serious
enough to get a popup of its own.

    """
    text = StringProperty()

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
        self.size = (0, 0)
        stackh = 0
        for bone in self.imgbones:
            tex = self.board.closet.get_texture(bone.img)
            if tex.width > self.width:
                self.width = tex.width
            if tex.height > self.height:
                self.height = tex.height
            img = Image(
                texture=tex,
                size=tex.size,
                x=bone.off_x,
                y=bone.off_y + stackh)
            stackh += bone.stacking_height + bone.off_y
            self.add_widget(img)

    def on_touch_up(self, touch):
        """Create a real Pawn on top of the Spot I am on top of, along
        with a Thing for it to represent. Then disappear."""
        for spot in self.board.spotdict.itervalues():
            if self.collide_widget(spot):
                dimn = unicode(spot.board)
                placen = unicode(spot.place)
                th = self.board.closet.make_generic_thing(
                    dimn, placen)
                thingn = unicode(th)
                branch = spot.board.closet.branch
                tick = spot.board.closet.tick
                skel = spot.board.closet.skeleton[u"pawn_img"]
                if dimn not in skel:
                    skel[dimn] = Skeleton()
                skel[dimn][thingn] = Skeleton()
                for layer in xrange(0, len(self.imgnames)):
                    skel[dimn][thingn][layer] = Skeleton()
                    ptr = skel[dimn][thingn][layer][branch] = Skeleton()
                    ptr[tick] = Pawn.bonetypes.pawn_img(
                        dimension=dimn,
                        thing=thingn,
                        layer=layer,
                        branch=branch,
                        tick=tick,
                        img=self.imgnames[layer])
                pawn = Pawn(board=self.board, thing=th)
                pawn.set_interactive()
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
    touched = ObjectProperty(None, allownone=True)

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

    def on_touch_down(self, touch):
        if self.portaling > 0:
            for spot in self.board.spotdict.itervalues():
                if spot.collide_point(touch.x, touch.y):
                    self.touched = spot
                    break
        return (super(LiSELayout, self).on_touch_down(touch)
                or self.touched is not None)

    def on_touch_up(self, touch):
        if self.touched is not None and self.touched.collide_point(
                touch.x, touch.y):
            if self.portaling == 1:
                self.origspot = self.touched
                self.portaling = 2
                self.display_prompt(self.board.closet.get_text("@putportalto"))
            elif self.portaling == 2:
                destspot = self.touched
                origspot = self.origspot
                del self.origspot
                origplace = origspot.place
                destplace = destspot.place
                skeleton = self.board.closet.skeleton[u"portal"][
                    unicode(self.board)]
                if unicode(origplace) not in skeleton:
                    skeleton[unicode(origplace)] = {}
                if unicode(destplace) not in skeleton[unicode(origplace)]:
                    skeleton[unicode(origplace)][unicode(destplace)] = []
                branch = self.board.closet.branch
                tick = self.board.closet.tick
                skel = skeleton[unicode(origplace)][unicode(destplace)]
                if branch not in skel:
                    skel[branch] = []
                if tick not in skel[branch]:
                    skel[branch][tick] = Portal.bonetype(
                        dimension=unicode(self.board),
                        origin=unicode(origplace),
                        destination=unicode(destplace),
                        branch=branch,
                        tick_from=tick)
                port = Portal(self.board.closet,
                              self.board.dimension,
                              origplace, destplace)
                arrow = Arrow(board=self.board, portal=port)
                self.board.arrowdict[unicode(port)] = arrow
                self.board.content.remove_widget(origspot)
                self.board.content.add_widget(arrow)
                self.board.content.add_widget(origspot)
                self.portaling = 0
                self.touched = None
                self.dismiss_prompt()
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
        if len(swatches) < 1:
            return
        self.display_prompt(self.board.closet.get_text("@putplace"))
        Clock.schedule_once(self.dismiss_prompt, 5)
        place = self.board.closet.make_generic_place(self.board.dimension)
        closet = self.board.closet
        branch = closet.branch
        tick = closet.tick
        skeleton = closet.skeleton
        dimn = unicode(self.board)
        placen = unicode(place)
        for tab in (u"spot_img", u"spot_interactive", u"spot_coords"):
            if tab not in skeleton:
                skeleton[tab] = {}
            if dimn not in skeleton[tab]:
                skeleton[tab][dimn] = {}
            if placen not in skeleton[tab][dimn]:
                skeleton[tab][dimn][placen] = []
        if branch not in skeleton[u"spot_interactive"][dimn][placen]:
            skeleton[u"spot_interactive"][dimn][placen][branch] = []
        if branch not in skeleton[u"spot_coords"][dimn][placen]:
            skeleton[u"spot_coords"][dimn][placen][branch] = []
        i = 0
        for swatch in swatches:
            if i not in skeleton[u"spot_img"][dimn][placen]:
                skeleton[u"spot_img"][dimn][placen][i] = []
            if branch not in skeleton[u"spot_img"][dimn][placen][i]:
                skeleton[u"spot_img"][dimn][placen][i][branch] = []
            skeleton[u"spot_img"][dimn][placen][i][branch][
                tick] = Spot.bonetypes.spot_img(
                dimension=dimn,
                place=placen,
                layer=i,
                branch=branch,
                tick_from=tick,
                img=swatch.text,
                off_x=0,
                off_y=0)
            i += 1
        skeleton[u"spot_interactive"][dimn][placen][branch][
            tick] = Spot.bonetypes.spot_interactive(
            dimension=dimn,
            place=placen,
            branch=branch,
            tick_from=tick,
            tick_to=None)
        skeleton[u"spot_coords"][dimn][placen][branch][
            tick] = Spot.bonetypes.spot_coords(
            dimension=dimn,
            place=placen,
            branch=branch,
            tick_from=tick,
            x=min((
                (0.1 * self.width) + self.board.scroll_x *
                self.board.viewport_size[0],
                self.board.viewport_size[0] - 32)),
            y=min((
                (0.9 * self.height) + self.board.scroll_y *
                self.board.viewport_size[1],
                self.board.viewport_size[1] - 32)))
        spot = Spot(board=self.board, place=place)
        self.add_widget(spot)
        self.dismiss_popup()

    def new_pawn_with_swatches(self, swatches):
        """Given some iterable of Swatch widgets, make a dummy pawn, prompt
the user to place it, and dismiss the popup."""
        self.display_prompt(self.board.closet.get_text("@putthing"))
        self.add_widget(DummyPawn(
            board=self.board,
            imgnames=[swatch.text for swatch in swatches],
            callback=self.dismiss_prompt,
            pos=(self.width * 0.1, self.height * 0.9)))
        self.dismiss_popup()

    def show_spot_picker(self, categories):
        """Show a SwatchBox for the given tags. The chosen swatches will be
        used to build a Spot later.

        """
        cattexlst = [
            (cat, sorted(self.board.closet.textagdict[cat.strip("!?")]))
            for cat in categories]
        dialog = PickImgDialog(
            set_imgs=self.new_spot_with_swatches,
            cancel=self.dismiss_popup)
        dialog.ids.picker.closet = self.board.closet
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
        cattexlst = [
            (cat, sorted(self.board.closet.textagdict[cat]))
            for cat in categories]
        dialog = PickImgDialog(
            set_imgs=self.new_pawn_with_swatches,
            cancel=self.dismiss_popup)
        dialog.ids.picker.closet = self.board.closet
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
        self.display_prompt(self.board.closet.get_text("@putportalfrom"))
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
    dimension_name = StringProperty()
    character_name = StringProperty()
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
            else:
                try:
                    conn.cursor().execute("SELECT * FROM game;")
                    conn.close()
                except DatabaseError:
                    exit("The database contains data that does not "
                         "conform to the expected schema.")
        except IOError:
            closet.mkdb(self.dbfn, __path__[-1])
        self.closet = closet.load_closet(
            self.dbfn, self.lise_path, self.lang, True)
        self.closet.load_img_metadata()
        self.closet.uptick_skel()
        self.updater = Clock.schedule_interval(self.closet.update, 0.1)
        menu = self.closet.load_menu(self.menu_name)
        board = self.closet.load_board(self.dimension_name)
        charsheet = self.closet.load_charsheet(self.character_name)
        prompt = CueCard()
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
