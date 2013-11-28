from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.properties import (
    ObjectProperty,
    ListProperty,
    StringProperty)
from kivy.clock import Clock
from kivy.factory import Factory
from board import Pawn
from swatchbox import SwatchBox
from util import Skeleton


Factory.register('SwatchBox', cls=SwatchBox)


class CueCard(TextInput):
    """Widget that looks like TextInput but doesn't take input and can't be
clicked."""
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
    imgnames = ListProperty()
    board = ObjectProperty()
    callback = ObjectProperty()

    def __init__(self, **kwargs):
        """Fill myself with Images taken from self.textures"""
        super(DummyPawn, self).__init__(**kwargs)

        sized = False
        for imgn in self.imgnames:
            tex = self.board.closet.get_texture(imgn)
            if not sized:
                self.size = tex.size
                sized = True
            self.add_widget(Image(texture=tex, size=self.size))

    def on_touch_up(self, touch):
        """Create a real Pawn on top of the Spot that's been clicked, along
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

Mostly they can do what they like. Only one of them may grab a touch
event though: priority goes first to menus, then to charsheets, then
to the board.

    """
    menus = ListProperty()
    charsheets = ListProperty()
    board = ObjectProperty()
    prompt = ObjectProperty()

    def __init__(self, **kwargs):
        """Add board first, then menus and charsheets."""
        super(LiSELayout, self).__init__(**kwargs)
        self.add_widget(self.board)
        for menu in self.menus:
            self.add_widget(menu)
        for charsheet in self.charsheets:
            self.add_widget(charsheet)
        self.add_widget(self.prompt)

    def display_prompt(self, text):
        self.prompt.text = text

    def dismiss_prompt(self):
        self.prompt.text = ''

    def dismiss_popup(self):
        self._popup.dismiss()

    def new_pawn_with_swatches(self, swatches):
        self.display_prompt(self.board.closet.get_text("@putthing"))
        self.add_widget(DummyPawn(
            board=self.board,
            imgnames=[swatch.text for swatch in swatches],
            callback=self.dismiss_prompt,
            pos=(self.width * 0.1, self.height * 0.9)))
        self.dismiss_popup()

    def show_pic_picker(self):
        dialog = PickImgDialog(
            set_imgs=self.new_pawn_with_swatches,
            cancel=self.dismiss_popup,
            closet=self.board.closet)
        self._popup = Popup(
            title="Select some images",
            content=dialog,
            size_hint=(0.9, 0.9))
        self._popup.open()

    def ins_tex(self, path, filename):
        self.dismiss_popup()


class LoadImgDialog(FloatLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()


class PickImgDialog(FloatLayout):
    closet = ObjectProperty()
    what_for = ObjectProperty(None)
    set_imgs = ObjectProperty()
    cancel = ObjectProperty()


Factory.register('LiSELayout', cls=LiSELayout)
Factory.register('LoadImgDialog', cls=LoadImgDialog)
Factory.register('PickImgDialog', cls=PickImgDialog)


class LiSEApp(App):
    closet = ObjectProperty()
    menu_name = StringProperty()
    dimension_name = StringProperty()
    character_name = StringProperty()

    def build(self):
        self.closet.uptick_skel()
        self.updater = Clock.schedule_interval(self.closet.update, 0.1)
        menu = self.closet.load_menu(self.menu_name)
        board = self.closet.load_board(self.dimension_name)
        charsheet = self.closet.load_charsheet(self.character_name)
        prompt = CueCard(pos_hint={'x': 0.1, 'top': 1},
                         size=(400, 30),
                         size_hint=(None, None))
        return LiSELayout(
            menus=[menu],
            board=board,
            charsheets=[charsheet],
            prompt=prompt)
