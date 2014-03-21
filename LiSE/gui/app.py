from os import sep

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    ObjectProperty,
    ListProperty,
    DictProperty,
    StringProperty,
    NumericProperty,
    AliasProperty
)
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup

from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.logger import Logger

from sqlite3 import connect, OperationalError

from LiSE.gui.board import (
    Pawn,
    Spot,
    Arrow)
from LiSE.gui.board.gamepiece import GamePiece
from LiSE.gui.board.arrow import get_points

from LiSE.gui.kivybits import TouchlessWidget, ClosetLabel
from LiSE.gui.charsheet import CharSheetAdder

from LiSE.util import TimestreamException
from LiSE.model import Thing
from LiSE.orm import SaveableMetaclass, mkdb, load_closet
from LiSE import __path__

_ = lambda x: x


def get_categorized_images(closet, tags):
    """Get images with the given tags, and return them in a dict keyed by
    those tags."""
    r = {}
    for tag in tags:
        r[tag] = closet.get_imgs_tagged([tag]).values()
    return r


class BoardView(ScrollView):
    board = ObjectProperty()

    def _set_scroll_x(self, x):
        self.board.bone = self.board.bone._replace(x=x)
        self.board._trigger_set_bone()

    def _set_scroll_y(self, y):
        self.board.bone = self.board.bone._replace(y=y)
        self.board._trigger_set_bone()

    scroll_x = AliasProperty(
        lambda self: self.board.bone.x if self.board else 0,
        _set_scroll_x,
        cache=False)

    scroll_y = AliasProperty(
        lambda self: self.board.bone.y if self.board else 0,
        _set_scroll_y,
        cache=False)

    def on_touch_down(self, touch):
        for preemptor in 'menu', 'charsheet', 'portaling':
            if preemptor in touch.ud:
                self.do_scroll_x = self.do_scroll_y = False
        if self.do_scroll_x:
            self.do_scroll_x = self.do_scroll_y = (
                not self.board.pawnlayout.on_touch_down(touch))
        if self.do_scroll_x:
            self.do_scroll_x = self.do_scroll_y = (
                not self.board.spotlayout.on_touch_down(touch))
        if self.board.on_touch_down(touch):
            return True
        return super(BoardView, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        self.do_scroll_x = self.do_scroll_y = True
        return super(BoardView, self).on_touch_up(touch)


class FrobSwatch(Button):
    """A :class:`Button` that contains both an :class:`Image` and
    some text."""
    box = ObjectProperty()
    """The :class:`SwatchBox` that I belong to."""
    img = ObjectProperty()
    """Image to show"""

    def __init__(self, **kwargs):
        """Bind ``self.img`` to ``self.upd_img``"""
        super(FrobSwatch, self).__init__(**kwargs)
        self.trigger_upd_image = Clock.create_trigger(self.upd_image)
        self.bind(img=self.trigger_upd_image)

    def upd_image(self, *args):
        """Make an ``Image`` to display ``self.img`` with."""
        Logger.debug("FooSwatch: upd_image with img {}".format(
            self.img))
        if not self.img:
            return
        if not self.box:
            self.trigger_upd_image()
            return
        image = Image(
            texture=self.img.texture,
            center=self.center,
            size=self.img.size)
        self.bind(center=image.setter('center'))
        self.add_widget(image)

    def on_box(self, *args):
        """Bind the box's state to its ``upd_selection`` method"""
        Logger.debug("FooSwatch: got box {}".format(
            self.box))
        self.bind(state=self.box.upd_selection)


class TogSwatch(ToggleButton, FrobSwatch):
    pass


class SwatchBox(GridLayout):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    closet = ObjectProperty()
    """Closet to get data from"""
    tag = StringProperty()
    """Tag of the images to be displayed herein."""
    max_sel = BoundedNumericProperty(1, min=1)
    """How many swatches may the user select at once?

    When exceeded, the oldest selection will wear off.

    """
    sellen = BoundedNumericProperty(0, min=0)
    selection = ListProperty([])
    pile = ListProperty([])

    def __init__(self, **kwargs):
        """Get the imgs for ``tag``, make a swatch for each, and add them to
        me.

        """
        if 'tag' not in kwargs:
            raise ValueError("img tag required")
        super(SwatchBox, self).__init__(**kwargs)
        imgs = self.closet.get_imgs_with_tag(kwargs['tag'])
        Logger.debug("SwatchBox: tag {} has {} imgs".format(
            kwargs['tag'], len(imgs)))
        for imgn in imgs:
            img = self.closet.get_img(imgn)
            box = TogSwatch(
                box=self, img=img, size_hint=(None, None))
            self.add_widget(box)
        self.height = self.children[0].height * (
            (len(self.children) % self.cols) + 1)

    def upd_selection(self, togswatch, state):
        """Make sure self.selection has the togswatch in it if it's pressed,
        and not if it isn't."""
        if state == 'normal':
            while togswatch in self.selection:
                self.selection.remove(togswatch)
        else:
            if togswatch not in self.selection:
                self.selection.append(togswatch)

    def on_selection(self, *args):
        """Make sure the pile stays sync'd with the selection"""
        lv = len(self.selection)
        if lv > self.sellen:
            self.pile.append(
                self.selection[-1].texture, self.selection[-1].xoff,
                self.selection[-1].yoff, self.selection[-1].stackh)
        elif lv < self.sellen:
            try:
                self.pile.pop()
            except IndexError:
                pass
        self.sellen = lv
        if self.sellen > self.max_sel:
            if self.sellen != self.max_sel + 1:
                raise ValueError(
                    "Seems like you somehow selected >1 at once?")
            oldsel = self.selection.pop(0)
            oldsel.state = 'normal'
            self.sellen -= 1

    def undo(self, *args):
        """Put the last pressed swatch back to normal."""
        try:
            swatch = self.selection.pop()
            swatch.state = 'normal'
        except IndexError:
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


class DummyPawn(GamePiece):
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


class ConfirmOrCancel(BoxLayout):
    """Show a confirm button and a cancel button. 30px high"""
    cancel = ObjectProperty()
    """To be called when my cancel button is pressed"""
    confirm = ObjectProperty()
    """To be called when my confirm button is pressed"""


class SpriteMenuContent(StackLayout):
    """Menu shown when a place or thing is to be created."""
    closet = ObjectProperty()
    """Closet to make things with and get text from."""
    cancel = ObjectProperty()
    """Callable for when user presses cancel button"""
    confirm = ObjectProperty()
    """Callable for when user presses confirm button"""
    preview = ObjectProperty()
    """Optional thing to wedge between the title and the swatchbox.

    The intent is that it will show what the sprite will look like
    if you press Confirm right now.

    """
    swatchbox = ObjectProperty()
    """Swatches go in here to begin with. I'll show them surrounded by a
    titlebar and cancel/confirm buttons."""

    def __init__(self, **kwargs):
        """Do as for StackLayout, then add child widgets"""
        def finalize(*args):
            """Put myself together.

            Waits for ``self.swatchbox`` and, if provided in keywords,
            ``self.preview``

            """
            def delay_for_preview(*args):
                """If I have preview, wait for it to get assigned properly
                before triggering ``finalize``"""
                if not self.preview:
                    Clock.schedule_once(delay_for_preview, 0)
            if 'preview' in kwargs:
                delay_for_preview()

            if not self.swatchbox:
                Clock.schedule_once(self.finalize, 0)
                return

            self.confirm_cancel = ConfirmOrCancel(
                confirm=lambda: self.confirm(),
                cancel=lambda: self.cancel())
            if self.preview:
                self.add_widget(self.preview)
            self.add_widget(self.swatchbox)
            self.add_widget(self.confirm_cancel)
        super(SpriteMenuContent, self).__init__(**kwargs)

    def upd_selection(self, togswatch, state):
        """Respond to the selection or deselection of one of the options"""
        if state == 'normal':
            while togswatch in self.selection:
                self.selection.remove(togswatch)
        else:
            if togswatch not in self.selection:
                self.selection.append(togswatch)


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
    playspeed = BoundedNumericProperty(0, min=-0.999, max=0.999)
    """How fast time is advancing. If negative, it's \'advancing\' into
    the past."""

    def __init__(self, **kwargs):
        """Make a trigger for draw_arrow, then initialize as for
        FloatLayout."""
        self._trigger_draw_arrow = Clock.create_trigger(self.draw_arrow)
        super(LiSELayout, self).__init__(**kwargs)

    def handle_adbut(self, charsheet, i):
        """Open the popup for adding something to the charsheet."""
        adder = CharSheetAdder(charsheet=charsheet, insertion_point=i)
        adder.open()

    def draw_arrow(self, *args):
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
        _ = self.app.closet.get_text
        self.display_prompt(_(
            "Draw a line between the places to connect with a portal."))
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
                    'dummyarrow': TouchlessWidget()}
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
        return super(LiSELayout, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        """If I'm currently in the process of connecting two Places with a
        Portal, check whether the touch collides a Spot that isn't the
        one I started at. If so, make the Portal.

        """
        if self.portaling == 2:
            self.portaling = 0
            if touch != self._touch:
                return
            ud = self.portal_d
            ud['dummyarrow'].canvas.clear()
            self.remove_widget(ud['dummyspot'])
            self.board.remove_widget(ud['dummyarrow'])
            self.dismiss_prompt()
            destspot = None
            for spot in self.board.spotlayout.children:
                if spot.collide_point(*self.board.spotlayout.to_local(
                        *touch.pos)) and spot is not ud['origspot']:
                    destspot = spot
                    break
            if destspot is None:
                ud['dummyarrow'].canvas.clear()
                self.dismiss_prompt()
                return True
            origplace = ud['origspot'].place
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

    def get_swatch_view(self, sections, cols=5):
        """Return a ``ScrollView``, to be used in a popup, for the user to
        select a graphic for something (not necessarily a Thing) that
        they want to make.

        ``sections`` is a list of pairs, in which the first item is a
        section header, and the second is a list of tags of images to
        be swatched under that header.

        """
        hostn = unicode(self.board.host)
        if hostn not in self.app.closet.skeleton[u"place"]:
            self.app.closet.skeleton[u"place"][hostn] = {}
        swatch_menu_scrollview = ScrollView(
            do_scroll_x=False)
        swatch_menu_swatches = BoxLayout(orientation='vertical')
        swatch_menu_scrollview.add_widget(swatch_menu_swatches)
        for (headtxt, tag) in sections:
            content = BoxLayout(orientation='vertical', size_hint_y=None)
            header = ClosetLabel(closet=self.app.closet, stringname=headtxt)
            content.add_widget(header)
            pallet = SwatchBox(
                closet=self.app.closet,
                tag=tag,
                cols=cols,
                size_hint_y=None)
            content.add_widget(pallet)
            swatch_menu_swatches.add_widget(content)
        return swatch_menu_scrollview

    def graphic_menu_confirm(self, validator, confirmer, namebox, swatches_view):
        """Validate the name, compose a graphic from the selected
        images, and pass those to the callback.

        """
        vmesg = validator(namebox.text)
        # if the validator returns a message, it indicates failure
        if vmesg:
            old_color = namebox.background_color

            def unred(*args):
                """namebox has been turned red. turn it back."""
                namebox.background_color = old_color

            namebox.background_color = [1, 0, 0, 1]
            namebox.hint_text = vmesg
            Clock.schedule_once(unred, 0.5)
        else:
            swatchl = []
            for swatchbox in swatches_view.children[0].children[0].children:
                if isinstance(swatchbox, SwatchBox):
                    swatchl.extend(swatchbox.selection)
            graphic = self.mk_graphic_from_img_list([
                swatch.img for swatch in swatchl])
            return confirmer(namebox.text, graphic)

    def show_pawn_menu(self):
        """Show the menu to pick what graphic to give to the Pawn the user
        wants to make.

        """
        obsrvd = unicode(self.board.facade.observed)
        if obsrvd not in self.app.closet.skeleton[u"thing"]:
            self.app.closet.skeleton[u"thing"][obsrvd] = {}
        if obsrvd not in self.app.closet.skeleton[u"thing_loc"]:
            self.app.closet.skeleton[u"thing_loc"][obsrvd] = {}

        namebox = TextInput(
            hint_text=_('Enter a unique thing name'), multiline=False,
            size_hint_y=None, height=34, font_size=20)
        swatches = self.get_swatch_view([('Body', 'base'),
                                         ('Clothes', 'body')])
        popcont = BoxLayout(orientation='vertcal')
        popcont.add_widget(namebox)
        popcont.add_widget(swatches)
        pawnmenu = Popup(title=_('Select Thing\'s Appearance'),
                         content=popcont)
        popcont.add_widget(ConfirmOrCancel(
            confirm=lambda: self.graphic_menu_confirm(
                self.new_pawn_with_name_and_swatches, namebox, swatches),
            cancel=lambda: pawnmenu.dismiss()))
        pawnmenu.open()
        return pawnmenu

    def show_spot_menu(self):
        """Show the menu to pick the name and graphic for a new Spot"""
        def validator(text):
            if text == '':
                return _('You need to enter a name here')
            elif text in self.app.closet.skeleton[u'place'][
                    unicode(self.board.host)]:
                return _('That name is already used, choose another')
            else:
                return None

        hst = unicode(self.board.host)
        if hst not in self.app.closet.skeleton[u"place"]:
            self.app.closet.skeleton[u"place"][hst] = {}

        namebox = TextInput(
            hint_text=_('Enter a unique place name'), multiline=False,
            size_hint_y=None, height=34, font_size=20)
        swatches = self.get_swatch_view([('', 'pixel_city')])
        popcont = BoxLayout(orientation='vertical')
        popcont.add_widget(namebox)
        popcont.add_widget(swatches)
        spotmenu = Popup(title=_("Select Place's Appearance"),
                         content=popcont)

        def confirmer(name, graphic):
            spotmenu.dismiss()
            self.new_spot_with_name_and_graphic(name, graphic)

        popcont.add_widget(ConfirmOrCancel(
            confirm=lambda: self.graphic_menu_confirm(
                validator, confirmer, namebox, swatches),
            cancel=lambda: spotmenu.dismiss()))
        spotmenu.open()
        return spotmenu

    def mk_graphic_from_img_list(self, imgl, offx=0, offy=0):
        """Make a new graphic from the list of images; return its name."""
        grafbone = self.app.closet.create_graphic(offx=offx, offy=offy)
        i = 0
        for img in imgl:
            self.app.closet.add_img_to_graphic(img.name, grafbone.name, i)
            i += 1
        return grafbone.name

    def new_pawn_with_name_and_graphic(self, thing_name, graphic_name):
        """Finish positioning a newly created Pawn for a newly created Thing.

        The user has requested a new Thing, given its name, and picked
        a graphic. The Thing has been created, but not placed
        anywhere. So make a Pawn for it, but don't put it on any
        Spot. Let the user drag it there.

        """
        _ = self.app.closet.get_text
        self.display_prompt(_(
            "Drag this to a spot"))
        dummy = DummyPawn(
            thing_name=thing_name,
            board=self.board,
            graphic_name=graphic_name)

        def cb():
            """Throw out the dummy so it doesn't get in the way of the real
            Pawn"""
            self.board.pawnlayout.remove_widget(dummy)
            self.dismiss_prompt()
        dummy.callback = cb
        dummy.pos = self.center_of_view_on_board()
        self.board.pawnlayout.add_widget(dummy)

    def new_spot_with_name_and_graphic(self, place_name, graphic_name):
        """Place a new Spot for a newly created Place.

        The user has requested a new Place, given it a name, and
        picked a graphic for it. The Place has been created. It needs
        a Spot, but we don't know where it should go exactly, so we'll
        just put it in the middle of the viewport. The user may drag
        it where they like.

        """
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

    def normal_speed(self, forward=True):
        """Advance time at a sensible rate."""
        if forward:
            self.playspeed = 0.1
        else:
            self.playspeed = -0.1

    def pause(self):
        """Halt the flow of time."""
        if hasattr(self, 'updater'):
            Clock.unschedule(self.updater)

    def update(self, ticks):
        """Advance time if possible. Otherwise pause."""
        try:
            self.app.closet.time_travel_inc_tick(ticks)
        except TimestreamException:
            self.pause()

    def on_playspeed(self, *args):
        """Change the interval of updates to match the playspeed."""
        self.pause()
        if self.playspeed > 0:
            ticks = 1
            interval = self.playspeed
        elif self.playspeed < 0:
            ticks = -1
            interval = -self.playspeed
        else:
            return
        self.updater = lambda dt: self.update(ticks)
        Clock.schedule_interval(self.updater, interval)

    def go_to_branch(self, bstr):
        """Switch to a different branch of the timestream."""
        self.app.closet.time_travel(int(bstr), self.app.closet.tick)

    def go_to_tick(self, tstr):
        """Go to a different tick of the current branch of the timestream."""
        self.app.closet.time_travel(self.app.closet.branch, int(tstr))


class LoadImgDialog(FloatLayout):
    """Dialog for adding img files to the database."""
    load = ObjectProperty()
    cancel = ObjectProperty()


class PickImgDialog(FloatLayout):
    """Dialog for associating imgs with something, perhaps a Pawn.

    """
    categorized_images = ObjectProperty()
    set_imgs = ObjectProperty()
    cancel = ObjectProperty()


class LiSEApp(App):
    """LiSE, run as a standalone application, and not a library.

    As it's a Kivy app, this implements the things required of the App
    class. I also keep \"globals\" here.

    """
    closet = ObjectProperty()
    """The interface to the ORM."""
    dbfn = StringProperty(allownone=True)
    """Name of the database file to use."""
    lgettext = ObjectProperty()
    """gettext function"""
    observer_name = StringProperty()
    """Name of the Character whose view on the world we display presently."""
    observed_name = StringProperty()
    """Name of the Character we are presently observing.

    This character contains all the Portals and Things that may be
    shown to the user presently. We will not necessarily show *all* of
    these; but any that are not in the observed character will *not*
    be shown.

    """
    host_name = StringProperty()
    """Name of the Character that shows all the Places we'll display.

    This is influential: we can only display Portals that connect
    Places herein; and we can only display Things that are in those
    Portals or Places.

    """

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
        l = LiSELayout(app=self)
        from kivy.core.window import Window
        from kivy.modules import inspector
        inspector.create_inspector(Window, l)
        l.board.finalize()
        return l

    def on_pause(self):
        """Sync the database with the current state of the game."""
        self.closet.save_game()

    def stop(self, *largs):
        """Sync the database, wrap up the game, and halt."""
        self.closet.save_game()
        self.closet.end_game()
        super(LiSEApp, self).stop(*largs)
