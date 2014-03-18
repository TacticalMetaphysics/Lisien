from os import sep

from kivy.app import App
from kivy.clock import Clock
from kivy.properties import (
    BoundedNumericProperty,
    ObjectProperty,
    ListProperty,
    StringProperty)
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
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
    tags = ListProperty([])
    """List for use in SwatchBox"""

    def __init__(self, **kwargs):
        """Bind ``self.img`` to ``self.upd_img``"""
        super(FrobSwatch, self).__init__(**kwargs)
        self.trigger_upd_image = Clock.create_trigger(self.upd_image)
        self.bind(img=self.trigger_upd_image)

    def upd_image(self, *args):
        """Make an ``Image`` to display ``self.img`` with."""
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
        self.bind(state=self.box.upd_selection)


class TogSwatch(ToggleButton, FrobSwatch):
    pass


class SwatchBox(ScrollView):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    cols = NumericProperty()
    """Number of columns, as for ``GridLayout``"""
    closet = ObjectProperty()
    """Closet to get data from"""
    tags = ListProperty([])
    """Image tags to be used as categories. If supplied,
    ``categorized_images`` is not necessary.

    """
    categorized_images = DictProperty()
    """Lists of images, keyed by the name to use for each list
    when displaying its images to the user.

    Overrides ``tags``.

    """
    sellen = NumericProperty(0)
    selection = ListProperty([])

    def __init__(self, **kwargs):
        def finalize(*args):
            """For each category in ``cattexlst``, construct a grid of grouped
            Swatches displaying the images therein.

            """
            def wait_for_cats(*args):
                """Assign ``self.categorized_images`` based on
                ``kwargs['tags']`` if present. Otherwise,
                ``kwargs['categorized_images']`` must be
                present, so wait for it.

                """
                if (
                        'categorized_images' in kwargs
                        and not self.categorized_images):
                    Clock.schedule_once(wait_for_cats, 0)
                    return
                if 'tags' in kwargs:
                    self.categorized_images = get_categorized_images(
                        kwargs['closet'], kwargs['tags'])
                else:
                    if not self.categorized_images:
                        raise ValueError("SwatchBox requires either ``tags``_"
                                         " or ``categorized_images``")
            wait_for_cats()

            if not self.cols and self.closet and self.categorized_images:
                Clock.schedule_once(finalize, 0)
                return
            cats = GridLayout(cols=self.cols, size_hint_y=None)
            self.add_widget(cats)
            i = 0
            h = 0
            for (catname, images) in self.categorized_images:
                l = ClosetLabel(closet=self.closet,
                                stringname=catname.strip('!?'),
                                size_hint_y=None)
                cats.add_widget(l)
                cats.rows_minimum[i] = l.font_size * 2
                h += cats.rows_minimum[i]
                i += 1
                layout = StackLayout(size_hint_y=None)
                for image in images:
                    fakelabel = Label(text=image.name)
                    fakelabel.texture_update()
                    w = fakelabel.texture.size[0]
                    kwargs = {
                        'box': self,
                        'text': image.name,
                        'image': image,
                        'width': w + l.font_size * 2}
                    if catname[0] == '!':
                        swatch = TogSwatch(**kwargs)
                    elif catname[0] == '?':
                        swatch = FrobSwatch(**kwargs)
                    else:
                        kwargs['group'] = catname
                        swatch = TogSwatch(**kwargs)
                    layout.add_widget(swatch)

                    def upd_from_swatch(swatch, state):
                        """When the swatch notices it's been pressed, put it in
                        the pile."""
                        # It seems weird I'm handling the state in two places.
                        bone = self.closet.skeleton[u"img"][swatch.img_name]
                        if (
                                state == 'down' and
                                swatch.img_name not in self.pile.names):
                            self.pile.bones.append(bone)
                        elif (
                                state == 'normal' and
                                swatch.img_name in self.pile.names):
                            self.pile.bones.remove(bone)
                    swatch.bind(state=upd_from_swatch)
                layout.minimum_width = 500
                cats.add_widget(layout)
                cats.rows_minimum[i] = (len(images) / 5) * 100
                h += cats.rows_minimum[i]
                i += 1
            cats.height = h

        kwargs['orientation'] = 'vertical'
        super(SwatchBox, self).__init__(**kwargs)
        finalize()

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
                self.selection[-1].display_texture, self.selection[-1].xoff,
                self.selection[-1].yoff, self.selection[-1].stackh)
        elif lv < self.sellen:
            try:
                self.pile.pop()
            except IndexError:
                pass
        self.sellen = lv

    def undo(self, *args):
        """Put the last pressed swatch back to normal."""
        try:
            swatch = self.selection.pop()
            swatch.state = 'normal'
        except IndexError:
            pass


class MenuTextInput(TextInput):
    closet = ObjectProperty()

    def __init__(self, **kwargs):
        super(MenuTextInput, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not self.closet:
            Clock.schedule_once(self.finalize, 0)
            return
        self.rehint_registrar(self.rehint)
        self.rehint()

    def rehint(self, *args):
        self.text = ''
        self.hint_text = self.hint_getter()

    def on_focus(self, *args):
        if not self.focus:
            try:
                self.value_setter(self.text)
            except ValueError:
                pass
            self.rehint()
        super(MenuTextInput, self).on_focus(*args)

    def rehint_registrar(self, reh):
        raise NotImplementedError(
            "Abstract method")

    def hint_getter(self):
        raise NotImplementedError(
            "Abstract method")

    def value_setter(self, v):
        raise NotImplementedError(
            "Abstract method")


class MenuBranchInput(MenuTextInput):
    def rehint_registrar(self, reh):
        self.closet.register_branch_listener(reh)

    def hint_getter(self):
        return str(self.closet.branch)

    def value_setter(self, v):
        w = int(v)
        if w < 0:
            raise ValueError
        self.closet.branch = w


class MenuTickInput(MenuTextInput):
    def rehint_registrar(self, reh):
        self.closet.register_tick_listener(reh)

    def hint_getter(self):
        return str(self.closet.tick)

    def value_setter(self, v):
        w = int(v)
        if w < 0:
            raise ValueError
        self.closet.tick = w


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
    __metaclass__ = LiSEWidgetMetaclass
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
    __metaclass__ = LiSEWidgetMetaclass
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
                    'dummyarrow': TouchlessWidget()}
                self.board.arrowlayout.add_widget(
                    self.portal_d['dummyarrow'])
                self._touch = touch
                del touch.ud['spot']
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
            self.board.remove_widget(ud['dummyarrow'])
            self.dismiss_prompt()
            destspot = None
            for spot in self.board.spotlayout.children:
                if self.portal_d['origspot'] is not spot and\
                   spot.on_touch_up(touch):
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
            arrow.trigger_repoint()
        else:
            return super(LiSELayout, self).on_touch_up(touch)

    def display_prompt(self, text):
        """Put the text in the cue card"""
        self.ids.prompt.ids.l.text = text

    def dismiss_prompt(self, *args):
        """Blank out the cue card"""
        self.ids.prompt.text = ''

    def get_swatch_view(self, sections):
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
        swatch_menu_swatches = ScrollView(do_scroll_x=False)
        for (headtxt, tags) in sections:
            content = BoxLayout(orientation='vertical', size_hint_y=None)
            header = ClosetLabel(closet=self.closet, stringname=headtxt)
            content.add_widget(header)
            pallet = SpriteMenuContent(
                closet=self.app.closet,
                swatchbox=SwatchBox(
                    closet=self.app.closet,
                    tags=tags))
            content.add_widget(pallet)
            swatch_menu_swatches.add_widget(content)
        return swatch_menu_swatches

    def graphic_menu_confirm(self, cb, namebox, swatches):
        """Validate the name, compose a graphic from the selected
        images, and pass those to the callback.

        """
        if self.validate_name(namebox.text):
            swatchl = []
            for swatchbox in swatches:
                swatchl.extend(swatchbox.selection)
            graphic = self.mk_graphic_from_list(swatchl)
            return cb(namebox.text, graphic)
        else:
            old_color = namebox.background_color

            def unred(*args):
                """namebox has been turned red. turn it back."""
                namebox.background_color = old_color

            namebox.background_color = [1, 0, 0, 1]
            namebox.hint_text = _("Another Thing has that name already. "
                                  "Use a different name.")
            Clock.schedule_once(unred, 0.5)

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
            size_hint_y=None, height=34, font_size=30)
        swatches = self.get_swatch_view([('Body', 'base'),
                                         ('Clothes', 'body')])
        popcont = BoxLayout()
        popcont.add_widget(namebox)
        popcont.add_widget(swatches)
        pawnmenu = Popup(title=_('Select Thing\'s Appearance'),
                         content=popcont)
        popcont.add_widget(ConfirmOrCancel(
            confirm=lambda: self.graphic_menu_confirm(
                self.new_pawn_with_name_and_swatches, namebox, swatches),
            cancel=lambda: pawnmenu.close()))
        pawnmenu.open()
        return pawnmenu

    def show_spot_menu(self):
        """Show the menu to pick the name and graphic for a new Spot"""
        hst = unicode(self.board.host)
        if hst not in self.app.closet.skeleton[u"place"]:
            self.app.closet.skeleton[u"place"][hst] = {}

        namebox = TextInput(
            hint_text=_('Enter a unique place name'), multiline=False,
            size_hint_y=None, height=34, font_size=30)
        swatches = self.get_swatch_view([('', '?pixelcity')])
        popcont = BoxLayout()
        popcont.add_widget(namebox)
        popcont.add_widget(swatches)
        spotmenu = Popup(title=_("Select Place's Appearance"),
                         content=popcont)
        popcont.add_widget(ConfirmOrCancel(
            confirm=lambda: self.graphic_menu_confirm(
                self.new_spot_with_name_and_swatches, namebox, swatches),
            cancel=lambda: spotmenu.close()))
        spotmenu.open()
        return spotmenu

    def mk_graphic_from_img_list(self, imgl, offx=0, offy=0):
        """Make a new graphic from the list of images; return its name."""
        grafbone = self.closet.create_graphic(offx=offx, offy=offy)
        i = 0
        for img in imgl:
            self.closet.add_img_to_graphic(img.name, grafbone.name, i)
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
        screen."""
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
    __metaclass__ = LiSEWidgetMetaclass
    kv = """
<LoadImgDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        FileChooserListView:
            id: filechooser
        BoxLayout:
            size_hint_y: None
            height: 30
            ClosetButton:
                closet: root.closet
                stringname: _("Cancel")
                on_release: root.cancel()
            ClosetButton:
                closet: root.closet
                stringname: _("Load")
                on_release: root.load(filechooser.path, filechooser.selection)
    """
    load = ObjectProperty()
    cancel = ObjectProperty()


class PickImgDialog(FloatLayout):
    """Dialog for associating imgs with something, perhaps a Pawn.

In lise.kv this is given a SwatchBox with texdict=root.texdict."""
    __metaclass__ = LiSEWidgetMetaclass
    kv = """
<PickImgDialog>:
    BoxLayout:
        size: root.size
        pos: root.pos
        orientation: "vertical"
        SwatchBox:
            id: picker
            categorized_images: root.categorized_images
        BoxLayout:
            size_hint_y: None
            height: 30
            Button:
                text: _("Cancel")
                on_release: root.cancel()
            Button:
                text: _("Confirm")
                on_release: root.set_imgs(picker.selection)
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
