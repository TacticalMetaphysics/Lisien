from kivy.properties import (
    BooleanProperty,
    BoundedNumericProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.modalview import ModalView
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.logger import Logger

from LiSE.util import RedundantRuleError

from .dummy import Dummy
from .configurator import PawnConfigDialog
from .board.arrow import ArrowWidget


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
    reciprocal_portal = BooleanProperty(False)
    grabbed = ObjectProperty(None, allownone=True)
    selection = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    keep_selection = BooleanProperty(False)
    engine = ObjectProperty()
    tick_results = DictProperty({})
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ListProperty(['master', 0])
    rules_per_frame = BoundedNumericProperty(10, min=1)

    def toggle_pawn_config(self):
        """Show or hide the pop-over where you can configure the dummy pawn"""
        if not hasattr(self, '_pawn_config'):
            return
        if hasattr(self, '_popover'):
            dummything = self.ids.dummything
            self.ids.thingtab.remove_widget(dummything)
            dummything.clear()
            if self._pawn_config.prefix:
                dummything.prefix = self._pawn_config.prefix
                dummything.num = self._dummynum(dummything.prefix) + 1
            if self._pawn_config.imgpaths:
                dummything.paths = self._pawn_config.imgpaths
            else:
                dummything.paths = ['atlas://rltiles/base/unseen']
            self.ids.thingtab.add_widget(dummything)
            self._popover.remove_widget(self._pawn_config)
            self._popover.dismiss()
            del self._popover
        else:
            self._pawn_config.prefix = self.ids.dummything.prefix
            self._popover = ModalView()
            self._popover.add_widget(self._pawn_config)
            self._popover.open()

    def toggle_reciprocal(self):
        """Flip my ``reciprocal_portal`` boolean, and draw (or stop drawing)
        an extra arrow on the appropriate button to indicate the
        fact.

        """
        self.reciprocal_portal = not self.reciprocal_portal
        if self.reciprocal_portal:
            assert(not hasattr(self, 'revarrow'))
            self.revarrow = ArrowWidget(
                board=self.board,
                origin=self.ids.emptyright,
                destination=self.ids.emptyleft
            )
            self.ids.portaladdbut.add_widget(self.revarrow)
        else:
            if hasattr(self, 'revarrow'):
                self.ids.portaladdbut.remove_widget(self.revarrow)
                del self.revarrow

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
                if self.ids.portaladdbut.state == 'down':
                    self.origspot = self.selection_candidates.pop(0)
                    self.protodest = Dummy(
                        pos=touch.pos,
                        size=(0, 0)
                    )
                    self.board.add_widget(self.protodest)
                    self.selection = self.protodest
                    # why do I need this next?
                    self.protodest.on_touch_down(touch)
                    self.protoportal = ArrowWidget(
                        origin=self.origspot,
                        destination=self.protodest
                    )
                    self.board.add_widget(self.protoportal)
                    if self.reciprocal_portal:
                        self.protoportal2 = ArrowWidget(
                            destination=self.origspot,
                            origin=self.protodest
                        )
                        self.board.add_widget(self.protoportal2)
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
        if hasattr(self, 'protodest'):
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            try:
                destspot = next(self.board.spots_at(*touch.pos))
                orig = self.origspot.remote
                dest = destspot.remote
                if not (
                    orig.name in self.board.character.portal and
                    dest.name in self.board.character.portal[orig.name]
                ):
                    port = self.board.character.new_portal(
                        orig.name,
                        dest.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}->{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.board.add_widget(self.board.make_arrow(port))
                if (
                    hasattr(self, 'protoportal2') and not (
                        orig.name in self.board.character.preportal and
                        dest.name in self.board.character.preportal[orig.name]
                    )
                ):
                    deport = self.board.character.new_portal(
                        dest.name,
                        orig.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}<-{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.board.add_widget(self.board.make_arrow(deport))
            except StopIteration:
                pass
            self.board.remove_widget(self.protoportal)
            if hasattr(self, 'protoportal2'):
                self.board.remove_widget(self.protoportal2)
                del self.protoportal2
            self.board.remove_widget(self.protodest)
            del self.protoportal
            del self.protodest
            touch.pop()
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
        if not self.keep_selection and not (
                self.ids.timemenu.collide_point(*touch.pos) or
                self.ids.charmenu.collide_point(*touch.pos) or
                self.ids.charsheet.collide_point(*touch.pos)
        ):
            if hasattr(self.selection, 'selected'):
                Logger.debug('ELiDELayout: unselecting')
                self.selection.selected = False
            self.selection = None
        self.keep_selection = False

    def _dummynum(self, name):
        num = 0
        for nodename in self.board.character.node:
            nodename = str(nodename)
            if not nodename.startswith(name):
                continue
            try:
                nodenum = int(nodename.lstrip(name))
            except ValueError:
                continue
            num = max((nodenum, num))
        return num

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
            if dummy == self.ids.dummything:
                dummy.paths = ['atlas://rltiles/base/unseen']
                self._pawn_config = PawnConfigDialog(layout=self)
            if dummy == self.ids.dummyplace:
                dummy.paths = ['orb.png']
            dummy.num = self._dummynum(dummy.prefix) + 1
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
            tick=self.timeupd,
        )

    def timeupd(self, *args):
        Logger.debug('ELiDELayout: timeupd')
        if self.engine.branch != self.branch:
            self.engine.branch = self.branch
        if self.engine.tick != self.tick:
            self.engine.tick = self.tick

        def timeprop(*args):
            if not (
                    self.engine.branch == self.branch and
                    self.engine.tick == self.tick
            ):
                Logger.debug('timeprop: cycling')
                Clock.schedule_once(timeprop, 0.001)
                return
            Logger.debug('timeprop: time {}->{}'.format(
                self.time, self.engine.time)
            )
            self.time = self.engine.time
            self.ids.board._trigger_update()

        Clock.schedule_once(timeprop, 0)

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
        except RedundantRuleError:
            self.tick += 1

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
