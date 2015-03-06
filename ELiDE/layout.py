from functools import partial
from kivy.properties import (
    BooleanProperty,
    BoundedNumericProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.modalview import ModalView
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.logger import Logger

from LiSE.util import RedundantRuleError
from LiSE.character import CharStatCache

from .dummy import Dummy
from .configurator import PawnConfigDialog, SpotConfigDialog
from .board.arrow import Arrow, ArrowWidget
from .board.spot import Spot
from .board.pawn import Pawn
from .rulesview import RulesView
from .funcwin import FuncsEdWindow
from .statwin import StatWindow
from .stringwin import StringsEdWindow
from .charsel import CharListView

from gorm.xjson import json_load


def try_json_load(obj):
    """Attempt to interpret the argument as JSON. If this fails, return
    the object.

    """
    try:
        return json_load(obj)
    except (TypeError, ValueError):
        return obj


class ELiDELayout(FloatLayout):
    """A master layout that contains one board and some menus
    and charsheets.

    This contains three elements: a scrollview (containing the board),
    a menu, and the time control panel. This class has some support methods
    for handling interactions with the menu and the character sheet,
    but if neither of those happen, the scrollview handles touches on its
    own.

    """
    character = ObjectProperty()
    character_name = StringProperty()
    engine = ObjectProperty()
    dummies = ListProperty()
    _touch = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    grabbing = BooleanProperty(True)
    reciprocal_portal = BooleanProperty(False)
    grabbed = ObjectProperty(None, allownone=True)
    selection = ObjectProperty(None, allownone=True)
    selection_candidates = ListProperty([])
    selected_remote = ObjectProperty()
    keep_selection = BooleanProperty(False)
    engine = ObjectProperty()
    tick_results = DictProperty({})
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ListProperty(['master', 0])
    rules_per_frame = BoundedNumericProperty(10, min=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stat_cfg_layout = StatWindow(layout=self)
        self._trigger_reremote = Clock.create_trigger(self.reremote)
        self.bind(selection=self._trigger_reremote)
        self._trigger_reremote()

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
        self._strings_ed_window = StringsEdWindow(layout=self)
        self._funcs_ed_window = FuncsEdWindow(layout=self)
        self._rulesbox = BoxLayout(orientation='vertical')
        self._rulesview = RulesView(engine=self.engine)
        self._rulesbox.add_widget(self._rulesview)
        below_rulesbox = BoxLayout(size_hint_y=0.05)
        self._rulesbox.add_widget(below_rulesbox)
        self._new_rule_name = TextInput(
            hint_text='New rule name',
            write_tab=False
        )
        below_rulesbox.add_widget(self._new_rule_name)

        def new_rule(*args):
            if not self.engine.rule.db.haverule(self._new_rule_name.text):
                new = self.engine.rule.new_empty(self._new_rule_name.text)
                self._rulesview.rulebook.append(new)
                view = self._rulesview._list.adapter.get_view(
                    self._rulesview._list.adapter.data.index(new)
                )
                self._rulesview._list.adapter.select_list([view])
                self._rulesview.rule = new
            self._new_rule_name.text = ''

        self._new_rule_but = Button(
            text='+',
            on_press=new_rule
        )
        below_rulesbox.add_widget(self._new_rule_but)
        below_rulesbox.add_widget(
            Button(
                text='Close',
                on_press=self.toggle_rules_view
            )
        )

        def select_character(char):
            if char == self.character:
                return
            self.character = char
            self.character_name = char.name
            self.toggle_char_list()

        def new_character(but):
            self.toggle_char_list()
            charn = self._new_char_name.text
            self.character = self.engine.new_character(charn)
            self.character_name = charn

        self._charlist = CharListView(
            layout=self,
            set_char=select_character
        )
        self._charbox = BoxLayout(orientation='vertical')
        self._charbox.add_widget(self._charlist)
        below_charbox = BoxLayout(size_hint_y=0.05)
        self._charbox.add_widget(below_charbox)
        self._new_char_name = TextInput(
            hint_text='New character name',
            write_tab=False
        )
        below_charbox.add_widget(self._new_char_name)
        self._new_char_but = Button(
            text='+',
            on_press=new_character
        )
        below_charbox.add_widget(self._new_char_but)
        below_charbox.add_widget(
            Button(
                text='Cancel',
                on_press=self.toggle_char_list
            )
        )

        @self.engine.on_time
        def board_upd(*args):
            Clock.schedule_once(self.ids.board.update, 0)

    def toggle_char_list(self, *args):
        if hasattr(self, '_popover'):
            self._popover.remove_widget(self._charbox)
            self._popover.dismiss()
            del self._popover
        else:
            self._charlist.adapter.data = list(self.engine.character)
            self._charlist.adapter.select_list(
                [self._charlist.adapter.get_view(
                    self._charlist.adapter.data.index(self.character_name)
                )]
            )
            self._popover = ModalView()
            self._popover.add_widget(self._charbox)
            self._popover.open()

    def toggle_rules_view(self, *args):
        if hasattr(self, '_popover'):
            self._popover.remove_widget(self._rulesbox)
            self._popover.dismiss()
            del self._popover
        else:
            if (
                    self.selected_remote is None or
                    isinstance(self.selected_remote, CharStatCache)
            ):
                self._rulesview.rulebook = self.character.rulebook
            else:
                self._rulesview.rulebook = self.selected_remote.rulebook
            self._popover = ModalView()
            self._popover.add_widget(self._rulesbox)
            self._popover.open()

    def toggle_funcs_editor(self, functyp):
        if hasattr(self, '_popover'):
            self._popover.remove_widget(self._funcs_ed_window)
            self._popover.dismiss()
            del self._popover
        else:
            self._funcs_ed.store = getattr(self.engine, functyp)
            self._funcs_ed.table = functyp
            self._popover = ModalView()
            self._popover.add_widget(self._funcs_ed_window)
            self._popover.open()

    def toggle_strings_editor(self):
        if hasattr(self, '_popover'):
            self._popover.remove_widget(self._strings_ed_window)
            self._popover.dismiss()
            del self._popover
        else:
            self._popover = ModalView()
            self._popover.add_widget(self._strings_ed_window)
            self._popover.bind(
                on_size=self._strings_ed_window._trigger_layout
            )
            self._popover.open()

    def set_remote_value(self, remote, k, v):
        if v is None:
            del remote[k]
        else:
            remote[k] = try_json_load(v)

    def toggle_stat_cfg(self, *args):
        if hasattr(self, '_popover'):
            self._popover.remove_widget(self._stat_cfg_layout)
            self._popover.dismiss()
            del self._popover
        else:
            self._stat_cfg.remote = self.selected_remote
            self._stat_cfg.set_value = lambda k, v: self.set_remote_value(
                self.ids.charsheet.remote, k, v
            )
            self._popover = ModalView()
            self._popover.add_widget(self._stat_cfg_layout)
            self._popover.open()

    def reremote(self, *args):
        if 'charsheet' not in self.ids:
            Clock.schedule_once(self.reremote, 0)
            return
        self.selected_remote = self._get_selected_remote()

    def _get_selected_remote(self):
        Logger.debug('ELiDELayout: getting remote...')
        if self.character is None:
            return {}
        elif self.selection is None:
            return self.character.stat
        elif hasattr(self.selection, 'remote'):
            return self.selection.remote
        elif (
            hasattr(self.selection, 'portal') and
            self.selection.portal is not None
        ):
            return self.selection.portal
        return {}

    def set_stat(self):
        key = self._newstatkey.text
        value = self._newstatval.text
        if not (key and value):
            # TODO implement some feedback to the effect that
            # you need to enter things
            return
        self.ids.charsheet.remote[key] = value
        self._newstatkey.text = ''
        self._newstatval.text = ''

    def delete_selection(self):
        if self.selection is None:
            return
        if isinstance(self.selection, Arrow):
            arr = self.selection
            self.selection = None
            o = arr.origin.name
            d = arr.destination.name
            self.ids.board.remove_widget(arr)
            del self.ids.board.arrow[o][d]
            arr.portal.delete()
        elif isinstance(self.selection, Spot):
            spot = self.selection
            spot.canvas.clear()
            self.selection = None
            self.ids.board.remove_widget(spot)
            del self.ids.board.spot[spot.name]
            spot.remote.delete()
        else:
            assert(isinstance(self.selection, Pawn))
            pawn = self.selection
            for canvas in (
                    self.ids.board.pawnlayout.canvas.after,
                    self.ids.board.pawnlayout.canvas.before,
                    self.ids.board.pawnlayout.canvas
            ):
                if pawn.group in canvas.children:
                    canvas.remove(pawn.group)
            self.selection = None
            self.ids.board.remove_widget(pawn)
            del self.ids.board.character.thing[pawn.name]
            pawn.remote.delete()

    def toggle_spot_config(self):
        """Show the dialog where you select graphics and a name for a place,
        or hide it if already showing.

        """
        if not hasattr(self, '_spot_config'):
            return
        if hasattr(self, '_popover'):
            dummyplace = self.ids.dummyplace
            self.ids.placetab.remove_widget(dummyplace)
            dummyplace.clear()
            if self._spot_config.prefix:
                dummyplace.prefix = self._spot_config.prefix
                dummyplace.num = self._dummynum(dummyplace.prefix) + 1
            dummyplace.paths = self._spot_config.imgpaths
            self.ids.placetab.add_widget(dummyplace)
            self._popover.remove_widget(self._spot_config)
            self._popover.dismiss()
            del self._popover
        else:
            self._spot_config.prefix = self.ids.dummyplace.prefix
            self._popover = ModalView()
            self._popover.add_widget(self._spot_config)
            self._popover.open()

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
                board=self.ids.board,
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
        if self.ids.timemenu.collide_point(*touch.pos):
            self.ids.timemenu.dispatch('on_touch_down', touch)
        if self.ids.charmenu.collide_point(*touch.pos):
            self.ids.charmenu.dispatch('on_touch_down', touch)
        if self.ids.charsheet.collide_point(*touch.pos):
            self.ids.charsheet.dispatch('on_touch_down', touch)
        if self._newstatkey.collide_point(*touch.pos):
            self._newstatkey.dispatch('on_touch_down', touch)
            self.keep_selection = True
        if self._newstatval.collide_point(*touch.pos):
            self._newstatval.dispatch('on_touch_down', touch)
            self.keep_selection = True
        if self._newstatbut.collide_point(*touch.pos):
            self._newstatbut.dispatch('on_touch_down', touch)
            self.keep_selection = True
        if self.ids.cfgstatbut.collide_point(*touch.pos):
            self.ids.cfgstatbut.dispatch('on_touch_down', touch)
            self.keep_selection = True
        if self.ids.dummyplace.collide_point(*touch.pos):
            self.ids.dummyplace.dispatch('on_touch_down', touch)
            return
        if self.ids.dummything.collide_point(*touch.pos):
            self.ids.dummything.dispatch('on_touch_down', touch)
            return
        if (
                self.ids.boardview.collide_point(*touch.pos)
                and not self.selection
                and not self.selection_candidates
        ):
            # if the board itself handles the touch, let it be
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            pawns = list(self.ids.board.pawns_at(*touch.pos))
            if pawns:
                self.selection_candidates = pawns
                return True
            spots = list(self.ids.board.spots_at(*touch.pos))
            if spots:
                self.selection_candidates = spots
                if self.ids.portaladdbut.state == 'down':
                    self.origspot = self.selection_candidates.pop(0)
                    self.protodest = Dummy(
                        pos=touch.pos,
                        size=(0, 0)
                    )
                    self.ids.board.add_widget(self.protodest)
                    self.selection = self.protodest
                    # why do I need this next?
                    self.protodest.on_touch_down(touch)
                    self.protoportal = ArrowWidget(
                        origin=self.origspot,
                        destination=self.protodest
                    )
                    self.ids.board.add_widget(self.protoportal)
                    if self.reciprocal_portal:
                        self.protoportal2 = ArrowWidget(
                            destination=self.origspot,
                            origin=self.protodest
                        )
                        self.ids.board.add_widget(self.protoportal2)
                return True
            arrows = list(self.ids.board.arrows_at(*touch.pos))
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
            if r:
                self.keep_selection = True
                return True
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
                destspot = next(self.ids.board.spots_at(*touch.pos))
                orig = self.origspot.remote
                dest = destspot.remote
                if not (
                    orig.name in self.ids.board.character.portal and
                    dest.name in self.ids.board.character.portal[orig.name]
                ):
                    port = self.ids.board.character.new_portal(
                        orig.name,
                        dest.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}->{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.ids.board.add_widget(self.ids.board.make_arrow(port))
                if (
                    hasattr(self, 'protoportal2') and not (
                        orig.name in self.ids.board.character.preportal and
                        dest.name in self.ids.board.character.preportal[orig.name]
                    )
                ):
                    deport = self.ids.board.character.new_portal(
                        dest.name,
                        orig.name
                    )
                    Logger.debug(
                        "ELiDELayout: new arrow for {}<-{}".format(
                            orig.name,
                            dest.name
                        )
                    )
                    self.ids.board.add_widget(self.ids.board.make_arrow(deport))
            except StopIteration:
                pass
            self.ids.board.remove_widget(self.protoportal)
            if hasattr(self, 'protoportal2'):
                self.ids.board.remove_widget(self.protoportal2)
                del self.protoportal2
            self.ids.board.remove_widget(self.protodest)
            del self.protoportal
            del self.protodest
            touch.pop()
        Logger.debug(
            'ELiDELayout: Touch pos {}. {havesel}{havecandid}{keepsel}'.format(
                touch.pos,
                havesel='Have selection. ' if self.selection else '',
                havecandid='Have selection candidates. ' if self.selection_candidates else '',
                keepsel='Keeping selection. ' if self.keep_selection else ''
            )
        )
        if not self.keep_selection and hasattr(self.selection, 'on_touch_up'):
            self.selection.dispatch('on_touch_up', touch)
        if self.ids.timemenu.collide_point(*touch.pos):
            self.ids.timemenu.dispatch('on_touch_up', touch)
            return True
        if self.ids.charmenu.collide_point(*touch.pos):
            self.ids.charmenu.dispatch('on_touch_up', touch)
            return True
        if self.ids.charsheet.collide_point(*touch.pos):
            self.ids.charsheet.dispatch('on_touch_up', touch)
            return True
        if not self.keep_selection and self.selection_candidates:
            touch.push()
            touch.apply_transform_2d(self.ids.boardview.to_local)
            while self.selection_candidates:
                candidate = self.selection_candidates.pop(0)
                if candidate.collide_point(*touch.pos):
                    if hasattr(self.selection, 'selected'):
                        self.selection.selected = False
                    if hasattr(self.selection, '_start'):
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
                self.selection.selected = False
            self.selection = None
        self.keep_selection = False

    def _dummynum(self, name):
        """Given some name, count how many nodes there already are whose name
        starts the same.

        """
        num = 0
        for nodename in self.character.node:
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
        def renum_dummy(dummy, *args):
            dummy.num = self._dummynum(dummy.prefix) + 1

        if 'board' not in self.ids or self.character is None:
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
                self._spot_config = SpotConfigDialog(layout=self)
            dummy.num = self._dummynum(dummy.prefix) + 1
            dummy.bind(prefix=partial(renum_dummy, dummy))
            dummy._numbered = True

    def spot_from_dummy(self, dummy):
        """Create a new :class:`board.Spot` instance, along with the
        underlying :class:`LiSE.Place` instance, and give it the name,
        position, and imagery of the provided dummy.

        """
        Logger.debug(
            "ELiDELayout: Making spot from dummy {} ({} #{})".format(
                dummy.name,
                dummy.prefix,
                dummy.num
            )
        )
        (x, y) = self.ids.boardview.to_local(*dummy.pos_up)
        x /= self.ids.board.width
        y /= self.ids.board.height
        self.ids.board.spotlayout.add_widget(
            self.ids.board.make_spot(
                self.ids.board.character.new_place(
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
        for spot in self.ids.board.spotlayout.children:
            if spot.collide_widget(dummy):
                whereat = spot
                break
        else:
            return
        whereat.add_widget(
            self.ids.board.make_pawn(
                self.ids.board.character.new_thing(
                    dummy.name,
                    whereat.place.name,
                    _image_paths=dummy.paths
                )
            )
        )
        dummy.num += 1

    def arrow_from_wid(self, wid):
        for spot in self.ids.board.spotlayout.children:
            if spot.collide_widget(wid):
                whereto = spot
                break
        else:
            return
        self.ids.board.arrowlayout.add_widget(
            self.ids.board.make_arrow(
                self.ids.board.character.new_portal(
                    self.grabbed.place.name,
                    whereto.place.name,
                    reciprocal=self.reciprocal_portal
                )
            )
        )

    def timeupd(self, *args):
        Logger.debug('ELiDELayout: timeupd({})'.format(self.time))
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
