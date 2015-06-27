from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.modalview import ModalView

from .stringwin import StringsEdWindow
from .funcwin import FuncsEdWindow
from .statwin import StatWindow
from .spritebuilder import PawnConfigDialog, SpotConfigDialog
from .rulesview import RulesBox
from .charsview import CharactersBox
from .board.spot import Spot
from .board.pawn import Pawn
from .board.arrow import ArrowWidget
from .util import try_json_load, remote_setter, dummynum


class CharMenu(BoxLayout):
    engine = ObjectProperty()
    time = ListProperty()
    selection = ObjectProperty(None, allownone=True)
    board = ObjectProperty()
    character = ObjectProperty()
    character_name = StringProperty()
    select_character = ObjectProperty()
    selected_remote = ObjectProperty()
    dummies = ListProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    portaladdbut = ObjectProperty()
    portaldirbut = ObjectProperty()
    spot_from_dummy = ObjectProperty()
    pawn_from_dummy = ObjectProperty()
    reciprocal_portal = BooleanProperty(False)
    charlist = ObjectProperty()
    charsbox = ObjectProperty()
    rulesview = ObjectProperty()
    rulesbox = ObjectProperty()
    funcs_ed_window = ObjectProperty()
    strings_ed_window = ObjectProperty()
    stat_cfg = ObjectProperty()
    spot_cfg = ObjectProperty()
    pawn_cfg = ObjectProperty()
    revarrow = ObjectProperty(None, allownone=True)
    popover = ObjectProperty()
    popover_shown = BooleanProperty(False)

    def open_popover(self):
        self.popover.open()
        self.popover_shown = True

    def dismiss_popover(self):
        self.popover.dismiss()
        self.popover_shown = False

    def toggle_popover(self):
        if self.popover_shown:
            self.dismiss_popover()
        else:
            self.open_popover()

    def delete_selection(self):
        """Delete both the selected widget and whatever it represents."""
        if self.selection is None:
            return
        if isinstance(self.selection, ArrowWidget):
            arr = self.selection
            self.selection = None
            self.board.rm_arrow(arr.origin.name, arr.destination.name)
            arr.portal.delete()
        elif isinstance(self.selection, Spot):
            spot = self.selection
            self.selection = None
            self.board.rm_spot(spot.name)
            spot.remote.delete()
        else:
            assert(isinstance(self.selection, Pawn))
            pawn = self.selection
            self.selection = None
            self.board.rm_pawn(pawn.name)
            pawn.remote.delete()

    def toggle_stat_cfg(self, *args):
        """Display or hide the configurator where you decide how to display an
        entity's stats, or add or delete stats.

        """
        if self.popover_shown:
            self.popover.remove_widget(self.stat_cfg)
        else:
            self.stat_cfg.remote = self.selected_remote
            self.stat_cfg.set_value = remote_setter(
                self.stat_cfg.remote
            )
            self.popover.add_widget(self.stat_cfg)
        self.toggle_popover()

    def toggle_charsbox(self, *args):
        """Display or hide the list you use to switch between characters."""
        if self.popover_shown:
            self.popover.remove_widget(self.charsbox)
        else:
            adapter = self.charsbox.charsview.adapter
            adapter.data = list(self.engine.character)
            adapter.select_list(
                [adapter.get_view(
                    adapter.data.index(self.character_name)
                )]
            )
            self.popover.add_widget(self.charsbox)
        self.toggle_popover()

    def toggle_rules_view(self, *args):
        """Display or hide the view for constructing rules out of cards."""
        if self.popover_shown:
            self.popover.remove_widget(self.rulesbox)
        else:
            if not hasattr(self.selected_remote, 'rulebook'):
                self.rulesview.rulebook = self.character.rulebook
            else:
                self.rulesview.rulebook = self.selected_remote.rulebook
            self.popover.add_widget(self.rulesbox)
        self.toggle_popover()

    def toggle_funcs_editor(self, functyp):
        """Display or hide the text editing window for functions."""
        if self.popover_shown:
            self.popover.remove_widget(self.funcs_ed_window)
        else:
            self.funcs_ed_window.store = getattr(self.engine, functyp)
            self.funcs_ed_window.table = functyp
            self.popover.add_widget(self.funcs_ed_window)
        self.toggle_popover()

    def toggle_strings_editor(self):
        """Display or hide the text editing window for strings."""
        if self.popover_shown:
            self.popover.remove_widget(self.strings_ed_window)
            self.popover.unbind(
                on_size=self.strings_ed_window._trigger_layout
            )
        else:
            self.popover.add_widget(self.strings_ed_window)
            self.popover.bind(
                on_size=self.strings_ed_window._trigger_layout
            )
        self.toggle_popover()

    def toggle_spot_cfg(self):
        """Show the dialog where you select graphics and a name for a place,
        or hide it if already showing.

        """
        if self.spot_cfg is None:
            Logger.warning("CharMenu: no spot config")
            return
        if self.popover_shown:
            dummyplace = self.dummyplace
            self.ids.placetab.remove_widget(dummyplace)
            dummyplace.clear()
            if self.spot_config.prefix:
                dummyplace.prefix = self._spot_config.prefix
                dummyplace.num = dummynum(
                    self.character, dummyplace.prefix
                ) + 1
            dummyplace.paths = self._spot_config.imgpaths
            self.ids.placetab.add_widget(dummyplace)
            self.popover.remove_widget(self.spot_cfg)
        else:
            self.spot_config.prefix = self.ids.dummyplace.prefix
            self.popover.add_widget(self.spot_cfg)
        self.toggle_popover()

    def toggle_pawn_cfg(self):
        """Show or hide the pop-over where you can configure the dummy pawn"""
        if self.pawn_cfg is None:
            Logger.warning("CharMenu: no pawn config")
            return
        if self.popover_shown:
            dummything = self.dummything
            self.ids.thingtab.remove_widget(dummything)
            dummything.clear()
            if self.pawn_config.prefix:
                dummything.prefix = self.pawn_config.prefix
                dummything.num = dummynum(
                    self.character, dummything.prefix
                ) + 1
            if self.pawn_config.imgpaths:
                dummything.paths = self.pawn_config.imgpaths
            else:
                dummything.paths = ['atlas://rltiles/base/unseen']
            self.ids.thingtab.add_widget(dummything)
            self.popover.remove_widget(self.pawn_cfg)
        else:
            self.pawn_config.prefix = self.ids.dummything.prefix
            self.popover.add_widget(self.pawn_cfg)
        self.toggle_popover()

    def toggle_reciprocal(self):
        """Flip my ``reciprocal_portal`` boolean, and draw (or stop drawing)
        an extra arrow on the appropriate button to indicate the
        fact.

        """
        self.reciprocal_portal = not self.reciprocal_portal
        if self.reciprocal_portal:
            assert(self.revarrow is None)
            self.revarrow = ArrowWidget(
                board=self.board,
                origin=self.ids.emptyright,
                destination=self.ids.emptyleft
            )
            self.ids.portaladdbut.add_widget(self.revarrow)
        else:
            if hasattr(self, 'revarrow'):
                self.ids.portaladdbut.remove_widget(self.revarrow)
                self.revarrow = None

    def new_character(self, but):
        charn = try_json_load(self.charsbox.ids.newname.text)
        self.select_character(self.engine.new_character(charn))
        self.charsbox.ids.newname.text = ''
        self.charsbox.charsview.adapter.data = list(
            self.engine.character.keys()
        )
        Clock.schedule_once(self.toggle_charsbox, 0.01)

    def new_rule(self, *args):
        new_rule_name = self.rulebox.ids.rulename.text
        if new_rule_name and new_rule_name not in self.engine.rule:
            new = self.engine.rule.new_empty(new_rule_name)
            rulesview = self.rulesbox.ids.rulesview
            rulesview.rulebook.append(new)

            def select_new(*args):
                view = rulesview._list.adapter.get_view(
                    rulesview._list.adapter.data.index(new)
                )
                rulesview._list.adapter.select_list([view])
                rulesview.rule = new
            Clock.schedule_once(select_new, 0.01)
        self.rulesbox.ids.rulename.text = ''

    def on_board(self, *args):
        if hasattr(self, '_boarded'):
            return
        if None in (
                self.board,
                self.character,
                self.character_name,
                self.selected_remote,
                self.spot_from_dummy,
                self.pawn_from_dummy,
                self.select_character,
                self.selected_remote
        ):
            Clock.schedule_once(self.on_board, 0)
            return
        self.popover = ModalView()
        self.strings_ed_window = StringsEdWindow(
            engine=self.engine,
            popover=self.popover,
            dismisser=self.toggle_strings_editor
        )
        self.funcs_ed_window = FuncsEdWindow(
            table='trigger',
            store=self.engine.trigger
        )

        def dismisser(*args):
            self.toggle_funcs_editor(self.funcs_ed_window.table)
        self.funcs_ed_window.dismisser = dismisser
        self.rulesbox = RulesBox(
            engine=self.engine,
            new_rule=self.new_rule,
            toggle_rules_view=self.toggle_rules_view
        )
        self.rulesview = self.rulesbox.rulesview
        self.rulesbox.bind(rulesview=self.setter('rulesview'))
        self.charsbox = CharactersBox(
            engine=self.engine,
            toggle_charsbox=self.toggle_charsbox,
            select_character=self.select_character,
            new_character=self.new_character,
            character_name=self.character_name
        )
        self.bind(character_name=self.charsbox.setter('character_name'))
        self.stat_cfg = StatWindow(
            remote=self.selected_remote,
            toggle_stat_cfg=self.toggle_stat_cfg,
            time=self.time
        )
        self.bind(
            time=self.stat_cfg.setter('time'),
            selected_remote=self.stat_cfg.setter('remote')
        )
        self.pawn_cfg = PawnConfigDialog(
            cb=self.toggle_pawn_cfg
        )
        self.spot_cfg = SpotConfigDialog(
            cb=self.toggle_spot_cfg
        )
        self._boarded = True

    def on_dummyplace(self, *args):
        if not self.dummyplace.paths:
            self.dummyplace.paths = ["orb.png"]

    def on_dummything(self, *args):
        if not self.dummything.paths:
            self.dummything.paths = ["atlas://rltiles/base.atlas/unseen"]

Builder.load_string("""
<CharMenu>:
    orientation: 'vertical'
    dummyplace: dummyplace
    dummything: dummything
    dummies: [dummyplace, dummything]
    portaladdbut: portaladdbut
    portaldirbut: portaldirbut
    Button:
        text: 'Character'
        on_press: root.toggle_charsbox()
    Button:
        text: 'Strings'
        on_press: root.toggle_strings_editor()
    Button:
        text: 'Triggers'
        on_press: root.toggle_funcs_editor('trigger')
    Button:
        text: 'Prereqs'
        on_press: root.toggle_funcs_editor('prereq')
    Button:
        text: 'Actions'
        on_press: root.toggle_funcs_editor('action')
    Button:
        text: 'Rules'
        on_press: root.toggle_rules_view()
    Button:
        text: 'Delete'
        on_press: root.delete_selection()
    BoxLayout:
        Widget:
            id: placetab
            Dummy:
                id: dummyplace
                center: placetab.center
                prefix: 'place'
                on_pos_up: root.spot_from_dummy(self)
        Button:
            text: 'cfg'
            on_press: root.toggle_spot_cfg()
    BoxLayout:
        orientation: 'vertical'
        ToggleButton:
            id: portaladdbut
            Widget:
                id: emptyleft
                center_x: portaladdbut.x + portaladdbut.width / 3
                center_y: portaladdbut.center_y
                size: (0, 0)
            Widget:
                id: emptyright
                center_x: portaladdbut.right - portaladdbut.width / 3
                center_y: portaladdbut.center_y
                size: (0, 0)
            ArrowWidget:
                board: root.board
                origin: emptyleft
                destination: emptyright
        Button:
            id: portaldirbut
            text: 'One-way' if root.reciprocal_portal else 'Two-way'
            on_press: root.toggle_reciprocal()
    BoxLayout:
        Widget:
            id: thingtab
            Dummy:
                id: dummything
                center: thingtab.center
                prefix: 'thing'
                on_pos_up: root.pawn_from_dummy(self)
        Button:
            text: 'cfg'
            on_press: root.toggle_pawn_cfg()
""")
