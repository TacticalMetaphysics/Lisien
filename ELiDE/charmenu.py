from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty,
    StringProperty
)
from kivy.uix.boxlayout import BoxLayout

from .board.spot import Spot
from .board.pawn import Pawn
from .board.arrow import ArrowWidget
from .util import try_json_load, remote_setter, dummynum


class CharMenu(BoxLayout):
    engine = ObjectProperty()
    branch = StringProperty()
    tick = NumericProperty()
    time = ReferenceListProperty(branch, tick)
    selection = ObjectProperty(None, allownone=True)
    board = ObjectProperty()
    character = ObjectProperty()
    character_name = StringProperty()
    select_character = ObjectProperty()
    selected_remote = ObjectProperty()
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    dummies = ReferenceListProperty(dummyplace, dummything)
    portaladdbut = ObjectProperty()
    portaldirbut = ObjectProperty()
    spot_from_dummy = ObjectProperty()
    pawn_from_dummy = ObjectProperty()
    reciprocal_portal = BooleanProperty(False)
    charlist = ObjectProperty()
    chars = ObjectProperty()
    rulesview = ObjectProperty()
    rulesbox = ObjectProperty()
    funcs = ObjectProperty()
    strings = ObjectProperty()
    stat_cfg = ObjectProperty()
    spot_cfg = ObjectProperty()
    pawn_cfg = ObjectProperty()
    revarrow = ObjectProperty(None, allownone=True)
    current = StringProperty()

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
        if self.current != 'statcfg':
            self.stat_cfg.remote = self.selected_remote
            self.stat_cfg.set_value = remote_setter(
                self.stat_cfg.remote
            )
        self.stat_cfg.toggle()

    def toggle_chars_screen(self, *args):
        """Display or hide the list you use to switch between characters."""
        if self.current != 'chars':
            adapter = self.chars.charsview.adapter
            adapter.data = list(self.engine.character)
            adapter.select_list(
                [adapter.get_view(
                    adapter.data.index(self.character_name)
                )]
            )
        self.chars.toggle()

    def toggle_rules(self, *args):
        """Display or hide the view for constructing rules out of cards."""
        if self.current != 'rules':
            if not hasattr(self.selected_remote, 'rulebook'):
                self.rules.rulebook = self.character.rulebook
            else:
                self.rules.rulebook = self.selected_remote.rulebook
        self.rules.toggle()

    def toggle_funcs_editor(self, functyp):
        """Display or hide the text editing window for functions."""
        if self.current != 'funcs':
            self.funcs.store = getattr(self.engine, functyp)
            self.funcs.table = functyp
        self.funcs.toggle()

    def toggle_strings_editor(self):
        """Display or hide the text editing window for strings."""
        self.strings.toggle()

    def toggle_spot_cfg(self):
        """Show the dialog where you select graphics and a name for a place,
        or hide it if already showing.

        """
        if self.spot_cfg is None:
            Logger.warning("CharMenu: no spot config")
            return
        if self.current == 'spotcfg':
            dummyplace = self.dummyplace
            self.ids.placetab.remove_widget(dummyplace)
            dummyplace.clear()
            if self.spot_cfg.prefix:
                dummyplace.prefix = self.spot_cfg.prefix
                dummyplace.num = dummynum(
                    self.character, dummyplace.prefix
                ) + 1
            dummyplace.paths = self.spot_cfg.imgpaths
            self.ids.placetab.add_widget(dummyplace)
        else:
            self.spot_cfg.prefix = self.ids.dummyplace.prefix
        self.spot_cfg.toggle()

    def toggle_pawn_cfg(self):
        """Show or hide the pop-over where you can configure the dummy pawn"""
        if self.pawn_cfg is None:
            Logger.warning("CharMenu: no pawn config")
            return
        if self.current == 'pawncfg':
            dummything = self.dummything
            self.ids.thingtab.remove_widget(dummything)
            dummything.clear()
            if self.pawn_cfg.prefix:
                dummything.prefix = self.pawn_cfg.prefix
                dummything.num = dummynum(
                    self.character, dummything.prefix
                ) + 1
            if self.pawn_cfg.imgpaths:
                dummything.paths = self.pawn_cfg.imgpaths
            else:
                dummything.paths = ['atlas://rltiles/base/unseen']
            self.ids.thingtab.add_widget(dummything)
        else:
            self.pawn_cfg.prefix = self.ids.dummything.prefix
        self.pawn_cfg.toggle()

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
        charn = try_json_load(self.chars.ids.newname.text)
        self.select_character(self.engine.new_character(charn))
        self.chars.ids.newname.text = ''
        self.chars.charsview.adapter.data = list(
            self.engine.character.keys()
        )
        Clock.schedule_once(self.toggle_chars_screen, 0.01)

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
                self.selected_remote,
                self.rules,
                self.chars,
                self.stat_cfg
        ):
            Clock.schedule_once(self.on_board, 0)
            return
        self.rulesview = self.rules.rulesview
        self.rules.bind(rulesview=self.setter('rulesview'))
        self.chars.character_name = self.character_name
        self.bind(character_name=self.chars.setter('character_name'))
        self.stat_cfg.time = self.time
        self.stat_cfg.remote = self.selected_remote
        self.bind(
            time=self.stat_cfg.setter('time'),
            selected_remote=self.stat_cfg.setter('remote')
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
    portaladdbut: portaladdbut
    portaldirbut: portaldirbut
    Button:
        text: 'Character'
        on_press: root.toggle_chars_screen()
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
        on_press: root.toggle_rules()
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
