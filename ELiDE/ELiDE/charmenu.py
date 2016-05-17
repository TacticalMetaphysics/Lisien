# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout

from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty,
    StringProperty
)
from .board.spot import Spot
from .board.pawn import Pawn
from .board.arrow import ArrowWidget
from .util import try_load, dummynum


class CharMenu(BoxLayout):
    screen = ObjectProperty()
    app = AliasProperty(
        lambda self: self.screen.app if self.screen else None,
        lambda self, v: None,
        bind=('screen',)
    )
    engine = AliasProperty(
        lambda self: self.screen.app.engine if self.screen else None,
        lambda self, v: None,
        bind=('screen',)
    )
    revarrow = ObjectProperty(None, allownone=True)
    dummyplace = ObjectProperty()
    dummything = ObjectProperty()
    dummies = ReferenceListProperty(dummyplace, dummything)

    def delete_selection(self):
        """Delete both the selected widget and whatever it represents."""
        selection = self.app.selection
        if selection is None:
            return
        if isinstance(selection, ArrowWidget):
            self.app.selection = None
            self.screen.board.rm_arrow(
                selection.origin.name,
                selection.destination.name
            )
            selection.portal.delete()
        elif isinstance(selection, Spot):
            self.app.selection = None
            self.screen.board.rm_spot(selection.name)
            selection.remote.delete()
        else:
            assert isinstance(selection, Pawn)
            self.app.selection = None
            self.screen.board.rm_pawn(selection.name)
            selection.remote.delete()

    def toggle_chars_screen(self, *args):
        """Display or hide the list you use to switch between characters."""
        if self.app.current != 'chars':
            adapter = self.app.chars.charsview.adapter
            adapter.data = list(self.app.character)
            adapter.select_list(
                [adapter.get_view(
                    adapter.data.index(self.app.character_name)
                )]
            )
        self.app.chars.toggle()

    def toggle_rules(self, *args):
        """Display or hide the view for constructing rules out of cards."""
        if self.app.manager.current != 'rules':
            if not hasattr(self.app.selected_remote, 'rulebook'):
                self.app.rules.rulebook = self.app.character.rulebook
            else:
                self.app.rules.rulebook = self.app.selected_remote.rulebook
        self.app.rules.toggle()

    def toggle_funcs_editor(self):
        """Display or hide the text editing window for functions."""
        self.app.funcs.toggle()

    def toggle_strings_editor(self):
        self.app.strings.toggle()

    def toggle_spot_cfg(self):
        """Show the dialog where you select graphics and a name for a place,
        or hide it if already showing.

        """
        if self.app.manager.current == 'spotcfg':
            dummyplace = self.screendummyplace
            self.ids.placetab.remove_widget(dummyplace)
            dummyplace.clear()
            if self.app.spotcfg.prefix:
                dummyplace.prefix = self.app.spotcfg.prefix
                dummyplace.num = dummynum(
                    self.app.character, dummyplace.prefix
                ) + 1
            dummyplace.paths = self.app.spotcfg.imgpaths
            self.ids.placetab.add_widget(dummyplace)
        else:
            self.app.spotcfg.prefix = self.ids.dummyplace.prefix
        self.app.spotcfg.toggle()

    def toggle_pawn_cfg(self):
        """Show or hide the pop-over where you can configure the dummy pawn"""
        if self.app.current == 'pawncfg':
            dummything = self.app.dummything
            self.ids.thingtab.remove_widget(dummything)
            dummything.clear()
            if self.app.pawncfg.prefix:
                dummything.prefix = self.app.pawncfg.prefix
                dummything.num = dummynum(
                    self.app.character, dummything.prefix
                ) + 1
            if self.app.pawncfg.imgpaths:
                dummything.paths = self.app.pawncfg.imgpaths
            else:
                dummything.paths = ['atlas://rltiles/base/unseen']
            self.ids.thingtab.add_widget(dummything)
        else:
            self.app.pawncfg.prefix = self.ids.dummything.prefix
        self.app.pawncfg.toggle()

    def toggle_reciprocal(self):
        """Flip my ``reciprocal_portal`` boolean, and draw (or stop drawing)
        an extra arrow on the appropriate button to indicate the
        fact.

        """
        self.screen.reciprocal_portal = not self.screen.reciprocal_portal
        if self.screen.reciprocal_portal:
            assert(self.revarrow is None)
            self.revarrow = ArrowWidget(
                board=self.screen.board,
                origin=self.ids.emptyright,
                destination=self.ids.emptyleft
            )
            self.ids.portaladdbut.add_widget(self.revarrow)
        else:
            if hasattr(self, 'revarrow'):
                self.ids.portaladdbut.remove_widget(self.revarrow)
                self.revarrow = None

    def new_character(self, but):
        charn = try_load(
            self.app.engine.json_load,
            self.app.chars.ids.newname.text
        )
        self.app.select_character(self.app.engine.new_character(charn))
        self.app.chars.ids.newname.text = ''
        self.app.chars.charsview.adapter.data = list(
            self.engine.character.keys()
        )
        Clock.schedule_once(self.toggle_chars_screen, 0.01)

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
        text: 'Functions'
        on_press: root.toggle_funcs_editor()
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
                on_pos_up: root.screen.spot_from_dummy(self)
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
                board: root.screen.board if root.screen else None
                origin: emptyleft
                destination: emptyright
        Button:
            id: portaldirbut
            text: 'One-way' if root.screen and root.screen.reciprocal_portal else 'Two-way'
            on_press: root.toggle_reciprocal()
    BoxLayout:
        Widget:
            id: thingtab
            Dummy:
                id: dummything
                center: thingtab.center
                prefix: 'thing'
                on_pos_up: root.screen.pawn_from_dummy(self)
        Button:
            text: 'cfg'
            on_press: root.toggle_pawn_cfg()
""")
