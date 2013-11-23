from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, ListProperty, StringProperty
from kivy.clock import Clock


class LiSELayout(FloatLayout):
    """A very tiny master layout that contains one board and some menus
and charsheets.

Mostly they can do what they like. Only one of them may grab a touch
event though: priority goes first to menus, then to charsheets, then
to the board.

    """
    menus = ListProperty()
    board = ObjectProperty()
    charsheets = ListProperty()

    def __init__(self, **kwargs):
        """Add board first, then menus and charsheets."""
        super(LiSELayout, self).__init__(**kwargs)
        self.add_widget(self.board)
        for menu in self.menus:
            self.add_widget(menu)
        for charsheet in self.charsheets:
            self.add_widget(charsheet)

    def on_touch_down(self, touch):
        """Poll menus, then charsheets, then the board. Once someone handles
the touch event, return."""
        for menu in self.menus:
            if menu.on_touch_down(touch):
                return
        for charsheet in self.charsheets:
            if charsheet.on_touch_down(touch):
                return
        self.board.on_touch_down(touch)


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
        layout = LiSELayout(menus=[menu], board=board, charsheets=[charsheet])
        return layout
