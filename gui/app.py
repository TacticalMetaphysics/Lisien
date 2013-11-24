from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, ListProperty, StringProperty
from kivy.clock import Clock
import os


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

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_pic_loader(self):
        content = LoadImgDialog(
            load=self.ins_tex,
            cancel=self.dismiss_popup)
        self._popup = Popup(title="Select an image file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def ins_tex(self, path, filename):
        self.board.closet.ins_texture(os.path.join(path, filename[0]), filename[0])
        self.dismiss_popup()


class LoadImgDialog(FloatLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()


from kivy.factory import Factory
Factory.register('LiSELayout', cls=LiSELayout)
Factory.register('LoadImgDialog', cls=LoadImgDialog)


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
        return LiSELayout(menus=[menu], board=board, charsheets=[charsheet])
