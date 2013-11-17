from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock


class LiSELayout(FloatLayout):
    menu = ObjectProperty()
    board = ObjectProperty()
    charsheet = ObjectProperty()

    def __init__(self, **kwargs):
        super(LiSELayout, self).__init__(**kwargs)
        for wid in [self.board, self.menu, self.charsheet]:
            self.add_widget(wid)

    def on_touch_down(self, touch):
        if self.menu.collide_point(touch.x, touch.y):
            touch.grab(self.menu)
        elif self.charsheet.collide_point(touch.x, touch.y):
            touch.grab(self.charsheet)
        else:
            touch.grab(self.board)


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
        layout = LiSELayout(menu=menu, board=board, charsheet=charsheet)
        return layout
