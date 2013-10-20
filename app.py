from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock


class LiSEApp(App):
    closet = ObjectProperty()
    menu_name = StringProperty()
    dimension_name = StringProperty()
    character_name = StringProperty()

    def build(self):
        self.updater = Clock.schedule_interval(self.closet.update, 0.1)
        menu = self.closet.load_menu(self.menu_name)
        board = self.closet.load_board(self.dimension_name)
        charsheet = self.closet.load_charsheet(self.character_name)
        layout = FloatLayout()
        layout.add_widget(board)
        layout.add_widget(menu)
        layout.add_widget(charsheet)
        return layout
