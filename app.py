from kivy.app import App
from kivy.uix.floatlayout import FloatLayout


class LiSEApp(App):
    def __init__(self, closet, menu_name, dimension_name, character_name):
        self.closet = closet
        self.menu_name = menu_name
        self.dimension_name = dimension_name
        self.character_name = character_name
        App.__init__(self)

    def build(self):
        menu = self.closet.load_menu(self.menu_name)
        board = self.closet.load_board(self.dimension_name)
        charsheet = self.closet.load_charsheet(self.character_name)
        layout = FloatLayout()
        layout.add_widget(board)
        layout.add_widget(menu)
        layout.add_widget(charsheet)
        return layout
