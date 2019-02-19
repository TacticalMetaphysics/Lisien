from kivy.properties import(
    ObjectProperty,
    ListProperty,
    StringProperty
)
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior


class Calendar(GridLayout):
    engine = ObjectProperty()
    turn_blocks = ListProperty()  # (title, [turns])
    menu_options = ListProperty() # (title, viability_check_func, select_func)


class CalendarToggleButton(RecycleDataViewBehavior, Button):
    index = None

    def on_touch_up(self, touch):
        if super().on_touch_up(touch):
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.state = 'down' if is_selected else 'normal'


class CalendarMenuLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass