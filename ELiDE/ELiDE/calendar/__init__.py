from kivy.properties import(
    DictProperty,
    ObjectProperty,
    ListProperty,
    BooleanProperty,
    BoundedNumericProperty,
    NumericProperty
)
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.modalview import ModalView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.lang import Builder


class CalendarWidget(RecycleDataViewBehavior, Widget):
    turn = NumericProperty()
    key = ObjectProperty()
    """The key to set in the entity"""
    value = ObjectProperty()
    """The value you want to set the key to"""

    def on_value(self, *args):
        # do I want to do some validation at this point?
        # Maybe I should validate on the proxy objects and catch that in Calendar,
        # display an error message?
        calendar = self.parent.parent
        my_dict = calendar.idx[(self.key, self.value)]
        if my_dict[self.key] != self.value:
            my_dict[self.key] = self.value
        if calendar.entity[self.key] != self.value:
            calendar.entity[self.key] = self.value

    def on_parent(self, *args):
        turn = self.parent.parent.entity.engine.turn
        self.disabled = self.turn >= turn


class CalendarDropMenuButton(CalendarWidget, Button):
    options = ListProperty()
    modalview = ObjectProperty()
    columns = BoundedNumericProperty(1, min=1)

    def on_columns(self, *args):
        if not self.modalview:
            self.modalview = ModalView()
        if self.modalview.children:
            container = self.modalview.children[0]
        else:
            container = GridLayout(cols=self.columns)
            self.modalview.add_widget(container)
        container.size = container.minimum_size

    def on_options(self, *args):
        if not self.modalview:
            Clock.schedule_once(self.on_options, 0)
            return
        if not self.modalview.children:
            container = GridLayout(cols=self.columns)
            self.modalview.add_widget(container)
        else:
            container = self.modalview.children[0]
        for option in self.options:
            if type(option) is tuple:
                text, value = option
                self.modalview.add_widget(Button(text=text, on_release=self.setter('value')))
            else:
                self.modalview.add_widget(Button(text=str(option), on_release=self.setter('value')))
        container.size = container.minimum_size



class CalendarToggleButton(RecycleDataViewBehavior, Button):
    index = None

    def on_touch_up(self, touch):
        if super().on_touch_up(touch):
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.state = 'down' if is_selected else 'normal'


class CalendarMenuLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class Calendar(RecycleView):
    entity = ObjectProperty()
    idx = DictProperty()

    def get_track(self):
        """Get a dictionary that can be used to submit my changes to ``LiSE.Engine.apply_choices``

        You'll need to at least put the result in a list in a dictionary under the key ``'tracks'``.
        This is to support the possibility of multiple calendars, even multiple simultaneous players,
        and the payload might have to contain credentials or something like that.

        """
        changes = []
        track = {'entity': self.entity, 'changes': changes}
        if not self.data:
            return track
        turn = self.entity.engine.turn
        last = self.data[0]
        accumulator = []
        for datum in self.data:
            if 'turn' not in datum:
                continue
            trn = datum['turn']
            if trn < turn:
                continue
            if trn > last:
                if trn > last + 1:
                    changes.extend([[]] * (trn - last - 1))
                changes.append(accumulator)
                accumulator = []
                last = trn
            accumulator.append((datum['key'], datum['value']))
        return track


class CalendarScreen(Screen):
    calendar_cls = ObjectProperty(Calendar)
    calendar = ObjectProperty()
    entity = ObjectProperty()
    toggle = ObjectProperty()
    data = ListProperty()
    omit_empty = BooleanProperty(False)
    """Whether to skip rows in which you have nothing to do"""
    show_turn_labels = BooleanProperty(True)
    """Whether to put some labels on the left showing what turn a row refers to"""
    header = ListProperty()
    """Stuff to show at the top of the calendar. Strings or widgets, as many as you've got columns
    
    Make sure to account for the turn labels' column if it's present.
    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trigger_reinit = Clock.create_trigger(self._reinit)
        self.bind(calendar_cls=self._trigger_reinit)
        self._trigger_reinit()

    def _reinit(self, *args):
        self.clear_widgets()
        cal = self.calendar = self.calendar_cls(
            entity=self.entity,
            content=self.content,
            omit_empty=self.omit_empty,
            show_turn_labels=self.show_turn_labels,
            header=self.header
        )
        self.bind(
            entity=cal.setter('entity'),
            content=cal.setter('content'),
            omit_empty=cal.setter('omit_empty'),
            show_turn_labels=cal.setter('show_turn_labels'),
            header=cal.setter('header')
        )
        self.add_widget(cal)
        self.add_widget(Button(text='Close', on_release=self.toggle))


Builder.load_string("""
<Calendar>:
    key_viewclass: 'widget'
    RecycleGridLayout:
        cols: 3
        size_hint_y: None
        default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        orientation: 'horizontal'
""")


if __name__ == '__main__':
    from kivy.app import App
    class CalendarTestApp(App):
        def build(self):
            self.wid = Calendar()
            return self.wid

        def on_start(self):
            # it seems like the calendar blanks out its data sometime after initialization
            data = []
            for i in range(7):
                for j in range(3):
                    data.append({'widget': 'Button', 'text': f'row{i} col{j}'})
            self.wid.data = data

    CalendarTestApp().run()