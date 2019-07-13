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
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trigger_update_disabledness = Clock.create_trigger(self._update_disabledness)

    def _update_disabledness(self, *args):
        self.disabled = self.turn > self.parent.parent.entity.engine.turn

    def on_value(self, *args):
        # do I want to do some validation at this point?
        # Maybe I should validate on the proxy objects and catch that in Calendar,
        # display an error message?
        if not self.parent:
            return
        calendar = self.parent.parent
        my_dict = calendar.idx[(self.key, self.value)]
        if my_dict[self.key] != self.value:
            my_dict[self.key] = self.value
        if calendar.entity[self.key] != self.value:
            calendar.entity[self.key] = self.value

    def on_parent(self, *args):
        if not self.parent:
            return
        self._trigger_update_disabledness()
        self.parent.parent.entity.engine.time.connect(self._trigger_update_disabledness, weak=False)


class CalendarLabel(CalendarWidget, Label):

    def __init__(self, **kwargs):
        if 'text' not in kwargs or not kwargs['text']:
            kwargs['text'] = ''
        super().__init__(**kwargs)


class CalendarSlider(Slider, CalendarWidget):
    pass


class CalendarTextInput(CalendarWidget, TextInput):
    pass


class CalendarToggleButton(CalendarWidget, ToggleButton):
    pass


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
    _control2wid = {
        'slider': 'CalendarSlider',
        'toggle': 'CalendarToggleButton',
        'textinput': 'CalendarTextInput'
    }
    cols = NumericProperty()
    entity = ObjectProperty()
    idx = DictProperty()

    def get_track(self):
        """Get a dictionary that can be used to submit my changes to ``LiSE.Engine.apply_choices``

        You'll need to at least put the result in a list in a dictionary under the key ``'tracks'``.
        This is to support the possibility of multiple calendars, even multiple simultaneous players,
        and the payload might have to contain credentials or something like that.

        If a data dictionary does not have the key 'turn', it will not be included in the track.
        You can use this to add labels and other non-input widgets to the calendar.

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

    def from_history(self, history, start_turn=None, headers=True, turn_labels=True, key=lambda x: str(x)):
        # It should be convenient to style the calendar using data from the core;
        # not sure what the API should be like
        control2wid = self._control2wid
        if start_turn is None:
            start_turn = self.entity.engine.turn
        curturn = start_turn
        endturn = curturn + len(next(iter(history.values())))
        data = []
        stats = sorted((stat for stat in history if not stat.startswith('_')), key=key)
        if headers:
            if turn_labels:
                data.append({'widget': 'CalendarLabel', 'text': ''})
            for stat in stats:
                if stat.startswith('_'):
                    continue
                data.append({'widget': 'CalendarLabel', 'text': str(stat)})
        cols = len(data)
        iters = {stat: iter(values) for (stat, values) in history.items()}
        for turn in range(curturn, endturn):
            if turn_labels:
                data.append({'widget': 'CalendarLabel', 'text': str(turn)})
            if '_config' in iters:
                config = next(iters['_config'])
            else:
                config = None
            for stat in stats:
                datum = {'key': stat, 'value': next(iters[stat])}
                if config and stat in config and 'control' in config[stat]:
                    configstat = config[stat]
                    datum['widget'] = control2wid.get(configstat['control'], 'CalendarLabel')
                    if 'min' in configstat:
                        datum['min'] = configstat['min']
                    if 'max' in configstat:
                        datum['max'] = configstat['max']
                else:
                    datum['widget'] = 'CalendarLabel'
                data.append(datum)
        (self.cols, self.data) = (cols, data)


class CalendarScreen(Screen):
    calendar = ObjectProperty()
    entity = ObjectProperty()
    toggle = ObjectProperty()
    data = ListProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calendar = Calendar(
            entity=self.entity,
            data=self.data
        )
        self.from_history = self.calendar.from_history
        self.add_widget(self.calendar)
        self.add_widget(Button(text='Close', on_release=self.toggle, size_hint_y=0.1))
        self.bind(data=self.calendar.setter('data'))
        self.bind(entity=self.calendar.setter('entity'))

Builder.load_string("""
<Calendar>:
    key_viewclass: 'widget'
    RecycleGridLayout:
        cols: root.cols
        size_hint_y: None
        default_size: None, dp(56)
        default_size_hint: 1, None
        height: self.minimum_height
        orientation: 'horizontal'
<CalendarLabel>:
    text: str(self.value)
<CalendarSlider>:
    padding: 25
    Label:
        center_x: root.center_x
        y: root.center_y
        text: str(root.value)
        size: self.texture_size
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