from functools import partial
from kivy.properties import(
    DictProperty,
    ObjectProperty,
    OptionProperty,
    ListProperty,
    BooleanProperty,
    BoundedNumericProperty,
    NumericProperty,
    StringProperty
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
from kivy.clock import Clock
from kivy.lang import Builder

from ELiDE.util import trigger


class CalendarWidget(RecycleDataViewBehavior, Widget):
    """Base class for widgets within a Calendar

    Shows the value of its ``key`` at a particular ``turn``, and sets
    it at that turn if the value changes.

    """
    turn = NumericProperty()
    """What turn I'm displaying the stat's value for"""
    key = ObjectProperty()
    """The key to set in the entity"""
    value = ObjectProperty(allownone=True)
    """The value you want to set the key to"""

    def _update_disabledness(self, *args, **kwargs):
        if not self.parent:
            return
        self.disabled = self.turn < self.parent.parent.entity.engine.turn

    def _trigger_update_disabledness(self, *args, **kwargs):
        if hasattr(self, '_scheduled_update_disabledness'):
            Clock.unschedule(self._scheduled_update_disabledness)
        self._scheduled_update_disabledness = Clock.schedule_once(self._update_disabledness)

    def _set_value(self):
        entity = self.parent.parent.entity
        entity = getattr(entity, 'stat', entity)
        entity[self.key] = self.value

    def on_value(self, *args):
        # do I want to do some validation at this point?
        # Maybe I should validate on the proxy objects and catch that in Calendar,
        # display an error message?
        if not self.parent:
            return
        calendar = self.parent.parent
        my_dict = calendar.idx[(self.turn, self.key)]
        entity = calendar.entity
        update_mode = calendar.update_mode
        if my_dict['value'] != self.value:
            my_dict['value'] = self.value
            if update_mode == 'batch':
                calendar.changed = True
            elif update_mode == 'present':
                if self.turn == entity.engine.turn:
                    self._set_value()
                else:
                    calendar.changed = True
            else:
                eng = entity.engine
                now = eng.turn
                if now == self.turn:
                    self._set_value()
                else:
                    eng.turn = self.turn
                    self._set_value()
                    eng.turn = now

    def on_parent(self, *args):
        if not self.parent:
            return
        self._trigger_update_disabledness()
        self.parent.parent.entity.engine.time.connect(self._trigger_update_disabledness)


class CalendarLabel(CalendarWidget, Label):

    def __init__(self, **kwargs):
        if 'text' not in kwargs or not kwargs['text']:
            kwargs['text'] = ''
        super().__init__(**kwargs)


class CalendarSlider(Slider, CalendarWidget):
    pass


class CalendarTextInput(CalendarWidget, TextInput):

    def _parse_text(self, *args):
        print('_parse_text')
        from ast import literal_eval
        try:
            v = literal_eval(self.text)
        except (TypeError, ValueError, SyntaxError):
            v = self.text
        self.value = self.hint_text = v
        self.text = ''
    _trigger_parse_text = trigger(_parse_text)


class CalendarOptionButton(CalendarWidget, Button):
    options = ListProperty()
    modalview = ObjectProperty()
    cols = BoundedNumericProperty(1, min=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._make_modalview()
        self._update_modalview()
        self.bind(cols=self._make_modalview)
        self.bind(options=self._update_modalview)
        self.bind(on_release=self.modalview.open)

    def _make_modalview(self, *args):
        if not self.modalview:
            self.modalview = ModalView()
        if self.modalview.children:
            container = self.modalview.children[0]
        else:
            container = GridLayout(cols=self.cols)
            self.modalview.add_widget(container)
        container.size = container.minimum_size
        self._update_modalview()

    def _update_modalview(self, *args):
        if not self.modalview:
            Clock.schedule_once(self.on_options, 0)
            return
        if not self.modalview.children:
            container = GridLayout(cols=self.cols)
            self.modalview.add_widget(container)
        else:
            container = self.modalview.children[0]
        for option in self.options:
            if type(option) is tuple:
                text, value = option
                container.add_widget(Button(
                    size_hint_y=None, height=30,
                    text=text, on_release=partial(self._set_value_and_close, text)))
            else:
                container.add_widget(Button(
                    text=str(option), on_release=partial(
                    self._set_value_and_close, str(option)),
                    size_hint_y=None, height=30))
        container.size = container.minimum_size

    def _set_value_and_close(self, val, *args):
        self.value = val
        self.modalview.dismiss()


class CalendarToggleButton(CalendarWidget, ToggleButton):
    index = None
    true_text = StringProperty('True')
    false_text = StringProperty('False')

    def on_state(self, *args):
        self.value = self.state == 'down'
        self.text = self.true_text if self.value else self.false_text


class CalendarMenuLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class AbstractCalendar(RecycleView):
    _control2wid = {
        'slider': 'CalendarSlider',
        'togglebutton': 'CalendarToggleButton',
        'textinput': 'CalendarTextInput',
        'option': 'CalendarOptionButton'
    }
    cols = NumericProperty(1)
    """Number of columns to display, default 1"""
    entity = ObjectProperty()
    """The LiSE proxy object to display the stats of"""
    idx = DictProperty()
    """Dictionary mapping ``key, turn`` pairs to their widgets"""
    changed = BooleanProperty(False)
    """Whether there are changes yet to be committed to the LiSE core"""
    update_mode = OptionProperty('batch', options=['batch', 'present', 'all'])
    """How to go about submitting changes to the LiSE core. Options:
    
    * ``'batch'`` (default): don't submit changes automatically. You have to call
    ``get_track`` and apply the changes using the ``LiSE.handle`` method
    ``apply_choices``, eg.
        ``engine_proxy.handle('apply_choices', choices=calendar.get_track())``
    * ``'present'``: immediately apply changes that affect the current turn,
    possibly wiping out anything in the future -- so you still have to use
    ``get_track`` and ``apply_choices``. However, if you're using a calendar
    in the same interface as another control widget for the same stat,
    ``'present'`` will ensure that the two widgets always display the same value.
    * ``'all'``: apply every change immediately to the LiSE core. Should only be
    used when the LiSE core is in planning mode.
    
    """
    headers = BooleanProperty(True)
    """Whether to display the name of the stat above its column, default ``True``"""
    turn_labels = BooleanProperty(True)
    """Whether to display the turn of the value before its row, default ``True``"""
    turn_label_transformer = ObjectProperty(str)
    """A function taking the turn number and returning a string to represent it
    
    Defaults to ``str``, but you might use this to display eg. the day of the
    week instead.
    
    """

    def on_data(self, *args):
        idx = self.idx
        for item in self.data:
            if 'key' in item and 'turn' in item:
                idx[(item['turn'], item['key'])] = item

    def get_track(self):
        """Get a dictionary that can be used to submit my changes to ``LiSE.Engine.apply_choices``

        If a data dictionary does not have the key 'turn', it will not be included in the track.
        You can use this to add labels and other non-input widgets to the calendar.

        """
        changes = []
        track = {'entity': self.entity, 'changes': changes}
        if not self.data:
            return track
        for datum in self.data:
            if 'turn' in datum:
                break
        else:
            # I don't know *why* the calendar has no actionable data in it but here u go
            return track
        last = self.entity.engine.turn
        accumulator = []
        for datum in self.data:
            if 'turn' not in datum:
                continue
            trn = datum['turn']
            if trn < last:
                continue
            if trn > last:
                if trn > last + 1:
                    changes.extend([[]] * (trn - last - 1))
                changes.append(accumulator)
                accumulator = []
                last = trn
            accumulator.append((datum['key'], datum['value']))
        changes.append(accumulator)
        return track


class Agenda(AbstractCalendar):
    def from_schedule(self, schedule, start_turn=None, key=str):
        # It should be convenient to style the calendar using data from the core;
        # not sure what the API should be like
        control2wid = self._control2wid
        if start_turn is None:
            start_turn = self.entity.engine.turn
        curturn = start_turn
        endturn = curturn + len(next(iter(schedule.values())))
        data = []
        stats = sorted((stat for stat in schedule if not stat.startswith('_')), key=key)
        headers = self.headers
        turn_labels = self.turn_labels
        if headers:
            if turn_labels:
                data.append({'widget': 'Label', 'text': ''})
            for stat in stats:
                if stat.startswith('_'):
                    continue
                data.append({'widget': 'Label', 'text': str(stat), 'bold': True})
        cols = len(data)
        iters = {stat: iter(values) for (stat, values) in schedule.items()}
        for turn in range(curturn, endturn):
            if turn_labels:
                data.append({'widget': 'Label', 'text': self.turn_label_transformer(turn), 'bold': True})
            if '_config' in iters:
                config = next(iters['_config'])
            else:
                config = None
            for stat in stats:
                datum = {'key': stat, 'value': next(iters[stat]), 'turn': turn}
                if config and stat in config and 'control' in config[stat]:
                    datum.update(config[stat])
                    datum['widget'] = control2wid.get(datum.pop('control', None), 'CalendarLabel')
                    if datum['widget'] == 'CalendarToggleButton':
                        if datum['value']:
                            datum['text'] = config[stat].get('true_text', 'True')
                            datum['state'] = 'down'
                        else:
                            datum['text'] = config[stat].get('false_text', 'False')
                            datum['state'] = 'normal'
                    elif datum['widget'] == 'CalendarTextInput':
                        datum['hint_text'] = str(datum['value'])
                else:
                    datum['widget'] = 'CalendarLabel'
                data.append(datum)
        (self.cols, self.data, self.changed) = (cols, data, False)


class Calendar(AbstractCalendar):
    multicol = BooleanProperty(False)

    def from_schedule(self, schedule, start_turn=None, key=str):
        control2wid = self._control2wid
        if start_turn is None:
            start_turn = self.entity.engine.turn
        curturn = start_turn
        endturn = curturn + len(next(iter(schedule.values())))
        data = []
        stats = sorted((stat for stat in schedule if not stat.startswith('_')), key=key)
        iters = {stat: iter(values) for (stat, values) in schedule.items()}
        headers = self.headers
        turn_labels = self.turn_labels
        for turn in range(curturn, endturn):
            if turn_labels:
                data.append({'widget': 'Label', 'text': self.turn_label_transformer(turn), 'bold': True})
            if '_config' in iters:
                config = next(iters['_config'])
            else:
                config = None
            for stat in stats:
                if headers:
                    data.append({'widget': 'Label', 'text': str(stat), 'bold': True})
                datum = {'key': stat, 'value': next(iters[stat]), 'turn': turn}
                if config and stat in config and 'control' in config[stat]:
                    datum.update(config[stat])
                    datum['widget'] = control2wid.get(datum.pop('control', None), 'CalendarLabel')
                    if datum['widget'] == 'CalendarToggleButton':
                        if datum['value']:
                            datum['text'] = config[stat].get('true_text', 'True')
                            datum['state'] = 'down'
                        else:
                            datum['text'] = config[stat].get('false_text', 'False')
                            datum['state'] = 'normal'
                    elif datum['widget'] == 'CalendarTextInput':
                        datum['hint_text'] = str(datum['value'])
                else:
                    datum['widget'] = 'CalendarLabel'
                data.append(datum)
        if self.multicol:
            self.cols = endturn - curturn
        self.data = data


Builder.load_string("""
<Agenda>:
    key_viewclass: 'widget'
    RecycleGridLayout:
        cols: root.cols
        size_hint: None, None
        default_size: dp(84), dp(36)
        default_size_hint: None, None
        size: self.minimum_size
        orientation: 'tb-lr'
<Calendar>:
    turn_labels: False
    key_viewclass: 'widget'
    RecycleGridLayout:
        cols: 1
        orientation: 'lr-tb'
        default_size_hint: 1, None
        default_size: dp(84), dp(36)
        size: self.minimum_size
<CalendarLabel>:
    text: str(self.value) if self.value is not None else ''
<CalendarSlider>:
    padding: 5
    Label:
        x: root.center_x - (self.width / 2)
        y: root.center_y
        text: str(root.value)
        size: self.texture_size
<CalendarOptionButton>:
    text: str(self.value)
<CalendarTextInput>:
    multiline: False
    on_text_validate: self._trigger_parse_text()
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