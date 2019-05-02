from kivy.properties import(
    ObjectProperty,
    ListProperty,
    BooleanProperty,
    BoundedNumericProperty
)
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.modalview import ModalView
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.clock import Clock


class CalendarWidget(Widget):
    key = ObjectProperty()
    """The key to set in the entity"""
    value = ObjectProperty()
    """The value you want to set the key to"""

    def on_value(self, *args):
        # do I want to do some validation at this point?
        # Maybe I should validate on the proxy objects and catch that in Calendar,
        # display an error message?
        self.parent.entity[self.key] = self.value


class CalendarDropMenuButton(CalendarWidget, Button):
    options = ListProperty()
    modalview = ObjectProperty()
    columns = BoundedNumericProperty(min=1)

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


class Calendar(GridLayout):
    entity = ObjectProperty()
    content = ListProperty()
    """For each turn, a list of ``CalendarWidget``s to put in the row for that turn, or ``None``
    
    Everything else will get filled with the result of my ``make_fill_wid`` method.
    
    Instead of a list alone, each item may be a pair of an integer and then a list, in which case
    the integer is the turn number. Turns in the past will be disabled. The items will be shown
    in whatever order you give them.
    
    """
    omit_empty = BooleanProperty(False)
    """Whether to skip rows in which you have nothing to do"""
    show_turn_labels = BooleanProperty(True)
    """Whether to put some labels on the left showing what turn a row refers to"""
    header = ListProperty()
    """Stuff to show at the top of the calendar. Strings or widgets, as many as you've got columns
    
    Make sure to account for the turn labels' column if it's present.
    
    """

    def make_turn_label(self, turn):
        return Label(text=str(turn))

    def make_fill_wid(self, turn, row, col):
        return Widget()

    def on_content(self, *args):
        self.clear_widgets()  # TODO: Incremental updates for Calendar
        add_widget = self.add_widget
        label = self.make_turn_label
        columns = self.cols
        if self.header:
            if len(self.header) != columns:
                raise ValueError("Header has {} cols, should be {}".format(len(self.header), columns))
            for i, something in enumerate(self.header):
                if isinstance(something, Widget):
                    add_widget(something)
                else:
                    add_widget(label(something))
        data_iter = iter(self.content)
        datum = next(data_iter)
        past = type(datum) is tuple
        turn = datum[0] if past else self.entity.engine.turn
        make_fill_wid = self.make_fill_wid
        for i, something in enumerate(datum):
            if something is None:
                add_widget(make_fill_wid(turn, 0, i))
            else:
                add_widget(something)

        omit_empty = self.omit_empty
        show_turn_labels = self.show_turn_labels
        wids_per_row = columns
        if show_turn_labels:
            wids_per_row -= 1
        for j, (trn, datum) in enumerate((data_iter if past else enumerate(data_iter, start=turn))):
            if not datum:
                if not omit_empty:
                    if show_turn_labels:
                        add_widget(label(trn))
                    for k in range(columns):
                        add_widget(make_fill_wid(trn, j, k))
                continue
            if show_turn_labels:
                add_widget(label(trn))
            if len(datum) != wids_per_row:
                raise ValueError("Need {} widgets, got {}".format(wids_per_row, len(datum)))
            for k, something in enumerate(datum):
                if something is None:
                    add_widget(make_fill_wid(trn, j, k))
                else:
                    if turn > trn:
                        something.disabled = True
                    add_widget(something)

    def get_track(self):
        """Get a dictionary that can be used to submit my changes to ``LiSE.Engine.apply_choices``

        You'll need to at least put the result in a list in a dictionary under the key ``'tracks'``.
        This is to support the possibility of multiple calendars, even multiple simultaneous players,
        and the payload might have to contain credentials or something like that.

        """
        changes = []
        track = {'entity': self.entity, 'changes': changes}
        data_iter = iter(self.content)
        datum = next(data_iter)
        past = type(datum) is tuple
        turn = self.entity.engine.turn
        if past:
            last = datum[0]
            for (i, row) in sorted(self.content):
                if i < turn:
                    continue
                if i > last + 1:
                    changes.extend([[]] * (i - last - 1))
                changes.append([(wid.key, wid.value) for wid in row])
                last = i
        else:
            changes.append([(wid.key, wid.value) for wid in datum])
            changes.extend([(wid.key, wid.value) for wid in row] for row in data_iter)
        return track


class CalendarToggleButton(RecycleDataViewBehavior, Button):
    index = None

    def on_touch_up(self, touch):
        if super().on_touch_up(touch):
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        self.state = 'down' if is_selected else 'normal'


class CalendarMenuLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass