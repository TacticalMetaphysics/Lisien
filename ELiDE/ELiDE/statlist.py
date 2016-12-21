# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com
"""Grid of current values for some entity. Can be changed by the
user. Autoupdates when there's a change for any reason.

"""
from functools import partial
from kivy.properties import (
    BooleanProperty,
    DictProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    ReferenceListProperty,
    StringProperty,
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.lang import Builder
from .util import trigger


class StatRowListItem(Widget):
    key = ObjectProperty()
    value = ObjectProperty(None, allownone=True)
    reg = ObjectProperty()
    unreg = ObjectProperty()
    sett = ObjectProperty()
    config = DictProperty()

    def set_value(self, *args):
        self.sett(self.key, self.value)

    def on_parent(self, *args):
        if self.parent is None:
            self.unreg(self)
        else:
            self.reg(self)

class StatRowLabel(StatRowListItem, Label):
    pass


class StatRowTextInput(StatRowListItem, TextInput):
    def __init__(self, **kwargs):
        kwargs['multiline'] = False
        super().__init__(**kwargs)

        def lost_focus(self, *args):
            if not self.focus:
                self.upd_value()

        self.bind(
            on_enter=self.upd_value,
            on_text_validate=self.upd_value,
            on_focus=lost_focus
        )

    @trigger
    def upd_value(self, *args):
        if self.text == '':
            self.parent.value = None
        else:
            self.parent.value = self.text
        self.parent.set_value()
        self.text = ''


class StatRowToggleButton(StatRowListItem, ToggleButton):
    def on_touch_up(self, *args):
        if self.parent is None:
            return
        if (
            self.state == 'normal' and self.parent.value == 0
        ) or (
            self.state == 'down' and self.parent.value == 1
        ):
            return
        if self.state == 'normal':
            self.parent.value = 0
        else:
            self.parent.value = 1
        self.parent.set_value()


class StatRowSlider(StatRowListItem, Slider):
    need_set = BooleanProperty(False)

    def __init__(self, **kwargs):
        if 'text' in kwargs:
            del kwargs['text']
        super().__init__(**kwargs)

    def on_value(self, *args):
        self.need_set = True

    def on_touch_up(self, *args):
        if self.need_set:
            self.parent.value = self.value
            self.parent.set_value()
            self.need_set = False


class StatRowListItemContainer(BoxLayout):
    key = ObjectProperty()
    value = ObjectProperty(None, allownone=True)
    reg = ObjectProperty()
    unreg = ObjectProperty()
    sett = ObjectProperty()
    config = DictProperty()
    control = OptionProperty('readout', options=['readout', 'textinput', 'togglebutton', 'slider'])
    licls = {
        'readout': StatRowLabel,
        'textinput': StatRowTextInput,
        'togglebutton': StatRowToggleButton,
        'slider': StatRowSlider
    }

    def set_value(self, *args):
        self.sett(self.key, self.value)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            key=self.remake,
            value=self.remake,
            control=self.remake,
            config=self.remake,
            parent=self.remake
        )

    @trigger
    def remake(self, *args):
        if not hasattr(self, 'label'):
            self.label = Label(text=str(self.key))
            
            def updlabel(*args):
                self.label.text = str(self.key)
            self.bind(key=updlabel)
            self.add_widget(self.label)
        if hasattr(self, 'wid'):
            self.remove_widget(self.wid)
            del self.wid
        cls = self.licls[self.control]
        self.wid = cls(
            key=self.key,
            value=self.value,
            reg=self.reg,
            unreg=self.unreg,
            sett=self.sett,
            config=self.config
        )
        self.bind(
            key=self.wid.setter('key'),
            value=self.wid.setter('value'),
            reg=self.wid.setter('reg'),
            unreg=self.wid.setter('unreg'),
            sett=self.wid.setter('sett'),
            config=self.wid.setter('config')
        )
        self.add_widget(self.wid)


default_cfg = {
    'true_text': '1',
    'false_text': '0',
    'min': 0.0,
    'max': 1.0
}


class AbstractStatListView(RecycleView):
    control = DictProperty({})
    config = DictProperty({})
    mirror = DictProperty({})
    remote = ObjectProperty()
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)
    engine = ObjectProperty()

    def __init__(self, **kwargs):
        self._listeners = {}
        self.bind(
            branch=self.refresh_mirror,
            tick=self.refresh_mirror,
            remote=self.refresh_mirror,
            mirror=self._trigger_upd_data
        )
        super().__init__(**kwargs)

    def del_key(self, k):
        if k not in self.mirror:
            raise KeyError
        del self.remote[k]
        del self.mirror[k]
        if k in self.control:
            del self.remote['_control'][k]
            del self.control[k]
        if k in self.config:
            del self.remote['_config'][k]
            del self.config[k]

    def set_value(self, k, v):
        if self.engine is None or self.remote is None:
            self._trigger_set_value(k, v)
            return
        if v is None:
            del self.remote[k]
            del self.mirror[k]
        else:
            try:
                vv = self.engine.json_load(v)
            except (TypeError, ValueError):
                vv = v
            self.remote[k] = self.mirror[k] = vv

    def _trigger_set_value(self, k, v, *args):
        todo = partial(self.set_value, k, v)
        Clock.unschedule(todo)
        Clock.schedule_once(todo, 0)

    def init_control_config(self, key):
        if key not in self.control:
            self.set_control(key, 'readout')
        if key not in self.config:
            cfgd = dict(self.config)
            cfgd[key] = default_cfg
            self.remote['_config'] = cfgd
        else:
            cfgd = dict(self.config)
            for option in default_cfg:
                if option not in cfgd[key]:
                    cfgd[key][option] = default_cfg[option]
            self.remote['_config'] = cfgd

    def set_control(self, key, control):
        if '_control' not in self.mirror:
            ctrld = {key: control}
        else:
            ctrld = dict(self.control)
            ctrld[key] = control
        self.remote['_control'] = self.mirror['_control'] = self.control = ctrld

    def set_config(self, key, option, value):
        if '_config' not in self.mirror:
            self.remote['_config'] \
                = self.config \
                = {key: {option: value}}
        elif key in self.config:
            newcfg = dict(self.config)
            newcfg[key][option] = value
            self.remote['_config'] = self.config = newcfg
        else:
            newcfg = dict(default_cfg)
            newcfg[option] = value
            self.remote['_config'][key] = self.config = newcfg

    def set_configs(self, key, d):
        if '_config' in self.mirror:
            self.mirror['_config'][key] = self.remote['_config'][key] = d
        else:
            self.mirror['_config'] = self.remote['_config'] = {key: d}
        self.config[key] = d

    def iter_data(self):
        for (k, v) in self.mirror.items():
            if (
                not (isinstance(k, str) and k[0] == '_') and
                k not in (
                    'character',
                    'name',
                    'location',
                    'next_location',
                    'locations',
                    'arrival_time',
                    'next_arrival_time'
                )
            ):
                yield k, v

    @trigger
    def refresh_mirror(self, *args):
        Logger.debug('{}: refreshing mirror'.format(type(self)))
        if self.remote is None:
            return
        new = dict(self.remote)
        if self.mirror != new:
            self.mirror = new

    def munge(self, k, v):
        return {
            'key': k,
            'value': v,
            'reg': self._reg_widget,
            'unreg': self._unreg_widget,
            'sett': self.set_value,
            'control': self.control.get(k, 'readout'),
            'config': self.config.get(k, default_cfg)
        }

    def upd_data(self, *args):
        if (
                '_control' in self.mirror and
                self.control != self.mirror['_control']
        ):
            self.unbind(control=self._trigger_upd_data)
            self.control = dict(self.mirror['_control'])
            self.bind(control=self._trigger_upd_data)
        if (
                '_config' in self.mirror and
                self.config != self.mirror['_config']
        ):
            self.unbind(config=self._trigger_upd_data)
            self.config = dict(self.mirror['_config'])
            self.bind(config=self._trigger_upd_data)
        self.data = [self.munge(k, v) for k, v in self.iter_data()]
    _trigger_upd_data = trigger(upd_data)

    def _reg_widget(self, w, *args):
        if not self.mirror:
            Clock.schedule_once(partial(self._reg_widget, w), 0)
            return

        def listen(*args):
            if w.key not in self.mirror:
                return
            if w.value != self.mirror[w.key]:
                w.value = self.mirror[w.key]
        self._listeners[w.key] = listen
        self.bind(mirror=listen)

    def _unreg_widget(self, w):
        if w.key in self._listeners:
            self.unbind(mirror=self._listeners[w.key])


class StatListView(AbstractStatListView):
    pass


Builder.load_string(
    """
<StatRowListItem>:
    height: 30
<StatRowLabel>:
    text: str(self.value)
<StatRowTextInput>:
    text: str(self.value)
    multiline: False
<StatRowToggleButton>:
    text: self.config['true_text'] if self.value else self.config['false_text']
    state: 'down' if self.value else 'normal'
<StatRowSlider>:
    min: self.config['min']
    max: self.config['max']
<StatListView>:
    viewclass: 'StatRowListItemContainer'
    RecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
"""
)
