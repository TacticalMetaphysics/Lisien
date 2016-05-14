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
    ReferenceListProperty,
    StringProperty,
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.uix.listview import (
    SelectableView,
    ListView,
    ListItemLabel,
    ListItemButton,
    CompositeListItem
)
from kivy.adapters.dictadapter import DictAdapter
from kivy.lang import Builder
from .util import trigger


class StatRowTextInput(TextInput, SelectableView):
    def __init__(self, **kwargs):
        kwargs['multiline'] = False
        super().__init__(**kwargs)

        def lost_focus(self, *args):
            if not self.focus:
                self.upd_value()

        self.bind(
            on_enter=self._upd_value,
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


class StatRowToggleButton(ToggleButtonBehavior, ListItemButton):
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


class StatRowSlider(Slider, SelectableView):
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


class StatRowListItem(CompositeListItem):
    key = ObjectProperty()
    value = ObjectProperty(None, allownone=True)
    reg = ObjectProperty()
    unreg = ObjectProperty()
    setter = ObjectProperty()

    def set_value(self, *args):
        self.setter(self.key, self.value)

    def on_parent(self, *args):
        if self.parent is None:
            self.unreg(self)
        else:
            self.reg(self)


control_cls = {
    'readout': lambda v: {
        'cls': ListItemLabel,
        'kwargs': {
            'text': str(v)
        }
    },
    'textinput': lambda v: {
        'cls': StatRowTextInput,
        'kwargs': {
            'text': str(v),
            'multiline': False
        }
    },
    'togglebutton': lambda v: {
        'cls': StatRowToggleButton,
        'kwargs': {
            'state': 'down' if v else 'normal'
        }
    },
    'slider': lambda v: {
        'cls': StatRowSlider,
        'kwargs': {'value': v}
    }
}


default_cfg = {
    'true_text': '1',
    'false_text': '0',
    'min': 0.0,
    'max': 1.0
}


class StatListView(ListView):
    control = DictProperty({})
    config = DictProperty({})
    mirror = DictProperty({})
    remote = ObjectProperty()
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)
    engine = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['adapter'] = self.get_adapter()
        self._listeners = {}
        super().__init__(**kwargs)

    def on_remote(self, *args):
        if self.remote is not None:
            self.mirror = dict(self.remote)

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
            self.remote[k] = vv
            self.mirror[k] = vv

    def _trigger_set_value(self, k, v, *args):
        todo = partial(self.set_value, k, v)
        Clock.unschedule(todo)
        Clock.schedule_once(todo, 0)

    def on_time(self, *args):
        super().on_time(*args)
        self.upd_data()

    def on_mirror(self, *args):
        self.upd_data()
        self.sortkeys()

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

    def get_adapter(self):
        return DictAdapter(
            data=self.get_data(),
            cls=StatRowListItem,
            args_converter=lambda i, kv: {
                'key': kv[0],
                'value': kv[1],
                'reg': self._reg_widget,
                'unreg': self._unreg_widget,
                'setter': self._trigger_set_value,
                'cls_dicts': self.get_cls_dicts(*kv)
            },
            selection_mode='multiple',
            allow_empty_selection=True
        )

    def set_control(self, key, control):
        if '_control' not in self.mirror:
            ctrld = {key: control}
        else:
            ctrld = dict(self.control)
            ctrld[key] = control
        self.remote['_control'] = self.control = ctrld
        self.canvas.after.clear()

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

    def get_cls_dicts(self, key, value):
        control_type = self.control.get(key, 'readout')
        cfg = self.config.get(key, {})
        keydict = {
            'cls': ListItemLabel,
            'kwargs': {'text': str(key)}
        }
        valdict = control_cls[control_type](value)
        if control_type == 'togglebutton':
            true_text = cfg.get('true_text', '1')
            false_text = cfg.get('false_text', '0')
            valdict['kwargs']['text'] = true_text if value else false_text
        elif control_type == 'slider':
            valdict['min'] = cfg.get('min', 0)
            valdict['max'] = cfg.get('max', 100)
        return [keydict, valdict]

    def get_data(self):
        return {
            k: (k, v) for (k, v) in self.mirror.items()
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
            )
        }

    @trigger
    def refresh_adapter(self, *args):
        self.adapter = self.get_adapter()

    @trigger
    def upd_data(self, *args):
        if (
                '_control' in self.mirror
        ):
            self.control = dict(self.mirror['_control'])
        if (
                '_config' in self.mirror
        ):
            self.config = dict(self.mirror['_config'])
        self.adapter.data = self.get_data()

    @trigger
    def sortkeys(self, *args):
        for key in self.mirror.keys():
            if key not in self.adapter.sorted_keys:
                self.adapter.sorted_keys = sorted(self.mirror.keys())
                return
        seen = set()
        for k in self.adapter.sorted_keys:
            if k not in seen and k not in self.mirror:
                self.adapter.sorted_keys.remove(k)
            seen.add(k)

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


Builder.load_string(
    """
<StatRowListItem>:
    height: 30
"""
)
