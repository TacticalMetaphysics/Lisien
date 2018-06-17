# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
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
    """Mixin class for widgets that show a stat's current value.

    """
    key = ObjectProperty()
    "The stat I am about. Same as in the LiSE entity."
    value = ObjectProperty(None, allownone=True)
    "The present value of the stat."
    gett = ObjectProperty()
    "Getter function: takes stat, returns its present value."
    sett = ObjectProperty()
    "Setter function: takes stat and value, sets stat=value."
    listen = ObjectProperty()
    "Function that takes a function and calls it when the stat changes."
    unlisten = ObjectProperty()
    "Function that takes a ``listen``ed function and stops it listening."
    config = DictProperty()
    "Dictionary of some parameters for how to present myself."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(value=self._bind_value)

    def _bind_value(self, *args):
        self.bind(value=self._push)
        self.unbind(value=self._bind_value)

    def on_listen(self, *args):
        self.listen(self._pull)
        self._pull()

    @trigger
    def _push(self, *args):
        self.sett(self.key, self.value)

    def _really_pull(self, *args):
        self.unbind(value=self._push)
        try:
            self.value = self.gett(self.key)
        except KeyError:
            Logger.info('StatRowListItem: {} deleted'.format(self.key))
        self.bind(value=self._push)

    def _pull(self, *args, **kwargs):
        Clock.unschedule(self._really_pull)
        Clock.schedule_once(self._really_pull, 0)


class StatRowLabel(StatRowListItem, Label):
    """Display the current value of a stat as text."""


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
        if self.text not in ('', self.value):
            self.value = self.text
        self.text = ''


class StatRowToggleButton(StatRowListItem, ToggleButton):
    def on_touch_up(self, *args):
        if self.parent is None:
            return
        if self.state == 'normal' and self.value != 0:
            self.value = 0
        elif self.state == 'down' and self.value != 1:
            self.value = 1


class StatRowSlider(StatRowListItem, Slider):
    need_set = BooleanProperty(False)

    def __init__(self, **kwargs):
        if 'text' in kwargs:
            del kwargs['text']
        kwargs['config'].setdefault('min', 0.0)
        kwargs['config'].setdefault('max', 1.0)
        kwargs['value'] = float(kwargs['gett'](kwargs['key']))
        super().__init__(**kwargs)

    def on_listen(self, *args):
        self.listen(self._pull)
        self._pull()

    def on_value(self, *args):
        self.need_set = True

    def on_touch_up(self, *args):
        if self.need_set:
            self._push()
            self.need_set = False


class StatRowListItemContainer(BoxLayout):
    key = ObjectProperty()
    reg = ObjectProperty()
    unreg = ObjectProperty()
    gett = ObjectProperty()
    sett = ObjectProperty()
    listen = ObjectProperty()
    unlisten = ObjectProperty()
    config = DictProperty()
    control = OptionProperty(
        'readout', options=['readout', 'textinput', 'togglebutton', 'slider']
    )
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
            gett=self.gett,
            sett=self.sett,
            config=self.config,
            listen=self.listen,
            unlisten=self.unlisten
        )
        self.bind(
            key=self.wid.setter('key'),
            gett=self.wid.setter('gett'),
            sett=self.wid.setter('sett'),
            config=self.wid.setter('config'),
            listen=self.wid.setter('listen'),
            unlisten=self.wid.setter('unlisten')
        )
        self.add_widget(self.wid)


default_cfg = {
    'true_text': '1',
    'false_text': '0',
    'min': 0.0,
    'max': 1.0
}


class BaseStatListView(RecycleView):
    mirror = DictProperty({})
    proxy = ObjectProperty()
    engine = ObjectProperty()
    app = ObjectProperty()

    def __init__(self, **kwargs):
        self._listeners = {}
        self.bind(
            proxy=self.refresh_mirror,
            mirror=self._trigger_upd_data
        )
        super().__init__(**kwargs)

    def on_app(self, *args):
        self.app.bind(
            branch=self.refresh_mirror,
            turn=self.refresh_mirror,
            tick=self.refresh_mirror
        )

    def del_key(self, k):
        if k not in self.mirror:
            raise KeyError
        del self.proxy[k]
        del self.mirror[k]
        if '_control' in self.proxy and k in self.proxy['_control']:
            del self.proxy['_control'][k]
            del self.mirror['_control'][k]
        else:
            assert '_control' not in self.mirror or k not in self.mirror['_control']
        if '_config' in self.proxy and k in self.proxy['_config']:
            del self.proxy['_config'][k]
            del self.mirror['_config'][k]
        else:
            assert '_config' not in self.mirror or k not in self.mirror['_config']

    def set_value(self, k, v):
        if self.engine is None or self.proxy is None:
            self._trigger_set_value(k, v)
            return
        if v is None:
            del self.proxy[k]
            del self.mirror[k]
        else:
            try:
                vv = self.engine.unpack(v)
            except (TypeError, ValueError):
                vv = v
            self.proxy[k] = self.mirror[k] = vv

    def _trigger_set_value(self, k, v, *args):
        todo = partial(self.set_value, k, v)
        Clock.unschedule(todo)
        Clock.schedule_once(todo, 0)

    def init_control_config(self, key):
        if key not in self.mirror['_control']:
            self.set_control(key, 'readout')
        if key not in self.mirror['_config']:
            cfgd = dict(self.mirror['_config'])
            cfgd[key] = default_cfg
            self.proxy['_config'] = cfgd
        else:
            cfgd = dict(self.mirror['_config'])
            for option in default_cfg:
                if option not in cfgd[key]:
                    cfgd[key][option] = default_cfg[option]
            self.proxy['_config'] = cfgd

    def set_control(self, key, control):
        if '_control' not in self.mirror:
            ctrld = {key: control}
        else:
            ctrld = dict(self.mirror['_control'])
            ctrld[key] = control
        self.proxy['_control'] \
            = self.mirror['_control'] \
            = ctrld

    def set_config(self, key, option, value):
        if '_config' not in self.mirror:
            newcfg = dict(default_cfg)
            newcfg[key] = {option: value}
            self.proxy['_config'] \
                = newcfg
        else:
            newcfg = dict(self.mirror['_config'])
            newcfg[option] = value
            self.proxy['_config'][key] = newcfg

    def set_configs(self, key, d):
        if '_config' in self.mirror:
            self.mirror['_config'][key] = self.proxy['_config'][key] = d
        else:
            self.mirror['_config'] = self.proxy['_config'] = {key: d}

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
        if self.proxy is None:
            return
        new = dict(self.proxy)
        if self.mirror != new:
            self.mirror = new

    def munge(self, k, v):
        return {
            'key': k,
            'reg': self._reg_widget,
            'unreg': self._unreg_widget,
            'gett': self.proxy.__getitem__,
            'sett': self.set_value,
            'listen': self.proxy.connect,
            'unlisten': self.proxy.disconnect,
            'control': self.mirror.get('_control', {}).get(k, 'readout'),
            'config': self.mirror.get('_config', {}).get(k, default_cfg)
        }

    def upd_data(self, *args):
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


class StatListView(BaseStatListView):
    pass


Builder.load_string(
    """
<StatRowListItem>:
    height: 30
<StatRowLabel>:
    text: str(self.value)
<StatRowTextInput>:
    hint_text: str(self.value)
    multiline: False
<StatRowToggleButton>:
    text: self.config['true_text'] if self.value else self.config['false_text']
    state: 'down' if self.value else 'normal'
<StatRowSlider>:
    min: self.config['min']
    max: self.config['max']
<StatListView>:
    viewclass: 'StatRowListItemContainer'
    app: app
    proxy: app.selected_proxy
    RecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
"""
)
