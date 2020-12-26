# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Grid of current values for some entity. Can be changed by the
user. Autoupdates when there's a change for any reason.

"""
from functools import partial
from kivy.properties import (
    BooleanProperty,
    DictProperty,
    ObjectProperty,
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
        except TypeError:
            value = self.gett(self.key)
            Logger.warning("StatRowListItem: couldn't set value {} because it is type {}".format(
                value, type(value)
            ))
        self.bind(value=self._push)

    def _pull(self, *args, **kwargs):
        if hasattr(self, '_scheduled_pull'):
            Clock.unschedule(self._scheduled_pull)
        self._scheduled_pull = Clock.schedule_once(self._really_pull, 0)


class StatRowLabel(StatRowListItem, Label):
    """Display the current value of a stat as text."""


class StatRowTextInput(StatRowListItem, TextInput):
    """Display the current value of a stat and accept text input to change it."""
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
    """Display the current value of a stat, 0 or 1, and let the user press a button to flip it."""
    true_text = StringProperty('1')
    """String to display when the stat is true."""
    false_text = StringProperty('0')
    """String to display when the stat is false."""

    def on_touch_up(self, *args):
        if self.parent is None:
            return
        if self.state == 'normal' and self.value != 0:
            self.value = 0
        elif self.state == 'down' and self.value != 1:
            self.value = 1


class StatRowSlider(StatRowListItem, Slider):
    """Display the current value of a numeric stat and let the user slide it."""
    need_set = BooleanProperty(False)
    """Internal. Usually False, becomes True briefly when the value has changed."""

    def __init__(self, **kwargs):
        self.value = kwargs['value']
        self.min = kwargs['min']
        self.max = kwargs['max']
        super().__init__(**kwargs)

    def _bind_value(self, *args):
        pass

    def _really_pull(self, *args):
        try:
            self.value = self.gett(self.key)
        except KeyError:
            Logger.info('StatRowListItem: {} deleted'.format(self.key))
        except TypeError:
            value = self.gett(self.key)
            Logger.warning("StatRowListItem: couldn't set value {} because it is type {}".format(
                value, type(value)
            ))

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
    """The name of a stat followed by a widget representing its value.

    The widget can be

    * :class:`StatRowLabel`
    * :class:`StatRowTextInput`
    * :class:`StatRowToggleButton`
    * :class:`StatRowSlider`

    """
    key = ObjectProperty()
    """The name of the stat"""
    reg = ObjectProperty()
    unreg = ObjectProperty()
    gett = ObjectProperty()
    """Function to get the current value of stats, taking the stat name as its argument"""
    sett = ObjectProperty()
    """Function to set the current value of stats, taking args (key, value)"""
    listen = ObjectProperty()
    """Function to register a listener to a LiSE entity"""
    unlisten = ObjectProperty()
    """Function to unregister a listener from a LiSE entity"""
    config = DictProperty()
    """Dictionary describing the configuration of this stat's widget.
    
    The key 'control' has the widget type as its value, which may be
    
    * 'readout'
    * 'textinput'
    * 'togglebutton'
    * 'slider'
    
    Other keys are specific to one widget type or another.
    
    """

    def set_value(self, *args):
        """Use my ``sett`` function to set my stat (``key``) to my new ``value``.

        This doesn't need arguments, but accepts any positional arguments provided,
        so that you can use this in kvlang

        """
        self.sett(self.key, self.value)

    def __init__(self, **kwargs):
        kwargs.setdefault('orientation', 'vertical')
        super().__init__(**kwargs)
        self.remake()

    def on_key(self, *args):
        self.remake()

    def on_config(self, *args):
        self.remake()

    def on_parent(self, *args):
        self.remake()

    def remake(self, *args):
        """Replace any existing child widget with the one described by my ``config``.

        This doesn't need arguments, but accepts any positional arguments provided,
        so that you can use this in kvlang

        """
        if not self.config:
            return
        if not all((self.key, self.gett, self.sett, self.listen, self.unlisten)):
            Clock.schedule_once(self.remake, 0)
            return
        if not hasattr(self, 'label'):
            self.label = Label(text=str(self.key))

            def updlabel(*args):
                self.label.text = str(self.key)
            self.bind(key=updlabel)
            self.add_widget(self.label)
        if hasattr(self, 'wid'):
            self.remove_widget(self.wid)
            del self.wid
        control = self.config['control']
        widkwargs = {
            'key': self.key,
            'gett': self.gett,
            'sett': self.sett,
            'listen': self.listen,
            'unlisten': self.unlisten
        }
        if control == 'slider':
            cls = StatRowSlider
            try:
               widkwargs['value'] = float(self.gett(self.key))
               widkwargs['min'] = float(self.config['min'])
               widkwargs['max'] = float(self.config['max'])
            except (KeyError, ValueError):
                return
        elif control == 'togglebutton':
            cls = StatRowToggleButton
            try:
                widkwargs['true_text'] = self.config['true_text']
                widkwargs['false_text'] = self.config['false_text']
            except KeyError:
                return
        elif control == 'textinput':
            cls = StatRowTextInput
        else:
            cls = StatRowLabel
        self.wid = cls(**widkwargs)
        self.bind(
            key=self.wid.setter('key'),
            gett=self.wid.setter('gett'),
            sett=self.wid.setter('sett'),
            listen=self.wid.setter('listen'),
            unlisten=self.wid.setter('unlisten')
        )
        if control == 'slider':
            self.unbind(config=self._toggle_update_config)
            self.bind(config=self._slider_update_config)
        elif control == 'togglebutton':
            self.unbind(config=self._slider_update_config)
            self.bind(config=self._toggle_update_config)
        else:
            self.unbind(config=self._slider_update_config)
            self.unbind(config=self._toggle_update_config)
        self.add_widget(self.wid)

    @trigger
    def _slider_update_config(self, *args):
        self.wid.min = self.config['min']
        self.wid.max = self.config['max']

    @trigger
    def _toggle_update_config(self, *args):
        self.wid.true_text = self.config['true_text']
        self.wid.false_text = self.config['false_text']


default_cfg = {
    'control': 'readout',
    'true_text': '1',
    'false_text': '0',
    'min': 0.0,
    'max': 1.0
}


class BaseStatListView(RecycleView):
    """Base class for widgets showing lists of stats and their values"""
    proxy = ObjectProperty()
    """A proxy object representing a LiSE entity"""
    engine = ObjectProperty()
    """A :class:`LiSE.proxy.EngineProxy` object"""
    app = ObjectProperty()
    """The Kivy app object"""
    _scheduled_set_value = DictProperty()

    def __init__(self, **kwargs):
        self._listeners = {}
        super().__init__(**kwargs)

    def on_proxy(self, *args):
        self.proxy.connect(self._trigger_upd_data, weak=False)
        self._trigger_upd_data()

    def del_key(self, k):
        """Delete the key and any configuration for it"""
        if k not in self.proxy:
            raise KeyError
        del self.proxy[k]
        if '_config' in self.proxy and k in self.proxy['_config']:
            del self.proxy['_config'][k]

    def set_value(self, k, v):
        """Set a value on the proxy, parsing it to a useful datatype if possible"""
        from ast import literal_eval
        if self.engine is None or self.proxy is None:
            self._trigger_set_value(k, v)
            return
        if v is None:
            del self.proxy[k]
        else:
            try:
                vv = literal_eval(v)
            except (TypeError, ValueError):
                vv = v
            self.proxy[k] = vv
        if (k, v) in self._scheduled_set_value:
            del self._scheduled_set_value[k, v]

    def _trigger_set_value(self, k, v, *args):
        todo = partial(self.set_value, k, v)
        if (k, v) in self._scheduled_set_value:
            Clock.unschedule(self._scheduled_set_value[k, v])
        self._scheduled_set_value[k, v] = Clock.schedule_once(todo, 0)

    def init_config(self, key):
        """Set the configuration for the key to something that will always work"""
        self.proxy['_config'].setdefault(key, default_cfg)

    def set_config(self, key, option, value):
        """Set a configuration option for a key"""
        if '_config' not in self.proxy:
            newopt = dict(default_cfg)
            newopt[option] = value
            self.proxy['_config'] = {key: newopt}
        else:
            if key in self.proxy['_config']:
                self.proxy['_config'][key][option] = value
            else:
                newopt = dict(default_cfg)
                newopt[option] = value
                self.proxy['_config'][key] = newopt

    def set_configs(self, key, d):
        """Set the whole configuration for a key"""
        if '_config' in self.proxy:
            self.proxy['_config'][key] = d
        else:
            self.proxy['_config'] = {key: d}

    def iter_data(self):
        """Iterate over key-value pairs that are really meant to be displayed"""
        invalid = {
            'character',
            'name',
            'location',
            'rulebooks'
        }
        for (k, v) in self.proxy.items():
            if (
                not (isinstance(k, str) and k[0] == '_') and
                k not in invalid
            ):
                yield k, v

    def munge(self, k, v):
        """Turn a key and value into a dictionary describing a widget to show"""
        if '_config' in self.proxy and k in self.proxy['_config']:
            config = self.proxy['_config'][k].unwrap()
        else:
            config = default_cfg
        return {
            'key': k,
            'reg': self._reg_widget,
            'unreg': self._unreg_widget,
            'gett': self.proxy.__getitem__,
            'sett': self.set_value,
            'listen': self.proxy.connect,
            'unlisten': self.proxy.disconnect,
            'config': config
        }

    def upd_data(self, *args):
        """Update to match new entity data"""
        data = [self.munge(k, v) for k, v in self.iter_data()]
        self.data = sorted(data, key=lambda d: d['key'])

    def _trigger_upd_data(self, *args, **kwargs):
        if hasattr(self, '_scheduled_upd_data'):
            Clock.unschedule(self._scheduled_upd_data)
        self._scheduled_upd_data = Clock.schedule_once(self.upd_data, 0)

    def _reg_widget(self, w, *args):
        if not self.proxy:
            Clock.schedule_once(partial(self._reg_widget, w), 0)
            return

        def listen(*args):
            if w.key not in self.proxy:
                return
            if w.value != self.proxy[w.key]:
                w.value = self.proxy[w.key]
        self._listeners[w.key] = listen
        self.proxy.connect(listen)

    def _unreg_widget(self, w):
        if w.key in self._listeners:
            self.proxy.disconnect(self._listeners[w.key])


class StatListView(BaseStatListView):
    """The type of StatListView that shows only stats and their values"""


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
    text: self.true_text if self.value else self.false_text
    state: 'down' if self.value else 'normal'
<StatRowSlider>:
    Label:
        center_x: root.center_x
        y: root.center_y
        text: str(root.value)
        size: self.texture_size
<StatListView>:
    viewclass: 'StatRowListItemContainer'
    app: app
    RecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
"""
)
