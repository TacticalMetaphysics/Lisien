# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
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

from kivy.clock import Clock
from kivy.properties import DictProperty, ObjectProperty
from kivy.uix.recycleview import RecycleView

default_cfg = {
	"control": "readout",
	"true_text": "1",
	"false_text": "0",
	"min": 0.0,
	"max": 1.0,
}


class BaseStatListView(RecycleView):
	"""Base class for widgets showing lists of stats and their values"""

	proxy = ObjectProperty()
	"""A proxy object representing a lisien entity"""
	engine = ObjectProperty()
	"""A :class:`lisien.proxy.EngineProxy` object"""
	app = ObjectProperty()
	"""The Kivy app object"""
	_scheduled_set_value = DictProperty()

	def __init__(self, **kwargs):
		self._listeners = {}
		super().__init__(**kwargs)

	def on_proxy(self, *_):
		self.proxy.connect(self._trigger_upd_data, weak=False)
		self._trigger_upd_data()

	def del_key(self, k):
		"""Delete the key and any configuration for it"""
		if k not in self.proxy:
			raise KeyError
		del self.proxy[k]
		if "_config" in self.proxy and k in self.proxy["_config"]:
			del self.proxy["_config"][k]

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

	def _trigger_set_value(self, k, v, *_):
		todo = partial(self.set_value, k, v)
		if (k, v) in self._scheduled_set_value:
			Clock.unschedule(self._scheduled_set_value[k, v])
		self._scheduled_set_value[k, v] = Clock.schedule_once(todo, 0)

	def init_config(self, key):
		"""Set the configuration for the key to something that will always work"""
		self.proxy["_config"].setdefault(key, default_cfg)

	def set_config(self, key, option, value):
		"""Set a configuration option for a key"""
		if "_config" not in self.proxy:
			newopt = dict(default_cfg)
			newopt[option] = value
			self.proxy["_config"] = {key: newopt}
		else:
			if key in self.proxy["_config"]:
				self.proxy["_config"][key][option] = value
			else:
				newopt = dict(default_cfg)
				newopt[option] = value
				self.proxy["_config"][key] = newopt

	def set_configs(self, key, d):
		"""Set the whole configuration for a key"""
		if "_config" in self.proxy:
			self.proxy["_config"][key] = d
		else:
			self.proxy["_config"] = {key: d}

	def iter_data(self):
		"""Iterate over key-value pairs that are really meant to be displayed"""
		invalid = {"character", "name", "location", "rulebooks"}
		for k, v in self.proxy.items():
			if not (isinstance(k, str) and k[0] == "_") and k not in invalid:
				yield k, v

	def munge(self, k, v):
		"""Turn a key and value into a dictionary describing a widget to show"""
		if "_config" in self.proxy and k in self.proxy["_config"]:
			config = self.proxy["_config"][k].unwrap()
		else:
			config = default_cfg
		return {
			"key": k,
			"reg": self._reg_widget,
			"unreg": self._unreg_widget,
			"gett": self.proxy.__getitem__,
			"sett": self.set_value,
			"listen": self.proxy.connect,
			"unlisten": self.proxy.disconnect,
			"config": config,
		}

	def upd_data(self, *_):
		"""Update to match new entity data"""
		data = [self.munge(k, v) for k, v in self.iter_data()]
		self.data = sorted(data, key=lambda d: d["key"])

	def _trigger_upd_data(self, *_, **__):
		if hasattr(self, "_scheduled_upd_data"):
			Clock.unschedule(self._scheduled_upd_data)
		self._scheduled_upd_data = Clock.schedule_once(self.upd_data, 0)

	def _reg_widget(self, w, *_):
		if not self.proxy:
			Clock.schedule_once(partial(self._reg_widget, w), 0)
			return

		def listen(*_):
			if w.key not in self.proxy:
				return
			if w.value != self.proxy[w.key]:
				w.value = self.proxy[w.key]

		self._listeners[w.key] = listen
		self.proxy.connect(listen)

	def _unreg_widget(self, w):
		if w.key in self._listeners:
			self.proxy.disconnect(self._listeners[w.key])
