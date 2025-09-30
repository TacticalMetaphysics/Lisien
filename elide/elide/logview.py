from __future__ import annotations

from logging import Handler

from kivy.logger import Logger
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen

from elide.util import load_kv


class LogViewHandler(Handler):
	def __init__(self, logview: LogView, level=0):
		self.logview = logview
		super().__init__(level)

	def emit(self, record):
		if hasattr(record, "message"):
			msg = record.message
		elif hasattr(record, "msg"):
			msg = record.msg
		else:
			Logger.warning("Can't format log record")
			return
		self.logview.data.append({"text": str(msg)})


class LogLabel(Label):
	pass


class LogView(RecycleView):
	"""View of a log, not necessarily in a file"""

	level = NumericProperty(10)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._handler = LogViewHandler(self, level=int(self.level))
		Logger.addHandler(self._handler)

	def on_level(self, *_):
		if not hasattr(self, "_handler"):
			return
		self._handler.level = int(self.level)

	def on_data(self, *_):
		self.scroll_y = 0.0


class LogScreen(Screen):
	toggle = ObjectProperty()

	def __init__(self, **kw):
		load_kv("logview.kv")
		super().__init__(**kw)
