from __future__ import annotations

from logging import Handler

from kivy.logger import Logger
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.label import Label
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen

from .util import store_kv


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


kv = """
<LogLabel>:
	halign: 'left'
	text_size: self.width, dp(56)
<LogView>:
	viewclass: 'LogLabel'
	do_scroll_x: False
	RecycleBoxLayout:
		orientation: 'vertical'
		height: self.minimum_height
		size_hint_y: None
		default_size_hint: 1, None
		default_size: None, dp(56)
<LogScreen>:
	name: 'log'
	BoxLayout:
		orientation: 'vertical'
		LogView:
			level: 10
			size_hint_y: 0.9
		Button:
			text: 'Close'
			on_release: root.toggle()
			size_hint_y: 0.1
"""
store_kv(
	__name__,
	kv,
)
