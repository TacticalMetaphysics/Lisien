from __future__ import annotations

from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen

from .util import store_kv


class RulebookList(RecycleView):
	pass


class RulebookItem(BoxLayout):
	pass


class RulebooksScreen(Screen):
	toggle = ObjectProperty()


kv = """
<RulebooksScreen>:
	BoxLayout:
		orientation: 'vertical'
		BoxLayout:
			orientation: 'horizontal'
			RecycleView:
				
			RecycleView:
		Button:
			text: 'Close'
			on_release: root.toggle()
"""
store_kv(__name__, kv)
