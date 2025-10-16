from __future__ import annotations

from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen


class RulebookList(RecycleView):
	pass


class RulebookItem(BoxLayout):
	pass


class RulebooksScreen(Screen):
	toggle = ObjectProperty()
