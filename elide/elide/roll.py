from kivy.properties import ObjectProperty
from kivy.uix.modalview import ModalView
from kivy.uix.tabbedpanel import TabbedPanel
from .util import store_kv


class Roller(TabbedPanel):
	engine = ObjectProperty()


class RollerView(ModalView):
	pass


kv = """
<Roller>:
	do_default_tab: False
	TabbedPanelItem:
		text: 'Polyhedral'
	TabbedPanelItem:
		text: 'Percentile'
	TabbedPanelItem:
		text: 'Pool'
<RollerView>:
	BoxLayout:
		orientation: 'vertical'
		Roller:
			engine: app.engine
			size_hint_y: 0.9
		Button:
			text: 'Close'
			size_hint_y: 0.1
			on_release: root.dismiss()
"""
store_kv(__name__, kv)
