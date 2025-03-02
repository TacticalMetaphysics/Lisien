from lisien import Engine
from lisien.examples import sickle

from .util import ELiDEAppTest, idle_until


class PythonEditorTest(ELiDEAppTest):
	def setUp(self):
		super().setUp()
		with Engine(self.prefix) as eng:
			sickle.install(eng)

	def _get_actions_box(self):
		app = self.app
		idle_until(
			lambda: hasattr(app, "mainscreen")
			and app.mainscreen.mainview
			and app.mainscreen.statpanel
			and hasattr(app.mainscreen, "gridview")
		)
		app.funcs.toggle()
		idle_until(
			lambda: "actions" in app.funcs.ids, 100, "Never got actions box"
		)
		actions_box = app.funcs.ids.actions
		idle_until(lambda: actions_box.editor, 100, "Never got FuncEditor")
		idle_until(lambda: actions_box.storelist, 100, "Never got StoreList")
		idle_until(
			lambda: actions_box.storelist.data, 100, "Never got StoreList data"
		)
		return actions_box


class TestShowCode(PythonEditorTest):
	def test_show_code(self):
		app = self.app
		self.Window.add_widget(app.build())
		actions_box = self._get_actions_box()
		last = actions_box.storelist.data[-1]["name"]
		actions_box.storelist.selection_name = last
		idle_until(
			lambda: "funname" in actions_box.editor.ids,
			100,
			"Never got function input widget",
		)
		idle_until(
			lambda: actions_box.editor.ids.funname.hint_text,
			100,
			"Never got function name",
		)
		idle_until(
			lambda: "code" in actions_box.editor.ids,
			100,
			"Never got code editor widget",
		)
		idle_until(
			lambda: actions_box.editor.ids.code.text,
			100,
			"Never got source code",
		)


class TestCreateAction(PythonEditorTest):
	def test_create_action(self):
		app = self.app
		self.Window.add_widget(app.build())
		actions_box = self._get_actions_box()
		actions_box.editor.ids.funname.text = "new_func"
		actions_box.editor.ids.code.text = 'return "Hello, world!"'
		app.stop()
		with Engine(self.prefix) as eng:
			assert hasattr(eng.action, "new_func")
