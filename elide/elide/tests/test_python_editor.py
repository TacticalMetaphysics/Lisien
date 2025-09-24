import pytest

from lisien import Engine
from lisien.examples import sickle

from .util import idle_until


def get_actions_box(elide_app):
	app = elide_app
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
	idle_until(lambda: actions_box.store, 100, "Never got FuncStoreProxy")
	return actions_box


@pytest.mark.usefixtures("sickle_sim")
def test_show_code(elide_app):
	app = elide_app
	actions_box = get_actions_box(app)
	idle_until(
		lambda: actions_box.storelist.data, 100, "Never got actions data"
	)
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


@pytest.mark.usefixtures("kivy")
def test_create_action(prefix, elide_app):
	app = elide_app
	actions_box = get_actions_box(app)
	actions_box.editor.ids.funname.text = "new_func"
	actions_box.editor.ids.code.text = 'return "Hello, world!"'
	app.stop()
	idle_until(lambda: app.stopped, 100, "Didn't stop the app")
	with Engine(app.play_path, workers=0) as eng:
		assert hasattr(eng.action, "new_func")
