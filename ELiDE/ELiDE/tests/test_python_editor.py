from LiSE import Engine
from LiSE.examples import kobold
from .util import ELiDEAppTest, window_with_widget, idle_until


class TestPythonEditor(ELiDEAppTest):
    def setUp(self):
        super().setUp()
        with Engine(self.prefix) as eng:
            kobold.inittest(eng)

    def test_show_code(self):
        app = self.app
        win = window_with_widget(app.build())
        idle_until(lambda: hasattr(app, 'mainscreen') and app.mainscreen.mainview and app.mainscreen.statpanel and hasattr(app.mainscreen, 'gridview'))
        app.funcs.toggle()
        idle_until(lambda: 'actions' in app.funcs.ids, 100, 'Never got actions box')
        actions_box = app.funcs.ids.actions
        idle_until(lambda: actions_box.editor, 100, 'Never got FuncEditor')
        idle_until(lambda: 'funname' in actions_box.editor.ids, 100, 'Never got function input widget')
        idle_until(lambda: actions_box.editor.ids.funname.text, 100, "Never got function name")
        idle_until(lambda: 'code' in actions_box.editor.ids, 100, "Never got code editor widget")
        idle_until(lambda: actions_box.editor.ids.code.text, 100, "Never got source code")
