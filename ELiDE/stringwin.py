from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.factory import Factory
from .stores import StringsEditor
Factory.register('StringsEditor', cls=StringsEditor)


class StringsEdWindow(BoxLayout):
    engine = ObjectProperty()
    popover = ObjectProperty()
    dismisser = ObjectProperty()

    def add_string(self, *args):
        ed = self.ids.editor
        ed.save()
        newname = self.ids.strname.text
        if newname in self.engine.string:
            return
        self.engine.string[newname] = ed.source = ''
        assert(newname in self.engine.string)
        self.ids.strname.text = ''
        ed.name = newname
        ed._trigger_redata_reselect()


Builder.load_string("""
<StringsEdWindow>:
    orientation: 'vertical'
    StringsEditor:
        id: editor
        table: 'strings'
        store: root.engine.string if root.engine else None
        size_hint_y: 0.95
    BoxLayout:
        orientation: 'horizontal'
        size_hint_y: 0.05
        TextInput:
            id: strname
            hint_tex: 'New string name'
        Button:
            id: newstr
            text: 'New'
            on_press: root.add_string()
        Button:
            id: close
            text: 'Close'
            on_press: root.dismisser()
""")
