from kivy.lang import Builder
from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    StringProperty,
    ListProperty
)
from kivy.uix.screenmanager import Screen

class FuncsEdScreen(Screen):
    funcs_ed = ObjectProperty()
    table = StringProperty()
    store = ObjectProperty()
    data = ListProperty()
    font_name = StringProperty('Roboto-Regular')
    font_size = NumericProperty(12)
    toggle = ObjectProperty()

    def add_func(self, *args):
        newname = self.ids.newfuncname.text
        self.ids.newfuncname.text = ''
        self.funcs_ed.save()
        self.funcs_ed.name = newname
        self.ids.funcs_ed.source = 'def {}({}):\n    pass'.format(
            newname,
            ', '.join(self.funcs_ed.params)
        )
        self.funcs_ed._trigger_redata_reselect()

    def dismiss(self, *args):
        self.funcs_ed._trigger_save()
        self.toggle()

    def subjtyp(self, val):
        if val == 'character':
            self.ids.char.active = True
        elif val == 'thing':
            self.ids.thing.active = True
        elif val == 'place':
            self.ids.place.active = True
        elif val == 'portal':
            self.ids.port.active = True

    def setchar(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'character'

    def setthing(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'thing'

    def setplace(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'place'

    def setport(self, active):
        if not active:
            return
        self.funcs_ed.subject_type = 'portal'


Builder.load_string("""
<FuncsEdScreen>:
    name: 'funcs'
    funcs_ed: funcs_ed
    data: funcs_ed.data if funcs_ed else []
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            id: subj_type_sel
            size_hint_y: 0.05
            BoxLayout:
                CheckBox:
                    id: char
                    group: 'subj_type'
                    size_hint_x: 0.05
                    on_active: root.setchar(self.active)
                Label:
                    text: 'Character'
                    size_hint_x: 0.95
            BoxLayout:
                CheckBox:
                    id: thing
                    group: 'subj_type'
                    size_hint_x: 0.05
                    on_active: root.setthing(self.active)
                Label:
                    text: 'Thing'
                    size_hint_x: 0.95
            BoxLayout:
                CheckBox:
                    id: place
                    group: 'subj_type'
                    size_hint_x: 0.05
                    on_active: root.setplace(self.active)
                Label:
                    text: 'Place'
                    size_hint_y: 0.95
            BoxLayout:
                CheckBox:
                    id: port
                    group: 'subj_type'
                    size_hint_x: 0.05
                    on_active: root.setport(self.active)
                Label:
                    text: 'Portal'
                    size_hint_y: 0.95
        FuncsEditor:
            id: funcs_ed
            size_hint_y: 0.9
            table: root.table
            store: root.store
            font_name: root.font_name
            font_size: root.font_size
            on_subject_type: root.subjtyp(self.subject_type)
        BoxLayout:
            size_hint_y: 0.05
            TextInput:
                id: newfuncname
                hint_text: 'New function name'
            Button:
                text: '+'
                on_press: root.add_func()
            Button:
                text: 'Close'
                on_press: root.dismiss()
""")
