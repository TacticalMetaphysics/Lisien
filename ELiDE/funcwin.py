from .stores import FuncsEditor
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput


class FuncsEdWindow(BoxLayout):
    layout = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

    def on_layout(self, *args):
        if self.layout is None:
            return
        if self.canvas is None:
            Clock.schedule_once(self.on_layout, 0)
            return

        funcs_ed = FuncsEditor(
            size_hint_y=0.9
        )

        def setchar(box, active):
            if active:
                funcs_ed.params = ['engine', 'character']

        def setthing(box, active):
            if active:
                funcs_ed.params = ['engine', 'character', 'thing']

        def setplace(box, active):
            if active:
                funcs_ed.params = ['engine', 'character', 'place']

        def setport(box, active):
            if active:
                funcs_ed.params = [
                    'engine', 'character', 'origin', 'destination'
                ]

        subj_type_sel = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.05
        )
        self.add_widget(subj_type_sel)

        charsel = BoxLayout()
        char = CheckBox(group='subj_type', size_hint_x=0.05)
        char.bind(active=setchar)
        charsel.add_widget(char)
        charl = Label(text='Character', size_hint_x=0.95)
        charsel.add_widget(charl)
        subj_type_sel.add_widget(charsel)

        thingsel = BoxLayout()
        thing = CheckBox(group='subj_type', size_hint_x=0.05)
        thing.bind(active=setthing)
        thingsel.add_widget(thing)
        thingl = Label(text='Thing', size_hint_x=0.95)
        thingsel.add_widget(thingl)
        subj_type_sel.add_widget(thingsel)

        placesel = BoxLayout()
        place = CheckBox(group='subj_type', size_hint_x=0.05)
        place.bind(active=setplace)
        placesel.add_widget(place)
        placel = Label(text='Place', size_hint_x=0.95)
        placesel.add_widget(placel)
        subj_type_sel.add_widget(placesel)

        portsel = BoxLayout()
        port = CheckBox(group='subj_type', size_hint_x=0.05)
        port.bind(active=setport)
        portsel.add_widget(port)
        portl = Label(text='Portal', size_hint_x=0.95)
        portsel.add_widget(portl)
        subj_type_sel.add_widget(portsel)

        def subjtyp(inst, val):
            if val == 'character':
                char.active = True
            elif val == 'thing':
                thing.active = True
            elif val == 'place':
                place.active = True
            elif val == 'portal':
                port.active = True

        funcs_ed.bind(subject_type=subjtyp)
        self.add_widget(funcs_ed)

        addclosefunc = BoxLayout(orientation='horizontal', size_hint_y=0.05)
        self.add_widget(addclosefunc)
        newfuncname = TextInput(hint_text='New function name')
        addclosefunc.add_widget(newfuncname)

        def add_func(*args):
            newname = newfuncname.text
            newfuncname.text = ''
            funcs_ed.save()
            funcs_ed.name = newname
            funcs_ed.source = 'def {}({}):\n    pass'.format(
                newname,
                ', '.join(funcs_ed.params)
            )
            funcs_ed._trigger_redata_reselect()

        addfuncbut = Button(
            text='New',
            on_press=add_func
        )
        addclosefunc.add_widget(addfuncbut)

        def dismiss_func(*args):
            funcs_ed._trigger_save()
            self.layout._popover.remove_widget(self)
            self.layout._popover.dismiss()
            del self.layout._popover

        closefuncbut = Button(text='Close', on_press=dismiss_func)
        addclosefunc.add_widget(closefuncbut)

        self.layout._funcs_ed = funcs_ed
