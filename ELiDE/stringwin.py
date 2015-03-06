from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from .stores import StringsEditor


class StringsEdWindow(BoxLayout):
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
        strings_ed = StringsEditor(
            table='strings',
            store=self.layout.engine.string,
            size_hint_y=0.95
        )
        self.add_widget(strings_ed)
        addclosestr = BoxLayout(
            orientation='horizontal',
            size_hint_y=0.05
        )
        self.add_widget(addclosestr)
        add_string_field = TextInput(
            hint_text='New string name'
        )
        addclosestr.add_widget(add_string_field)

        def add_string(*args):
            strings_ed.save()
            newname = add_string_field.text
            add_string_field.text = ''
            strings_ed.name = newname
            strings_ed.source = ''
            strings_ed._trigger_redata_reselect()

        add_string_but = Button(
            text='New',
            on_press=add_string
        )
        addclosestr.add_widget(add_string_but)

        def dismiss_str(*args):
            strings_ed.save()
            self.layout._popover.remove_widget(self)
            self.layout._popover.dismiss()
            del self.layout._popover

        close_string_but = Button(
            text='Close',
            on_press=dismiss_str
        )
        addclosestr.add_widget(close_string_but)

        self.layout._strings_ed = strings_ed
        self.layout._add_string_field = add_string_field
        self.layout._add_string_but = add_string_but
        self.layout._close_string_but = close_string_but
