from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from .statgrid import StatListViewConfigurator


class StatWindow(BoxLayout):
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
        cfg = StatListViewConfigurator(
            time=self.layout.time,
            size_hint_y=0.95
        )
        newstatkey = TextInput(
            multiline=False,
            write_tab=False,
            hint_text='New stat'
        )
        newstatval = TextInput(
            multiline=False,
            write_tab=False,
            hint_text='Value'
        )
        newstatbut = Button(
            text='+',
            on_press=lambda inst: self.layout.set_remote_value(
                cfg.remote,
                newstatkey.text,
                newstatval.text
            )
        )
        close_cfg_but = Button(
            text='Close',
            on_press=lambda inst: self.layout.toggle_stat_cfg()
        )
        buttons = BoxLayout(size_hint_y=0.05)
        buttons.add_widget(newstatkey)
        buttons.add_widget(newstatval)
        buttons.add_widget(newstatbut)
        buttons.add_widget(close_cfg_but)
        self.add_widget(buttons)
        self.add_widget(cfg)

        self.layout._stat_cfg = cfg
        self.layout._newstatkey = newstatkey
        self.layout._newstatval = newstatval
        self.layout._newstatbut = newstatbut
        self.layout._close_stat_cfg_but = close_cfg_but
