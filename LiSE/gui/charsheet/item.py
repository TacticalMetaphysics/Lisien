from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ListProperty,
    ObjectProperty,
    ReferenceListProperty
)
from kivy.clock import Clock
from kivy.logger import Logger


"""Common behavior for items that go in character sheets"""


class CharSheetItemButtonBox(BoxLayout):
    csitem = ObjectProperty()


class CharSheetItem(BoxLayout):
    charsheet = ObjectProperty()
    closet = ObjectProperty()
    mybone = ObjectProperty()
    csbone = ObjectProperty()
    content = ObjectProperty()
    buttons = ListProperty()
    item_class = ObjectProperty()
    item_kwargs = DictProperty()
    widspec = ReferenceListProperty(item_class, item_kwargs)

    def __init__(self, **kwargs):
        self._trigger_set_bone = Clock.create_trigger(self._set_my_bone)
        super(CharSheetItem, self).__init__(**kwargs)
        self.finalize()

    def _set_my_bone(self, *args):
        if not self.mybone:
            return
        self.closet.set_bone(self.mybone)

    def _set_i(self, i):
        if not self.csbone:
            Logger.debug("{0}.i set before {0}.csbone; why?".format(self))
            Clock.schedule_once(lambda dt: self.set_i(i), 0)
            return
        self.csbone = self.csbone._replace(idx=self.i)
        self._trigger_set_bone()

    i = AliasProperty(
        lambda self: self.csbone.idx if self.csbone else -1,
        _set_i,
        bind=('csbone',))

    def on_height(self, *args):
        if not self.csbone:
            Logger.debug("{0}.height set before {0}.csbone; why?".format(self))
            Clock.schedule_once(self.on_height, 0)
            return
        self.csbone = self.csbone._replace(height=self.height)
        self.content.height = self.height
        self.buttons.height = self.height
        self._trigger_set_bone()

    def on_touch_move(self, touch):
        if not ('sizer_i' in touch.ud and touch.ud['sizer_i'] == self.i):
            return
        if 'wid_before' in touch.ud:
            return
        touch.ud['wid_before'] = self
        touch.ud['wid_after'] = self.charsheet.csitems[self.i+1]

    def on_touch_up(self, touch):
        if 'wid_before' not in touch.ud:
            return
        if self not in (touch.ud['wid_before'], touch.ud['wid_after']):
            return
        if not self.mybone:
            return
        self.mybone = self.mybone._replace(height=self.height)
        self.closet.set_bone(self.csbone)

    def finalize(self, *args):
        if not self.item_class and self.item_kwargs:
            Clock.schedule_once(self.finalize, 0)
            return
        self.content = self.item_class(**self.item_kwargs)
        self.add_widget(self.content)
        buttonbox = CharSheetItemButtonBox(csitem=self)
        self.add_widget(buttonbox)
        self.buttons = buttonbox.children
        buttonbox.bind(children=self.setter('buttons'))
