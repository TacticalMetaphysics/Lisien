from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty
)
from kivy.clock import Clock
from kivy.logger import Logger

from LiSE.gui.kivybits import ClosetButton


class CharSheetItemButtonBox(BoxLayout):
    csitem = ObjectProperty()


class CharSheetItem(BoxLayout):
    csbone = ObjectProperty()
    content = ObjectProperty()
    sizer = ObjectProperty(None, allownone=True)
    adder = ObjectProperty(None, allownone=True)
    asbox = ObjectProperty(None, allownone=True)
    buttons = ListProperty()
    middle = ObjectProperty()
    item_class = ObjectProperty()
    item_kwargs = DictProperty()
    widspec = ReferenceListProperty(item_class, item_kwargs)
    charsheet = AliasProperty(
        lambda self: self.item_kwargs['charsheet']
        if self.item_kwargs else None,
        lambda self, v: None,
        bind=('item_kwargs',))
    closet = AliasProperty(
        lambda self: self.item_kwargs['charsheet'].character.closet
        if self.item_kwargs else None,
        lambda self, v: None,
        bind=('item_kwargs',))
    mybone = AliasProperty(
        lambda self: self.item_kwargs['mybone']
        if self.item_kwargs and 'mybone' in self.item_kwargs
        else None,
        lambda self, v: None,
        bind=('item_kwargs',))
    i = AliasProperty(
        lambda self: self.csbone.idx if self.csbone else -1,
        lambda self, v: None,
        bind=('csbone',))

    def __init__(self, **kwargs):
        self._trigger_set_bone = Clock.create_trigger(self.set_bone)
        kwargs['orientation'] = 'vertical'
        kwargs['size_hint_y'] = None
        super(CharSheetItem, self).__init__(**kwargs)
        self.finalize()

    def on_touch_down(self, touch):
        if self.sizer.collide_point(*touch.pos):
            touch.ud['sizer'] = self.sizer
            return True
        return super(CharSheetItem, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        if not ('sizer' in touch.ud and touch.ud['sizer'] is self.sizer):
            return
        touch.push()
        touch.apply_transform_2d(self.parent.parent.to_local)
        b = touch.y - self.sizer.height / 2
        h = self.top - b
        self.y = b
        self.height = h
        touch.pop()

    def set_bone(self, *args):
        if self.csbone:
            self.closet.set_bone(self.csbone)

    def finalize(self, *args):
        _ = lambda x: x
        if not (self.item_class and self.item_kwargs):
            Clock.schedule_once(self.finalize, 0)
            return
        self.middle = BoxLayout()
        self.content = self.item_class(**self.item_kwargs)
        self.buttonbox = BoxLayout(
            orientation='vertical',
            size_hint_x=0.2)
        self.buttons = [ClosetButton(
            closet=self.closet,
            symbolic=True,
            stringname=_('@del'),
            fun=self.charsheet.del_item,
            arg=self.i)]
        if self.i > 0:
            self.buttons.insert(0, ClosetButton(
                closet=self.closet,
                symbolic=True,
                stringname=_('@up'),
                fun=self.charsheet.move_it_up,
                arg=self.i,
                size_hint_y=0.1))
            if self.i+1 < len(self.charsheet.csitems):
                self.buttons.append(ClosetButton(
                    closet=self.closet,
                    symbolic=True,
                    stringname=_('@down'),
                    fun=self.charsheet.move_it_down,
                    arg=self.i,
                    size_hint_y=0.1))
        for button in self.buttons:
            self.buttonbox.add_widget(button)
        self.middle.add_widget(self.content)
        self.middle.add_widget(self.buttonbox)
        self.add_widget(self.middle)
        self.sizer = ClosetButton(
            closet=self.charsheet.character.closet,
            symbolic=True,
            stringname=_('@ud'),
            size_hint_x=0.2)
        self.adder = ClosetButton(
            closet=self.charsheet.character.closet,
            symbolic=True,
            stringname=_('@add'),
            fun=self.charsheet.add_item,
            arg=self.i+1,
            size_hint_x=0.8)
        self.asbox = BoxLayout(
            size_hint_y=None,
            height=40)
        self.asbox.add_widget(self.sizer)
        self.asbox.add_widget(self.adder)
        self.add_widget(self.asbox)
        self.height = self.csbone.height
        self.content.height = self.buttonbox.height = (
            self.height - self.asbox.height)
        self.charsheet.i2wid[self.i] = self
