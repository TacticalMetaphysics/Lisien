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


class Sizer(ClosetButton):
    spacer = ObjectProperty()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud['charsheet'] = self.charsheet
            touch.ud['sizer'] = self
            touch.grab(self)
            self.prior_y = self.y
            self.state = 'down'
            touch.ud['sizer_i'] = self.spacer.i
            return True

    def on_touch_move(self, touch):
        if 'sizer' not in touch.ud or touch.ud['sizer'] is not self:
            touch.ungrab(self)
            self.state = 'normal'
            return
        self.center_y = touch.pos
        return True

    def on_touch_up(self, touch):
        if 'sizer' not in touch.ud or touch.ud['sizer'] is not self:
            touch.ungrab(self)
            self.state = 'normal'
            return
        self.state = 'normal'
        return True


class Spacer(BoxLayout):
    csitem = ObjectProperty()
    charsheet = AliasProperty(
        lambda self: self.csitem.charsheet,
        lambda self, v: None,
        bind=('csitem',))
    i = NumericProperty()
    sizer = ObjectProperty()
    adder = ObjectProperty()

    def __init__(self, **kwargs):
        _ = lambda x: x
        kwargs['size_hint_y'] = None
        kwargs['height'] = 30
        kwargs['sizer'] = Sizer(
            spacer=self,
            size_hint_x=0.2)
        kwargs['adder'] = ClosetButton(
            closet=self.charsheet.character.closet,
            symbolic=True,
            stringname=_('@add'),
            fun=self.charsheet.add_item,
            arg=kwargs['i'],
            size_hint_y=0.8)
        super(Sizer, self).__init__(**kwargs)
        self.add_widget(self.sizer)
        self.add_widget(self.adder)


class CharSheetItemButtonBox(BoxLayout):
    csitem = ObjectProperty()


class CharSheetItem(BoxLayout):
    csbone = ObjectProperty()
    content = ObjectProperty()
    spacer = ObjectProperty()
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
        self._trigger_set_bone = Clock.create_trigger(
            lambda dt: self.closet.set_bone(self.csbone))
        kwargs['orientation'] = 'vertical'
        super(CharSheetItem, self).__init__(**kwargs)
        self.finalize()

    def upd_height(self, *args):
        if not self.csbone:
            Logger.debug("{0}.height set before {0}.csbone; why?".format(self))
            Clock.schedule_once(self.upd_height, 0)
            return
        elif self.i - 1 not in self.charsheet.i2wid:
            Logger.debug("{0} seems to be the 0th charsheet item, but {0}.i=1".format(self))
            Clock.schedule_once(self.upd_height, 0)
            return
        if self.sizer:
            self.height = self.sizer.top - self.y
            dh = self.height - self.csbone.height
            wid_before = self.charsheet.i2wid[self.i-1]
            wid_before.y += dh
            wid_before.height -= dh
        if self.height != self.csbone.height:
            self.csbone = self.csbone._replace(height=self.height)
            self.content.height = self.height
            self.buttons.height = self.height
            self._trigger_set_bone()

    def on_touch_move(self, touch):
        if not ('sizer_i' in touch.ud and touch.ud['sizer_i'] == self.i):
            return
        if 'wid_before' in touch.ud:
            return
        # crappy hack to get the widget seen before me in the charsheet
        touch.ud['wid_before'] = self.charsheet.i2wid[self.i-1]
        touch.ud['wid_after'] = self

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
        _ = lambda x: x
        if not self.item_class and self.item_kwargs:
            Clock.schedule_once(self.finalize, 0)
            return
        self.middle = BoxLayout()
        self.content = self.item_class(**self.item_kwargs)
        buttonbox = BoxLayout(
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
            buttonbox.add_widget(button)
        self.middle.add_widget(self.content)
        self.middle.add_widget(buttonbox)
        if self.csbone.idx > 0:
            self.spacer = Spacer(
                csitem=self,
                i=self.csbone.idx)
            self.add_widget(self.spacer)
            self.sizer.bind(top=self.upd_height)
        self.add_widget(self.middle)
        self.bind(height=self.upd_height)
        self.charsheet.i2wid[self.i] = self
