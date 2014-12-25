from kivy.properties import (
    StringProperty,
    ListProperty,
    ObjectProperty
)
from kivy.logger import Logger
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from .kivygarden.texturestack import ImageStack


class PalletBox(Widget):
    pallets = ListProperty()


class PawnConfigurator(BoxLayout):
    name = StringProperty()
    pallets = ListProperty()
    imgpaths = ListProperty([])
    on_press = ObjectProperty()

    def on_pallets(self, *args):
        for pallet in self.pallets:
            pallet.bind(selection=self._upd_imgpaths)

    def on_imgpaths(self, *args):
        Logger.debug(
            'PawnConfigurator: got imgpaths {}'.format(self.imgpaths)
        )
        if hasattr(self, '_imgstack'):
            self.ids.preview.remove_widget(self._imgstack)
        self._imgstack = ImageStack(
            paths=self.imgpaths,
            x=self.ids.preview.center_x - 16,
            y=self.ids.preview.y + 16
        )
        self.ids.preview.add_widget(self._imgstack)

    def _upd_imgpaths(self, *args):
        self.imgpaths = self._get_imgpaths()

    def _get_imgpaths(self):
        r = []
        for pallet in self.pallets:
            if pallet.selection:
                r.append(
                    'atlas://{}/{}'.format(
                        pallet.filename,
                        pallet.selection[0].name
                    )
                )
        return r


kv = """
#: import resource_find kivy.resources.resource_find
<SLabel@Label>:
    halign: 'center'
    size: self.texture_size
    size_hint: (None, None)
<PalletBox>:
    pallets: [base, body, arm, leg, hand1, hand2, boot, hair, beard, head]
    size_hint_y: None
    height: 6200
    SLabel:
        id: baselabel
        x: root.center_x - self.width / 2
        y: root.top - self.height
        text: 'Body'
    Pallet:
        id: base
        y: baselabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'base.atlas'
    SLabel:
        id: bodylabel
        x: root.center_x - self.width / 2
        y: base.y - self.height
        text: 'Basic clothes'
    Pallet:
        id: body
        y: bodylabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'body.atlas'
    SLabel:
        id: armlabel
        x: root.center_x - self.width / 2
        y: body.y - self.height
        text: 'Armwear'
    Pallet:
        id: arm
        y: armlabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'arm.atlas'
    SLabel:
        id: leglabel
        x: root.center_x - self.width / 2
        y: arm.y - self.height
        text: 'Legwear'
    Pallet:
        id: leg
        y: leglabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'leg.atlas'
    SLabel:
        id: hand1label
        x: root.center_x - self.width / 2
        y: leg.y - self.height
        text: 'Right hand'
    Pallet:
        id: hand1
        y: hand1label.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'hand1.atlas'
    SLabel:
        id: hand2label
        x: root.center_x - self.width / 2
        y: hand1.y - self.height
        text: 'Left hand'
    Pallet:
        id: hand2
        y: hand2label.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'hand2.atlas'
    SLabel:
        id: bootlabel
        x: root.center_x - self.width / 2
        y: hand2.y - self.height
        text: 'Boots'
    Pallet:
        id: boot
        y: bootlabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'boot.atlas'
    SLabel:
        id: hairlabel
        x: root.center_x - self.width / 2
        y: boot.y - self.height
        text: 'Hair'
    Pallet:
        id: hair
        y: hairlabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'hair.atlas'
    SLabel:
        id: beardlabel
        x: root.center_x - self.width / 2
        y: hair.y - self.height
        text: 'Beard'
    Pallet:
        id: beard
        y: beardlabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'beard.atlas'
    SLabel:
        id: headlabel
        x: root.center_x - self.width / 2
        y: beard.y - self.height
        text: 'Headwear'
    Pallet:
        id: head
        y: headlabel.y - self.minimum_height
        height: self.minimum_height
        width: root.width
        filename: 'head.atlas'
<PawnConfigurator>:
    id: pawnconf
    orientation: 'vertical'
    pallets: palletbox.pallets
    BoxLayout:
        id: namer
        size_hint_y: 0.1
        y: root.top - self.height
        width: root.width
        height: 50
        Button:
            text: 'Accept'
            on_press: root.on_press()
        Widget:
            id: preview
        TextInput:
            id: namer
            hint_text: 'enter a name'
    ScrollView:
        PalletBox:
            id: palletbox
"""
Builder.load_string(kv)
