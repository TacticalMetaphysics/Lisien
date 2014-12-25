from kivy.properties import (
    StringProperty,
    ListProperty,
    ObjectProperty
)
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from .kivygarden.texturestack import ImageStack


class Configurator(BoxLayout):
    prefix = StringProperty()
    pallets = ListProperty()
    imgpaths = ListProperty([])
    cb = ObjectProperty()

    def on_prefix(self, *args):
        self.ids.textbox.text = self.prefix

    def on_imgpaths(self, *args):
        if hasattr(self, '_imgstack'):
            self.ids.preview.remove_widget(self._imgstack)
        self._imgstack = ImageStack(
            paths=self.imgpaths,
            x=self.ids.preview.center_x - 16,
            y=self.ids.preview.center_y - 16
        )
        self.ids.preview.add_widget(self._imgstack)

    def on_pallets(self, *args):
        for pallet in self.pallets:
            pallet.bind(selection=self._upd_imgpaths)

    def _upd_imgpaths(self, *args):
        imgpaths = []
        for pallet in self.pallets:
            if pallet.selection:
                for selected in pallet.selection:
                    imgpaths.append(
                        'atlas://{}/{}'.format(
                            pallet.filename,
                            selected.name
                        )
                    )
        if imgpaths:
            self.imgpaths = imgpaths
        else:
            self.imgpaths = self._default_imgpaths()


configurator_kv = """
<Configurator>:
    textbox: textbox
    BoxLayout:
        Button:
            text: 'OK'
            on_press: root.cb()
        Widget:
            id: preview
        TextInput:
            id: textbox
            multiline: False
            hint_text: 'Enter name prefix'
"""
Builder.load_string(configurator_kv)


class ConfigDialog(BoxLayout):
    prefix = StringProperty()
    imgpaths = ListProperty()
    layout = ObjectProperty()

    def pressed(self):
        self.prefix = self.ids.configurator.prefix
        self.imgpaths = self.ids.configurator.imgpaths


class SpotConfigDialog(ConfigDialog):
    def pressed(self):
        super().pressed()
        self.layout.toggle_spot_config()

    def _default_imgpaths(self):
        return ['atlas://base.atlas/unseen']


class PawnConfigDialog(ConfigDialog):
    def pressed(self):
        super().pressed()
        self.layout.toggle_pawn_config()

    def _default_imgpaths(self):
        return ['orb.png']


slabel_kv = """
<SLabel@Label>:
    halign: 'center'
    size: self.texture_size
    size_hint: (None, None)
"""
Builder.load_string(slabel_kv)


pawn_configurator_kv = """
<PawnConfigDialog>:
    orientation: 'vertical'
    Configurator:
        id: configurator
        prefix: root.prefix
        pallets: [base, body, arm, leg, hand1, hand2, boot, hair, beard, head]
        cb: root.pressed
        size_hint_y: None
        height: 50
    ScrollView:
        PalletBox:
            pallets: [base, body, arm, leg, hand1, hand2, boot, hair, beard, head]
            size_hint_y: None
            height: 6200
            SLabel:
                id: baselabel
                x: root.center_x - self.width / 2
                y: 6200 - self.height
                text: 'Body'
            Pallet:
                id: base
                y: baselabel.y - self.minimum_height
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
                width: root.width
                filename: 'head.atlas'
"""
Builder.load_string(pawn_configurator_kv)


spot_configurator_kv = """
<SpotConfigDialog>:
    orientation: 'vertical'
    Configurator:
        id: configurator
        prefix: root.prefix
        pallets: [pallet]
        cb: root.pressed
        size_hint_y: None
        height: 50
    Pallet:
        id: pallet
        y: configurator.y - self.minimum_height
        width: root.width
        filename: 'dungeon.atlas'
"""
Builder.load_string(spot_configurator_kv)
