from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.boxlayout import BoxLayout


class RLTileBuilder(BoxLayout):
    dummypawn = ObjectProperty()

    def put_texs(self, *args):
        self.dummypawn.texs = self.ids.preview.texs


def non_none(l):
    for item in l:
        if item is not None:
            yield item


kv = """
#: import non_none ELiDE.rltile.non_none
<RLTileBuilder>:
    TextureStack:
        id: preview
        texs: non_none([pal.selection[0] if pal.selection else None for pal in (rlcloak, rlbase, rlhair, rlbeard, rlbody, rlhead, rlarm, rlhand, rlleg)])
    Pallet:
        id: rlcloak
        atlas_path: 'rltiles/cloak.atlas'
    Pallet:
        id: rlbase
        atlas_path: 'rltiles/base.atlas'
    Pallet:
        id: rlhair
        atlas_path: 'rltiles/hair.atlas'
    Pallet:
        id: rlbeard
        atlas_path: 'rltiles/beard.atlas'
    Pallet:
        id: rlbody
        atlas_path: 'rltiles/body.atlas'
    Pallet:
        id: rlhead
        atlas_path: 'rltiles/head.atlas'
    Pallet:
        id: rlarm
        atlas_path: 'rltiles/arm.atlas'
    Pallet:
        id: rlhand
        atlas_path: 'rltiles/hand.atlas'
    Pallet:
        id: rlleg
        atlas_path: 'rltiles/leg.atlas'
"""
Builder.load_string(kv)
