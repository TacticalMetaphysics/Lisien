from texturestack import TextureStack
from kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    AliasProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
    ReferenceListProperty)


class ImgStack(TextureStack):
    closet = ObjectProperty()
    imgs = ListProperty()

    def on_imgs(self, *args):
        for i in xrange(0, len(self.imgs)):
            if self.imgs[i].texture not in self.texture_rectangles:
                self[i] = self.imgs[i].texture

    def clear(self):
        self.imgs = []
        super(ImgStack, self).clear()


class GamePiece(ImgStack):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ("graphic", {
            "columns": {
                "name": "text not null",
                "offset_x": "integer not null default 0",
                "offset_y": "integer not null default 0"},
            "primary_key": ("name",)}),
        ("graphic_img", {
            "columns": {
                "graphic": "text not null",
                "layer": "integer not null default 0",
                "img": "text not null"},
            "primary_key": ("graphic", "layer"),
            "foreign_keys": {
                "graphic": ("graphic", "name"),
                "img": ("img", "name")},
            "checks": ["layer>=0"]})]
    graphic_name = StringProperty()
    _touch = ObjectProperty(None, allownone=True)
    imgs = ListProperty()
    offset_x = NumericProperty()
    offset_y = NumericProperty()
    graphic_bone = AliasProperty(
        lambda self: self._get_graphic_bone(),
        lambda self, v: None,
        bind=('graphic_name', 'closet'))
    offset = ReferenceListProperty(offset_x, offset_y)

    def _get_graphic_bone(self):
        if not (self.closet and self.graphic_name):
            return
        if self.graphic_name not in self.closet.skeleton[u"graphic"]:
            self.closet.load_game_piece(self.graphic_name)
        assert(self.graphic_name in self.closet.skeleton[u"graphic"])
        return self.closet.skeleton[u"graphic"][self.graphic_name]

    def __init__(self, **kwargs):
        kwargs['size_hint'] = (None, None)
        if 'board' in kwargs and 'closet' not in kwargs:
            kwargs['closet'] = kwargs['board'].host.closet
        super(GamePiece, self).__init__(**kwargs)
