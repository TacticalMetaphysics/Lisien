from LiSE.gui.texturestack import TextureStack
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.properties import (
    AliasProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
    ReferenceListProperty)
from kivy.clock import Clock


class ImgStack(TextureStack):
    """Several Img superimposed upon one another.

    LiSE uses tile graphics, and tile graphics are often made
    deliberately incomplete: perhaps there is a generic sprite for a
    human, which is then made to represent a specific person by
    layering different skins on top, and then clothes may be layered
    on top of that. ImgStack lets you assemble such a compound sprite
    easily, using a list-like interface, and thereafter treat the
    ImgStack much as you would a single image.

    """
    closet = ObjectProperty()
    imgs = ListProperty()

    def on_imgs(self, *args):
        if not self.canvas:
            Clock.schedule_once(self.on_imgs, 0)
            return
        super(ImgStack, self).clear()
        for img in self.imgs:
            self.append(img.texture)

    def clear(self):
        self.imgs = []
        super(ImgStack, self).clear()


class GamePiece(ImgStack):
    """A saveable ImgStack with optional offsets of its own."""
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
    graphic_bone = AliasProperty(
        lambda self: self._get_graphic_bone(),
        lambda self, v: None,
        bind=('graphic_name',))
    graphic_img_bones = AliasProperty(
        lambda self: self._get_graphic_img_bones(),
        lambda self, v: None,
        bind=('graphic_name',))
    offset_x = NumericProperty()
    offset_y = NumericProperty()
    offset = ReferenceListProperty(offset_x, offset_y)

    def __init__(self, **kwargs):
        kwargs['size_hint'] = (None, None)
        if 'board' in kwargs and 'closet' not in kwargs:
            kwargs['closet'] = kwargs['board'].host.closet
        super(GamePiece, self).__init__(**kwargs)
        self.bind(graphic_name=self._get_graphic_bone)
        self._get_graphic_bone()

    def _get_graphic_bone(self, *args):
        if not (self.closet and self.graphic_name):

            return
        if self.graphic_name not in self.closet.skeleton[u"graphic"]:
            self.closet.load_game_piece(self.graphic_name)
        if self.graphic_name not in self.closet.skeleton[u"graphic"]:
            raise KeyError("Graphic {} not loaded".format(self.graphic_name))
        return self.closet.skeleton[u"graphic"][self.graphic_name]

    def _get_graphic_img_bones(self, *args):
        if not (self.closet and self.graphic_name):
            return

    def upd_texs(self, *args):
        super(GamePiece, self).upd_texs()
        self.size = (self.width + self.offset_x, self.height + self.offset_y)
