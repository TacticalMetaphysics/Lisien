from collections import defaultdict
from LiSE.orm import Closet
from LiSE.orm import load_closet
from gui.img import Img
from kivy.atlas import Atlas
from kivy.logger import Logger
from kivy.core.image import Image
import sqlite3
import shelve


class KivyCloset(Closet):
    def __init__(self, connector, gettext=None, **kwargs):
        super(Closet, self).__init__(connector, gettext, Logger, **kwargs)
        if 'load_img_tags' in kwargs:
            self.load_imgs_tagged(kwargs['load_img_tags'])
        self.boardhand_d = {}
        self.calendar_d = {}
        self.color_d = {}
        self.board_d = {}
        self.menu_d = {}
        self.menuitem_d = {}
        self.style_d = {}
        self.img_d = {}
        self.img_tag_d = defaultdict(set)
        self.game_piece_d = defaultdict(list)

    def remember_img_bone(self, bone, r):
        """Load the ``Img`` for ``bone`` and keep it in ``r``"""
        r[bone.name] = Img(
            closet=self,
            name=bone.name,
            texture=Image(bone.path).texture
        )

    def load_imgs(self, names):
        """Load ``Img`` objects into my ``img_d``.
        
        These contain texture data and some metadata.

        """
        r = {}

        self.select_and_set(
            (
                Img.bonetypes["img"]._null()._replace(name=n)
                for n in names
            ),
            lambda bone: self.remember_img_bone(bone, r)
        )
        self.select_and_set(
            Img.bonetypes["img_tag"]._null()._replace(name=n)
            for n in names
        )
        self.img_d.update(r)
        return r

    def get_imgs(self, imgnames):
        """Return a dictionary full of images by the given names, loading
        them as needed."""
        r = {}
        unloaded = set()
        for imgn in imgnames:
            if imgn in self.img_d:
                r[imgn] = self.img_d[imgn]
            else:
                unloaded.add(imgn)
        if len(unloaded) > 0:
            r.update(self.load_imgs(unloaded))
        return r

    def get_img(self, imgn):
        """Get an ``Img`` and return it, loading if needed"""
        return self.get_imgs([imgn])[imgn]

    def load_menus(self, names):
        """Return a dictionary full of menus by the given names, loading them
        as needed."""
        r = {}
        for name in names:
            r[name] = self.load_menu(name)
        return r

    def load_menu(self, name):
        """Load and return the named menu"""
        self.load_menu_items(name)
        return Menu(closet=self, name=name)

    def load_menu_items(self, menu):
        """Load a dictionary of menu item infos. Don't return anything."""
        self.update_keybone(
            Menu.bonetypes["menu_item"]._null()._replace(menu=menu)
        )

    def get_imgs(self, names):
        """Return a dict of ``Img`` by name, loading as needed."""
        r = {}

        def iter_unhad():
            """Put the ones I have into ``r``; for each of the rest,
            yield a ``Bone`` to match it

            """
            for name in names:
                if name in self.img_d:
                    r[name] = self.img_d[name]
                else:
                    yield Img.bonetypes["img"]._null()._replace(
                        name=name)


        self.select_and_set(iter_unhad(), remember_img_bone)
        self.select_and_set(
            Img.bonetypes["img_tag"]._null()._replace(img=n)
            for n in names if n not in self.img_tag_d)
        return r

    def load_imgs_tagged(self, tags):
        """Load ``Img``s tagged thus, return as from ``get_imgs``"""
        boned = set()
        self.select_and_set(
            [
                Img.bonetypes["img_tag"]._null()._replace(tag=tag)
                for tag in tags
            ],
            lambda bone: boned.add(bone.img)
        )
        return get_imgs(boned)

    def get_imgs_with_tags(self, tags):
        """Get ``Img``s tagged thus, return as from ``get_imgs``"""
        r = {}
        unhad = set()
        for tag in tags:
            if tag in self.img_tag_d:
                r[tag] = get_imgs(self.img_tag_d[tag])
            else:
                unhad.add(tag)
        r.update(load_imgs_tagged(unhad))
        return r

    def get_imgs_with_tag(self, tag):
        return get_imgs_with_tags([tag])[tag]

    def iter_graphic_keybones(self, names):
        """Yield the ``graphic`` and ``graphic_img`` bones
        for each name in turn."""
        for name in names:
            yield GamePiece.bonetypes[
                u"graphic"]._null()._replace(name=name)
            yield GamePiece.bonetypes[
                u"graphic_img"]._null()._replace(graphic=name)

    def create_graphic(self, name=None, offx=0, offy=0):
        """Create a new graphic, but don't put any images in it yet.
        Return its bone.

        Graphics are really just headers that group imgs
        together. They hold the offset of the img, being some
        amount to move every img on each of the x and y axes
        (default 0, 0) -- this is used so that a Spot and a
        Pawn may have the same coordinates, yet neither will
        entirely cover the other.

        Every graphic has a unique name, which will be
        assigned for you if you don't provide it. You can get
        it from the bone returned.

        """
        if name is None:
            numeral = self.get_global(u'top_generic_graphic') + 1
            self.set_global(u'top_generic_graphic', numeral)
            name = "generic_graphic_{}".format(numeral)
        grafbone = GamePiece.bonetypes[
            u"graphic"](name=name,
                        offset_x=offx,
                        offset_y=offy)
        self.set_bone(grafbone)
        return grafbone

    def add_img_to_graphic(self, imgname, grafname, layer=None):
        """Put the named img in the named graphic at the given layer,
        or the new topmost layer if unspecified.

        img must already be loaded, graphic must already exist--use
        ``create_graphic`` if it does not.

        """
        if grafname not in self.skeleton[u"graphic"]:
            raise ValueError("No such graphic: {}".format(
                grafname))
        if imgname not in self.skeleton[u"img"]:
            raise ValueError("No such img: {}".format(
                imgname))
        if layer is None:
            layer = max(self.skeleton[u"graphic_img"].keys()) + 1
        imggrafbone = GamePiece.bonetypes[
            u"graphic_img"](
            graphic=grafname,
            img=imgname,
            layer=layer
            )
        self.set_bone(imggrafbone)

    def rm_graphic_layer(grafname, layer):
        """Delete the layer from the graphic.

        The img on that layer won't be there anymore.

        """
        if grafname not in self.skeleton[u"graphic"]:
            raise ValueError(
                "No such graphic: {}".format(grafname)
            )
        if grafname not in self.skeleton[u"graphic_img"]:
            raise ValueError(
                "No imgs for graphic: {}".format(
                    grafname
                )
            )
        if layer not in self.skeleton[
                u"graphic_img"][grafname]:
            raise ValueError(
                "Graphic {} does not have layer {}".format(
                    grafname,
                    layer
                )
            )

        self.del_bone(GamePiece.bonetypes[
            u"graphic_img"]._null()._replace(
            name=grafname,
            layer=layer)
        )
        if not self.skeleton[u"graphic_img"][grafname].keys():
            self.del_bone(GamePiece.bonetypes[
                u"graphic"]._null()._replace(
                    name=grafname
                )
            )

    def load_game_pieces(names):
        """Load graphics into game pieces. Return a dictionary
        with one game piece per name."""
        self.select_keybones(iter_graphic_keybones(names))
        r = {}
        for name in names:
            r[name] = GamePiece(closet=self, graphic_name=name)
        self.game_piece_d.update(r)
        return r

    def get_game_pieces(names):
        """Return a dictionary of one game piece per name,
        loading as needed."""
        r = {}
        unhad = set()
        for name in names:
            if name in self.game_piece_d:
                r[name] = self.game_piece_d[name]
            else:
                unhad.add(name)
        r.update(self.load_game_pieces(unhad))
        return r

    def load_img_metadata(self):
        """Get all the records to do with where images are, so maybe I can
        load them later"""
        self.select_class_all(Img)

    def load_gfx_metadata(self):
        """Get all the records to do with how to put ``Img``s together into
        ``GamePiece``s"""
        self.select_class_all(GamePiece)

    def get_charsheet(self, observer, observed):
        return CharSheet(facade=self.get_facade(observer, observed))



def load_kivy_closet(
        dbfn,
        gettext=None,
        load_characters=[u'Omniscient', u'Physical'],
        load_board=('Omniscient', 'Physical'),
        load_imgs_tagged=[],
        load_gfx=[]
):
    r = KivyCloset(
        connector=sqlite3.connect(dbfn, isolation_level=None),
        shelf=shelve.open(dbfn),
        gettext=gettext
    )
    r.load_timestream()
    if load_characters:
        r.load_characters(load_characters)
    r.load_img_metadata()
    r.load_board(*load_board)
    if load_imgs_tagged:
        r.load_imgs_tagged(load_img_tags)
    if load_gfx:
        r.load_gfx_metadata()
    return r
