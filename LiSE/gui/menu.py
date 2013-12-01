# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import os
from re import match, compile

from kivy.properties import (
    BooleanProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
    NumericProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

from kivybits import SaveableWidgetMetaclass


"""Menus that are actually just stacks of buttons.

I'll probably change over to drop menus at some point."""


ON_CLICK_RE = compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuButton(Button):
    """A button that does something. It is to be placed by a Menu."""
    symbolic = BooleanProperty()
    oncl = ObjectProperty()
    fargs = ListProperty()

    def on_press(self):
        self.oncl(self, *self.fargs)


class Menu(BoxLayout):
    """A stack of buttons that call functions assigned in the database."""
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ('menu_item',
         {'menu': 'text not null',
          'idx': 'integer not null',
          'text': 'text',
          'on_click': 'text not null',
          'closer': 'boolean not null default 1',
          'symbolic': 'boolean not null default 0'},
         ('menu', 'idx'),
         {},
         [])]
    closet = ObjectProperty()
    name = StringProperty()
    completedness = NumericProperty(0)

    def __unicode__(self):
        return self.name

    def __str__(self):
        return str(self.name)

    def on_closet(self, i, v):
        self.completedness += 1

    def on_name(self, i, v):
        self.completedness += 1

    def on_parent(self, i, v):
        self.completedness += 1

    def on_completedness(self, i, v):
        if v == 3:
            self.finalize()

    def finalize(self):
        """Create one ``MenuButton`` for each bone under my name in the
        Skeleton.

        Each ``MenuButton``'s function will be taken from the dictionary
        ``menu_cbs`` in my ``Closet``.

        """
        for bone in self.closet.skeleton["menu_item"][
                unicode(self)].iterbones():
            ocmatch = match(ON_CLICK_RE, bone.on_click)
            if ocmatch is not None:
                (ocfn, ocargs) = ocmatch.groups()
                (on_click_fun, ARG_RE) = self.closet.menu_cbs[ocfn]
                ocargm = match(ARG_RE, ocargs)
                if ocargm is None:
                    fargs = []
                else:
                    fargs = list(ocargm.groups())

                if bone.symbolic:
                    fontname = os.sep.join([
                        self.closet.lisepath, "gui",
                        "assets", "Entypo.ttf"])
                else:
                    fontname = 'DroidSans'

                it = MenuButton(
                    symbolic=bone.symbolic,
                    oncl=on_click_fun,
                    fargs=fargs,
                    text=self.closet.get_text(bone.text),
                    font_name=fontname)

                def retext(*args):
                    """When the ``Bone``'s string changes, update the
                    ``MenuButton``'s text to match."""
                    it.text = self.closet.get_text(bone.text)
                self.closet.register_text_listener(bone.text, retext)
                self.add_widget(it)
